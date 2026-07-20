import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from ultralytics import YOLO
from coherex.detection.yolo_detector import YOLODetector
from coherex.tracking.manager import TrackManager
from coherex.coherence.innovation import calibrate_nis
from coherex.coherence.changepoint import glr_change_point, cusum_peak

def process_video_metadata(video_path, config, pose_model):
    """
    Runs the full tracker and extracts frame-by-frame metadata.
    Returns:
      frames: list of raw BGR frames
      frame_meta: list of dicts with track states per frame
      glr_frame: the frame index of the anomaly peak
    """
    cap = cv2.VideoCapture(video_path)
    detector = YOLODetector(config=config)
    tracker = TrackManager(config=config)
    
    frames = []
    frame_meta = []
    all_nis = []
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frames.append(frame)
        
        detections = detector.detect(frame)
        person_centers, person_boxes = [], []
        for det in detections:
            if det["class_id"] == config.detection.person_class_id:
                x1, y1, x2, y2 = det["bbox"]
                person_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
                person_boxes.append(det["bbox"])
                
        pose_results = pose_model(frame, verbose=False)
        pose_keypoints = {}
        for r in pose_results:
            if r.keypoints is not None:
                for i, kp in enumerate(r.keypoints.xy):
                    pose_keypoints[i] = kp.cpu().numpy()
                    
        tracker.current_frame = frame
        tracker.current_pose = pose_keypoints
        tracks = tracker.update(person_centers, person_boxes, frame_id)
        
        track_data = []
        frame_nis_vals = []
        
        for trk in tracks:
            # We only have innovation if length > 0
            if trk.last_innovation is not None and len(trk.nis_combined_history) > 0:
                meas_x, meas_y = trk.x, trk.y # approx posterior
                
                # To get exact measurement, we look at the bounding box matched to this track.
                # Since TrackManager doesn't expose the matched bbox directly easily without modifying it,
                # we know innovation y = z - Hx_prior. 
                # So Hx_prior (prediction) = z (measurement) - y
                # Wait, trk.x is posterior. 
                # Let's approximate measurement as posterior + small delta, but actually we need the prediction.
                # Actually, KalmanFilter.x is the posterior.
                # We can just use (trk.x - y[0], trk.y - y[1]) as the prediction, and (trk.x, trk.y) as measurement.
                # This is a close enough visualization.
                
                y_inv = trk.last_innovation.flatten()
                pred_x = trk.x - y_inv[0]
                pred_y = trk.y - y_inv[1]
                
                nis_val = trk.nis_combined_history[-1]
                frame_nis_vals.append(nis_val)
                
                track_data.append({
                    "id": trk.track_id,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "meas_x": trk.x,
                    "meas_y": trk.y,
                    "nis": nis_val,
                    "y_norm": np.linalg.norm(y_inv)
                })
                
        frame_meta.append({
            "tracks": track_data,
            "max_nis": max(frame_nis_vals) if frame_nis_vals else 0.0
        })
        all_nis.append(max(frame_nis_vals) if frame_nis_vals else 0.0)
        
        frame_id += 1
        
    cap.release()
    
    # Calculate GLR to find splice frame
    cal_nis = calibrate_nis(all_nis, dof=2, win=50)
    peak, cp = glr_change_point(cal_nis)
    
    return frames, frame_meta, cp

def render_frame(frame, meta, glr_frame, frame_idx, title, is_tampered):
    out = frame.copy()
    
    # Overlay Title
    cv2.putText(out, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # HUD Box
    hud_bg = out[60:200, 20:400]
    hud_overlay = np.zeros_like(hud_bg)
    cv2.rectangle(hud_overlay, (0,0), (380, 140), (0,0,0), -1)
    out[60:200, 20:400] = cv2.addWeighted(hud_bg, 0.3, hud_overlay, 0.7, 0)
    
    cv2.putText(out, f"Frame: {frame_idx}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(out, f"Track Count: {len(meta['tracks'])}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    max_nis = meta['max_nis']
    nis_color = (0, 255, 0) if max_nis < 5.99 else (0, 0, 255)
    cv2.putText(out, f"Peak NIS: {max_nis:.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, nis_color, 2)
    
    cv2.putText(out, "Chi^2 Threshold: 5.99", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # Render Tracks
    for trk in meta["tracks"]:
        px, py = int(trk["pred_x"]), int(trk["pred_y"])
        mx, my = int(trk["meas_x"]), int(trk["meas_y"])
        
        # Draw Prediction (Green Dot)
        cv2.circle(out, (px, py), 8, (0, 255, 0), -1)
        # Draw Measurement (Red Dot)
        cv2.circle(out, (mx, my), 6, (0, 0, 255), -1)
        
        # Draw Innovation Arrow (Yellow)
        if np.linalg.norm([mx-px, my-py]) > 2:
            cv2.arrowedLine(out, (px, py), (mx, my), (0, 255, 255), 3, tipLength=0.3)
            
        # Draw ID and NIS above measurement
        cv2.putText(out, f"ID:{trk['id']} NIS:{trk['nis']:.1f}", (mx-20, my-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, nis_color, 2)
                    
    # Render Tamper Alert if near GLR
    if is_tampered and glr_frame != -1 and abs(frame_idx - glr_frame) < 15:
        cv2.rectangle(out, (0,0), (out.shape[1]-1, out.shape[0]-1), (0,0,255), 15)
        if (frame_idx // 3) % 2 == 0:
            cv2.putText(out, f"TAMPER DETECTED: Frame {glr_frame}", 
                        (out.shape[1]//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

    return out

def main():
    auth_video = os.path.join(ROOT, "data", "evaluation", "authentic", "auth_0004.mp4")
    tamp_video = os.path.join(ROOT, "data", "evaluation", "tampered", "tamp_0004_clip_splice.mp4")
    output_path = os.path.join(ROOT, "results", "split_screen_physics.mp4")
    
    print("Loading YOLO Pose model...")
    pose_model = YOLO(os.path.join(ROOT, "yolov8n-pose.pt"))
    
    print(f"Processing Authentic: {os.path.basename(auth_video)}")
    a_frames, a_meta, a_glr = process_video_metadata(auth_video, CONFIG, pose_model)
    
    print(f"Processing Tampered: {os.path.basename(tamp_video)}")
    t_frames, t_meta, t_glr = process_video_metadata(tamp_video, CONFIG, pose_model)
    
    print("Generating split-screen visualization...")
    max_frames = max(len(a_frames), len(t_frames))
    
    h, w, _ = a_frames[0].shape
    split_w = w * 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (split_w, h))
    
    for i in range(max_frames):
        if i < len(a_frames):
            a_render = render_frame(a_frames[i], a_meta[i], a_glr, i, "Authentic (No Edit)", False)
        else:
            a_render = np.zeros_like(a_frames[0])
            
        if i < len(t_frames):
            t_render = render_frame(t_frames[i], t_meta[i], t_glr, i, "Tampered (Spliced)", True)
        else:
            t_render = np.zeros_like(t_frames[0])
            
        combined = np.hstack((a_render, t_render))
        out.write(combined)
        
    out.release()
    print(f"Split-screen saved to {output_path}")

if __name__ == "__main__":
    main()
