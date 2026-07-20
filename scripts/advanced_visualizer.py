import os
import sys
import cv2
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from ultralytics import YOLO
from coherex.detection.yolo_detector import YOLODetector
from coherex.tracking.manager import TrackManager
from coherex.coherence.innovation import calibrate_nis
from coherex.coherence.changepoint import glr_change_point, cusum_peak

def draw_line_graph(canvas, data, color, y_label, max_val, thresh=None, title="", current_idx=0):
    h, w, _ = canvas.shape
    margin = 40
    # draw axes
    cv2.line(canvas, (margin, h - margin), (w - margin, h - margin), (255,255,255), 1)
    cv2.line(canvas, (margin, margin), (margin, h - margin), (255,255,255), 1)
    
    cv2.putText(canvas, title, (margin, margin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(canvas, f"{max_val:.1f}", (5, margin), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(canvas, "0", (15, h - margin), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    if len(data) < 2: return
    
    # Scale data
    n = len(data)
    dx = (w - 2 * margin) / max(1, (n - 1))
    
    # Draw threshold
    if thresh is not None and thresh <= max_val:
        ty = h - margin - int((thresh / max_val) * (h - 2 * margin))
        cv2.line(canvas, (margin, ty), (w - margin, ty), (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Threshold", (w - margin - 60, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
    pts = []
    for i, val in enumerate(data):
        if i > current_idx: break
        x = margin + int(i * dx)
        y = h - margin - int((min(val, max_val) / max_val) * (h - 2 * margin))
        pts.append((x, y))
        
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], color, 2, cv2.LINE_AA)
        
    if len(pts) > 0:
        cv2.circle(canvas, pts[-1], 4, color, -1)

def process_video_metadata(video_path, config, pose_model):
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
        
        # Standardize resolution
        frame = cv2.resize(frame, (1280, 720))
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
            if trk.last_innovation is not None and len(trk.nis_combined_history) > 0:
                y_inv = trk.last_innovation.flatten()
                pred_x = trk.x - y_inv[0]
                pred_y = trk.y - y_inv[1]
                nis_val = trk.nis_combined_history[-1]
                
                # Cap NIS for visualization stability, but keep real for logic
                vis_nis = min(nis_val, 100.0) 
                frame_nis_vals.append(vis_nis)
                
                track_data.append({
                    "id": trk.track_id,
                    "pred_x": pred_x, "pred_y": pred_y,
                    "meas_x": trk.x, "meas_y": trk.y,
                    "nis": nis_val,
                    "vis_nis": vis_nis
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
    
    glr_scores = np.zeros(len(cal_nis))
    total = np.sum(cal_nis)
    n = len(cal_nis)
    csum = 0.0
    for k in range(1, n):
        csum += cal_nis[k-1]
        n1, n2 = k, n-k
        stat = (csum**2)/n1 + ((total-csum)**2)/n2 - (total**2)/n
        glr_scores[k] = stat
        
    cp = int(np.argmax(glr_scores)) if n > 4 else -1
    
    return frames, frame_meta, all_nis, glr_scores, cp

def build_column(frames, meta, all_nis, glr_scores, cp, frame_idx, title, is_tampered, traj_history):
    if frame_idx < len(frames):
        frame = frames[frame_idx].copy()
        current_meta = meta[frame_idx]
    else:
        frame = np.zeros_like(frames[0])
        current_meta = {"tracks": [], "max_nis": 0}
        
    h, w, _ = frame.shape
    
    # Panel 1: Video
    v_panel = frame
    cv2.putText(v_panel, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # HUD
    cv2.putText(v_panel, f"Frame: {frame_idx}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(v_panel, f"Tracks: {len(current_meta['tracks'])}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(v_panel, f"NIS: {current_meta['max_nis']:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    # Panel 2: Trajectory
    t_panel = np.zeros((h, w, 3), dtype=np.uint8)
    # Fade in background slightly
    t_panel = cv2.addWeighted(t_panel, 0.8, v_panel, 0.2, 0)
    
    for trk in current_meta["tracks"]:
        tid = trk["id"]
        traj_history[tid].append((int(trk["meas_x"]), int(trk["meas_y"])))
        
        # Draw history on t_panel
        pts = traj_history[tid]
        for i in range(1, len(pts)):
            cv2.line(t_panel, pts[i-1], pts[i], (100,100,100), 2)
            
        px, py = int(trk["pred_x"]), int(trk["pred_y"])
        mx, my = int(trk["meas_x"]), int(trk["meas_y"])
        
        for panel in [v_panel, t_panel]:
            # Prediction
            cv2.circle(panel, (px, py), 6, (0, 255, 0), -1)
            # Measurement
            cv2.drawMarker(panel, (mx, my), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            
            # Innovation Vector
            if np.linalg.norm([mx-px, my-py]) > 2:
                cv2.arrowedLine(panel, (px, py), (mx, my), (0, 255, 255), 3, tipLength=0.3)
                
            cv2.putText(panel, f"ID:{tid}", (mx-20, my-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
    # Tamper Alert
    if is_tampered and cp != -1 and abs(frame_idx - cp) < 15:
        for p in [v_panel, t_panel]:
            cv2.rectangle(p, (0,0), (w-1, h-1), (0,0,255), 10)
            if (frame_idx // 3) % 2 == 0:
                cv2.putText(p, f"TAMPER (Frame {cp})", (w//2-150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # Panel 3: NIS Plot
    plot_h = h // 2
    nis_panel = np.zeros((plot_h, w, 3), dtype=np.uint8)
    draw_line_graph(nis_panel, all_nis, (0, 255, 255), "NIS", 50.0, thresh=5.99, title="Kalman NIS (Capped at 50)", current_idx=frame_idx)
    
    # Panel 4: GLR Plot
    glr_panel = np.zeros((plot_h, w, 3), dtype=np.uint8)
    max_glr = max(glr_scores) if len(glr_scores)>0 else 10.0
    draw_line_graph(glr_panel, glr_scores, (255, 100, 255), "GLR", max(10.0, max_glr), title="GLR Anomaly Score", current_idx=frame_idx)
    
    # Stack panels vertically
    # Row 1: Video
    # Row 2: Trajectory
    # Row 3: NIS
    # Row 4: GLR
    col = np.vstack((v_panel, t_panel, nis_panel, glr_panel))
    return col

def main():
    auth_video = os.path.join(ROOT, "data", "evaluation", "authentic", "auth_0010.mp4")
    tamp_video = os.path.join(ROOT, "data", "evaluation", "tampered", "extreme_teleport.mp4")
    output_path = os.path.join(ROOT, "results", "showcase_visualizer.mp4")
    
    print("Loading Pose model...")
    pose_model = YOLO(os.path.join(ROOT, "yolov8n-pose.pt"))
    
    print(f"Processing Authentic: {os.path.basename(auth_video)}")
    a_frames, a_meta, a_nis, a_glr, a_cp = process_video_metadata(auth_video, CONFIG, pose_model)
    
    print(f"Processing Tampered: {os.path.basename(tamp_video)}")
    t_frames, t_meta, t_nis, t_glr, t_cp = process_video_metadata(tamp_video, CONFIG, pose_model)
    
    print("Rendering 4-panel split-screen grid...")
    max_frames = max(len(a_frames), len(t_frames))
    
    h, w, _ = a_frames[0].shape
    
    # The output height is h + h + h//2 + h//2 = 3h
    out_h = 3 * h
    out_w = w * 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (out_w, out_h))
    
    a_traj_history = defaultdict(list)
    t_traj_history = defaultdict(list)
    
    for i in range(max_frames):
        a_col = build_column(a_frames, a_meta, a_nis, a_glr, a_cp, i, "Authentic Video", False, a_traj_history)
        t_col = build_column(t_frames, t_meta, t_nis, t_glr, t_cp, i, "Tampered Video", True, t_traj_history)
        
        grid = np.hstack((a_col, t_col))
        out.write(grid)
        
        if i % 20 == 0:
            print(f"Rendered {i}/{max_frames} frames...")
            
    out.release()
    print(f"Showcase generated at: {output_path}")

if __name__ == "__main__":
    main()
