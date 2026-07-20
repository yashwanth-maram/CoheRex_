# scripts/visualize_signals.py
"""
CoheRex 2.0 — Visual Evidence of Tampering
============================================

Extracts and plots the NIS, CUSUM, and GLR signals for an authentic video
and its tampered counterpart, side-by-side, to provide direct visual 
evidence of how the physics models react to manipulation.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.evaluate_dataset import evaluate_video
from coherex.config import CONFIG

def get_signals_for_video(video_path, config, pose_model):
    """Run pipeline and extract the raw frame-by-frame signals."""
    from coherex.detection.yolo_detector import YOLODetector
    from coherex.tracking.manager import TrackManager
    from coherex.trajectory.store import TrajectoryStore
    
    detector = YOLODetector(config=config)
    tracker = TrackManager(config=config)
    traj_store = TrajectoryStore(config=config)

    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    all_combined_nis = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        detections = detector.detect(frame)
        person_centers, person_boxes = [], []
        for det in detections:
            if det["class_id"] == config.detection.person_class_id:
                x1, y1, x2, y2 = det["bbox"]
                person_centers.append(((x1+x2)/2, (y1+y2)/2))
                person_boxes.append(det["bbox"])

        pose_results = pose_model(frame, verbose=False)
        pose_keypoints = {}
        for r in pose_results:
            if r.keypoints is None: continue
            for i, kp in enumerate(r.keypoints.xy):
                pose_keypoints[i] = kp.cpu().numpy()

        tracker.current_frame = frame
        tracker.current_pose = pose_keypoints
        tracks = tracker.update(person_centers, person_boxes, frame_id)
        traj_store.update_from_tracks(tracks)

        frame_nis_values = []
        for track in tracks:
            if hasattr(track, 'nis_combined_history') and track.nis_combined_history:
                frame_nis_values.append(track.nis_combined_history[-1])

        all_combined_nis.append(max(frame_nis_values) if frame_nis_values else 0.0)
        frame_id += 1
    cap.release()

    # Compute derived signals
    from coherex.coherence.innovation import calibrate_nis
    from coherex.coherence.changepoint import glr_normalized_dispersion, cusum_peak
    
    cal_nis = calibrate_nis(all_combined_nis, dof=config.physics.chi2_dof, win=config.physics.calibration_window)
    
    # Reconstruct CUSUM signal
    drift = config.changepoint.cusum_drift
    c_plus = np.zeros(len(cal_nis))
    for t in range(1, len(cal_nis)):
        c_plus[t] = max(0, c_plus[t-1] + cal_nis[t] - drift)
        
    # Get GLR peak
    glr_info = glr_normalized_dispersion(cal_nis)
    
    return {
        "nis": cal_nis,
        "cusum": c_plus,
        "glr_frame": glr_info["glr_index"],
        "glr_peak": glr_info["glr_peak"]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", required=True, help="Path to authentic video")
    parser.add_argument("--tamp", required=True, help="Path to tampered video")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "tamper_evidence.png"))
    parser.add_argument("--tamp-label", default="Tampered Video")
    args = parser.parse_args()

    print(f"[VISUALIZE] Loading models...")
    from ultralytics import YOLO
    pose_model = YOLO(os.path.join(ROOT, "yolov8n-pose.pt"))
    config = CONFIG

    print(f"[VISUALIZE] Processing Authentic: {args.auth}")
    sig_auth = get_signals_for_video(args.auth, config, pose_model)
    
    print(f"[VISUALIZE] Processing Tampered : {args.tamp}")
    sig_tamp = get_signals_for_video(args.tamp, config, pose_model)

    print(f"[VISUALIZE] Plotting results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')
    fig.suptitle("Visual Evidence of Physics-based Tampering Detection", fontsize=16, fontweight='bold')
    
    t_auth = np.arange(len(sig_auth['nis']))
    t_tamp = np.arange(len(sig_tamp['nis']))
    
    # Top-left: Authentic NIS
    axes[0, 0].plot(t_auth, sig_auth['nis'], color='blue', label="NIS")
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label="Expected (1.0)")
    axes[0, 0].set_title("Authentic Video - Normalized Innovation Squared (NIS)")
    axes[0, 0].set_ylabel("Calibrated NIS")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Bottom-left: Authentic CUSUM
    axes[1, 0].plot(t_auth, sig_auth['cusum'], color='green', label="CUSUM Statistic")
    axes[1, 0].set_title(f"Authentic Video - CUSUM\n(Peak: {np.max(sig_auth['cusum']):.1f}, GLR: {sig_auth['glr_peak']:.1f})")
    axes[1, 0].set_ylabel("Cumulative Sum")
    axes[1, 0].set_xlabel("Frame Number")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Top-right: Tampered NIS
    axes[0, 1].plot(t_tamp, sig_tamp['nis'], color='red', label="NIS")
    axes[0, 1].axhline(y=1.0, color='black', linestyle='--', label="Expected (1.0)")
    axes[0, 1].set_title(f"{args.tamp_label} - Normalized Innovation Squared (NIS)")
    axes[0, 1].set_ylabel("Calibrated NIS")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Bottom-right: Tampered CUSUM
    axes[1, 1].plot(t_tamp, sig_tamp['cusum'], color='darkorange', label="CUSUM Statistic")
    if sig_tamp['glr_frame'] > 0:
        axes[1, 1].axvline(x=sig_tamp['glr_frame'], color='red', linestyle=':', lw=2, label="GLR Detected Change-point")
    axes[1, 1].set_title(f"{args.tamp_label} - CUSUM\n(Peak: {np.max(sig_tamp['cusum']):.1f}, GLR: {sig_tamp['glr_peak']:.1f})")
    axes[1, 1].set_ylabel("Cumulative Sum")
    axes[1, 1].set_xlabel("Frame Number")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()
    
    print(f"\n[DONE] Saved visual evidence to: {args.output}")

if __name__ == "__main__":
    main()
