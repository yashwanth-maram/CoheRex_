import os
import sys
import cv2
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from scripts.evaluate_dataset import evaluate_video
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Annotate a video with tampering evidence")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Path to output annotated video")
    return parser.parse_args()

def annotate_video(video_path, output_path):
    print(f"Analyzing {video_path} for physical anomalies...")
    
    # 1. Run evaluation to find the exact tamper frame
    pose_model = YOLO(os.path.join(ROOT, "yolov8n-pose.pt"))
    features = evaluate_video(
        video_path=video_path,
        config=CONFIG,
        window_size=150,
        agents_mode="full",
        pose_model=pose_model
    )
    
    glr_frame = features.get("_glr_frame", -1)
    nis_peak = features.get("nis_peak", 0.0)
    
    if glr_frame == -1:
        print("No tampering detected (or no tracks found).")
        return
        
    print(f"Anomaly detected at frame {glr_frame} with Peak NIS {nis_peak:.2f}")
    
    # 2. Re-read video and write annotated version
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if output_path is None:
        base_name = os.path.basename(video_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(ROOT, "results", f"{name}_annotated{ext}")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw frame number
        cv2.putText(frame, f"Frame: {frame_id}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        # Annotate tampering
        if glr_frame != -1 and abs(frame_id - glr_frame) < 15:
            # Draw red border
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
            
            # Flash text
            if (frame_id // 3) % 2 == 0:
                text = f"TAMPER DETECTED: NIS Spike (Frame {glr_frame})"
                cv2.putText(frame, text, (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                            
        out.write(frame)
        frame_id += 1
        
    cap.release()
    out.release()
    print(f"Annotated video saved to {output_path}")

def main():
    args = parse_args()
    if not os.path.exists(args.video):
        print(f"Error: File not found: {args.video}")
        sys.exit(1)
        
    annotate_video(args.video, args.output)

if __name__ == "__main__":
    main()
