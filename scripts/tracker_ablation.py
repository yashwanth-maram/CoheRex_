import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from ultralytics import YOLO
from scripts.evaluate_dataset import evaluate_video

def main():
    video_path = os.path.join(ROOT, "data", "evaluation", "tampered", "tamp_0004_clip_splice.mp4")
    
    # Assuming from features_clean.csv that tamp_0004_clip_splice is from source_id src_002
    # The actual tamper frame was calculated manually or via manifest.
    # We will use frame 32 as the ground truth splice frame based on previous analysis.
    gt_splice_frame = 32

    print("="*80)
    print("Tracker Ablation Experiment: Bounding Box Only vs Bounding Box + Pose")
    print("="*80)
    print(f"Evaluating: {os.path.basename(video_path)}")
    print("Running Tracker A (No Pose)...")
    
    # Tracker A: No Pose
    # evaluate_video accepts pose_model, if we pass a dummy lambda it returns empty keypoints
    class DummyPoseModel:
        def __call__(self, frame, verbose=False):
            return []

    features_A = evaluate_video(
        video_path=video_path,
        config=CONFIG,
        window_size=150,
        agents_mode="full",
        pose_model=DummyPoseModel()
    )

    print("Running Tracker B (Pose)...")
    
    # Tracker B: Pose
    pose_model = YOLO(os.path.join(ROOT, "yolov8n-pose.pt"))
    features_B = evaluate_video(
        video_path=video_path,
        config=CONFIG,
        window_size=150,
        agents_mode="full",
        pose_model=pose_model
    )
    
    # Gather metrics
    # Note: features output contains num_tracks, nis_peak, _glr_frame
    
    def get_metrics(features):
        num_tracks = features.get("num_tracks", 0)
        nis_peak = features.get("nis_peak", 0.0)
        glr_frame = features.get("_glr_frame", -1)
        loc_error = abs(glr_frame - gt_splice_frame) if glr_frame != -1 else "N/A"
        return num_tracks, nis_peak, loc_error

    tracks_a, nis_a, loc_a = get_metrics(features_A)
    tracks_b, nis_b, loc_b = get_metrics(features_B)

    print("\n")
    print("| Metric                                   | No Pose (Tracker A) | Pose (Tracker B) |")
    print("|------------------------------------------|--------------------:|-----------------:|")
    print(f"| Number of Tracks (Fragmentation)         | {tracks_a:>19} | {tracks_b:>16} |")
    print(f"| Peak NIS                                 | {nis_a:>19.2f} | {nis_b:>16.2f} |")
    print(f"| Localization Error (frames)              | {str(loc_a):>19} | {str(loc_b):>16} |")
    print("\nExperiment Complete.")

if __name__ == "__main__":
    main()
