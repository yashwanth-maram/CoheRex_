import os
import sys
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from coherex.detection.yolo_detector import YOLODetector
from coherex.tracking.manager import TrackManager


def parse_args():
    p = argparse.ArgumentParser(description="Check NIS calibration on authentic videos")
    p.add_argument("--input", default=os.path.join(ROOT, "data", "evaluation", "authentic"),
                   help="Path to directory containing authentic videos")
    p.add_argument("--n", type=int, default=179, help="Number of videos to process")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Directory not found: {args.input}")
        return
        
    videos = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".mp4")])
    videos = videos[:args.n]
    
    if not videos:
        print("No videos found.")
        return
        
    detector = YOLODetector(config=CONFIG)
    
    all_nis = []
    
    print(f"Processing {len(videos)} videos for NIS calibration check...")
    for vpath in tqdm(videos, desc="Videos"):
        cap = cv2.VideoCapture(vpath)
        tracker = TrackManager(config=CONFIG)
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = detector.detect(frame)
            centers, bboxes = [], []
            for det in detections:
                if det["class_id"] == CONFIG.detection.person_class_id:
                    x1, y1, x2, y2 = det["bbox"]
                    centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
                    bboxes.append(det["bbox"])
                    
            tracker.current_frame = frame
            tracks = tracker.update(centers, bboxes, frame_id)
            
            # Collect NIS from tracks
            for track in tracks:
                if hasattr(track, 'nis_combined_history') and len(track.nis_combined_history) > 0:
                    # track.update only appends after warmup frames
                    latest = track.nis_combined_history[-1]
                    all_nis.append(latest)
                    
            frame_id += 1
        cap.release()
        
    all_nis = np.array(all_nis)
    
    if len(all_nis) == 0:
        print("No NIS values collected (maybe videos are empty or tracks didn't survive warmup).")
        return
        
    # Analyze
    median_nis = np.median(all_nis)
    pct95 = np.percentile(all_nis, 95)
    
    thresh = CONFIG.physics.chi2_threshold
    exceed_frac = np.mean(all_nis > thresh)
    
    print("\n--- NIS Calibration Results ---")
    print(f"Total NIS samples : {len(all_nis)}")
    print(f"Median NIS        : {median_nis:.4f} (Target: ~1.386)")
    print(f"95th Percentile   : {pct95:.4f} (Target: ~5.991)")
    print(f"Exceed Fraction   : {exceed_frac:.4f} (Target: ~0.0500)")
    print(f"Max NIS           : {np.max(all_nis):.4f}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    
    # Cap values for histogram visualization
    plot_max = max(10.0, pct95 * 1.5)
    nis_plot = all_nis[all_nis < plot_max]
    
    plt.hist(nis_plot, bins=50, density=True, alpha=0.6, color='blue', label='Empirical NIS')
    
    # Theoretical chi2(2)
    x = np.linspace(0, plot_max, 200)
    y = chi2.pdf(x, 2)
    plt.plot(x, y, 'r-', lw=2, label='Theoretical χ²(2)')
    
    plt.axvline(thresh, color='red', linestyle='--', label=f'95% Threshold ({thresh:.2f})')
    plt.axvline(median_nis, color='blue', linestyle='--', label=f'Empirical Median ({median_nis:.2f})')
    
    plt.xlim(0, plot_max)
    plt.title(f'NIS Distribution (N={len(all_nis)} samples from {len(videos)} authentic videos)')
    plt.xlabel('Combined NIS value')
    plt.ylabel('Density')
    plt.legend()
    
    out_path = os.path.join(ROOT, "results", "nis_calibration.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved histogram plot to {out_path}")


if __name__ == "__main__":
    main()
