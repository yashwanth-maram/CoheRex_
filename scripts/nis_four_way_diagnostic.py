"""
scripts/nis_four_way_diagnostic.py
====================================
Four-panel NIS diagnostic:
  1. NIS_CV  alone vs chi²(2)
  2. NIS_CA  alone vs chi²(2)
  3. min(NIS_CV, NIS_CA) — current combined statistic
  4. max(NIS_CV, NIS_CA)

Also prints mean / median / variance / 95th / 99th pct / exceed_frac
for each channel.

Usage:
    python scripts/nis_four_way_diagnostic.py --n 179
"""

import os
import sys
import cv2
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from coherex.detection.yolo_detector import YOLODetector
from coherex.tracking.manager import TrackManager


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=os.path.join(ROOT, "data", "evaluation", "authentic"),
                   help="Authentic video directory")
    p.add_argument("--n", type=int, default=179, help="Number of videos to process")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "nis_four_way.png"))
    return p.parse_args()


def describe(arr, name, thresh):
    """Print descriptive statistics for a NIS channel."""
    arr = np.array(arr)
    if len(arr) == 0:
        print(f"  {name}: NO DATA")
        return
    print(f"\n  [{name}]")
    print(f"    N          = {len(arr)}")
    print(f"    Mean       = {arr.mean():.4f}")
    print(f"    Median     = {np.median(arr):.4f}  (chi²(2) target: 1.386)")
    print(f"    Variance   = {arr.var():.4f}")
    print(f"    95th pct   = {np.percentile(arr, 95):.4f}  (chi²(2) target: 5.991)")
    print(f"    99th pct   = {np.percentile(arr, 99):.4f}")
    print(f"    Exceed frac= {np.mean(arr > thresh):.4f}  (target: 0.0500)")
    print(f"    Max        = {arr.max():.4f}")


def main():
    args = parse_args()
    videos = sorted([
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".mp4")
    ])[:args.n]

    if not videos:
        print("No videos found.")
        return

    detector = YOLODetector(config=CONFIG)
    thresh = CONFIG.physics.chi2_threshold

    # Separate collectors — ONLY frames where update actually occurred
    nis_cv_all  = []
    nis_ca_all  = []
    nis_min_all = []
    nis_max_all = []

    print(f"Processing {len(videos)} authentic videos …")
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

            for track in tracks:
                # Only collect if warmup passed AND this frame had a measurement update
                # (tracked by the presence of new NIS entries after warmup)
                if (hasattr(track, 'nis_cv_history') and
                        len(track.nis_cv_history) > 0 and
                        track.age > CONFIG.physics.nis_warmup_frames):
                    cv_val  = track.nis_cv_history[-1]
                    ca_val  = track.nis_ca_history[-1] if track.nis_ca_history else cv_val

                    # Skip zero entries (no measurement was fused this frame)
                    if cv_val == 0.0 and ca_val == 0.0:
                        continue

                    nis_cv_all.append(cv_val)
                    nis_ca_all.append(ca_val)
                    nis_min_all.append(min(cv_val, ca_val))
                    nis_max_all.append(max(cv_val, ca_val))

            frame_id += 1
        cap.release()

    print("\n" + "=" * 60)
    print("  NIS FOUR-WAY DIAGNOSTIC RESULTS")
    print("=" * 60)
    describe(nis_cv_all,  "NIS_CV  (constant-velocity)", thresh)
    describe(nis_ca_all,  "NIS_CA  (constant-accel)   ", thresh)
    describe(nis_min_all, "min(CV,CA) — current fusion", thresh)
    describe(nis_max_all, "max(CV,CA)                 ", thresh)

    # ── Four-panel plot ──────────────────────────────────────────────────────
    channels = [
        (nis_cv_all,  "NIS_CV  (single filter)",        "steelblue"),
        (nis_ca_all,  "NIS_CA  (single filter)",        "darkorange"),
        (nis_min_all, "min(NIS_CV, NIS_CA) — current",  "firebrick"),
        (nis_max_all, "max(NIS_CV, NIS_CA)",             "forestgreen"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    x_ref = np.linspace(1e-3, 15, 300)
    y_ref = chi2.pdf(x_ref, 2)

    for ax, (data, label, color) in zip(axes, channels):
        arr = np.array(data)
        if len(arr) == 0:
            ax.set_title(f"{label}\n(no data)")
            continue

        p95 = np.percentile(arr, 95)
        plot_max = max(12.0, p95 * 1.5)

        arr_plot = arr[arr < plot_max]
        ax.hist(arr_plot, bins=60, density=True, alpha=0.65,
                color=color, label="Empirical")
        ax.plot(x_ref[x_ref < plot_max],
                y_ref[x_ref < plot_max],
                "r-", lw=2, label="χ²(2) theory")
        ax.axvline(thresh, color="red", ls="--", lw=1, label=f"χ²₀.₉₅ = {thresh}")
        med = np.median(arr)
        ax.axvline(med, color=color, ls=":", lw=1.5, label=f"Empirical median = {med:.3f}")
        ax.set_title(f"{label}\nN={len(arr)}, exceed={np.mean(arr>thresh):.3f}")
        ax.set_xlabel("NIS value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.set_xlim(0, plot_max)

    fig.suptitle(
        "Four-Way NIS Diagnostic — 179 Authentic Videos\n"
        "Each filter vs. χ²(2) | min-fusion vs. max-fusion",
        fontsize=12
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
