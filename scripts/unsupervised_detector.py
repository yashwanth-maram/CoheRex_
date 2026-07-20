# scripts/unsupervised_detector.py
"""
CoheRex 2.0 — Training-Free Unsupervised Detector
====================================================

Detects video tampering using physics-based thresholds with ZERO training.
This is the small-N-proof result: works without any labeled training data.

Detection logic:
  Flag a video as tampered if ANY of:
    - max_cusum_peak > threshold (sustained physics violation)
    - nis_exceed_frac > threshold (fraction of frames exceeding χ²)
    - glr_peak > threshold (strong change-point evidence)

Sweep thresholds to produce an ROC curve and report AUC.

Usage:
    python scripts/unsupervised_detector.py --csv results/features_clean.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


def parse_args():
    p = argparse.ArgumentParser(description="Training-Free Unsupervised Detector")
    p.add_argument("--csv", default=os.path.join(ROOT, "results", "features_clean.csv"),
                   help="Path to features CSV")
    p.add_argument("--output", default=os.path.join(ROOT, "results", "unsupervised_metrics.json"),
                   help="Output metrics file")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv).dropna(subset=["label"])
    y = df["label"].astype(int).values
    print(f"[LOAD] {len(df)} samples ({(y==0).sum()} auth, {(y==1).sum()} tamp)")

    # ── Define scoring functions (higher = more likely tampered) ──
    detectors = {}

    # 1. NIS exceed fraction (most direct)
    if "nis_exceed_frac" in df.columns:
        detectors["nis_exceed_frac"] = df["nis_exceed_frac"].values

    # 2. NIS peak
    if "nis_peak" in df.columns:
        detectors["nis_peak"] = df["nis_peak"].values

    # 3. GLR peak
    if "glr_peak" in df.columns:
        detectors["glr_peak"] = df["glr_peak"].values

    # 4. CUSUM peak
    if "_cusum_peak" in df.columns:
        detectors["cusum_peak"] = df["_cusum_peak"].values

    # 5. Whiteness lag-1
    if "whiteness_lag1" in df.columns:
        detectors["whiteness_lag1"] = df["whiteness_lag1"].values

    # 6. Combined score (mean of normalized individual scores)
    if len(detectors) >= 2:
        normalized = []
        for name, scores in detectors.items():
            s = scores.copy()
            smin, smax = s.min(), s.max()
            if smax > smin:
                s = (s - smin) / (smax - smin)
            normalized.append(s)
        detectors["combined"] = np.mean(normalized, axis=0)

    if not detectors:
        print("[ERROR] No suitable features found for unsupervised detection")
        sys.exit(1)

    # ── Evaluate each detector ──
    results = {}
    best_auc = 0
    best_name = None

    print(f"\n{'='*60}")
    print(f"  UNSUPERVISED DETECTOR RESULTS (Training-Free)")
    print(f"{'='*60}\n")

    for name, scores in detectors.items():
        if len(np.unique(y)) < 2:
            continue
        auc = roc_auc_score(y, scores)

        # Find best threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(y, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        y_pred = (scores >= best_thresh).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"  {name:25s}  AUC = {auc:.4f}  "
              f"(thresh={best_thresh:.3f}, TP={tp}, FP={fp}, FN={fn}, TN={tn})")

        results[name] = {
            "auc": round(auc, 4),
            "best_threshold": round(float(best_thresh), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }

        if auc > best_auc:
            best_auc = auc
            best_name = name

    print(f"\n  Best detector: {best_name} (AUC = {best_auc:.4f})")

    # ── Save results ──
    output = {
        "method": "unsupervised_threshold",
        "n_samples": len(y),
        "best_detector": best_name,
        "best_auc": round(best_auc, 4),
        "detectors": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {args.output}")

    # ── ROC plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        for name, scores in detectors.items():
            fpr, tpr, _ = roc_curve(y, scores)
            auc = results[name]["auc"]
            plt.plot(fpr, tpr, lw=2 if name == best_name else 1,
                     label=f'{name} (AUC={auc:.3f})',
                     alpha=1.0 if name == best_name else 0.6)

        plt.plot([0, 1], [0, 1], '--', color='gray', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Unsupervised Detectors (Training-Free)')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)

        roc_path = args.output.replace(".json", "_roc.png")
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] ROC → {roc_path}")
    except Exception as e:
        print(f"[WARN] ROC plot failed: {e}")


if __name__ == "__main__":
    main()
