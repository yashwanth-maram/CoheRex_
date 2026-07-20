# scripts/pertype_eval.py
"""
CoheRex 2.0 — Per-Tamper-Type Evaluation
==========================================

Computes AUC per tamper type (deletion, duplication, speed, splice, reverse)
from the manifest metadata.

This reveals which manipulations CoheRex detects well and which it struggles
with — essential for honest reporting.

Usage:
    python scripts/pertype_eval.py --csv results/features_clean.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.metrics import roc_auc_score, confusion_matrix
from coherex.meta.feature_extractor import FEATURE_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Per-Type Tamper Evaluation")
    p.add_argument("--csv", default=os.path.join(ROOT, "results", "features_clean.csv"))
    p.add_argument("--output", default=os.path.join(ROOT, "results", "pertype_metrics.json"))
    return p.parse_args()


def compute_type_auc(df_auth, df_type, feature_cols):
    """Compute AUC for one tamper type vs all authentic samples."""
    if len(df_type) < 2 or len(df_auth) < 2:
        return None

    # Combine authentic + this type
    df_combined = pd.concat([df_auth, df_type])
    y = df_combined["label"].values

    if len(np.unique(y)) < 2:
        return None

    # Use combined score of physics features
    available = [f for f in feature_cols if f in df_combined.columns]
    if not available:
        return None

    # Normalize and average available features
    scores_list = []
    for feat in available:
        s = df_combined[feat].values.astype(float)
        smin, smax = np.nanmin(s), np.nanmax(s)
        if smax > smin:
            s = (s - smin) / (smax - smin)
        scores_list.append(s)

    combined_score = np.nanmean(scores_list, axis=0)

    try:
        return float(roc_auc_score(y, combined_score))
    except ValueError:
        return None


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv).dropna(subset=["label"])
    print(f"[LOAD] {len(df)} samples")

    if "tamper_type" not in df.columns:
        print("[ERROR] Column 'tamper_type' not found. Was manifest used during evaluation?")
        sys.exit(1)

    df_auth = df[df["label"] == 0]
    tamper_types = df.loc[df["label"] == 1, "tamper_type"].unique()
    tamper_types = [t for t in tamper_types if t not in ("none", "unknown")]

    print(f"[DATA] {len(df_auth)} authentic, tamper types: {', '.join(tamper_types)}")

    # Physics features to use for scoring
    physics_feats = ["nis_peak", "nis_topk_mean", "nis_exceed_frac",
                     "glr_peak", "whiteness_lag1", "spec_entropy"]

    results = {}

    print(f"\n{'='*60}")
    print(f"  PER-TYPE AUC (Combined Physics Score)")
    print(f"{'='*60}\n")
    print(f"  {'Type':<25s}  {'N':>4s}  {'AUC':>8s}")
    print(f"  {'-'*40}")

    for ttype in sorted(tamper_types):
        df_type = df[(df["label"] == 1) & (df["tamper_type"] == ttype)]
        auc = compute_type_auc(df_auth, df_type, physics_feats)

        n = len(df_type)
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"  {ttype:<25s}  {n:>4d}  {auc_str:>8s}")

        results[ttype] = {
            "n_samples": n,
            "auc": round(auc, 4) if auc is not None else None,
        }

    # Per-feature per-type breakdown
    print(f"\n  Per-Feature × Per-Type AUC:")
    print(f"  {'Feature':<20s}", end="")
    for ttype in sorted(tamper_types):
        print(f"  {ttype[:12]:>12s}", end="")
    print()
    print(f"  {'-'*(20 + 14*len(tamper_types))}")

    feature_type_aucs = {}
    for feat in physics_feats:
        if feat not in df.columns:
            continue
        print(f"  {feat:<20s}", end="")
        feature_type_aucs[feat] = {}

        for ttype in sorted(tamper_types):
            df_type = df[(df["label"] == 1) & (df["tamper_type"] == ttype)]
            df_combined = pd.concat([df_auth, df_type])
            y = df_combined["label"].values
            scores = df_combined[feat].values.astype(float)

            try:
                auc = roc_auc_score(y, scores)
                print(f"  {auc:>12.3f}", end="")
                feature_type_aucs[feat][ttype] = round(auc, 4)
            except (ValueError, Exception):
                print(f"  {'N/A':>12s}", end="")
                feature_type_aucs[feat][ttype] = None
        print()

    # ── Save ──
    output = {
        "n_authentic": len(df_auth),
        "per_type": results,
        "feature_type_aucs": feature_type_aucs,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
