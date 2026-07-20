# scripts/localize_eval.py
"""
CoheRex 2.0 — Temporal Localization Evaluation
================================================

Compares predicted change-point frames (from CUSUM/GLR) against
ground-truth tamper windows from the manifest.

Metrics:
  - Hit-rate: fraction of tampered videos where predicted change-point
              falls within the ground-truth tamper window
  - Temporal IoU: overlap between predicted and ground-truth windows

Usage:
    python scripts/localize_eval.py --csv results/features_clean.csv --manifest data/evaluation/manifest.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Temporal Localization Evaluation")
    p.add_argument("--csv", default=os.path.join(ROOT, "results", "features_clean.csv"))
    p.add_argument("--manifest", default=os.path.join(ROOT, "data", "evaluation", "manifest.csv"))
    p.add_argument("--output", default=os.path.join(ROOT, "results", "localization_metrics.json"))
    p.add_argument("--window-margin", type=int, default=10,
                   help="Margin (frames) around ground-truth for hit-rate")
    return p.parse_args()


def temporal_iou(pred_start, pred_end, gt_start, gt_end):
    """Compute temporal IoU between predicted and ground-truth windows."""
    if pred_start < 0 or gt_start < 0:
        return 0.0
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)

    # Filter to tampered videos only
    tampered = df[df["label"] == 1].copy()
    if len(tampered) == 0:
        print("[ERROR] No tampered videos found in CSV")
        sys.exit(1)

    print(f"[LOAD] {len(tampered)} tampered videos")

    # Check for required columns
    has_cusum = "_cusum_frame" in tampered.columns
    has_glr = "_glr_frame" in tampered.columns
    has_gt = ("tamper_start_frame" in tampered.columns and
              "tamper_end_frame" in tampered.columns)

    if not has_gt:
        print("[ERROR] Ground-truth columns (tamper_start_frame, tamper_end_frame) not found")
        sys.exit(1)

    # ── Evaluate localization ──
    margin = args.window_margin
    results = {"cusum": {"hits": 0, "ious": []}, "glr": {"hits": 0, "ious": []}}
    valid_count = 0

    for _, row in tampered.iterrows():
        gt_start = int(row.get("tamper_start_frame", -1))
        gt_end = int(row.get("tamper_end_frame", -1))

        if gt_start < 0 or gt_end < 0:
            continue
        valid_count += 1

        # CUSUM localization
        if has_cusum:
            cusum_frame = int(row.get("_cusum_frame", -1))
            if cusum_frame >= 0:
                hit = (gt_start - margin) <= cusum_frame <= (gt_end + margin)
                if hit:
                    results["cusum"]["hits"] += 1
                # Simple IoU: treat prediction as a point window
                pred_start = max(0, cusum_frame - margin)
                pred_end = cusum_frame + margin
                iou = temporal_iou(pred_start, pred_end, gt_start, gt_end)
                results["cusum"]["ious"].append(iou)

        # GLR localization
        if has_glr:
            glr_frame = int(row.get("_glr_frame", -1))
            if glr_frame >= 0:
                hit = (gt_start - margin) <= glr_frame <= (gt_end + margin)
                if hit:
                    results["glr"]["hits"] += 1
                pred_start = max(0, glr_frame - margin)
                pred_end = glr_frame + margin
                iou = temporal_iou(pred_start, pred_end, gt_start, gt_end)
                results["glr"]["ious"].append(iou)

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"  TEMPORAL LOCALIZATION RESULTS")
    print(f"  Valid tampered videos with ground-truth: {valid_count}")
    print(f"  Hit-rate margin: ±{margin} frames")
    print(f"{'='*60}\n")

    output = {"n_tampered": valid_count, "margin": margin}

    for method in ["cusum", "glr"]:
        r = results[method]
        hit_rate = r["hits"] / valid_count if valid_count > 0 else 0.0
        mean_iou = np.mean(r["ious"]) if r["ious"] else 0.0
        median_iou = np.median(r["ious"]) if r["ious"] else 0.0

        print(f"  {method.upper():8s}  Hit-rate: {hit_rate:.3f} ({r['hits']}/{valid_count})  "
              f"Mean IoU: {mean_iou:.3f}  Median IoU: {median_iou:.3f}")

        output[method] = {
            "hit_rate": round(hit_rate, 4),
            "mean_iou": round(mean_iou, 4),
            "median_iou": round(median_iou, 4),
            "hits": r["hits"],
        }

    # ── Per-type localization ──
    if "tamper_type" in tampered.columns:
        print(f"\n  Per-Type Localization (GLR):")
        for ttype in tampered["tamper_type"].unique():
            if ttype in ("none", "unknown"):
                continue
            subset = tampered[tampered["tamper_type"] == ttype]
            type_hits = 0
            type_ious = []
            for _, row in subset.iterrows():
                gt_start = int(row.get("tamper_start_frame", -1))
                gt_end = int(row.get("tamper_end_frame", -1))
                glr_frame = int(row.get("_glr_frame", -1))
                if gt_start < 0 or gt_end < 0 or glr_frame < 0:
                    continue
                hit = (gt_start - margin) <= glr_frame <= (gt_end + margin)
                if hit:
                    type_hits += 1
                pred_start = max(0, glr_frame - margin)
                pred_end = glr_frame + margin
                type_ious.append(temporal_iou(pred_start, pred_end, gt_start, gt_end))

            n = len(subset)
            hr = type_hits / n if n > 0 else 0.0
            miou = np.mean(type_ious) if type_ious else 0.0
            print(f"    {ttype:<25s}  Hit-rate: {hr:.3f}  Mean IoU: {miou:.3f}  (n={n})")

    # ── Save ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {args.output}")


if __name__ == "__main__":
    main()
