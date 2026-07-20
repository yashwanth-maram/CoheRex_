"""
compute_paper_stats.py — Compute blocking statistics for the paper
=====================================================================
T1: Cluster-bootstrap CIs for VIFFD (kinematic / pixel / hybrid)
T2: DeLong test — VIFFD pixel vs hybrid
T3: VIFFD largest-group share
"""

import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from scripts.delong import delong_roc_test
from scripts.enhanced_classifier import (
    engineer_features, select_features, run_stacked_ensemble,
    bootstrap_ci
)

SEED = 42
N_TRIALS = 30
B = 2000


def youdens_j_threshold(y_true, y_score):
    """Find threshold maximizing Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return thresholds[best_idx]


def run_viffd_modality(csv_path, label):
    """Run pipeline on a VIFFD CSV, return y, oof_final, groups, stats."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    if "scene_id" in df.columns:
        df = df[df["scene_id"] != "unknown"].copy()
        df = df[df["scene_id"] != "legacy_samples"].copy()
        df["camera_id"] = df["scene_id"].str.extract(r'VIRAT_S_(\d{4})')[0]
        df["camera_id"] = df["camera_id"].fillna(df["scene_id"])
        df = df.reset_index(drop=True)

    print(f"  Rows: {len(df)}, Label dist: {dict(df['label'].value_counts())}")

    df = engineer_features(df)
    feats = select_features(df, strategy="x_plus_engineered")
    print(f"  Features: {len(feats)}")

    group_col = "camera_id" if "camera_id" in df.columns else "scene_id"
    y, oof_l0, oof_final, groups = run_stacked_ensemble(
        df, feats, group_col=group_col, n_trials=N_TRIALS, seed=SEED
    )

    auc = roc_auc_score(y, oof_final)
    ci_lo, ci_hi = bootstrap_ci(y, oof_final, groups, n_boot=B, seed=SEED)

    # Balanced accuracy at Youden's J
    thresh = youdens_j_threshold(y, oof_final)
    y_pred = (oof_final >= thresh).astype(int)
    bal_acc = balanced_accuracy_score(y, y_pred)

    print(f"\n  AUC:          {auc:.4f}")
    print(f"  95% CI:       ({ci_lo:.4f} - {ci_hi:.4f})")
    print(f"  Bal.Acc (J):  {bal_acc:.4f}  (threshold={thresh:.4f})")

    return y, oof_final, groups, {
        "label": label, "auc": float(auc),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "bal_acc_j": float(bal_acc), "threshold_j": float(thresh)
    }


if __name__ == "__main__":
    results_dir = os.path.join(ROOT, "results")

    # ─────────────────────────────────────────
    # T3: VIFFD largest-group share
    # ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  T3: VIFFD LARGEST-GROUP SHARE")
    print("="*60)
    df_v = pd.read_csv(os.path.join(results_dir, "viffd_features_hybrid.csv"))
    vc = df_v.scene_id.value_counts()
    print(f"  Largest group: '{vc.index[0]}' -> {vc.iloc[0]}/{len(df_v)} = {vc.iloc[0]/len(df_v)*100:.1f}%")
    print(f"  Total groups: {len(vc)}")

    # ─────────────────────────────────────────
    # T1: Cluster-bootstrap CIs for VIFFD
    # ─────────────────────────────────────────
    viffd_csvs = {
        "VIFFD Pixel":     os.path.join(results_dir, "viffd_features_pixel.csv"),
        "VIFFD Kinematic": os.path.join(results_dir, "viffd_features_physics.csv"),
        "VIFFD Hybrid":    os.path.join(results_dir, "viffd_features_hybrid.csv"),
    }

    all_results = {}
    oof_scores = {}

    for label, csv_path in viffd_csvs.items():
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} not found")
            continue
        y, oof, groups, stats = run_viffd_modality(csv_path, label)
        all_results[label] = stats
        oof_scores[label] = (y, oof, groups)

    # ─────────────────────────────────────────
    # T2: DeLong test — pixel vs hybrid
    # ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  T2: DELONG TEST — VIFFD PIXEL vs HYBRID")
    print("="*60)

    if "VIFFD Pixel" in oof_scores and "VIFFD Hybrid" in oof_scores:
        y_pixel, oof_pixel, _ = oof_scores["VIFFD Pixel"]
        y_hybrid, oof_hybrid, _ = oof_scores["VIFFD Hybrid"]

        n = min(len(y_pixel), len(y_hybrid))
        assert np.array_equal(y_pixel[:n], y_hybrid[:n]), "Label mismatch!"

        p_val = delong_roc_test(y_pixel[:n], oof_pixel[:n], oof_hybrid[:n])
        auc_pix = roc_auc_score(y_pixel[:n], oof_pixel[:n])
        auc_hyb = roc_auc_score(y_hybrid[:n], oof_hybrid[:n])
        print(f"  Pixel AUC:  {auc_pix:.4f}")
        print(f"  Hybrid AUC: {auc_hyb:.4f}")
        print(f"  DeLong p:   {p_val:.4f}")
        if p_val > 0.05:
            print("  -> NOT significant (p > 0.05). Hybrid ~ Pixel on VIFFD.")
        else:
            print(f"  -> SIGNIFICANT (p = {p_val:.4f}). Check direction!")

        all_results["DeLong_pixel_vs_hybrid"] = {
            "pixel_auc": float(auc_pix), "hybrid_auc": float(auc_hyb),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05)
        }

    # ─────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────
    print("\n\n" + "="*80)
    print(" VIFFD TABLE II — Cluster-bootstrap CIs (B=2000)")
    print("="*80)
    print(f"{'Modality':<25} | {'AUC':>8} | {'95% CI':>20} | {'Bal.Acc(J)':>10}")
    print("-"*80)
    for label in ["VIFFD Pixel", "VIFFD Kinematic", "VIFFD Hybrid"]:
        if label in all_results:
            r = all_results[label]
            ci_str = f"({r['ci_lo']:.4f} - {r['ci_hi']:.4f})"
            print(f"{r['label']:<25} | {r['auc']:>8.4f} | {ci_str:>20} | {r['bal_acc_j']:>10.4f}")
    print("="*80)

    # Save to JSON
    out_path = os.path.join(results_dir, "paper_stats_viffd.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
