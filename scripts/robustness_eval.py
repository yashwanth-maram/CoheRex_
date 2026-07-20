"""
robustness_eval.py - Generate the definitive Recompression Robustness Table
=============================================================================

This script implements the core thesis experiment:
1. Loads Clean, CRF28, CRF32, and CRF36 feature CSVs.
2. Performs 6-fold GroupKFold (grouped by scene_id) on the CLEAN dataset.
3. For each fold, trains XGBoost on the clean training scenes.
4. Evaluates the fold's model on the clean holdout scene AND the matching
   recompressed holdout scenes.
5. Pools the OOF predictions and computes AUC + Bootstrap CIs.
6. Prints the 3x4 table: (Pixel, Physics, Hybrid) x (Clean, CRF28, CRF32, CRF36).

Usage:
    python scripts/robustness_eval.py
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

try:
    import xgboost as xgb
except ImportError:
    print("[ERROR] XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only evaluate on the 177 videos with known scene_id (the honest subset)
MANIFEST_PATH = os.path.join(ROOT, "data", "evaluation", "manifest.csv")


def get_features_by_set(df, feature_set):
    """Select features based on channel prefix."""
    numeric_cols = set(df.select_dtypes(include="number").columns)
    all_cols = list(df.columns)
    if feature_set == "pixel":
        return [c for c in all_cols if c in numeric_cols and c.startswith("pix_") and df[c].std() > 0]
    elif feature_set == "physics":
        return [c for c in all_cols if c in numeric_cols and c.startswith("phys_") and df[c].std() > 0]
    elif feature_set == "hybrid":
        return [c for c in all_cols if c in numeric_cols and c.startswith(("pix_", "phys_", "scene_")) and df[c].std() > 0]
    return []


def bootstrap_ci(y_true, y_score, n_boot=2000, seed=42, ci=0.95):
    """Compute bootstrap confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    alpha = (1 - ci) / 2
    return np.percentile(boots, alpha * 100), np.percentile(boots, (1 - alpha) * 100)


def evaluate_pipeline(df_clean, dict_df_crfs, feature_set, seed=42):
    """
    Run GroupKFold on df_clean, evaluate on df_clean and dict_df_crfs.
    Returns pooled OOF predictions for clean and each CRF.
    """
    features = get_features_by_set(df_clean, feature_set)
    if not features:
        return {}

    # Honest subset: only rows with known scene_id
    df_clean = df_clean[df_clean["scene_id"] != "unknown"].reset_index(drop=True)
    
    # Sort everything by video_name to ensure alignment
    df_clean = df_clean.sort_values("video_name").reset_index(drop=True)
    
    y = df_clean["label"].values
    groups = df_clean["scene_id"].values
    
    X_clean = df_clean[features].fillna(0).values
    
    X_crfs = {}
    for crf, df_crf in dict_df_crfs.items():
        df_crf_subset = df_crf[df_crf["video_name"].isin(df_clean["video_name"])].copy()
        df_crf_subset = df_crf_subset.sort_values("video_name").reset_index(drop=True)
        # Ensure alignment
        assert (df_crf_subset["video_name"].values == df_clean["video_name"].values).all(), f"Alignment failed for {crf}"
        X_crfs[crf] = df_crf_subset[features].fillna(0).values

    # Pre-allocate OOF prediction arrays
    oof_clean = np.zeros(len(y))
    oof_crfs = {crf: np.zeros(len(y)) for crf in dict_df_crfs.keys()}

    gkf = GroupKFold(n_splits=df_clean["scene_id"].nunique())
    
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    for train_idx, test_idx in gkf.split(X_clean, y, groups):
        clf = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            random_state=seed, eval_metric="auc", verbosity=0
        )
        
        # Train on CLEAN
        clf.fit(X_clean[train_idx], y[train_idx])
        
        # Predict on CLEAN holdout
        oof_clean[test_idx] = clf.predict_proba(X_clean[test_idx])[:, 1]
        
        # Predict on CRF holdouts
        for crf in dict_df_crfs.keys():
            oof_crfs[crf][test_idx] = clf.predict_proba(X_crfs[crf][test_idx])[:, 1]

    # Compute metrics
    results = {}
    
    def score(y_true, y_pred):
        auc = roc_auc_score(y_true, y_pred)
        lo, hi = bootstrap_ci(y_true, y_pred, seed=seed)
        return {"auc": auc, "lo": lo, "hi": hi}
        
    results["Clean"] = score(y, oof_clean)
    for crf in dict_df_crfs.keys():
        results[f"CRF{crf}"] = score(y, oof_crfs[crf])
        
    return results


def format_cell(res):
    if not res:
        return "---"
    return f"{res['auc']:.4f}\n({res['lo']:.2f}-{res['hi']:.2f})"


def main():
    print("==================================================")
    print(" CoheRex 2.0 — Recompression Robustness Evaluator")
    print("==================================================\n")
    
    csv_clean = os.path.join(ROOT, "results", "features_hybrid.csv")
    csv_crf28 = os.path.join(ROOT, "results", "features_hybrid_crf28.csv")
    csv_crf32 = os.path.join(ROOT, "results", "features_hybrid_crf32.csv")
    csv_crf36 = os.path.join(ROOT, "results", "features_hybrid_crf36.csv")
    csv_temporal = os.path.join(ROOT, "results", "features_hybrid_temporal.csv")
    
    # Check if files exist
    for f in [csv_clean]: # We only strictly require clean for a partial run
        if not os.path.exists(f):
            print(f"[ERROR] Missing required CSV: {f}")
            print("Please run build_features.py and merge pixel+physics CSVs first.")
            sys.exit(1)
            
    df_clean = pd.read_csv(csv_clean)
    
    dict_df_crfs = {}
    if os.path.exists(csv_crf28): dict_df_crfs["28"] = pd.read_csv(csv_crf28)
    if os.path.exists(csv_crf32): dict_df_crfs["32"] = pd.read_csv(csv_crf32)
    if os.path.exists(csv_crf36): dict_df_crfs["36"] = pd.read_csv(csv_crf36)
    if os.path.exists(csv_temporal): dict_df_crfs["temporal"] = pd.read_csv(csv_temporal)
    
    # If manifest exists, inject missing scene_id metadata
    if os.path.exists(MANIFEST_PATH):
        manifest = pd.read_csv(MANIFEST_PATH)
        scene_map = dict(zip(manifest['clip_name'], manifest['scene_id']))
        for df in [df_clean] + list(dict_df_crfs.values()):
            if "scene_id" not in df.columns or (df["scene_id"] == "unknown").any():
                df["scene_id"] = df["video_name"].map(lambda v: scene_map.get(v, "unknown"))
    
    print(f"Loaded Clean dataset ({len(df_clean)} rows)")
    print(f"Loaded conditions: {list(dict_df_crfs.keys())}")
    
    honest_count = sum(df_clean['scene_id'] != 'unknown')
    print(f"\n[INFO] Running evaluation on honest subset: N={honest_count} (known scene_id)")
    
    table_data = []
    has_temporal = "temporal" in dict_df_crfs
    headers = ["Feature Set", "Clean", "CRF 28", "CRF 32", "CRF 36"]
    if has_temporal:
        headers.append("Temporal Interp")
    
    for fset in ["pixel", "physics", "hybrid"]:
        print(f"\nEvaluating {fset.upper()} features...")
        res = evaluate_pipeline(df_clean, dict_df_crfs, fset)
        
        if not res:
            row = [fset.capitalize()] + ["Missing"] * (len(headers) - 1)
        else:
            row = [
                fset.capitalize(),
                format_cell(res.get("Clean")),
                format_cell(res.get("CRF28")),
                format_cell(res.get("CRF32")),
                format_cell(res.get("CRF36")),
            ]
            if has_temporal:
                row.append(format_cell(res.get("CRFtemporal")))
        table_data.append(row)
        
    print("\n\n" + "="*80)
    print(" TABLE 1: ROBUSTNESS UNDER POST-PROCESSING (AUC & 95% CI)")
    print("="*80 + "\n")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\n* Trained on Clean, tested on respective post-processing condition.")
    print("* Temporal Interp = minterpolate 30->25->30 fps (motion-compensated blending).")
    print("* Evaluation: Leave-one-scene-out GroupKFold (6 scenes, N=177).")


if __name__ == "__main__":
    main()
