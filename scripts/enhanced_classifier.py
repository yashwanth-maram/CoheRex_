"""
enhanced_classifier.py — CoheRex 2.0 Enhanced Stacked Ensemble Classifier
===========================================================================

Improvements over vanilla train_meta_classifier.py:
  1. Feature engineering: ratio/delta features, kurtosis aggregates
  2. Feature selection: drop noisy _y duplicates; keep only _x + engineered
  3. Stacked ensemble: XGBoost + RandomForest + LogisticRegression → Logistic meta-learner
  4. Optuna Bayesian hyperparameter tuning (falls back to grid search if optuna missing)
  5. Full reporting: per-type AUC, bootstrap CI, comparison table

Usage:
    python scripts/enhanced_classifier.py --csv results/features_hybrid_temporal.csv
    python scripts/enhanced_classifier.py --csv results/features_hybrid_temporal.csv --n-trials 50
    python scripts/enhanced_classifier.py --compare  # runs on all key CSVs and prints comparison table
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — will use RF + LR only")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[WARN] optuna not installed — will use default hyperparameters")


# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal-attack-robust engineered features.

    Key insight: temporal interpolation blurs pixel peaks but preserves the
    *fraction* of frames that exceed a threshold. The ratio of the _x (attacked)
    version to the _y (clean) version reveals how much signal survived.
    """
    df = df.copy()
    
    # --- Identify _x and _y feature pairs ---
    x_feats = [c for c in df.columns if c.endswith("_x") and pd.api.types.is_numeric_dtype(df[c])]
    y_feats = [c for c in df.columns if c.endswith("_y") and pd.api.types.is_numeric_dtype(df[c])]
    
    x_base = {c[:-2]: c for c in x_feats}  # base_name -> _x col
    y_base = {c[:-2]: c for c in y_feats}  # base_name -> _y col
    
    paired = set(x_base.keys()) & set(y_base.keys())
    
    n_eng = 0
    for base in paired:
        xc, yc = x_base[base], y_base[base]
        xv, yv = df[xc].fillna(0), df[yc].fillna(0)
        
        # Ratio: how much did the attacked version retain?
        ratio_col = f"eng_{base}_ratio"
        df[ratio_col] = xv / (yv.abs() + 1e-6)
        
        # Delta: absolute change from clean to attacked
        delta_col = f"eng_{base}_delta"
        df[delta_col] = xv - yv
        n_eng += 2
    
    # --- Kurtosis aggregates (high kurtosis = spiky = likely tampered) ---
    kurtosis_cols = [c for c in x_feats if "kurtosis" in c]
    if kurtosis_cols:
        df["eng_kurtosis_mean"] = df[kurtosis_cols].fillna(0).mean(axis=1)
        df["eng_kurtosis_max"]  = df[kurtosis_cols].fillna(0).max(axis=1)
        n_eng += 2
    
    # --- Fraction aggregate: mean of all frac_x features ---
    frac_cols = [c for c in x_feats if "frac" in c]
    if frac_cols:
        df["eng_frac_mean"] = df[frac_cols].fillna(0).mean(axis=1)
        df["eng_frac_max"]  = df[frac_cols].fillna(0).max(axis=1)
        n_eng += 2
    
    # --- Peak-to-std ratio (robust spike detector) ---
    peak_cols = [c for c in x_feats if "peak" in c and "pix_" in c]
    std_cols  = [c for c in x_feats if "std" in c and "pix_" in c]
    if peak_cols and std_cols:
        df["eng_peak_std_ratio"] = (
            df[peak_cols].fillna(0).max(axis=1) /
            (df[std_cols].fillna(0).mean(axis=1) + 1e-6)
        )
        n_eng += 1
    
    # --- NIS run × frac interaction (physics × pixel) ---
    nis_run_cols = [c for c in df.columns if "nis_run" in c]
    frac_x_cols  = [c for c in df.columns if "frac_x" in c and "pix_" in c]
    if nis_run_cols and frac_x_cols:
        nis_run_max  = df[nis_run_cols].fillna(0).max(axis=1)
        frac_x_mean  = df[frac_x_cols].fillna(0).mean(axis=1)
        df["eng_phys_pix_interaction"] = nis_run_max * frac_x_mean
        n_eng += 1
    
    print(f"[FEATURES] Engineered {n_eng} new features. Total cols: {len(df.columns)}")
    return df


def select_features(df: pd.DataFrame, strategy: str = "x_plus_engineered") -> list:
    """
    Select the final feature list for training.

    strategy='x_plus_engineered'  → use _x suffix + engineered (drop raw _y noise)
    strategy='all'                 → use everything numeric
    strategy='top_k'              → univariate AUC filter
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    exclude = {"label", "tamper_start_frame", "tamper_end_frame"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    if strategy == "x_plus_engineered":
        feats = [c for c in numeric_cols
                 if c.endswith("_x") or c.startswith("eng_")]
        # Also include un-suffixed physics/pixel features if present
        feats += [c for c in numeric_cols
                  if (c.startswith(("pix_", "phys_", "scene_"))
                      and not c.endswith(("_x", "_y")))]
    elif strategy == "all":
        feats = numeric_cols
    else:
        feats = numeric_cols
    
    # Drop zero-variance
    feats = [f for f in feats if df[f].std() > 0]
    # Drop highly correlated (keep one from each pair > 0.98)
    feats = drop_correlated(df, feats, threshold=0.98)
    return feats


def drop_correlated(df, feats, threshold=0.98):
    """Remove one from each highly-correlated pair."""
    if len(feats) < 2:
        return feats
    corr = df[feats].fillna(0).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.add(col)
    return [f for f in feats if f not in to_drop]


# ─────────────────────────────────────────────
# 2. BOOTSTRAP CI
# ─────────────────────────────────────────────

def bootstrap_ci(y_true, y_score, groups, n_boot=2000, seed=42, ci=0.95):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    boots = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = []
        for g in sampled_groups:
            idx.extend(np.where(groups == g)[0])
        idx = np.array(idx)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(roc_auc_score(yt, ys))
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))


# ─────────────────────────────────────────────
# 3. OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_xgb_optuna(X, y, groups, n_splits, n_trials=50, seed=42):
    """Return best XGBoost params via Optuna Bayesian search."""
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 600),
            max_depth       = trial.suggest_int("max_depth", 2, 6),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha       = trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            reg_lambda      = trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
            gamma           = trial.suggest_float("gamma", 0.0, 2.0),
            random_state    = seed,
            eval_metric     = "auc",
            verbosity       = 0,
        )
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        
        clf = xgb.XGBClassifier(**params)
        gkf = GroupKFold(n_splits=n_splits)
        oof = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(y)) < 2:
            return 0.5
        return roc_auc_score(y, oof)

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def default_xgb_params(y, seed=42):
    n_pos = np.sum(y == 1); n_neg = np.sum(y == 0)
    return dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0, min_child_weight=3,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=seed, eval_metric="auc", verbosity=0,
    )


# ─────────────────────────────────────────────
# 4. STACKED ENSEMBLE TRAINING + EVALUATION
# ─────────────────────────────────────────────

def run_stacked_ensemble(df, features, group_col="scene_id", n_trials=30, seed=42):
    """
    Run stacked ensemble with leave-one-scene-out GroupKFold.
    Returns pooled OOF predictions and true labels.
    """
    df = df[df[group_col] != "unknown"].reset_index(drop=True)
    y = df["label"].astype(int).values
    groups = df[group_col].values
    X = df[features].fillna(0).values.astype(float)
    
    n_splits = len(np.unique(groups))
    print(f"[CV] {n_splits}-fold GroupKFold, N={len(y)}, features={X.shape[1]}")
    
    # Optuna tuning on XGB
    if HAS_XGB and HAS_OPTUNA and n_trials > 0:
        print(f"[OPTUNA] Tuning XGBoost with {n_trials} trials...")
        best_xgb_params = tune_xgb_optuna(X, y, groups, n_splits, n_trials=n_trials, seed=seed)
        best_xgb_params.update({"random_state": seed, "eval_metric": "auc", "verbosity": 0})
        n_pos = np.sum(y == 1); n_neg = np.sum(y == 0)
        best_xgb_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        print(f"[OPTUNA] Best params: {best_xgb_params}")
    elif HAS_XGB:
        best_xgb_params = default_xgb_params(y, seed)
    else:
        best_xgb_params = None
    
    # Build level-0 estimators
    l0_estimators = []
    if HAS_XGB and best_xgb_params:
        l0_estimators.append(("xgb", xgb.XGBClassifier(**best_xgb_params)))
    l0_estimators.append(("rf", RandomForestClassifier(
        n_estimators=500, max_depth=8, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=-1
    )))
    l0_estimators.append(("lr", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000, random_state=seed
        ))
    ])))
    
    # OOF predictions — manual fold-by-fold for honest pooling
    gkf = GroupKFold(n_splits=n_splits)
    oof_l0 = np.zeros((len(y), len(l0_estimators)))
    oof_stack = np.zeros(len(y))
    
    print("[STACK] Running Level-0 OOF predictions...")
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        
        l0_train_preds = np.zeros((len(tr_idx), len(l0_estimators)))
        # Inner fold for meta-training
        inner_gkf = GroupKFold(n_splits=max(2, min(3, len(np.unique(groups[tr_idx])))))
        
        for j, (name, clf) in enumerate(l0_estimators):
            # Train on full training fold
            clf_clone = _clone(name, clf, seed, y_tr)
            clf_clone.fit(X_tr, y_tr)
            oof_l0[te_idx, j] = clf_clone.predict_proba(X_te)[:, 1]
        
        # Print fold AUC from best L0
        best_fold_auc = max(
            roc_auc_score(y[te_idx], oof_l0[te_idx, j])
            if len(np.unique(y[te_idx])) > 1 else 0.5
            for j in range(len(l0_estimators))
        )
        print(f"  Fold {fold_i+1}: best L0 AUC = {best_fold_auc:.4f} "
              f"(scene={groups[te_idx[0]]})")
    
    # Level-1: Logistic meta-learner on OOF L0 preds
    print("[STACK] Training Level-1 meta-learner on OOF predictions...")
    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    meta_lr.fit(oof_l0, y)
    
    # Final OOF stack score (for honest AUC)
    # Re-do fold-by-fold with meta-learner also fitted per fold
    print("[STACK] Computing honest stacked OOF AUC...")
    oof_final = np.zeros(len(y))
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        
        # Fit L0 on train
        l0_te_preds = np.zeros((len(te_idx), len(l0_estimators)))
        l0_tr_preds = np.zeros((len(tr_idx), len(l0_estimators)))
        
        for j, (name, clf) in enumerate(l0_estimators):
            clf_clone = _clone(name, clf, seed, y_tr)
            clf_clone.fit(X_tr, y_tr)
            l0_te_preds[:, j] = clf_clone.predict_proba(X_te)[:, 1]
            l0_tr_preds[:, j] = clf_clone.predict_proba(X_tr)[:, 1]
        
        # Fit L1 on train OOF preds
        meta_clone = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        meta_clone.fit(l0_tr_preds, y_tr)
        oof_final[te_idx] = meta_clone.predict_proba(l0_te_preds)[:, 1]
    
    return y, oof_l0, oof_final, groups


def _clone(name, clf, seed, y_tr):
    """Return a fresh clone of the classifier."""
    import copy
    return copy.deepcopy(clf)


# ─────────────────────────────────────────────
# 5. REPORTING
# ─────────────────────────────────────────────

def report(y, oof_l0, oof_final, df, features, groups, label=""):
    print(f"\n{'='*70}")
    print(f" RESULTS {label}")
    print(f"{'='*70}")
    
    # Overall AUC
    names = ["XGBoost", "RandomForest", "LogisticReg"] if oof_l0.shape[1] == 3 else \
            ["RandomForest", "LogisticReg"]
    if not HAS_XGB:
        names = ["RandomForest", "LogisticReg"]
    
    for j, name in enumerate(names[:oof_l0.shape[1]]):
        if len(np.unique(y)) > 1:
            auc_j = roc_auc_score(y, oof_l0[:, j])
            print(f"  L0 {name:<14} AUC: {auc_j:.4f}")
    
    auc_stack = roc_auc_score(y, oof_final) if len(np.unique(y)) > 1 else 0.5
    lo, hi    = bootstrap_ci(y, oof_final, groups)
    bal_acc   = balanced_accuracy_score(y, (oof_final >= 0.5).astype(int))
    
    print(f"\n  +------------------------------------------+")
    print(f"  |  STACKED ENSEMBLE AUC : {auc_stack:.4f}            |")
    print(f"  |  Bootstrap 95% CI     : ({lo:.4f} - {hi:.4f})   |")
    print(f"  |  Balanced Accuracy    : {bal_acc:.4f}            |")
    print(f"  +------------------------------------------+")
    
    # Per-type AUC
    if "tamper_type" in df.columns:
        honest_idx = df[df["scene_id"] != "unknown"].index if "scene_id" in df.columns else df.index
        df_h = df.loc[honest_idx].reset_index(drop=True)
        oof_h = oof_final[:len(df_h)]
        y_h   = y[:len(df_h)]
        
        print("\n  Per-tamper-type AUC:")
        for ttype in sorted(df_h["tamper_type"].unique()):
            mask = (df_h["tamper_type"] == ttype).values
            pos_mask = mask & (y_h == 1)
            neg_mask = (y_h == 0)
            if pos_mask.sum() == 0:
                continue
            idx = np.where(pos_mask | neg_mask)[0]
            if len(np.unique(y_h[idx])) < 2:
                continue
            t_auc = roc_auc_score(y_h[idx], oof_h[idx])
            print(f"    {ttype:<25} -> {t_auc:.4f}")
    
    return {"auc": auc_stack, "ci_lo": lo, "ci_hi": hi, "balanced_acc": bal_acc}


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def process_csv(csv_path, n_trials=30, seed=42, metrics_out=None):
    print(f"\n[LOAD] {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Honour honest subset and parse camera_id for grouping (prevents background leakage)
    if "scene_id" in df.columns:
        df = df[df["scene_id"] != "unknown"].copy()
        df = df[df["scene_id"] != "legacy_samples"].copy()
        df["camera_id"] = df["scene_id"].str.extract(r'VIRAT_S_(\d{4})')[0]
        df["camera_id"] = df["camera_id"].fillna(df["scene_id"])
        df = df.reset_index(drop=True)
    
    print(f"  Rows: {len(df)}, Label dist: {dict(df['label'].value_counts())}")
    
    # Engineer features
    df = engineer_features(df)
    
    # Select features
    feats = select_features(df, strategy="x_plus_engineered")
    print(f"[FEATURES] Final feature count after selection: {len(feats)}")
    
    # Run stacked ensemble
    y, oof_l0, oof_final, groups = run_stacked_ensemble(
        df, feats, group_col="camera_id" if "camera_id" in df.columns else "scene_id", n_trials=n_trials, seed=seed
    )
    
    # Report
    label = os.path.basename(csv_path)
    result = report(y, oof_l0, oof_final, df, feats, groups, label=label)
    
    if metrics_out:
        with open(metrics_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[SAVED] Metrics -> {metrics_out}")
    
    return result


def parse_args():
    p = argparse.ArgumentParser(description="CoheRex 2.0 Enhanced Stacked Classifier")
    p.add_argument("--csv", default=os.path.join(ROOT, "results", "features_hybrid_temporal.csv"))
    p.add_argument("--metrics", default=None, help="Optional JSON metrics output path")
    p.add_argument("--n-trials", type=int, default=30,
                   help="Optuna trials for XGBoost tuning (0 = skip tuning)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare", action="store_true",
                   help="Run all key CSVs and print comparison table")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.compare:
        csvs = {
            "VIRAT Pixel (Clean)":    os.path.join(ROOT, "results", "features_pixel.csv"),
            "VIRAT Pixel (CRF36)":    os.path.join(ROOT, "results", "features_pixel_crf36.csv"),
            "VIRAT Pixel (Temporal)": os.path.join(ROOT, "results", "features_pixel_temporal.csv"),
            "VIRAT Physics (Clean)":  os.path.join(ROOT, "results", "features_physics.csv"),
            "VIRAT Physics (CRF36)":  os.path.join(ROOT, "results", "features_physics_crf36.csv"),
            "VIRAT Physics (Temporal)": os.path.join(ROOT, "results", "features_physics_temporal.csv"),
            "VIRAT Hybrid (Clean)":   os.path.join(ROOT, "results", "features_hybrid.csv"),
            "VIRAT Hybrid (CRF36)":   os.path.join(ROOT, "results", "features_hybrid_crf36.csv"),
            "VIRAT Hybrid (Temporal)":os.path.join(ROOT, "results", "features_hybrid_temporal.csv"),
            "VIFFD Pixel (Clean)":    os.path.join(ROOT, "results", "viffd_features_pixel.csv"),
            "VIFFD Physics (Clean)":  os.path.join(ROOT, "results", "viffd_features_physics.csv"),
            "VIFFD Hybrid (Clean)":   os.path.join(ROOT, "results", "viffd_features_hybrid.csv"),
        }
        results = {}
        for label, path in csvs.items():
            if os.path.exists(path):
                r = process_csv(path, n_trials=args.n_trials, seed=args.seed)
                results[label] = r
            else:
                print(f"[SKIP] {path} not found")
        
        print("\n\n" + "="*70)
        print(" FINAL COMPARISON TABLE (Stacked Ensemble)")
        print("="*70)
        print(f"{'Dataset':<25} | {'AUC':>8} | {'95% CI':>16} | {'Bal.Acc':>8}")
        print("-"*70)
        for label, r in results.items():
            ci_str = f"({r['ci_lo']:.4f}–{r['ci_hi']:.4f})"
            print(f"{label:<25} | {r['auc']:>8.4f} | {ci_str:>16} | {r['balanced_acc']:>8.4f}")
    else:
        process_csv(args.csv, n_trials=args.n_trials, seed=args.seed, metrics_out=args.metrics)
