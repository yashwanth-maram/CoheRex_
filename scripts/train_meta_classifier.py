# scripts/train_meta_classifier.py
"""
CoheRex 2.0 — Meta-Classifier Training with Leak-Free Evaluation
===================================================================

Supports detection modes:
  --model logreg   : LogisticRegression (principled for small N)
  --model rf       : RandomForest (legacy)
  --model xgboost  : XGBoost (default, best AUC)
  --model oneclass : One-class physics prior (manipulation-agnostic)

Feature-set ablation:
  --feature-set pixel   : only pix_* features (the accuracy engine)
  --feature-set physics : only phys_* features (the robustness engine)
  --feature-set hybrid  : pix_* + phys_* + scene_* (default)

Evaluation:
  - GroupKFold (groups = scene_id) — no scene leakage
  - Pooled OOF AUC + bootstrap 95% CI as primary metric
  - Balanced accuracy (handles class imbalance honestly)
  - Per-tamper-type AUC breakdown
  - All results saved to results/metrics.json

Usage:
    python scripts/train_meta_classifier.py --csv results/features_hybrid.csv
    python scripts/train_meta_classifier.py --csv results/features_hybrid.csv --feature-set pixel
    python scripts/train_meta_classifier.py --csv results/features_hybrid.csv --model xgboost --feature-set hybrid
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

from sklearn.model_selection import GroupKFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve,
    classification_report, accuracy_score, balanced_accuracy_score,
)

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Canonical feature set (from feature_extractor.py)
try:
    from coherex.meta.feature_extractor import FEATURE_NAMES
except ImportError:
    FEATURE_NAMES = []  # Will auto-detect from CSV


def parse_args():
    p = argparse.ArgumentParser(description="CoheRex 2.0 Meta-Classifier Trainer")
    p.add_argument("--csv", default=os.path.join(ROOT, "results", "features_hybrid.csv"),
                   help="Path to features CSV")
    p.add_argument("--output", default=os.path.join(ROOT, "data", "models", "meta_classifier.pkl"),
                   help="Where to save the trained model bundle")
    p.add_argument("--metrics", default=os.path.join(ROOT, "results", "metrics.json"),
                   help="Where to save metrics JSON")
    p.add_argument("--model", default="xgboost",
                   choices=["logreg", "oneclass", "rf", "xgboost"],
                   help="Model type (default: xgboost)")
    p.add_argument("--feature-set", default="hybrid",
                   choices=["pixel", "physics", "hybrid"],
                   help="Feature channel: pixel (pix_*), physics (phys_*), hybrid (all)")
    p.add_argument("--group-col", default="scene_id",
                   help="Column to use for GroupKFold grouping (default: scene_id)")
    p.add_argument("--n-splits", type=int, default=None,
                   help="Number of CV splits (default: number of groups = leave-one-out)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def select_features(df, feature_set):
    """Select features based on channel prefix."""
    all_cols = list(df.columns)
    
    if feature_set == "pixel":
        features = [c for c in all_cols if c.startswith("pix_")]
    elif feature_set == "physics":
        features = [c for c in all_cols if c.startswith("phys_")]
    elif feature_set == "hybrid":
        features = [c for c in all_cols if c.startswith(("pix_", "phys_", "scene_"))]
    else:
        features = []
    
    # Fallback: if no pix_/phys_ prefixed columns, use FEATURE_NAMES (legacy CSV)
    if not features and FEATURE_NAMES:
        features = [f for f in FEATURE_NAMES if f in all_cols]
    
    # Last resort: use all numeric columns except metadata
    if not features:
        meta = {"video_name", "label", "source_id", "scene_id", "tamper_type",
                "tamper_start_frame", "tamper_end_frame", "physics_available"}
        features = [c for c in all_cols if c not in meta and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]
    
    return features


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
    lo = np.percentile(boots, alpha * 100)
    hi = np.percentile(boots, (1 - alpha) * 100)
    return float(lo), float(hi)


def per_type_auc(df, y_proba):
    """Compute AUC per tamper type."""
    results = {}
    if "tamper_type" not in df.columns:
        return results

    # Get authentic samples
    auth_mask = df["label"] == 0
    auth_proba = y_proba[auth_mask]
    auth_labels = np.zeros(auth_mask.sum())

    for ttype in df.loc[df["label"] == 1, "tamper_type"].unique():
        if ttype in ("none", "unknown"):
            continue
        tamp_mask = (df["label"] == 1) & (df["tamper_type"] == ttype)
        tamp_proba = y_proba[tamp_mask]
        tamp_labels = np.ones(tamp_mask.sum())

        combined_y = np.concatenate([auth_labels, tamp_labels])
        combined_p = np.concatenate([auth_proba, tamp_proba])

        if len(np.unique(combined_y)) < 2 or len(combined_y) < 4:
            results[ttype] = None
            continue
        results[ttype] = float(roc_auc_score(combined_y, combined_p))

    return results


def grouped_oof_logreg(X, y, groups, n_splits=5, seed=42):
    """GroupKFold with LogisticRegression, returns pooled OOF predictions."""
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)

    gkf = GroupKFold(n_splits=actual_splits)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=seed, solver="lbfgs")
    proba = cross_val_predict(clf, X, y, groups=groups, cv=gkf,
                              method="predict_proba")[:, 1]
    return proba


def grouped_oof_rf(X, y, groups, n_splits=5, seed=42):
    """GroupKFold with RandomForest."""
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)

    gkf = GroupKFold(n_splits=actual_splits)
    clf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                 min_samples_leaf=1, class_weight="balanced",
                                 random_state=seed, n_jobs=-1)
    proba = cross_val_predict(clf, X, y, groups=groups, cv=gkf,
                              method="predict_proba")[:, 1]
    return proba


def grouped_oof_xgboost(X, y, groups, n_splits=5, seed=42):
    """GroupKFold with XGBoost using Nested CV for hyperparameter tuning."""
    if not HAS_XGBOOST:
        print("[ERROR] XGBoost not installed. Run: pip install xgboost")
        sys.exit(1)
    
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    
    outer_cv = GroupKFold(n_splits=actual_splits)
    proba = np.zeros(len(y))
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [300, 500],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }
    
    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, y_train = X[train_idx], y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        X_test = X[test_idx]
        groups_train = groups.iloc[train_idx] if hasattr(groups, 'iloc') else groups[train_idx]
        
        # Compute scale_pos_weight for class imbalance on training fold
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        inner_cv = GroupKFold(n_splits=3)
        base_clf = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="auc",
            verbosity=0,
            n_jobs=1
        )
        
        grid = GridSearchCV(base_clf, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train, groups=groups_train)
        
        best_clf = grid.best_estimator_
        proba[test_idx] = best_clf.predict_proba(X_test)[:, 1]
        
    return proba


def oneclass_scores(df, X, y, groups):
    """One-class physics prior scoring."""
    from coherex.meta.physics_prior import fit_and_score_grouped
    return fit_and_score_grouped(X, y.values, groups.values,
                                 method="elliptic", contamination=0.05)


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv).dropna(subset=["label"])
    print(f"[LOAD] {len(df)} total samples from {args.csv}")

    # Filter out videos without track evidence (only for legacy CSVs)
    if "physics_available" in df.columns:
        df["physics_available"] = df["physics_available"].astype(str).str.lower() == "true"
        dropped = len(df[~df["physics_available"]])
        df = df[df["physics_available"]].copy()
        if dropped > 0:
            print(f"[FILTER] Dropped {dropped} samples with no physics available (empty scenes).")
    elif "phys_nis_peak" in df.columns:
        # New-style CSV: drop if all physics features are 0
        pass  # Keep all samples — pixel features work even without physics
    else:
        # Legacy: drop if nis_peak == -1.0
        if "nis_peak" in df.columns:
            dropped = len(df[df["nis_peak"] == -1.0])
            df = df[df["nis_peak"] != -1.0].copy()
            if dropped > 0:
                print(f"[FILTER] Dropped {dropped} samples with missing physics (-1.0).")

    # Select features based on --feature-set
    available_features = select_features(df, args.feature_set)
    if not available_features:
        print(f"[ERROR] No features found for feature-set '{args.feature_set}'.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    # Remove constant features (zero variance)
    valid_features = [f for f in available_features if pd.api.types.is_numeric_dtype(df[f]) and df[f].std() > 0]
    dropped_const = len(available_features) - len(valid_features)
    if dropped_const > 0:
        print(f"[FILTER] Dropped {dropped_const} constant features.")
    available_features = valid_features

    X = df[available_features].fillna(0).values.astype(float)
    y = df["label"].astype(int)

    # Groups for GroupKFold
    if args.group_col in df.columns:
        groups = df[args.group_col]
    elif "source_id" in df.columns:
        print(f"[WARN] Group column '{args.group_col}' not found. Falling back to source_id.")
        groups = df["source_id"]
    else:
        print(f"[WARN] No group column found. Using video_name as group.")
        groups = df["video_name"]

    n_groups = groups.nunique()
    # Auto-set n_splits to number of groups (leave-one-group-out) if not specified
    n_splits = args.n_splits if args.n_splits is not None else n_groups
    n_splits = min(n_splits, n_groups)

    print(f"\n[DATA] {len(y)} samples, {(y==0).sum()} authentic, {(y==1).sum()} tampered")
    print(f"[DATA] {n_groups} groups ({args.group_col}), {n_splits}-fold CV")
    print(f"[DATA] Feature set: {args.feature_set} ({len(available_features)} features)")

    # ── Sanity checks ──
    print(f"\n[SANITY] Class distribution:")
    print(y.value_counts().to_string())
    print(f"\n[FEATURES] Using {len(available_features)} features:")
    for f in available_features:
        print(f"    {f}")

    # ── Train / evaluate ──
    metrics = {"model": args.model, "feature_set": args.feature_set,
               "n_samples": len(y), "n_groups": n_groups,
               "n_splits": n_splits, "features": available_features}

    if args.model == "logreg":
        print(f"\n[TRAIN] LogisticRegression with GroupKFold (k={n_splits})...")
        y_proba = grouped_oof_logreg(X, y, groups, n_splits=n_splits, seed=args.seed)

    elif args.model == "rf":
        print(f"\n[TRAIN] RandomForest with GroupKFold (k={n_splits})...")
        y_proba = grouped_oof_rf(X, y, groups, n_splits=n_splits, seed=args.seed)

    elif args.model == "xgboost":
        print(f"\n[TRAIN] XGBoost with GroupKFold (k={n_splits})...")
        y_proba = grouped_oof_xgboost(X, y, groups, n_splits=n_splits, seed=args.seed)

    elif args.model == "oneclass":
        print(f"\n[TRAIN] One-class physics prior (GroupKFold)...")
        y_proba = oneclass_scores(df, X, y, groups)

    # ── Compute metrics ──
    auc = roc_auc_score(y, y_proba)
    lo, hi = bootstrap_ci(y.values, y_proba, seed=args.seed)

    print(f"\n{'='*60}")
    print(f"  POOLED OOF AUC: {auc:.4f}  [95% CI: {lo:.4f} – {hi:.4f}]")
    print(f"{'='*60}")

    metrics["auc"] = round(auc, 4)
    metrics["auc_ci_lo"] = round(lo, 4)
    metrics["auc_ci_hi"] = round(hi, 4)

    # Calculate Youden's J threshold
    fpr_roc, tpr_roc, thresholds = roc_curve(y, y_proba)
    j_scores = tpr_roc - fpr_roc
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n  Confusion Matrix (threshold=0.5):")
    # Confusion matrix at 0.5 threshold
    y_pred_05 = (y_proba >= 0.5).astype(int)
    cm_05 = confusion_matrix(y, y_pred_05, labels=[0, 1])
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()
    
    n = tn_05 + fp_05 + fn_05 + tp_05
    accuracy_05 = (tp_05 + tn_05) / n if n > 0 else 0
    bal_accuracy_05 = balanced_accuracy_score(y, y_pred_05)
    precision_05 = tp_05 / (tp_05 + fp_05) if (tp_05 + fp_05) else 0.0
    recall_05 = tp_05 / (tp_05 + fn_05) if (tp_05 + fn_05) else 0.0
    specificity_05 = tn_05 / (tn_05 + fp_05) if (tn_05 + fp_05) else 0.0
    f1_05 = (2 * precision_05 * recall_05 / (precision_05 + recall_05)) if (precision_05 + recall_05) else 0.0

    print(f"                   Pred Auth   Pred Tamp")
    print(f"  True Authentic    TN={tn_05:<5}    FP={fp_05}")
    print(f"  True Tampered     FN={fn_05:<5}    TP={tp_05}")
    print(f"\n  Accuracy          : {accuracy_05:.4f}")
    print(f"  Balanced Accuracy : {bal_accuracy_05:.4f}")
    print(f"  Precision         : {precision_05:.4f}")
    print(f"  Recall            : {recall_05:.4f}")
    print(f"  Specificity       : {specificity_05:.4f}")
    print(f"  F1-score          : {f1_05:.4f}")
    
    print(f"\n  Confusion Matrix (threshold=Youden's J: {optimal_threshold:.4f}):")
    # Confusion matrix at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / n if n > 0 else 0
    bal_accuracy = balanced_accuracy_score(y, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"                   Pred Auth   Pred Tamp")
    print(f"  True Authentic    TN={tn:<5}    FP={fp}")
    print(f"  True Tampered     FN={fn:<5}    TP={tp}")
    print(f"\n  Accuracy          : {accuracy:.4f}")
    print(f"  Balanced Accuracy : {bal_accuracy:.4f}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  Specificity       : {specificity:.4f}")
    print(f"  F1-score          : {f1:.4f}")

    metrics["optimal_threshold"] = round(optimal_threshold, 4)
    metrics["accuracy"] = round(accuracy, 4)
    metrics["balanced_accuracy"] = round(bal_accuracy, 4)
    metrics["precision"] = round(precision, 4)
    metrics["recall"] = round(recall, 4)
    metrics["specificity"] = round(specificity, 4)
    metrics["f1"] = round(f1, 4)
    metrics["f1_05"] = round(f1_05, 4)
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp),
                                    "fn": int(fn), "tp": int(tp)}

    # Per-type AUC
    type_aucs = per_type_auc(df, y_proba)
    if type_aucs:
        print(f"\n  Per-Type AUC:")
        for ttype, tauc in sorted(type_aucs.items()):
            auc_str = f"{tauc:.4f}" if tauc is not None else "N/A (too few samples)"
            print(f"    {ttype:<25s}  AUC = {auc_str}")
    metrics["per_type_auc"] = {k: round(v, 4) if v is not None else None
                                for k, v in type_aucs.items()}

    # ── Save metrics ──
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVED] Metrics -> {args.metrics}")

    # ── Save model bundle (for logreg and rf) ──
    if args.model in ("logreg", "rf"):
        import joblib
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        # Retrain on full data for the saved model
        if args.model == "logreg":
            final_clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                           random_state=args.seed, solver="lbfgs")
        else:
            final_clf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                               min_samples_leaf=1, class_weight="balanced",
                                               random_state=args.seed, n_jobs=-1)
        final_clf.fit(X, y)

        bundle = {
            "model": final_clf,
            "feature_names": available_features,
            "cv_auc": auc,
            "cv_auc_ci": (lo, hi),
            "model_type": args.model,
        }
        joblib.dump(bundle, args.output)
        print(f"[SAVED] Model bundle -> {args.output}")

        # Feature importances
        if hasattr(final_clf, 'feature_importances_'):
            importances = final_clf.feature_importances_
        elif hasattr(final_clf, 'coef_'):
            importances = np.abs(final_clf.coef_[0])
        else:
            importances = None

        if importances is not None:
            indices = np.argsort(importances)[::-1]
            print(f"\n  Feature Importances:")
            for i in indices:
                print(f"    {available_features[i]:<25s}  {importances[i]:.4f}")

    # ── ROC curve ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='b', lw=2,
                 label=f'{args.model} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], '--', color='r', lw=1, label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'CoheRex 2.0 ROC — {args.model} (GroupKFold)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)

        roc_path = os.path.join(ROOT, "results", f"roc_{args.model}.png")
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] ROC curve -> {roc_path}")
    except Exception as e:
        print(f"[WARN] Could not save ROC plot: {e}")

    print(f"\n[DONE] All results in {os.path.dirname(args.metrics)}/")


if __name__ == "__main__":
    main()
