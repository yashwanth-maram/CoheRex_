import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.train_meta_classifier import select_features
from sklearn.model_selection import GroupKFold
import xgboost as xgb

def fast_grouped_oof_xgboost(X, y, groups, n_splits=5, seed=42):
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    outer_cv = GroupKFold(n_splits=actual_splits)
    proba = np.zeros(len(y))
    
    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, y_train = X[train_idx], y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        X_test = X[test_idx]
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        clf = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="auc",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            verbosity=0,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        proba[test_idx] = clf.predict_proba(X_test)[:, 1]
    return proba


def bootstrap_test(y_true, y_pred1, y_pred2, groups, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    diffs = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = []
        for g in sampled_groups:
            idx.extend(np.where(groups == g)[0])
        idx = np.array(idx)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc1 = roc_auc_score(y_true[idx], y_pred1[idx])
        auc2 = roc_auc_score(y_true[idx], y_pred2[idx])
        diffs.append(auc2 - auc1) # Model 2 - Model 1
    
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lo = np.percentile(diffs, 2.5)
    ci_hi = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0) if mean_diff > 0 else np.mean(diffs >= 0)
    p_value = min(p_value * 2, 1.0) # two-tailed
    return mean_diff, ci_lo, ci_hi, p_value

# 1. VIRAT Test
print("=== VIRAT: Hybrid vs Pixel ===")
df_virat = pd.read_csv('results/features_hybrid.csv').dropna(subset=['label'])
df_virat = df_virat[df_virat['scene_id'] != 'legacy_samples'].copy()
df_virat['camera_id'] = df_virat['scene_id'].str.extract(r'VIRAT_S_(\d{4})')[0]
df_virat['camera_id'] = df_virat['camera_id'].fillna(df_virat['scene_id']) # fallback just in case

y_virat = df_virat['label'].astype(int)
groups_virat = df_virat['camera_id']

# Pixel
feats_virat_pix = select_features(df_virat, 'pixel')
X_virat_pix = df_virat[feats_virat_pix].fillna(0).values.astype(float)
preds_virat_pix = fast_grouped_oof_xgboost(X_virat_pix, y_virat, groups_virat, n_splits=6)
auc_virat_pix = roc_auc_score(y_virat, preds_virat_pix)
print(f"VIRAT Pixel AUC: {auc_virat_pix:.4f}")

# Hybrid
feats_virat_hyb = select_features(df_virat, 'hybrid')
feats_virat_hyb = [f for f in feats_virat_hyb if f != 'scene_id']
X_virat_hyb = df_virat[feats_virat_hyb].fillna(0).values.astype(float)
preds_virat_hyb = fast_grouped_oof_xgboost(X_virat_hyb, y_virat, groups_virat, n_splits=6)
auc_virat_hyb = roc_auc_score(y_virat, preds_virat_hyb)
print(f"VIRAT Hybrid AUC: {auc_virat_hyb:.4f}")

diff_v, lo_v, hi_v, p_v = bootstrap_test(y_virat.values, preds_virat_pix, preds_virat_hyb, groups_virat.values)
print(f"Diff (Hybrid - Pixel): {diff_v:.4f} [95% CI: {lo_v:.4f} to {hi_v:.4f}], p={p_v:.4f}")


# 2. VIFFD Test
print("\n=== VIFFD: Gated Hybrid vs Pixel ===")
df_viffd_pix = pd.read_csv('results/viffd_features_pixel.csv').dropna(subset=['label'])
df_viffd_gated = pd.read_csv('results/viffd_features_gated_hybrid.csv').dropna(subset=['label'])

# Ensure alignment
df_viffd_pix = df_viffd_pix.sort_values('video_name').reset_index(drop=True)
df_viffd_gated = df_viffd_gated.sort_values('video_name').reset_index(drop=True)
assert all(df_viffd_pix['video_name'] == df_viffd_gated['video_name'])

y_viffd = df_viffd_pix['label'].astype(int)
groups_viffd = df_viffd_pix['scene_id']

# Pixel
feats_viffd_pix = select_features(df_viffd_pix, 'pixel')
X_viffd_pix = df_viffd_pix[feats_viffd_pix].fillna(0).values.astype(float)
preds_viffd_pix = fast_grouped_oof_xgboost(X_viffd_pix, y_viffd, groups_viffd)
auc_viffd_pix = roc_auc_score(y_viffd, preds_viffd_pix)
print(f"VIFFD Pixel AUC: {auc_viffd_pix:.4f}")

# Gated Hybrid
feats_viffd_gated = select_features(df_viffd_gated, 'hybrid')
feats_viffd_gated = [f for f in feats_viffd_gated if f != 'scene_id']
X_viffd_gated = df_viffd_gated[feats_viffd_gated].fillna(0).values.astype(float)
preds_viffd_gated = fast_grouped_oof_xgboost(X_viffd_gated, y_viffd, groups_viffd)
auc_viffd_gated = roc_auc_score(y_viffd, preds_viffd_gated)
print(f"VIFFD Gated Hybrid AUC: {auc_viffd_gated:.4f}")

diff_viffd, lo_viffd, hi_viffd, p_viffd = bootstrap_test(y_viffd.values, preds_viffd_pix, preds_viffd_gated, groups_viffd.values)
print(f"Diff (Gated Hybrid - Pixel): {diff_viffd:.4f} [95% CI: {lo_viffd:.4f} to {hi_viffd:.4f}], p={p_viffd:.4f}")
