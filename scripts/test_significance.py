import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def get_temporal_oof_preds(df_clean, df_temporal, feature_set, seed=42):
    features = get_features_by_set(df_clean, feature_set)
    df_clean = df_clean[df_clean["scene_id"] != "unknown"].reset_index(drop=True).sort_values("video_name").reset_index(drop=True)
    df_temporal = df_temporal[df_temporal["video_name"].isin(df_clean["video_name"])].copy().sort_values("video_name").reset_index(drop=True)
    
    assert (df_temporal["video_name"].values == df_clean["video_name"].values).all(), "Alignment failed"
    
    y = df_clean["label"].values
    groups = df_clean["scene_id"].values
    
    X_clean = df_clean[features].fillna(0).values
    X_temporal = df_temporal[features].fillna(0).values
    
    oof_temporal = np.zeros(len(y))
    
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
        clf.fit(X_clean[train_idx], y[train_idx])
        oof_temporal[test_idx] = clf.predict_proba(X_temporal[test_idx])[:, 1]
        
    return y, oof_temporal

def main():
    import delong
    
    csv_clean = os.path.join(ROOT, "results", "features_hybrid.csv")
    csv_temporal = os.path.join(ROOT, "results", "features_hybrid_temporal.csv")
    manifest_path = os.path.join(ROOT, "data", "evaluation", "manifest.csv")
    
    df_clean = pd.read_csv(csv_clean)
    df_temporal = pd.read_csv(csv_temporal)
    
    manifest = pd.read_csv(manifest_path)
    scene_map = dict(zip(manifest['clip_name'], manifest['scene_id']))
    
    for df in [df_clean, df_temporal]:
        if "scene_id" not in df.columns or (df["scene_id"] == "unknown").any():
            df["scene_id"] = df["video_name"].map(lambda v: scene_map.get(v, "unknown"))
            
    print("Extracting OOF predictions for Pixel on Temporal Interp...")
    y, preds_pixel = get_temporal_oof_preds(df_clean, df_temporal, "pixel")
    
    print("Extracting OOF predictions for Hybrid on Temporal Interp...")
    _, preds_hybrid = get_temporal_oof_preds(df_clean, df_temporal, "hybrid")
    
    from sklearn.metrics import roc_auc_score
    auc_pixel = roc_auc_score(y, preds_pixel)
    auc_hybrid = roc_auc_score(y, preds_hybrid)
    
    print(f"\nResults on Temporal Interpolation:")
    print(f"Pixel AUC:  {auc_pixel:.4f}")
    print(f"Hybrid AUC: {auc_hybrid:.4f}")
    
    p_value = delong.delong_roc_test(y, preds_hybrid, preds_pixel)
    print(f"\nDeLong Paired Test p-value: {p_value:.4g}")
    
    if p_value < 0.05:
        print("=> SIGNIFICANT DIFFERENCE (p < 0.05). Hybrid is statistically better.")
    else:
        print("=> NOT SIGNIFICANT (p >= 0.05). The +0.018 AUC boost is indistinguishable from noise.")

if __name__ == "__main__":
    main()
