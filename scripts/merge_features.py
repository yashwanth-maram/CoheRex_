"""
merge_features.py - Merge pixel and physics CSVs into a hybrid CSV
====================================================================

Takes pixel-sourced features (pix_*) from the pixel CSV and
physics-sourced features (phys_*, scene_*) from the physics CSV.
Metadata comes from the pixel CSV. No _x/_y suffixes.

Usage:
    python scripts/merge_features.py --pixel results/features_pixel.csv \
                                     --physics results/features_physics.csv \
                                     --out results/features_hybrid.csv
"""

import os
import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Merge pixel and physics CSVs")
    p.add_argument("--pixel", type=str, required=True, help="Path to pixel features CSV")
    p.add_argument("--physics", type=str, required=True, help="Path to physics features CSV")
    p.add_argument("--out", type=str, required=True, help="Path to output hybrid CSV")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.pixel):
        print(f"[ERROR] Pixel CSV not found: {args.pixel}")
        return
    if not os.path.exists(args.physics):
        print(f"[ERROR] Physics CSV not found: {args.physics}")
        return

    df_pix = pd.read_csv(args.pixel)
    df_phys = pd.read_csv(args.physics)

    print(f"Loaded pixel features: {len(df_pix)} rows, {len(df_pix.columns)} cols")
    print(f"Loaded physics features: {len(df_phys)} rows, {len(df_phys.columns)} cols")

    # Metadata columns (keep from pixel CSV only)
    meta_cols = ["video_name", "label", "source_id", "scene_id", "tamper_type",
                 "tamper_start_frame", "tamper_end_frame"]

    # Extract pixel-sourced features from pixel CSV
    pix_features = [c for c in df_pix.columns if c.startswith("pix_")]

    # Extract physics-sourced features from physics CSV
    phys_features = [c for c in df_phys.columns if c.startswith("phys_")]
    scene_features = [c for c in df_phys.columns if c.startswith("scene_") and c != "scene_id"]

    print(f"  Pixel features from pixel CSV: {len(pix_features)}")
    print(f"  Physics features from physics CSV: {len(phys_features)}")
    print(f"  Scene features from physics CSV: {len(scene_features)}")

    # Build join key: basename (extension-agnostic)
    df_pix["_basename"] = df_pix["video_name"].apply(lambda x: str(x).rsplit(".", 1)[0])
    df_phys["_basename"] = df_phys["video_name"].apply(lambda x: str(x).rsplit(".", 1)[0])

    # Select only the columns we need from each
    meta_available = [c for c in meta_cols if c in df_pix.columns]
    df_pix_select = df_pix[meta_available + pix_features + ["_basename"]].copy()
    df_phys_select = df_phys[phys_features + scene_features + ["_basename"]].copy()

    # Merge
    df_hybrid = pd.merge(df_pix_select, df_phys_select, on="_basename", how="inner")
    df_hybrid = df_hybrid.drop(columns=["_basename"])

    # Verify no _x/_y suffixes
    bad_cols = [c for c in df_hybrid.columns if c.endswith("_x") or c.endswith("_y")]
    if bad_cols:
        print(f"[ERROR] Merge produced duplicate columns: {bad_cols}")
        return

    print(f"Merged hybrid features: {len(df_hybrid)} rows, {len(df_hybrid.columns)} cols")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df_hybrid.to_csv(args.out, index=False)
    print(f"[DONE] Saved {args.out}")


if __name__ == "__main__":
    main()
