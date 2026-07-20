"""
sample_sanity_check.py — Quick 20-clip sanity check for NIS scale fix
=====================================================================

Creates a temporary directory structure with symlinks/copies of 10 authentic
+ 10 tampered clips, then runs evaluate_dataset.py on that subset.
After evaluation, prints a summary table of nis_peak values per class
to verify they're in a sane range (single digits to low tens).

Usage:
    python scripts/sample_sanity_check.py
"""

import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

AUTH_DIR = os.path.join(ROOT, "data", "evaluation", "authentic")
TAMP_DIR = os.path.join(ROOT, "data", "evaluation", "tampered")
SAMPLE_DIR = os.path.join(ROOT, "data", "evaluation_sample")
SAMPLE_AUTH = os.path.join(SAMPLE_DIR, "authentic")
SAMPLE_TAMP = os.path.join(SAMPLE_DIR, "tampered")
OUTPUT_CSV = os.path.join(ROOT, "results", "features_sample.csv")

# Pick 10 authentic + 10 tampered — all from the freshly rebuilt activity-aware
# dataset (auth_0001–auth_0052, tamp_0001–tamp_0125). Every clip here was
# selected from a source window with >=60% person-presence by the build script.
AUTH_PICKS = [
    "auth_0001.mp4",   # src_005 VIRAT_S_010000 78% person
    "auth_0003.mp4",   # src_009 VIRAT_S_010000 100% person
    "auth_0006.mp4",   # src_012 VIRAT_S_010000 88% person
    "auth_0009.mp4",   # src_013 VIRAT_S_010000 100% person
    "auth_0013.mp4",   # src_023 VIRAT_S_010000 100% person
    "auth_0020.mp4",   # mid-dataset, activity-aware
    "auth_0030.mp4",   # mid-dataset, activity-aware
    "auth_0042.mp4",   # src_047 100% person (55 windows)
    "auth_0049.mp4",   # src_051 95% person
    "auth_0052.mp4",   # src_052 82% person (last source)
]
TAMP_PICKS = [
    "tamp_0001_frame_deletion.mp4",      # src_005, frame_deletion
    "tamp_0004_clip_splice.mp4",         # src_005, clip_splice
    "tamp_0009_clip_splice.mp4",         # src_009, clip_splice
    "tamp_0014_clip_splice.mp4",         # src_012, clip_splice
    "tamp_0030_reverse_segment.mp4",     # src_013 area, reverse_segment
    "tamp_0055_reverse_segment.mp4",     # mid-dataset, reverse_segment
    "tamp_0079_clip_splice.mp4",         # src_038, clip_splice 100% person
    "tamp_0104_clip_splice.mp4",         # src_047, clip_splice 100% person
    "tamp_0114_clip_splice.mp4",         # src_049, clip_splice 100% person
    "tamp_0125_reverse_segment.mp4",     # src_052, last tamp clip
]


def setup_sample():
    """Create sample directory with hard-links to the selected clips."""
    # Clean up any previous sample run
    if os.path.exists(SAMPLE_DIR):
        shutil.rmtree(SAMPLE_DIR)
    os.makedirs(SAMPLE_AUTH, exist_ok=True)
    os.makedirs(SAMPLE_TAMP, exist_ok=True)

    linked = 0
    for fname in AUTH_PICKS:
        src = os.path.join(AUTH_DIR, fname)
        dst = os.path.join(SAMPLE_AUTH, fname)
        if os.path.exists(src):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            linked += 1
        else:
            print(f"  [WARN] Missing: {src}")

    for fname in TAMP_PICKS:
        src = os.path.join(TAMP_DIR, fname)
        dst = os.path.join(SAMPLE_TAMP, fname)
        if os.path.exists(src):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            linked += 1
        else:
            print(f"  [WARN] Missing: {src}")

    print(f"[SAMPLE] Linked {linked} clips into {SAMPLE_DIR}")
    return linked


def run_evaluation():
    """Run evaluate_dataset.py on the sample directory."""
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts", "evaluate_dataset.py"),
        "--input", SAMPLE_DIR,
        "--output", OUTPUT_CSV,
    ]
    print(f"\n[RUN] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def analyze_results():
    """Print a summary of nis_peak values by class."""
    if not os.path.exists(OUTPUT_CSV):
        print("[ERROR] Output CSV not found!")
        return False

    df = pd.read_csv(OUTPUT_CSV)
    
    # Filter to physics-available clips only
    df_phys = df[df["physics_available"].astype(str).str.lower() == "true"]
    
    print("\n" + "=" * 70)
    print("NIS PEAK SANITY CHECK -- 20-Clip Sample")
    print("=" * 70)
    
    print(f"\nTotal clips evaluated: {len(df)}")
    print(f"Physics-available clips: {len(df_phys)}")
    
    if len(df_phys) == 0:
        print("[WARN] No physics-available clips -- cannot assess NIS range.")
        return False
    
    # Per-class stats
    for label, name in [(0, "Authentic"), (1, "Tampered")]:
        subset = df_phys[df_phys["label"] == label]
        if len(subset) == 0:
            print(f"\n  {name}: no clips with physics data")
            continue
        
        vals = subset["nis_peak"].values
        print(f"\n  {name} (n={len(subset)}):")
        print(f"    nis_peak  min={vals.min():.2f}  median={np.median(vals):.2f}  "
              f"max={vals.max():.2f}  mean={vals.mean():.2f}")
    
    # Per-clip details
    print(f"\n{'Video':40s} {'Label':8s} {'nis_peak':>10s} {'glr_peak':>10s} {'physics':>8s}")
    print("-" * 80)
    for _, row in df.iterrows():
        nis_p = row.get("nis_peak", -1)
        glr_p = row.get("glr_peak", -1)
        phys = row.get("physics_available", False)
        label_str = "auth" if row["label"] == 0 else "tamp"
        nis_str = f"{nis_p:.2f}" if nis_p != -1 else "N/A"
        glr_str = f"{glr_p:.2f}" if glr_p != -1 else "N/A"
        print(f"  {row['video_name']:38s} {label_str:8s} {nis_str:>10s} {glr_str:>10s} {str(phys):>8s}")
    
    # Sanity verdict
    max_nis = df_phys["nis_peak"].max()
    # 100.0 == 50*dof is the explicit ceiling set by calibrate_nis clamp.
    # A value of exactly 100.0 means the clamp activated (controlled), not blow-up.
    all_sane = max_nis <= 100.0

    print(f"\n{'=' * 70}")
    if all_sane:
        print(f"[PASS] All nis_peak values <= {max_nis:.2f} (<= 100 ceiling)")
    else:
        print(f"[FAIL] Max nis_peak = {max_nis:.2f} -- still in blow-up range!")
    print(f"{'=' * 70}\n")
    
    return all_sane


if __name__ == "__main__":
    print("[SANITY CHECK] Setting up 20-clip sample ...")
    n = setup_sample()
    if n == 0:
        print("[ERROR] No clips found!")
        sys.exit(1)
    
    rc = run_evaluation()
    if rc != 0:
        print(f"[ERROR] evaluate_dataset.py exited with code {rc}")
        sys.exit(rc)
    
    passed = analyze_results()
    sys.exit(0 if passed else 1)
