"""
Complete Feature Extraction Pipeline (resume from dataset-complete state)
========================================================================
Assumes:
  - data/evaluation/           = 295 clean clips (85 auth + 210 tamp) -- DONE
  - data/evaluation_temporal/  = 295 temporal clips                   -- DONE
  - data/evaluation_recompressed/crf36/ = EMPTY (cleared)             -- NEEDS BUILD

Steps:
  1. Build CRF36 recompressed copies from clean clips
  2. Extract pixel features: clean, crf36, temporal
  3. Extract physics features: clean, crf36, temporal
  4. Build merged hybrid feature CSVs
  5. Run enhanced stacked ensemble classifier
"""
import os
import subprocess
import sys

MANIFEST = "data/evaluation/manifest.csv"

def run(label, cmd):
    print(f"\n{'='*70}")
    print(f"[STEP] {label}")
    print(f"CMD:   {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nFAILED (exit={result.returncode}). Stopping.")
if __name__ == "__main__":

    # ── Step 1: CRF36 Recompression ──────────────────────────────────────────
    run("CRF36 Recompression",
        ["python", "scripts/create_recompressed.py",
         "--input",  "data/evaluation",
         "--output", "data/evaluation_recompressed",
         "--crf-levels", "36",
         "--manifest", MANIFEST,
         "--resume"])

    # ── Step 2: Pixel Feature Extraction ─────────────────────────────────────
    run("Pixel features - CLEAN",
        ["python", "scripts/extract_pixel.py",
         "--input",    "data/evaluation",
         "--out",      "cache/pixel_clean",
         "--manifest", MANIFEST,
         "--resume"])

    run("Pixel features - CRF36",
        ["python", "scripts/extract_pixel.py",
         "--input",    "data/evaluation_recompressed/crf36",
         "--out",      "cache/pixel_crf36",
         "--manifest", MANIFEST,
         "--resume"])

    run("Pixel features - TEMPORAL",
        ["python", "scripts/extract_pixel.py",
         "--input",    "data/evaluation_temporal",
         "--out",      "cache/pixel_temporal",
         "--manifest", MANIFEST,
         "--resume"])

    # ── Step 3: Physics Feature Extraction ───────────────────────────────────
    run("Physics features - CLEAN",
        ["python", "scripts/extract_physics.py",
         "--input",    "data/evaluation",
         "--out",      "cache/physics_clean",
         "--manifest", MANIFEST,
         "--no-pose",
         "--resume"])

    run("Physics features - CRF36",
        ["python", "scripts/extract_physics.py",
         "--input",    "data/evaluation_recompressed/crf36",
         "--out",      "cache/physics_crf36",
         "--manifest", MANIFEST,
         "--no-pose",
         "--resume"])

    run("Physics features - TEMPORAL",
        ["python", "scripts/extract_physics.py",
         "--input",    "data/evaluation_temporal",
         "--out",      "cache/physics_temporal",
         "--manifest", MANIFEST,
         "--no-pose",
         "--resume"])

    # ── Step 4: Build merged feature CSVs ────────────────────────────────────
    run("Build features - CLEAN",
        ["python", "scripts/build_features.py",
         "--cache", "cache/physics_clean",
         "--out",   "results/features_physics.csv"])

    run("Build features - CRF36",
        ["python", "scripts/build_features.py",
         "--cache", "cache/physics_crf36",
         "--out",   "results/features_physics_crf36.csv"])

    run("Build features - TEMPORAL",
        ["python", "scripts/build_features.py",
         "--cache", "cache/physics_temporal",
         "--out",   "results/features_physics_temporal.csv"])

    # Pixel CSVs (from cache npz)
    run("Build pixel features - CLEAN",
        ["python", "scripts/build_features.py",
         "--cache", "cache/pixel_clean",
         "--out",   "results/features_pixel.csv"])

    run("Build pixel features - CRF36",
        ["python", "scripts/build_features.py",
         "--cache", "cache/pixel_crf36",
         "--out",   "results/features_pixel_crf36.csv"])

    run("Build pixel features - TEMPORAL",
        ["python", "scripts/build_features.py",
         "--cache", "cache/pixel_temporal",
         "--out",   "results/features_pixel_temporal.csv"])

    # Merge pixel + physics into hybrid
    run("Merge hybrid - CLEAN",
        ["python", "scripts/merge_features.py",
         "--pixel",   "results/features_pixel.csv",
         "--physics", "results/features_physics.csv",
         "--out",     "results/features_hybrid.csv"])

    run("Merge hybrid - CRF36",
        ["python", "scripts/merge_features.py",
         "--pixel",   "results/features_pixel_crf36.csv",
         "--physics", "results/features_physics_crf36.csv",
         "--out",     "results/features_hybrid_crf36.csv"])

    run("Merge hybrid - TEMPORAL",
        ["python", "scripts/merge_features.py",
         "--pixel",   "results/features_pixel_temporal.csv",
         "--physics", "results/features_physics_temporal.csv",
         "--out",     "results/features_hybrid_temporal.csv"])

    # ── Step 5: Train + Evaluate ─────────────────────────────────────────────
    run("Enhanced Stacked Ensemble - Compare All",
        ["python", "-X", "utf8",
         "scripts/enhanced_classifier.py",
         "--compare",
         "--n-trials", "20"])

    print("\n" + "="*70)
    print("ALL STEPS COMPLETE! Check results/ for final AUC numbers.")
    print("="*70)
