"""
create_temporal_interpolated.py - Generate temporally interpolated video copies
===============================================================================

Creates copies of evaluation videos processed through ffmpeg's minterpolate
filter (motion-compensated frame interpolation). This simulates the realistic
threat of temporal post-processing (frame-rate conversion, slow-motion, TV
motion smoothing) that blends adjacent frames and erases inter-frame
discontinuities that pixel-level forgery detectors rely on.

Pipeline: input → minterpolate 30→25fps → back to 30fps → output
This forces every frame through motion-compensated interpolation, blending
adjacent frames and destroying hard frame boundaries.

Usage:
    python scripts/create_temporal_interpolated.py [--resume]
"""

import os
import sys
import subprocess
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.path.join(ROOT, "venv", "Scripts", "ffmpeg.exe")
MANIFEST = os.path.join(ROOT, "data", "evaluation", "manifest.csv")

INPUT_DIR = os.path.join(ROOT, "data", "evaluation")
OUTPUT_DIR = os.path.join(ROOT, "data", "evaluation_temporal")


def find_videos(input_dir):
    """Find all mp4 files in authentic/ and tampered/ subdirectories."""
    videos = []
    for subdir in ["authentic", "tampered"]:
        dirpath = os.path.join(input_dir, subdir)
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if fname.lower().endswith(".mp4"):
                videos.append((subdir, fname, os.path.join(dirpath, fname)))
    return videos


def temporal_interpolate(src_path, dst_path):
    """
    Apply temporal interpolation that blends adjacent frames.
    
    Strategy: Convert 30fps → 25fps using minterpolate (motion-compensated),
    then back to original fps. This forces every frame through interpolation,
    blending adjacent frames and destroying the hard inter-frame discontinuities
    that pixel features detect.
    
    Using mi_mode=blend for maximum frame blending (worst case for pixel features).
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # Two-pass: interpolate down to 25fps (blending frames), then back to 30fps
    # This ensures every single frame is a blend of neighbors
    cmd = [
        FFMPEG, "-y",
        "-i", src_path,
        "-vf", "minterpolate=fps=25:mi_mode=blend,minterpolate=fps=30:mi_mode=blend",
        "-c:v", "libx264",
        "-crf", "18",       # high quality — we want to isolate temporal effect only
        "-preset", "veryfast",
        "-an",
        dst_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {os.path.basename(src_path)}")
        return False
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(src_path)}: {e}")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    
    print("=" * 60)
    print(" Temporal Interpolation Generator")
    print(" Threat model: frame-rate conversion with motion blending")
    print("=" * 60)
    
    videos = find_videos(INPUT_DIR)
    
    # Filter to manifest videos only (the honest 177)
    if os.path.exists(MANIFEST):
        manifest = pd.read_csv(MANIFEST)
        manifest_names = set(manifest['clip_name'])
        videos = [v for v in videos if v[1] in manifest_names]
    
    print(f"\n[TEMPORAL] Found {len(videos)} videos")
    print(f"[TEMPORAL] Output: {OUTPUT_DIR}")
    print(f"[TEMPORAL] Resume: {'ON' if args.resume else 'OFF'}\n")
    
    done = 0
    skipped = 0
    errors = 0
    
    for subdir, fname, src_path in tqdm(videos, desc="Interpolating"):
        dst_dir = os.path.join(OUTPUT_DIR, subdir)
        dst_path = os.path.join(dst_dir, fname)
        
        if args.resume and os.path.exists(dst_path):
            skipped += 1
            continue
        
        if temporal_interpolate(src_path, dst_path):
            done += 1
        else:
            errors += 1
            print(f"  [FAIL] {fname}")
    
    print(f"\n[DONE] {done} interpolated, {skipped} skipped, {errors} errors")
    print(f"       Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
