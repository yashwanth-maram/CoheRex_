# scripts/download_dataset.py
"""
Dataset Organization & Verification
=====================================

Verifies and organizes source video files placed under data/sources/<scene_id>/.
Records a sources.csv manifest for downstream use by build_dataset.py.

Usage:
    1. Place your source videos under data/sources/<scene_id>/<video>.mp4
       Example structure:
         data/sources/scene_01/virat_01.mp4
         data/sources/scene_01/virat_02.mp4
         data/sources/scene_02/virat_03.mp4
         ...
    
    2. Run: python scripts/download_dataset.py
    
    This will verify all videos are readable and produce data/sources/sources.csv.
"""

import os
import sys
import csv
import cv2
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SOURCES_DIR = os.path.join(ROOT, "data", "sources")
SOURCES_CSV = os.path.join(SOURCES_DIR, "sources.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Verify and catalog source videos")
    p.add_argument("--sources-dir", default=SOURCES_DIR,
                   help=f"Directory containing scene subdirs (default: {SOURCES_DIR})")
    p.add_argument("--output", default=SOURCES_CSV,
                   help=f"Output CSV path (default: {SOURCES_CSV})")
    return p.parse_args()


def verify_video(path):
    """Verify a video file is readable and return metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    
    meta = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    
    if meta["frames"] < 30:
        return None  # Too short
    
    return meta


def main():
    args = parse_args()
    sources_dir = args.sources_dir
    
    if not os.path.isdir(sources_dir):
        print(f"[ERROR] Sources directory not found: {sources_dir}")
        print(f"        Please create it and place source videos under")
        print(f"        {sources_dir}/<scene_id>/<video>.mp4")
        sys.exit(1)
    
    # Scan for scene directories
    scene_dirs = sorted([
        d for d in os.listdir(sources_dir)
        if os.path.isdir(os.path.join(sources_dir, d))
    ])
    
    if not scene_dirs:
        print(f"[ERROR] No scene subdirectories found in {sources_dir}")
        print(f"        Expected structure: {sources_dir}/<scene_id>/<video>.mp4")
        sys.exit(1)
    
    print(f"[SCAN] Found {len(scene_dirs)} scene directories")
    
    entries = []
    source_id = 0
    
    for scene_id in scene_dirs:
        scene_path = os.path.join(sources_dir, scene_id)
        video_files = sorted([
            f for f in os.listdir(scene_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ])
        
        if not video_files:
            print(f"  [WARN] No videos in {scene_id}/")
            continue
        
        for vname in video_files:
            vpath = os.path.join(scene_path, vname)
            meta = verify_video(vpath)
            
            if meta is None:
                print(f"  [SKIP] {scene_id}/{vname} — unreadable or too short")
                continue
            
            source_id += 1
            entry = {
                "source_id": f"src_{source_id:03d}",
                "scene_id": scene_id,
                "filename": vname,
                "path": os.path.relpath(vpath, ROOT),
                "frames": meta["frames"],
                "fps": meta["fps"],
                "width": meta["width"],
                "height": meta["height"],
            }
            entries.append(entry)
            print(f"  [OK] {scene_id}/{vname}: {meta['frames']} frames "
                  f"@ {meta['fps']:.1f} fps, {meta['width']}x{meta['height']}")
    
    if not entries:
        print("\n[ERROR] No valid source videos found!")
        sys.exit(1)
    
    # Write sources.csv
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = ["source_id", "scene_id", "filename", "path", "frames", "fps", "width", "height"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)
    
    # Summary
    scenes = set(e["scene_id"] for e in entries)
    print(f"\n{'='*60}")
    print(f"SOURCE VIDEO CATALOG")
    print(f"{'='*60}")
    print(f"  Total sources  : {len(entries)}")
    print(f"  Total scenes   : {len(scenes)}")
    print(f"  Scenes         : {', '.join(sorted(scenes))}")
    print(f"  Catalog saved  : {args.output}")
    print(f"\n  Next step: python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
