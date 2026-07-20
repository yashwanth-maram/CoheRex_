"""
create_recompressed.py - Generate recompressed video copies at CRF 28/32/36
============================================================================

Creates copies of all evaluation videos re-encoded with libx264 at various
CRF levels. Preserves frame rate and frame count (no -r or fps filter).

Usage:
    python scripts/create_recompressed.py --input data/evaluation --output data/evaluation_recompressed
"""

import os
import sys
import argparse
import subprocess
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Create recompressed video copies")
    p.add_argument("--input", type=str, default=os.path.join(ROOT, "data", "evaluation"),
                   help="Input evaluation directory (with authentic/ and tampered/ subdirs)")
    p.add_argument("--output", type=str, default=os.path.join(ROOT, "data", "evaluation_recompressed"),
                   help="Output directory for recompressed videos")
    p.add_argument("--crf-levels", type=str, default="36",
                   help="Comma-separated CRF levels (default: 36)")
    p.add_argument("--preset", type=str, default="veryfast",
                   help="x264 encoding preset (default: veryfast)")
    p.add_argument("--resume", action="store_true",
                   help="Skip videos that already exist in the output directory")
    p.add_argument("--manifest", type=str, default=os.path.join(ROOT, "data", "evaluation", "manifest.csv"),
                   help="Path to manifest.csv to filter videos (skip unknowns)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="Number of parallel ffmpeg workers")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


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


def recompress_video(src_path, dst_path, crf, preset="veryfast", dry_run=False):
    """Re-encode a video with libx264 at the given CRF level.
    
    IMPORTANT: No -r or -vf fps= flags. Same frame rate and frame count
    as the original. We only change the compression quality.
    """
    cmd = [
        os.path.join(ROOT, "venv", "Scripts", "ffmpeg.exe"), "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-an",  # strip audio (surveillance footage, irrelevant)
        dst_path
    ]
    
    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {os.path.basename(src_path)}")
        return False
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Install it and add to PATH.")
        sys.exit(1)


def verify_frame_count(src_path, dst_path):
    """Verify that recompressed video has the same frame count as original."""
    def get_frame_count(path):
        cmd = [
            os.path.join(ROOT, "venv", "Scripts", "ffprobe.exe"), "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return int(result.stdout.strip())
        except Exception:
            return -1
    
    src_frames = get_frame_count(src_path)
    dst_frames = get_frame_count(dst_path)
    
    if src_frames != dst_frames and src_frames > 0:
        print(f"  [WARN] Frame count mismatch: {src_frames} -> {dst_frames} ({os.path.basename(src_path)})")
        return False
    return True


def process_video_task(args):
    src_path, dst_path, crf, preset, dry_run = args
    if recompress_video(src_path, dst_path, crf, preset, dry_run):
        return args, True
    return args, False

def main():
    import pandas as pd
    import concurrent.futures
    from tqdm import tqdm
    
    args = parse_args()
    crf_levels = [int(c.strip()) for c in args.crf_levels.split(",")]
    
    videos = find_videos(args.input)
    
    if os.path.exists(args.manifest):
        manifest = pd.read_csv(args.manifest)
        manifest_names = set(manifest['clip_name'])
        videos = [v for v in videos if v[1] in manifest_names]
        
    if not videos:
        print(f"[ERROR] No videos found in {args.input}")
        sys.exit(1)
    
    print(f"[RECOMPRESS] Found {len(videos)} videos (manifest filtered)")
    print(f"[RECOMPRESS] CRF levels: {crf_levels}")
    print(f"[RECOMPRESS] Preset: {args.preset}")
    print(f"[RECOMPRESS] Workers: {args.workers}")
    print(f"[RECOMPRESS] Resume: {'ON' if args.resume else 'OFF'}")
    if args.dry_run:
        print("[RECOMPRESS] DRY RUN - no files will be created\n")
    
    tasks = []
    skipped = 0
    
    for crf in crf_levels:
        for subdir, fname, src_path in videos:
            dst_dir = os.path.join(args.output, f"crf{crf}", subdir)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, fname)
            
            if args.resume and os.path.exists(dst_path):
                skipped += 1
                continue
                
            tasks.append((src_path, dst_path, crf, args.preset, args.dry_run))

    if not tasks:
        print(f"\n[DONE] All {skipped} videos already recompressed.")
        sys.exit(0)
        
    print(f"\n[RECOMPRESS] Launching {len(tasks)} ffmpeg tasks...\n")
    
    done = 0
    errors = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video_task, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Encoding"):
            task_args, success = future.result()
            if success:
                done += 1
            else:
                errors += 1
                src_path = task_args[0]
                print(f"  [FAIL] {os.path.basename(src_path)}")
    
    print(f"\n[DONE] {done} recompressed, {skipped} skipped, {errors} errors")
    print(f"       Output: {args.output}/")
    print(f"       Next: extract pixel features for each CRF level")
# Removed old sequential main body


if __name__ == "__main__":
    main()
