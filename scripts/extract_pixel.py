"""
extract_pixel.py — CoheRex 2.0 Fast Pixel Feature Extraction
=============================================================

Runs purely pixel-based forensic signals (no YOLO, no PyTorch).
Because this requires no ML models, it is very fast and can be parallelized
across all CPU cores.

Extracts pixel-residual SEQUENCES to .npz cache files.

Saves per video:
  - Pixel-residual SEQUENCES (not means!):
      frame_diff:     mean|Δpixel| between adjacent frames
      ssim_drop:      1 - SSIM between adjacent frames
      flow_spike:     Farnebäck flow magnitude variance per frame-pair (low-res)
      color_hist_dist: Bhattacharyya distance of color histograms (adjacent frames)
  - Metadata: label, tamper_type, source_id, scene_id, fps, total_frames

Usage:
    python scripts/extract_pixel.py --input data/evaluation --out cache/pixel_clean
"""

import cv2
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import concurrent.futures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MANIFEST = os.path.join(ROOT, "data", "evaluation", "manifest.csv")


def parse_args():
    p = argparse.ArgumentParser(description="CoheRex 2.0 — Fast Pixel Extraction")
    p.add_argument("--input", type=str, required=True,
                   help="Input directory (must contain authentic/ and tampered/ subdirs)")
    p.add_argument("--out", type=str, required=True,
                   help="Output cache directory (e.g., cache/pixel_clean)")
    p.add_argument("--manifest", type=str, default=MANIFEST,
                   help="Path to manifest.csv for metadata")
    p.add_argument("--resume", action="store_true",
                   help="Skip videos that already have .npz files")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="Number of parallel workers")
    return p.parse_args()


def load_manifest(manifest_path):
    manifest = {}
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            manifest[row["clip_name"]] = row.to_dict()
    return manifest


def compute_color_histogram(frame, nbins=64):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [nbins, nbins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def process_video(args):
    fname, vpath, label, meta, out_path = args

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return fname, False, "Could not open video"
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_diff_seq = []
    ssim_drop_seq = []
    flow_spike_seq = []
    color_hist_dist_seq = []

    prev_gray = None
    prev_hist = None
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale aggressively for speed while retaining structural anomalies
        gray_small = cv2.resize(gray, (320, 240))
        # Flow can be even smaller
        gray_flow = cv2.resize(gray, (160, 120))

        if prev_gray is not None:
            # Frame difference
            diff = np.mean(np.abs(gray_small.astype(np.float32) - prev_gray.astype(np.float32)))
            frame_diff_seq.append(float(diff))

            # SSIM
            try:
                from skimage.metrics import structural_similarity as ssim_func
                score, _ = ssim_func(prev_gray, gray_small, full=True)
                ssim_drop_seq.append(float(1.0 - score))
            except Exception:
                ssim_drop_seq.append(0.0)

            # Flow variance (on even smaller res)
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray_flow, gray_flow, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_spike_seq.append(float(np.var(flow_mag)))
            except Exception:
                flow_spike_seq.append(0.0)

        # Color hist
        curr_hist = compute_color_histogram(frame)
        if prev_hist is not None:
            bhatt_dist = cv2.compareHist(
                prev_hist.reshape(-1, 1).astype(np.float32),
                curr_hist.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA
            )
            color_hist_dist_seq.append(float(bhatt_dist))
        
        prev_hist = curr_hist
        prev_gray = gray_small
        prev_gray_flow = gray_flow
        frame_id += 1

    cap.release()

    # Save to npz
    try:
        np.savez_compressed(
            out_path,
            frame_diff=np.array(frame_diff_seq, dtype=np.float32),
            ssim_drop=np.array(ssim_drop_seq, dtype=np.float32),
            flow_spike=np.array(flow_spike_seq, dtype=np.float32),
            color_hist_dist=np.array(color_hist_dist_seq, dtype=np.float32),
            fps=np.float32(fps),
            total_frames=np.int32(frame_id),
            label=np.int32(label),
            source_id=np.array(meta.get("source_id", "unknown")),
            scene_id=np.array(meta.get("scene_id", "unknown")),
            tamper_type=np.array(meta.get("tamper_type", "none" if label == 0 else "unknown")),
            tamper_start_frame=np.int32(meta.get("tamper_start_frame", -1)),
            tamper_end_frame=np.int32(meta.get("tamper_end_frame", -1)),
            video_name=np.array(fname)
        )
    except Exception as e:
        return fname, False, str(e)

    return fname, True, frame_id


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = load_manifest(args.manifest)

    def collect(directory, label):
        entries = []
        if not os.path.isdir(directory):
            return entries
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(".mp4") or fname.lower().endswith(".avi") or fname.lower().endswith(".mov"):
                entries.append((fname, os.path.join(directory, fname), label))
        return entries

    videos = collect(os.path.join(args.input, "authentic"), 0) + \
             collect(os.path.join(args.input, "tampered"), 1)
             
    if not videos:
        print("[ERROR] No videos found.")
        sys.exit(1)

    print(f"\n[PIXEL] Processing {len(videos)} videos -> {args.out}/")
    print(f"[PIXEL] Workers: {args.workers}")
    print(f"[PIXEL] Resume: {'ON' if args.resume else 'OFF'}\n")

    tasks = []
    for fname, vpath, label in videos:
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(args.out, base + ".npz")
        if args.resume and os.path.exists(out_path):
            continue
        meta = manifest.get(fname, {})
        tasks.append((fname, vpath, label, meta, out_path))

    if not tasks:
        print("[DONE] All videos already extracted.")
        sys.exit(0)

    # Smoke test on the first 5 videos
    print(f"[SMOKE TEST] Running on first {min(5, len(tasks))} videos sequentially...")
    for i in range(min(5, len(tasks))):
        fname, success, msg = process_video(tasks[i])
        if success:
            print(f"  [OK] {fname} ({msg} frames)")
        else:
            print(f"  [FAIL] {fname}: {msg}")
            print("[ERROR] Smoke test failed. Aborting.")
            sys.exit(1)
            
    if len(tasks) <= 5:
        print("\n[DONE] Smoke test completed all tasks.")
        sys.exit(0)

    print("\n[PIXEL] Smoke test passed. Launching full parallel extraction...\n")

    # Full run
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, task): task for task in tasks[5:]}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting"):
            results.append(future.result())

    errors = sum(1 for _, success, _ in results if not success)
    print(f"\n[DONE] Extracted {len(tasks)} videos ({errors} errors)")
    print(f"       Next: python scripts/build_features.py --cache {args.out} --out results/features_pixel.csv")


if __name__ == "__main__":
    main()
