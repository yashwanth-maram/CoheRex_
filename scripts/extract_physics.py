"""
extract_raw.py — CoheRex 2.0 Heavy Extraction (run ONCE)
=========================================================

Runs the full YOLO + pose + tracking + Kalman + pixel-residual pipeline
on each video and saves raw SEQUENCES to .npz cache files.

This is the expensive step (~3-4 hours on 470 videos).
After this, feature iteration is instant via build_features.py.

Saves per video:
  - Pixel-residual SEQUENCES (not means!):
      frame_diff:     mean|Δpixel| between adjacent frames
      ssim_drop:      1 - SSIM between adjacent frames
      flow_spike:     Farnebäck flow magnitude variance per frame-pair
      color_hist_dist: Bhattacharyya distance of color histograms (adjacent frames)
  - Physics per-track sequences:
      nis_cv_*:       NIS from CV filter per track
      nis_combined_*: combined NIS per track
      innov_*:        innovation vectors per track
      track_meta:     (track_id, length, mean_bbox_h, first_frame, last_frame)
  - Pose keypoints per track
  - Metadata: label, tamper_type, source_id, scene_id, fps

Usage:
    python scripts/extract_raw.py --input data/evaluation --out cache/clean
    python scripts/extract_raw.py --input data/evaluation_recompressed/crf32 --out cache/crf32
"""

import cv2
import os
import sys
import concurrent.futures
import argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG
from coherex.detection.yolo_detector import YOLODetector
from coherex.tracking.manager import TrackManager
from coherex.trajectory.store import TrajectoryStore
from coherex.coherence.mcv import MotionCoherenceEngine
from coherex.integrity.motion_agent import MotionAgent
from ultralytics import YOLO


MANIFEST = os.path.join(ROOT, "data", "evaluation", "manifest.csv")


def parse_args():
    p = argparse.ArgumentParser(description="CoheRex 2.0 — Heavy Raw Extraction")
    p.add_argument("--input", type=str, required=True,
                   help="Input directory (must contain authentic/ and tampered/ subdirs)")
    p.add_argument("--out", type=str, required=True,
                   help="Output cache directory (e.g., cache/clean)")
    p.add_argument("--manifest", type=str, default=MANIFEST,
                   help="Path to manifest.csv for metadata")
    p.add_argument("--resume", action="store_true",
                   help="Skip videos that already have .npz files")
    p.add_argument("--workers", type=int, default=12,
                   help="Number of parallel workers")
    p.add_argument("--shard", type=int, default=0,
                   help="Shard index (0-indexed)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards")
    p.add_argument("--no-pose", action="store_true", default=True,
                   help="Disable pose estimation to speed up extraction (default: True)")
    return p.parse_args()


def load_manifest(manifest_path):
    """Load manifest.csv into a dict keyed by clip_name."""
    manifest = {}
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            manifest[row["clip_name"]] = row.to_dict()
    return manifest


def compute_color_histogram(frame, nbins=64):
    """Compute a normalized HSV color histogram for the full frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [nbins, nbins], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_video(video_path, config, pose_model, detector):
    """
    Run the full pipeline on a single video, return raw sequences.
    
    Returns dict with all raw sequences for caching.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    STRIDE = 3                      # 30fps -> 10fps effective
    dt = STRIDE / fps               # pass this dt into the Kalman predict step
    
    tracker = TrackManager(config=config, dt=dt)
    traj_store = TrajectoryStore(config=config)
    motion_agent = MotionAgent(config=config)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    # ── Physics collectors ──
    per_frame_nis = []        # max NIS across tracks per frame (for GLR/CUSUM)
    all_innovations = []      # innovation vectors (for whiteness)
    frame_track_counts = []   # track count per frame
    birth_frames = []
    death_frames = []

    # ── Per-track data ──
    track_nis_cv = {}         # track_id → list of nis_cv values
    track_nis_combined = {}   # track_id → list of combined nis values
    track_innov = {}          # track_id → list of innovation vectors
    track_bbox_heights = {}   # track_id → list of bbox heights
    track_positions = {}      # track_id → list of (x, y) positions
    track_first_frame = {}    # track_id → first frame
    track_last_frame = {}     # track_id → last frame

    # ── Pose per track ──
    track_keypoints = {}      # track_id → list of keypoint arrays

    # ── Legacy motion scores ──
    all_motion_scores = []
    all_raw_mcvs = []
    all_confidences = []

    frame_idx = -1
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % STRIDE != 0:
            continue
        frame_id = frame_idx // STRIDE   # keep the logical frame counter dense

        # ── Detection ──
        detections = detector.detect(frame)
        person_centers, person_boxes = [], []
        for det in detections:
            if det["class_id"] == config.detection.person_class_id:
                x1, y1, x2, y2 = det["bbox"]
                person_centers.append(((x1 + x2) / 2, y2))
                person_boxes.append(det["bbox"])

        # ── Pose ──
        pose_dets = []
        if pose_model is not None:
            pose_results = pose_model(frame, verbose=False)
            for r in pose_results:
                if r.keypoints is None or r.boxes is None:
                    continue
                for i, kp in enumerate(r.keypoints.xy):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    pose_dets.append(((x1 + x2) / 2.0, y2, kp.cpu().numpy()))

        # ── Tracking ──
        tracker.current_frame = frame
        tracker.current_pose = pose_dets
        tracks = tracker.update(
            detections=person_centers,
            bboxes=person_boxes,
            frame_id=frame_id
        )
        traj_store.update_from_tracks(tracks)

        # ── Match tracks to pose detections (spatial, Hungarian) ──
        matched_kps = {}
        if pose_dets and tracks:
            cost = np.zeros((len(tracks), len(pose_dets)))
            for ti, tr in enumerate(tracks):
                for pi, (px, py, _) in enumerate(pose_dets):
                    cost[ti, pi] = np.hypot(tr.x - px, tr.y - py)
            row_ind, col_ind = linear_sum_assignment(cost)
            for ti, pi in zip(row_ind, col_ind):
                if cost[ti, pi] <= config.tracking.max_association_distance:
                    matched_kps[tracks[ti].track_id] = pose_dets[pi][2]

        # ── Track births ──
        for track in tracks:
            if track.first_seen_frame == frame_id:
                birth_frames.append(frame_id)
        frame_track_counts.append(len(tracks))

        # ── Per-track physics signals ──
        frame_nis_values = []
        motion_scores = []

        for track in tracks:
            state = traj_store.get_state(track.track_id)
            if state is None:
                continue

            tid = track.track_id

            # Initialize per-track storage
            if tid not in track_nis_cv:
                track_nis_cv[tid] = []
                track_nis_combined[tid] = []
                track_innov[tid] = []
                track_bbox_heights[tid] = []
                track_positions[tid] = []
                track_keypoints[tid] = []
                track_first_frame[tid] = frame_id

            track_last_frame[tid] = frame_id

            # Store bbox height
            bbox = tracker.current_bboxes.get(
                next((i for i, (cx, cy) in enumerate(person_centers)
                      if abs(cx - track.x) < 50 and abs(cy - track.y) < 50), None),
                None
            )
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                track_bbox_heights[tid].append(y2 - y1)

            # Store position
            track_positions[tid].append((track.x, track.y))

            # NIS from track
            if hasattr(track, 'nis_combined_history') and len(track.nis_combined_history) > 0:
                latest_nis = track.nis_combined_history[-1]
                if track.age >= config.physics.min_track_length_for_nis:
                    frame_nis_values.append(latest_nis)

                # Sync to trajectory state
                if len(track.nis_combined_history) > len(state.nis_history):
                    state.update_nis(
                        nis_combined=latest_nis,
                        nis_cv=track.nis_cv_history[-1] if track.nis_cv_history else None,
                        nis_ca=track.nis_ca_history[-1] if track.nis_ca_history else None,
                        innovation=track.innov_history[-1] if track.innov_history else None,
                    )

            # Store per-track NIS sequences
            if track.nis_cv_history:
                track_nis_cv[tid] = list(track.nis_cv_history)
            if track.nis_combined_history:
                track_nis_combined[tid] = list(track.nis_combined_history)
            if track.innov_history:
                track_innov[tid] = list(track.innov_history)

            # Innovations for whiteness
            if track.innov_history and track.age >= config.physics.min_track_length_for_nis:
                all_innovations.append(track.innov_history[-1])

            all_confidences.append(track.confidence)

            # Pose keypoints
            kp = matched_kps.get(tid)
            track_keypoints[tid].append(kp if kp is not None else None)

            # Motion agent score (legacy)
            m_score = motion_agent.evaluate(state, track.confidence,
                                            track=track, current_frame=frame_id)
            motion_scores.append(m_score)

            # MCV (legacy)
            mcv_engine = MotionCoherenceEngine(
                window_size=config.motion.mcv_window_size,
                noise_floor=config.motion.mcv_noise_floor,
            )
            raw_mcv = mcv_engine.compute(state, track.confidence)
            all_raw_mcvs.append(abs(raw_mcv))

        # Max NIS across tracks for this frame
        if frame_nis_values:
            per_frame_nis.append(max(frame_nis_values))

        if motion_scores:
            all_motion_scores.append(sum(motion_scores) / len(motion_scores))

        frame_id += 1

    cap.release()

    # ── Build per-track metadata array ──
    track_meta = []
    for tid in track_nis_cv.keys():
        length = len(track_nis_combined.get(tid, []))
        mean_h = float(np.mean(track_bbox_heights.get(tid, [0]))) if track_bbox_heights.get(tid) else 0.0
        track_meta.append({
            "track_id": tid,
            "length": length,
            "mean_bbox_h": mean_h,
            "first_frame": track_first_frame.get(tid, 0),
            "last_frame": track_last_frame.get(tid, 0),
        })

    return {
        # Physics: per-frame aggregated NIS
        "per_frame_nis": np.array(per_frame_nis, dtype=np.float32),
        "innovations": np.array(all_innovations, dtype=object),
        # Physics: per-track sequences (stored as object arrays)
        "track_nis_cv": track_nis_cv,
        "track_nis_combined": track_nis_combined,
        "track_innov": track_innov,
        "track_positions": track_positions,
        "track_meta": track_meta,
        # Pose
        "track_keypoints": track_keypoints,
        # Scene-level
        "frame_track_counts": np.array(frame_track_counts, dtype=np.int32),
        "birth_frames": np.array(birth_frames, dtype=np.int32),
        "death_frames": np.array(death_frames, dtype=np.int32),
        # Legacy
        "motion_scores": np.array(all_motion_scores, dtype=np.float32),
        "raw_mcvs": np.array(all_raw_mcvs, dtype=np.float32),
        "confidences": np.array(all_confidences, dtype=np.float32),
        # Meta
        "fps": fps,
        "total_frames": frame_id,
    }



# --- GLOBALS FOR WORKER PROCESSES ---
_worker_pose_model = None
_worker_detector = None
_worker_initialized = False

def init_worker(no_pose):
    global _worker_pose_model, _worker_detector, _worker_initialized
    if _worker_initialized:
        return
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    from coherex.detection.yolo_detector import YOLODetector
    from ultralytics import YOLO
    
    if not no_pose:
        pose_path = os.path.join(ROOT, "yolov8n-pose.pt")
        if not os.path.exists(pose_path):
            import urllib.request
            urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt", pose_path)
        _worker_pose_model = YOLO(pose_path)
    
    _worker_detector = YOLODetector(config=CONFIG)
    _worker_initialized = True

def process_video(args):
    fname, vpath, label, meta, out_path, no_pose = args
    try:
        init_worker(no_pose)
        raw = extract_video(vpath, CONFIG, _worker_pose_model, _worker_detector)
        
        # Add metadata from manifest
        raw["label"] = label
        raw["source_id"] = meta.get("source_id", "unknown")
        raw["scene_id"] = meta.get("scene_id", "unknown")
        raw["tamper_type"] = meta.get("tamper_type", "none" if label == 0 else "unknown")
        raw["tamper_start_frame"] = meta.get("tamper_start_frame", -1)
        raw["tamper_end_frame"] = meta.get("tamper_end_frame", -1)
        raw["video_name"] = fname

        np.savez_compressed(
            out_path,
            per_frame_nis=raw["per_frame_nis"],
            frame_track_counts=raw["frame_track_counts"],
            birth_frames=raw["birth_frames"],
            death_frames=raw["death_frames"],
            motion_scores=raw["motion_scores"],
            raw_mcvs=raw["raw_mcvs"],
            confidences=raw["confidences"],
            fps=np.float32(raw["fps"]),
            total_frames=np.int32(raw["total_frames"]),
            label=np.int32(raw["label"]),
            source_id=np.array(raw["source_id"]),
            scene_id=np.array(raw["scene_id"]),
            tamper_type=np.array(raw["tamper_type"]),
            tamper_start_frame=np.int32(raw["tamper_start_frame"]),
            tamper_end_frame=np.int32(raw["tamper_end_frame"])
        )
        return fname, True, ""
    except Exception as e:
        return fname, False, str(e)

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Load manifest
    manifest = load_manifest(args.manifest)

    # Shared models (created ONCE)
    print(f"\n[EXTRACT] Loading YOLOv8n... (pose={not args.no_pose})")
    pose_model = YOLO(os.path.join(ROOT, "yolov8n-pose.pt")) if not args.no_pose else None
    detector = YOLODetector(config=CONFIG)

    # Gather videos
    auth_dir = os.path.join(args.input, "authentic")
    tamp_dir = os.path.join(args.input, "tampered")

    def collect(directory, label):
        entries = []
        if not os.path.isdir(directory):
            print(f"  [WARN] Directory not found: {directory}")
            return entries
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(".mp4") or fname.lower().endswith(".avi") or fname.lower().endswith(".mov"):
                entries.append((fname, os.path.join(directory, fname), label))
        return entries

    videos = collect(auth_dir, 0) + collect(tamp_dir, 1)
    
    # Filter to manifest only
    manifest_names = set(manifest.keys())
    videos = [v for v in videos if v[0] in manifest_names]
    
    # Sharding
    if args.num_shards > 1:
        videos = videos[args.shard :: args.num_shards]
        print(f"[EXTRACT] Shard {args.shard+1}/{args.num_shards}: {len(videos)} videos")

    if not videos:
        print("[ERROR] No videos found.")
        sys.exit(1)

    print(f"\n[EXTRACT] Processing {len(videos)} videos → {args.out}/")
    print(f"[EXTRACT] Workers: {args.workers}")
    print(f"[EXTRACT] Resume mode: {'ON' if args.resume else 'OFF'}\n")

    tasks = []
    for fname, vpath, label in videos:
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(args.out, base + ".npz")
        if args.resume and os.path.exists(out_path):
            continue
        meta = manifest.get(fname, {})
        tasks.append((fname, vpath, label, meta, out_path, args.no_pose))

    if not tasks:
        print("[EXTRACT] All files already cached. Done.")
        return

    errors = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_video, tasks), total=len(tasks), desc="Extracting"))
        
    for fname, ok, err in results:
        if not ok:
            print(f"  [ERROR] {fname}: {err}")
            errors += 1
            
    print(f"\n[DONE] Extracted {len(tasks)} videos ({errors} errors)")

if __name__ == "__main__":
    main()
