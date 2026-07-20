# scripts/build_dataset.py
"""
CoheRex 2.0 — Dataset Builder
================================

Builds the evaluation dataset from source videos organized by scene.

For each source video:
  - Cuts authentic clips (unmodified segments)
  - Produces tampered clips with 5 tampering types
  - Records source_id on every clip (for GroupKFold, leak-free evaluation)
  - Logs a manifest CSV with full ground-truth metadata

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --recompress        # also emit re-encoded copies
    python scripts/build_dataset.py --clips-per-source 4 --clip-duration 5

Requires:
    data/sources/sources.csv (from scripts/download_dataset.py)
    OR source videos directly in data/sources/<scene_id>/*.mp4
"""

import cv2
import os
import sys
import csv
import random
import argparse
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.detection.yolo_detector import YOLODetector
from coherex.config import CONFIG

SOURCES_CSV = os.path.join(ROOT, "data", "sources", "sources.csv")
EVAL_DIR    = os.path.join(ROOT, "data", "evaluation")
AUTH_DIR    = os.path.join(EVAL_DIR, "authentic")
TAMP_DIR    = os.path.join(EVAL_DIR, "tampered")
RECOMP_DIR  = os.path.join(ROOT, "data", "evaluation_recompressed")
MANIFEST    = os.path.join(EVAL_DIR, "manifest.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Build CoheRex evaluation dataset")
    p.add_argument("--sources-csv", default=SOURCES_CSV,
                   help="Path to sources.csv catalog")
    p.add_argument("--clips-per-source", type=int, default=3,
                   help="Number of authentic clips per source video")
    p.add_argument("--clip-duration", type=float, default=5.0,
                   help="Clip duration in seconds")
    p.add_argument("--tampers-per-type", type=int, default=1,
                   help="Number of tampered variants per type per source")
    p.add_argument("--recompress", action="store_true",
                   help="Also emit re-encoded copies for robustness testing")
    p.add_argument("--crf", type=int, nargs="+", default=[28, 32, 36],
                   help="CRF values for recompression (default: 28 32 36)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-person-frac", type=float, default=0.6,
                   help="Minimum fraction of sampled frames in a window that "
                        "must contain a detected person (default: 0.6). "
                        "Set to 0 to disable activity filtering entirely.")
    p.add_argument("--activity-stride", type=int, default=3,
                   help="Check every Nth frame when scoring windows for "
                        "person presence (default: 3 — balances speed vs "
                        "accuracy; the detector still runs at native "
                        "confidence_threshold from config.py)")
    p.add_argument("--activity-step-sec", type=float, default=1.0,
                   help="Granularity (seconds) of candidate window starts "
                        "when scanning a source video for activity")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n_frames, fps


def write_frames(frames, out_path, fps, frame_size=None):
    """Write a list of frames to an mp4 file."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    size = frame_size or (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    for f in frames:
        if f.shape[1] != size[0] or f.shape[0] != size[1]:
            f = cv2.resize(f, size)
        writer.write(f)
    writer.release()


def _longest_true_run(bool_arr):
    """Return (start, end) [end exclusive] of the longest True run, or None."""
    best = (0, 0)
    cur_start = None
    for i, v in enumerate(bool_arr):
        if v and cur_start is None:
            cur_start = i
        elif not v and cur_start is not None:
            if i - cur_start > best[1] - best[0]:
                best = (cur_start, i)
            cur_start = None
    if cur_start is not None and len(bool_arr) - cur_start > best[1] - best[0]:
        best = (cur_start, len(bool_arr))
    return best if best[1] > best[0] else None


def find_active_windows(src_path, n_frames, fps, clip_duration, detector, person_class_id,
                         min_person_frac=0.6, stride=3, step_sec=1.0):
    """
    Score every candidate clip-length window in a source video by what
    fraction of its (strided) frames contain a detected person, using the
    SAME detector/confidence threshold the evaluation pipeline uses — so a
    window that passes here is a window the pipeline can actually track.

    Returns a list of dicts sorted by person_frac descending:
        {start_sec, person_frac, active_range}
    where active_range = (start_frame, end_frame) — the longest contiguous
    span of person-present frames WITHIN this window, in frame indices
    relative to the window itself (not the source video). Tamper functions
    should target their edit inside active_range, not anywhere in the clip
    — a window can average >= min_person_frac presence while still having
    long dead stretches where the person hasn't entered/has left frame,
    and an edit placed there is invisible to a motion-based detector no
    matter how good the detector is.

    Empty list means no window in this source meets the activity bar at
    all (caller should skip the source, not force a clip out of it).
    """
    clip_len = int(round(clip_duration * fps))
    if n_frames < clip_len:
        return []

    presence = np.zeros(n_frames, dtype=bool)
    last = False
    
    cap = cv2.VideoCapture(src_path)
    for i in range(n_frames):
        ret, f = cap.read()
        if not ret:
            break
        if i % stride == 0:
            dets = detector.detect(f)
            last = any(d["class_id"] == person_class_id for d in dets)
        presence[i] = last
    cap.release()

    step_frames = max(1, int(round(step_sec * fps)))
    windows = []
    for start_f in range(0, n_frames - clip_len + 1, step_frames):
        window_presence = presence[start_f:start_f + clip_len]
        frac = window_presence.mean()
        if frac >= min_person_frac:
            active_range = _longest_true_run(window_presence)
            windows.append(dict(start_sec=start_f / fps, person_frac=float(frac),
                                 active_range=active_range))

    windows.sort(key=lambda w: -w["person_frac"])
    return windows


def extract_segment(src_path, fps, start_sec, duration_sec):
    """Extract a clean segment directly from video file without loading all frames."""
    cap = cv2.VideoCapture(src_path)
    start_frame = int(start_sec * fps)
    num_frames = int(duration_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    seg = []
    for _ in range(num_frames):
        ret, f = cap.read()
        if not ret:
            break
        seg.append(f)
    cap.release()
    return seg if len(seg) > 30 else None


def recompress_video(src_path, dst_path, crf=32):
    """Re-encode a video with ffmpeg at the given CRF."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", src_path,
            "-c:v", "libx264", "-crf", str(crf),
            "-c:a", "copy", dst_path
        ], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Tampering functions
# ──────────────────────────────────────────────────────────────────────────────

def _pick_start(rng, n_frames, edit_len, active_range=None):
    """
    Pick where to place an edit of length edit_len within n_frames.

    If active_range=(lo, hi) is given (the span where a person is actually
    present), constrain the pick to overlap it — otherwise the edit can
    land in a dead stretch of the clip and be physically undetectable no
    matter how good the detector is. Falls back to the full clip if the
    active range can't fit the edit.
    """
    default_lo, default_hi = 10, max(11, n_frames - edit_len - 10)
    if active_range is not None:
        lo, hi = active_range
        lo = max(lo, 10)
        hi = min(hi - edit_len, n_frames - edit_len - 10)
        if hi > lo:
            return rng.randint(lo, hi)
    return rng.randint(default_lo, default_hi)


def tamper_frame_deletion(frames, fps, seed=0, active_range=None):
    """Remove 5–20 consecutive frames starting at a random interior point."""
    rng = random.Random(seed)
    n_del = rng.randint(5, min(20, len(frames) // 4))
    start = _pick_start(rng, len(frames), n_del, active_range)
    result = frames[:start] + frames[start + n_del:]
    return result, start, start + n_del


def tamper_frame_duplication(frames, fps, seed=0, active_range=None):
    """Duplicate a 1-second segment at a random interior point."""
    rng = random.Random(seed)
    dup_len = int(fps * 1.5)
    start = _pick_start(rng, len(frames), dup_len, active_range)
    dup = frames[start:start + dup_len]
    result = frames[:start] + dup + dup + frames[start + dup_len:]
    return result, start, start + dup_len


def tamper_speed_manipulation(frames, fps, seed=0, active_range=None):
    """Simulate 1.5× speed in a 2-second window by dropping every 2nd frame."""
    rng = random.Random(seed)
    win_len = int(fps * 2.0)
    start = _pick_start(rng, len(frames), win_len, active_range)
    fast = frames[start:start + win_len:2]
    pad = [frames[start + win_len - 1]] * len(fast)
    result = frames[:start] + fast + pad + frames[start + win_len:]
    return result, start, start + win_len


def tamper_clip_splice(frames_a, other_path, fps, seed=0, active_range=None):
    """Insert 2–3 seconds from other_path into frames_a."""
    rng = random.Random(seed)
    splice_len = int(fps * rng.uniform(2.0, 3.0))
    insert_at = _pick_start(rng, len(frames_a), splice_len, active_range)
    
    other_n_frames, other_fps = get_video_info(other_path)
    b_start_sec = rng.randint(0, max(0, other_n_frames - splice_len - 1)) / other_fps
    
    foreign = extract_segment(other_path, other_fps, b_start_sec, splice_len / other_fps)
    if foreign is None:
        raise ValueError("Could not extract foreign segment")
        
    h, w = frames_a[0].shape[:2]
    foreign = [cv2.resize(f, (w, h)) for f in foreign]
    result = frames_a[:insert_at] + foreign + frames_a[insert_at:]
    return result, insert_at, insert_at + splice_len


def tamper_reverse_segment(frames, fps, seed=0, active_range=None):
    """Reverse a 1.5-second segment — creates impossible temporal flow."""
    rng = random.Random(seed)
    rev_len = int(fps * 1.5)
    start = _pick_start(rng, len(frames), rev_len, active_range)
    rev_seg = frames[start:start + rev_len][::-1]
    result = frames[:start] + rev_seg + frames[start + rev_len:]
    return result, start, start + rev_len


TAMPER_FNS = {
    "frame_deletion":     tamper_frame_deletion,
    "frame_duplication":  tamper_frame_duplication,
    "speed_manipulation": tamper_speed_manipulation,
    "clip_splice":        None,  # needs frames from another source
    "reverse_segment":    tamper_reverse_segment,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load source catalog ──
    if not os.path.exists(args.sources_csv):
        print(f"[ERROR] Sources catalog not found: {args.sources_csv}")
        print(f"        Run: python scripts/download_dataset.py")
        sys.exit(1)

    import pandas as pd
    sources = pd.read_csv(args.sources_csv)
    print(f"[LOAD] {len(sources)} source videos from {sources['scene_id'].nunique()} scenes")

    os.makedirs(AUTH_DIR, exist_ok=True)
    os.makedirs(TAMP_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    detector = YOLODetector(config=CONFIG)

    manifest = []
    auth_count = 0
    tamp_count = 0

    source_ids = sources["source_id"].tolist()

    # ── Generate clips lazily per source ──
    for _, row in sources.iterrows():
        src_id = row["source_id"]
        src_path = os.path.join(ROOT, row["path"])
        if not os.path.exists(src_path):
            print(f"  [WARN] Missing: {row['path']}")
            continue
            
        print(f"\n[LOAD] {src_id}: {row['path']}")
        n_frames, fps = get_video_info(src_path)
        total_dur = n_frames / fps

        # ── Find windows with sustained person presence ──
        if args.min_person_frac > 0:
            active_windows = find_active_windows(
                src_path, n_frames, fps, args.clip_duration, detector,
                CONFIG.detection.person_class_id,
                min_person_frac=args.min_person_frac,
                stride=args.activity_stride,
                step_sec=args.activity_step_sec,
            )
            if not active_windows:
                print(f"  [SKIP] {src_id}: no window with >= "
                      f"{args.min_person_frac:.0%} person-presence found "
                      f"— this source has no usable sustained activity.")
                continue
            print(f"  [ACTIVITY] {src_id}: {len(active_windows)} qualifying "
                  f"window(s), best person_frac="
                  f"{active_windows[0]['person_frac']:.0%}")
        else:
            active_windows = None  # activity filtering disabled

        # ── Authentic clips ──
        if active_windows is not None:
            # Spread picks across the best-scoring windows, spaced apart
            # so repeated clips_per_source picks aren't all the same window.
            chosen_starts = []
            for w in active_windows:
                if all(abs(w["start_sec"] - s) >= args.clip_duration
                       for s in chosen_starts):
                    chosen_starts.append(w["start_sec"])
                if len(chosen_starts) >= args.clips_per_source:
                    break
        else:
            step = max(args.clip_duration + 0.5,
                       (total_dur - args.clip_duration) / max(1, args.clips_per_source))
            chosen_starts = [i * step for i in range(args.clips_per_source)]

        for start in chosen_starts:
            seg = extract_segment(src_path, fps, start, args.clip_duration)
            if seg is None:
                continue
            auth_count += 1
            clip_name = f"auth_{auth_count:04d}.mp4"
            out_path = os.path.join(AUTH_DIR, clip_name)
            write_frames(seg, out_path, fps)

            manifest.append({
                "clip_name": clip_name,
                "source_id": src_id,
                "scene_id": row["scene_id"],
                "label": 0,
                "tamper_type": "none",
                "tamper_start_frame": -1,
                "tamper_end_frame": -1,
            })
            print(f"  {clip_name} — {len(seg)} frames from {src_id}")

        # ── Tampered clips ──
        for ttype, fn in TAMPER_FNS.items():
            for variant in range(args.tampers_per_type):
                seed = hash((src_id, ttype, variant)) % (2**31)

                # Get a base segment — prefer an activity-qualifying window
                # (same ones authentic clips use) over a blind random start,
                # so frame_deletion/duplication/speed/reverse tampers land
                # on footage that actually has something to tamper with.
                # Also carry the window's active_range through to the
                # tamper function itself — without this, the base *segment*
                # can pass the activity bar on average while the specific
                # edit still lands in a dead stretch inside it (e.g. the
                # person leaves frame partway through), making the tamper
                # label physically undetectable regardless of pipeline
                # quality.
                rng = random.Random(seed)
                active_range = None
                if active_windows:
                    chosen_window = rng.choice(active_windows)
                    start_sec = chosen_window["start_sec"]
                    active_range = chosen_window["active_range"]
                else:
                    start_sec = rng.uniform(0, max(0.1, total_dur - args.clip_duration - 1))
                base_seg = extract_segment(src_path, fps, start_sec, args.clip_duration)
                if base_seg is None:
                    continue

                try:
                    if ttype == "clip_splice":
                        # Load another source randomly for splice
                        other_ids = [s for s in source_ids if s != src_id]
                        if not other_ids:
                            other_ids = source_ids  # fallback
                        other_id = rng.choice(other_ids)
                        other_row = sources[sources["source_id"] == other_id].iloc[0]
                        other_path = os.path.join(ROOT, other_row["path"])
                        
                        modified, t_start, t_end = tamper_clip_splice(
                            base_seg, other_path, fps, seed=seed,
                            active_range=active_range
                        )
                    else:
                        modified, t_start, t_end = fn(base_seg, fps, seed=seed,
                                                       active_range=active_range)
                except Exception as e:
                    print(f"  [ERROR] {ttype}/{src_id}/v{variant}: {e}")
                    continue

                tamp_count += 1
                clip_name = f"tamp_{tamp_count:04d}_{ttype}.mp4"
                out_path = os.path.join(TAMP_DIR, clip_name)
                write_frames(modified, out_path, fps)

                manifest.append({
                    "clip_name": clip_name,
                    "source_id": src_id,
                    "scene_id": row["scene_id"],
                    "label": 1,
                    "tamper_type": ttype,
                    "tamper_start_frame": t_start,
                    "tamper_end_frame": t_end,
                })
                print(f"  {clip_name} — {len(modified)} frames, {ttype} [{t_start}:{t_end}]")

        # Memory automatically managed since we stream from disk.
        pass
    # ── Write manifest ──
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    fieldnames = ["clip_name", "source_id", "scene_id", "label",
                  "tamper_type", "tamper_start_frame", "tamper_end_frame"]
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in manifest:
            writer.writerow(entry)

    # ── Recompression (for headline robustness experiment) ──
    if args.recompress:
        print(f"\n[RECOMPRESS] Creating re-encoded copies ...")
        for crf in args.crf:
            crf_dir = os.path.join(RECOMP_DIR, f"crf{crf}")
            os.makedirs(os.path.join(crf_dir, "authentic"), exist_ok=True)
            os.makedirs(os.path.join(crf_dir, "tampered"), exist_ok=True)

            for entry in manifest:
                label_dir = "authentic" if entry["label"] == 0 else "tampered"
                src_path = os.path.join(EVAL_DIR, label_dir, entry["clip_name"])
                dst_path = os.path.join(crf_dir, label_dir, entry["clip_name"])

                if os.path.exists(src_path):
                    ok = recompress_video(src_path, dst_path, crf=crf)
                    if not ok:
                        print(f"  [WARN] ffmpeg failed for {entry['clip_name']} CRF={crf}")

            print(f"  CRF {crf}: done → {crf_dir}")

    # ── Summary ──
    scenes = set(e["scene_id"] for e in manifest)
    tamper_types = set(e["tamper_type"] for e in manifest if e["label"] == 1)
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Authentic : {auth_count:>4}")
    print(f"  Tampered  : {tamp_count:>4}")
    print(f"  Total     : {auth_count + tamp_count:>4}")
    print(f"  Scenes    : {len(scenes)}")
    print(f"  Sources   : {len(source_ids)}")
    print(f"  Types     : {', '.join(sorted(tamper_types))}")
    print(f"  Manifest  : {MANIFEST}")
    print(f"\n  Next step: python scripts/evaluate_dataset.py")


if __name__ == "__main__":
    main()