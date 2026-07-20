"""
build_features.py — CoheRex 2.0 Fast Feature Builder (runs in SECONDS)
=======================================================================

Reads .npz cache files from extract_raw.py and computes the full feature
CSV. This is where you iterate — every experiment costs seconds, not hours.

Feature channels:
  pix_*    — pixel-residual extreme-value features (the accuracy engine)
  phys_*   — physics-based NIS/GLR/whiteness features (the robustness engine)
  scene_*  — scene-level track dynamics

Usage:
    python scripts/build_features.py --cache cache/clean --out results/features_hybrid.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.coherence.innovation import calibrate_nis, whiteness_lag1, whiteness_lag2
from coherex.coherence.changepoint import glr_normalized_dispersion, cusum_peak
from coherex.coherence.aggregate import extreme_features, multi_scale_features, \
    track_count_discontinuity, appearance_burst
from coherex.config import CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="CoheRex 2.0 — Fast Feature Builder")
    p.add_argument("--cache", type=str, required=True,
                   help="Cache directory containing .npz files")
    p.add_argument("--out", type=str, default="results/features_hybrid.csv",
                   help="Output feature CSV path")
    p.add_argument("--min-track-len", type=int, default=15,
                   help="Minimum track length for physics features (quality gate)")
    p.add_argument("--min-bbox-h", type=float, default=60.0,
                   help="Minimum mean bbox height for physics features (quality gate)")
    return p.parse_args()


def extreme(seq, thr=None):
    """Compute extreme-value statistics from a 1D signal."""
    a = np.asarray(seq, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(peak=0.0, topk=0.0, frac=0.0, run=0)
    
    if thr is None:
        thr = np.percentile(a, 95)
    
    k = max(1, a.size // 10)
    topk = np.sort(a)[::-1][:k]
    
    over = a > thr
    run = mx = 0
    for v in over:
        run = run + 1 if v else 0
        mx = max(mx, run)
    
    return dict(
        peak=float(a.max()),
        topk=float(topk.mean()),
        frac=float(over.mean()),
        run=int(mx),
    )


def build_pixel_features(npz):
    """Build pixel-residual extreme-value features from cached sequences."""
    features = {}

    # Frame-diff channel — the sharpest deletion/insertion detector
    fd = npz.get("frame_diff", np.array([]))
    if fd.size > 0:
        fd_stats = extreme(fd)
        features["pix_fd_peak"] = fd_stats["peak"]
        features["pix_fd_topk"] = fd_stats["topk"]
        features["pix_fd_frac"] = fd_stats["frac"]
        features["pix_fd_run"] = fd_stats["run"]
        
        # First-difference peak: captures the sharpest jump in frame-diff
        if fd.size > 1:
            fd_diff = np.abs(np.diff(fd))
            features["pix_fd_diff_peak"] = float(fd_diff.max())
            features["pix_fd_diff_topk"] = float(np.sort(fd_diff)[::-1][:max(1, len(fd_diff)//10)].mean())
        else:
            features["pix_fd_diff_peak"] = 0.0
            features["pix_fd_diff_topk"] = 0.0
        
        # Additional stats
        features["pix_fd_std"] = float(np.std(fd))
        features["pix_fd_kurtosis"] = float(_kurtosis(fd))
    else:
        for k in ["pix_fd_peak", "pix_fd_topk", "pix_fd_frac", "pix_fd_run",
                   "pix_fd_diff_peak", "pix_fd_diff_topk", "pix_fd_std", "pix_fd_kurtosis"]:
            features[k] = 0.0

    # SSIM-drop channel
    sd = npz.get("ssim_drop", np.array([]))
    if sd.size > 0:
        sd_stats = extreme(sd)
        features["pix_sd_peak"] = sd_stats["peak"]
        features["pix_sd_topk"] = sd_stats["topk"]
        features["pix_sd_frac"] = sd_stats["frac"]
        features["pix_sd_run"] = sd_stats["run"]
        features["pix_sd_std"] = float(np.std(sd))
    else:
        for k in ["pix_sd_peak", "pix_sd_topk", "pix_sd_frac", "pix_sd_run", "pix_sd_std"]:
            features[k] = 0.0

    # Flow-spike channel
    fv = npz.get("flow_spike", np.array([]))
    if fv.size > 0:
        fv_stats = extreme(fv)
        features["pix_fv_peak"] = fv_stats["peak"]
        features["pix_fv_topk"] = fv_stats["topk"]
        features["pix_fv_frac"] = fv_stats["frac"]
        features["pix_fv_kurtosis"] = float(_kurtosis(fv))
        
        # Flow acceleration (first-difference of flow)
        if fv.size > 1:
            fv_diff = np.abs(np.diff(fv))
            features["pix_fv_accel_peak"] = float(fv_diff.max())
        else:
            features["pix_fv_accel_peak"] = 0.0
    else:
        for k in ["pix_fv_peak", "pix_fv_topk", "pix_fv_frac",
                   "pix_fv_kurtosis", "pix_fv_accel_peak"]:
            features[k] = 0.0

    # Appearance-discontinuity channel (splice detector)
    chd = npz.get("color_hist_dist", np.array([]))
    if chd.size > 0:
        chd_stats = extreme(chd)
        features["pix_appear_peak"] = chd_stats["peak"]
        features["pix_appear_topk"] = chd_stats["topk"]
        features["pix_appear_frac"] = chd_stats["frac"]
    else:
        for k in ["pix_appear_peak", "pix_appear_topk", "pix_appear_frac"]:
            features[k] = 0.0

    return features


def build_physics_features(npz, min_track_len=15, min_bbox_h=60.0):
    """Build physics-based features from cached sequences."""
    features = {}
    
    chi2_thresh = CONFIG.physics.chi2_threshold
    ms_windows = CONFIG.physics.multi_scale_windows

    # ── NIS Channel (from per-frame aggregated NIS) ──
    per_frame_nis = npz.get("per_frame_nis", np.array([]))
    
    if per_frame_nis.size > 0:
        # Calibrate NIS
        cal_nis = calibrate_nis(
            per_frame_nis.tolist(),
            dof=CONFIG.physics.chi2_dof,
            win=CONFIG.physics.calibration_window
        )
        
        # Multi-scale extreme features
        ms_nis = multi_scale_features(cal_nis, chi2_thresh, windows=ms_windows)
        features["phys_nis_peak"] = ms_nis["ms_peak"]
        features["phys_nis_topk"] = ms_nis["ms_topk_mean"]
        features["phys_nis_frac"] = ms_nis["ms_exceed_frac"]
        features["phys_nis_run"] = ms_nis["ms_longest_run"]
        
        # GLR change-point
        glr_info = glr_normalized_dispersion(cal_nis)
        features["phys_glr_peak"] = glr_info["glr_peak"]
        features["phys_glr_disp"] = glr_info["glr_dispersion"]
        features["phys_glr_frame"] = glr_info["glr_index"]
        
        # CUSUM
        cusum_p, cusum_f = cusum_peak(cal_nis)
        features["phys_cusum_peak"] = cusum_p
        features["phys_cusum_frame"] = cusum_f
        
        # Additional NIS stats
        nis_stats = extreme(cal_nis, thr=chi2_thresh)
        features["phys_nis_relative_spike"] = nis_stats["peak"] / (np.median(cal_nis) + 1e-6) if len(cal_nis) > 0 else 0.0
        features["phys_nis_mad_zscore"] = _mad_zscore_peak(cal_nis)
    else:
        for k in ["phys_nis_peak", "phys_nis_topk", "phys_nis_frac", "phys_nis_run",
                   "phys_glr_peak", "phys_glr_disp", "phys_glr_frame",
                   "phys_cusum_peak", "phys_cusum_frame",
                   "phys_nis_relative_spike", "phys_nis_mad_zscore"]:
            features[k] = 0.0

    # ── Whiteness Channel ──
    innovations = npz.get("innovations", np.array([]))
    if hasattr(innovations, 'tolist'):
        innovations = innovations.tolist()
    if innovations is not None and len(innovations) > 0:
        try:
            features["phys_w_lag1"] = whiteness_lag1(innovations)
            features["phys_w_lag2"] = whiteness_lag2(innovations)
        except Exception:
            features["phys_w_lag1"] = 0.0
            features["phys_w_lag2"] = 0.0
    else:
        features["phys_w_lag1"] = 0.0
        features["phys_w_lag2"] = 0.0

    # ── Multi-step prediction residual (predK_peak) ──
    # For quality-gated tracks: predict K frames ahead, measure position error
    track_positions = npz.get("track_positions", np.array({}, dtype=object)).item() \
        if isinstance(npz.get("track_positions", None), np.ndarray) else {}
    track_meta = npz.get("track_meta", np.array([], dtype=object))
    if isinstance(track_meta, np.ndarray):
        try:
            track_meta = track_meta.item() if track_meta.ndim == 0 else track_meta.tolist()
        except Exception:
            track_meta = []

    max_pred_error = 0.0
    for tmeta in (track_meta if isinstance(track_meta, list) else []):
        tid = tmeta.get("track_id")
        tlen = tmeta.get("length", 0)
        mean_h = tmeta.get("mean_bbox_h", 0)
        
        # Quality gate
        if tlen < min_track_len or mean_h < min_bbox_h:
            continue
        
        positions = track_positions.get(tid, [])
        if len(positions) < 10:
            continue
        
        # Multi-step prediction: predict K=5 frames ahead from each point
        K = 5
        positions_arr = np.array(positions)
        for i in range(len(positions_arr) - K - 1):
            # Velocity estimate from frames i-1 to i
            if i < 1:
                continue
            vx = positions_arr[i, 0] - positions_arr[i-1, 0]
            vy = positions_arr[i, 1] - positions_arr[i-1, 1]
            
            # Predict K frames ahead using constant velocity
            pred_x = positions_arr[i, 0] + vx * K
            pred_y = positions_arr[i, 1] + vy * K
            
            # Actual position K frames ahead
            actual_x = positions_arr[i + K, 0]
            actual_y = positions_arr[i + K, 1]
            
            # Prediction error (normalized by bbox height for scale invariance)
            error = np.hypot(pred_x - actual_x, pred_y - actual_y) / max(mean_h, 1.0)
            max_pred_error = max(max_pred_error, error)
    
    features["phys_predK_peak"] = max_pred_error

    # ── Legacy features ──
    motion_scores = npz.get("motion_scores", np.array([]))
    features["phys_motion_min"] = float(np.min(motion_scores)) if motion_scores.size > 0 else 1.0
    features["phys_motion_mean"] = float(np.mean(motion_scores)) if motion_scores.size > 0 else 1.0
    
    raw_mcvs = npz.get("raw_mcvs", np.array([]))
    features["phys_max_mcv"] = float(np.max(raw_mcvs)) if raw_mcvs.size > 0 else 0.0

    return features


def build_scene_features(npz):
    """Build scene-level track dynamics features."""
    features = {}
    
    frame_track_counts = npz.get("frame_track_counts", np.array([]))
    birth_frames = npz.get("birth_frames", np.array([]))
    death_frames = npz.get("death_frames", np.array([]))
    total_frames = int(npz.get("total_frames", 0))
    
    if frame_track_counts.size > 0:
        tc = track_count_discontinuity(frame_track_counts.tolist())
        features["scene_tc_max_jump"] = tc["tc_max_jump"]
        features["scene_tc_jump_frac"] = tc["tc_jump_frac"]
    else:
        features["scene_tc_max_jump"] = 0
        features["scene_tc_jump_frac"] = 0.0
    
    ab = appearance_burst(birth_frames.tolist(), death_frames.tolist(), total_frames)
    features["scene_birth_burst"] = ab["birth_burst"]
    features["scene_death_burst"] = ab["death_burst"]
    
    # Track dynamics
    track_meta = npz.get("track_meta", np.array([], dtype=object))
    if isinstance(track_meta, np.ndarray):
        try:
            track_meta = track_meta.item() if track_meta.ndim == 0 else track_meta.tolist()
        except Exception:
            track_meta = []
    
    if isinstance(track_meta, list) and len(track_meta) > 0:
        lengths = [t.get("length", 0) for t in track_meta]
        features["scene_num_tracks"] = len(track_meta)
        features["scene_max_track_len"] = max(lengths) if lengths else 0
        features["scene_mean_track_len"] = float(np.mean(lengths)) if lengths else 0.0
        features["scene_track_ratio"] = features["scene_mean_track_len"] / (features["scene_max_track_len"] + 1)
    else:
        features["scene_num_tracks"] = 0
        features["scene_max_track_len"] = 0
        features["scene_mean_track_len"] = 0.0
        features["scene_track_ratio"] = 0.0
    
    # Detector confidence
    confidences = npz.get("confidences", np.array([]))
    features["scene_avg_confidence"] = float(np.mean(confidences)) if confidences.size > 0 else 0.0
    
    return features


def _kurtosis(arr):
    """Compute excess kurtosis (robust to small samples)."""
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 4:
        return 0.0
    m = a.mean()
    s = a.std()
    if s < 1e-10:
        return 0.0
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


def _mad_zscore_peak(arr):
    """Peak MAD z-score of a sequence."""
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return 0.0
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad < 1e-10:
        return 0.0
    return float(np.max(np.abs(a - med) / (1.4826 * mad)))


# ── Feature channel definitions (for --feature-set flag) ──

PIXEL_FEATURES = [
    "pix_fd_peak", "pix_fd_topk", "pix_fd_frac", "pix_fd_run",
    "pix_fd_diff_peak", "pix_fd_diff_topk", "pix_fd_std", "pix_fd_kurtosis",
    "pix_sd_peak", "pix_sd_topk", "pix_sd_frac", "pix_sd_run", "pix_sd_std",
    "pix_fv_peak", "pix_fv_topk", "pix_fv_frac", "pix_fv_kurtosis", "pix_fv_accel_peak",
    "pix_appear_peak", "pix_appear_topk", "pix_appear_frac",
]

PHYSICS_FEATURES = [
    "phys_nis_peak", "phys_nis_topk", "phys_nis_frac", "phys_nis_run",
    "phys_glr_peak", "phys_glr_disp",
    "phys_cusum_peak",
    "phys_nis_relative_spike", "phys_nis_mad_zscore",
    "phys_w_lag1", "phys_w_lag2",
    "phys_predK_peak",
    "phys_motion_min", "phys_motion_mean", "phys_max_mcv",
]

SCENE_FEATURES = [
    "scene_num_tracks", "scene_max_track_len", "scene_mean_track_len",
    "scene_track_ratio", "scene_avg_confidence",
]

HYBRID_FEATURES = PIXEL_FEATURES + PHYSICS_FEATURES + SCENE_FEATURES

META_COLUMNS = [
    "video_name", "label", "source_id", "scene_id",
    "tamper_type", "tamper_start_frame", "tamper_end_frame",
]


def main():
    args = parse_args()

    npz_files = sorted(glob(os.path.join(args.cache, "*.npz")))
    if not npz_files:
        print(f"[ERROR] No .npz files found in {args.cache}")
        sys.exit(1)

    print(f"[BUILD] Processing {len(npz_files)} cached videos from {args.cache}/")
    print(f"[BUILD] Track quality gate: len>={args.min_track_len}, bbox_h>={args.min_bbox_h}px")

    rows = []
    for npz_path in npz_files:
        npz = np.load(npz_path, allow_pickle=True)

        # Build all feature channels
        pix = build_pixel_features(npz)
        phys = build_physics_features(npz, args.min_track_len, args.min_bbox_h)
        scene = build_scene_features(npz)

        # Combine
        row = {}
        row.update(pix)
        row.update(phys)
        row.update(scene)

        # Metadata
        row["video_name"] = str(npz.get("video_name", os.path.basename(npz_path).replace(".npz", ".mp4")))
        row["label"] = int(npz.get("label", -1))
        row["source_id"] = str(npz.get("source_id", "unknown"))
        row["scene_id"] = str(npz.get("scene_id", "unknown"))
        row["tamper_type"] = str(npz.get("tamper_type", "unknown"))
        row["tamper_start_frame"] = int(npz.get("tamper_start_frame", -1))
        row["tamper_end_frame"] = int(npz.get("tamper_end_frame", -1))

        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns: metadata first, then features
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    df = df[META_COLUMNS + sorted(feature_cols)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n[DONE] Saved {len(df)} rows × {len(feature_cols)} features to {args.out}")
    print(f"       Pixel features:  {len([c for c in feature_cols if c.startswith('pix_')])}")
    print(f"       Physics features: {len([c for c in feature_cols if c.startswith('phys_')])}")
    print(f"       Scene features:   {len([c for c in feature_cols if c.startswith('scene_')])}")
    print(f"\n       Next: python scripts/train_meta_classifier.py --csv {args.out} --feature-set hybrid")


if __name__ == "__main__":
    main()
