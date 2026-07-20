# coherex/coherence/aggregate.py
"""
Extreme-Value & Multi-Scale Feature Aggregation
================================================

Replaces mean-based video-level feature aggregation with
extreme-value statistics (Lever 1) and multi-scale temporal
windows (Lever 5).

Why extreme values work:
    Tampering is a *localized burst*; averaging a 10-frame edit
    over a 90-frame clip hides it. Extreme-value statistics
    (peak, top-k mean, exceed fraction, longest run) capture
    the forensic signal that means miss.

Multi-scale windows:
    Short windows (15 frames) catch frame deletions.
    Long windows (150 frames) catch slow speed-ramps.
    Feature = max over scales for each statistic.
"""

import numpy as np


def extreme_features(series, thresh):
    """
    Compute extreme-value statistics from a 1D signal (Lever 1).

    Args:
        series: 1D array of values (e.g., NIS, GLR, innovation magnitude)
        thresh: Threshold for exceed_frac and longest_run computation
                (e.g., chi2_threshold = 5.991 for NIS)

    Returns:
        dict with keys:
            peak: Maximum value in the series
            topk_mean: Mean of top 10% of values
            exceed_frac: Fraction of values exceeding threshold
            longest_run: Longest consecutive run of values above threshold
    """
    a = np.asarray(series, float)
    if a.size == 0:
        return dict(peak=0.0, topk_mean=0.0, exceed_frac=0.0, longest_run=0)

    k = max(1, a.size // 10)  # top 10%
    topk = np.sort(a)[::-1][:k]

    over = a > thresh
    longest = cur = 0
    for v in over:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)

    return dict(
        peak=float(a.max()),
        topk_mean=float(topk.mean()),
        exceed_frac=float(over.mean()),
        longest_run=int(longest),
    )


def multi_scale_features(series, thresh, windows=(15, 45, 150)):
    """
    Compute extreme-value features at multiple temporal scales (Lever 5).

    For each window size, slide the window over the signal, compute
    extreme features in each window, then take the MAX over all windows
    and positions for each statistic.

    Short windows catch frame deletions; long windows catch slow speed-ramps.

    Args:
        series: 1D array of values
        thresh: Threshold for exceed/run computation
        windows: Tuple of window sizes (in frames)

    Returns:
        dict with keys:
            ms_peak: Max peak across all scales
            ms_topk_mean: Max top-k mean across all scales
            ms_exceed_frac: Max exceed fraction across all scales
            ms_longest_run: Max longest run across all scales
    """
    a = np.asarray(series, float)
    if a.size == 0:
        return dict(ms_peak=0.0, ms_topk_mean=0.0, ms_exceed_frac=0.0, ms_longest_run=0)

    best = dict(ms_peak=0.0, ms_topk_mean=0.0, ms_exceed_frac=0.0, ms_longest_run=0)

    for w in windows:
        if a.size < w:
            # If signal is shorter than window, use full signal
            ef = extreme_features(a, thresh)
            best["ms_peak"] = max(best["ms_peak"], ef["peak"])
            best["ms_topk_mean"] = max(best["ms_topk_mean"], ef["topk_mean"])
            best["ms_exceed_frac"] = max(best["ms_exceed_frac"], ef["exceed_frac"])
            best["ms_longest_run"] = max(best["ms_longest_run"], ef["longest_run"])
            continue

        # Slide window with step = w//2
        step = max(1, w // 2)
        for start in range(0, len(a) - w + 1, step):
            segment = a[start:start + w]
            ef = extreme_features(segment, thresh)
            best["ms_peak"] = max(best["ms_peak"], ef["peak"])
            best["ms_topk_mean"] = max(best["ms_topk_mean"], ef["topk_mean"])
            best["ms_exceed_frac"] = max(best["ms_exceed_frac"], ef["exceed_frac"])
            best["ms_longest_run"] = max(best["ms_longest_run"], ef["longest_run"])

    return best


def track_count_discontinuity(track_counts):
    """
    Detect sudden changes in the number of active tracks (scene-level signal).

    Splice/insertion tends to cause abrupt track count changes.

    Args:
        track_counts: 1D array of track counts per frame

    Returns:
        dict with keys:
            tc_max_jump: Maximum absolute frame-to-frame change in track count
            tc_jump_frac: Fraction of frames with track count change > 1
    """
    a = np.asarray(track_counts, float)
    if a.size < 2:
        return dict(tc_max_jump=0, tc_jump_frac=0.0)

    diffs = np.abs(np.diff(a))
    return dict(
        tc_max_jump=int(diffs.max()),
        tc_jump_frac=float((diffs > 1).mean()),
    )


def appearance_burst(birth_frames, death_frames, total_frames):
    """
    Detect bursts of track appearances/disappearances.

    Splice edits cause sudden appearance/disappearance of multiple tracks.

    Args:
        birth_frames: List of frame indices where tracks first appeared
        death_frames: List of frame indices where tracks were terminated
        total_frames: Total number of frames in the video

    Returns:
        dict with keys:
            birth_burst: Maximum births in any 5-frame window
            death_burst: Maximum deaths in any 5-frame window
    """
    if total_frames < 1:
        return dict(birth_burst=0, death_burst=0)

    win = 5

    def max_burst(frame_list):
        if not frame_list:
            return 0
        counts = np.zeros(total_frames)
        for f in frame_list:
            if 0 <= f < total_frames:
                counts[f] += 1
        if len(counts) < win:
            return int(counts.max()) if counts.size > 0 else 0
        rolling = np.convolve(counts, np.ones(win), mode='valid')
        return int(rolling.max()) if rolling.size > 0 else 0

    return dict(
        birth_burst=max_burst(birth_frames),
        death_burst=max_burst(death_frames),
    )
