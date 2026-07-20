# coherex/coherence/changepoint.py
"""
Change-Point Detection for Temporal Tampering Localization
===========================================================

Two change-point detectors for localizing the frame at which
tampering occurs:

1. CUSUM (Page's test) — cumulative sum control chart
   - Detects sustained shifts in the signal mean
   - Returns (peak_statistic, peak_index)

2. GLR (Generalized Likelihood Ratio) — Lever 4
   - Most powerful detector for a single abrupt shift at unknown location
   - Higher detection power → higher AUC on abrupt edits
   - Sharper localization peak
"""

import numpy as np


def cusum_peak(signal, drift=0.5):
    """
    Two-sided CUSUM (Page's test) on a standardized signal.

    Detects sustained departures from the signal's mean in either direction.
    Returns the peak accumulated evidence of a shift and the frame index
    where it occurs.

    Args:
        signal: 1D array of values (e.g., innovation magnitudes, NIS values)
        drift: Allowance parameter — larger values make the test less
               sensitive to small shifts but more specific

    Returns:
        tuple: (peak_statistic, peak_index)
            peak_statistic: Maximum of the positive and negative CUSUM paths
            peak_index: Frame index of the peak (-1 if signal too short)
    """
    x = np.asarray(signal, float)
    if x.size < 3:
        return 0.0, -1
    mu, sd = float(x.mean()), float(x.std()) + 1e-9
    z = (x - mu) / sd
    s_pos = np.zeros_like(z)
    s_neg = np.zeros_like(z)
    for t in range(1, len(z)):
        s_pos[t] = max(0.0, s_pos[t - 1] + z[t] - drift)
        s_neg[t] = max(0.0, s_neg[t - 1] - z[t] - drift)
    s = np.maximum(s_pos, s_neg)
    idx = int(np.argmax(s))
    return float(s[idx]), idx


def glr_change_point(x):
    """
    Gaussian GLR test for a single mean shift at unknown location (Lever 4).

    The generalized-likelihood-ratio test is the most powerful detector
    for a single abrupt shift at unknown location. It scans all possible
    split points and returns the one with maximum evidence of a mean change.

    Assumes Gaussian data with unknown but equal variance in both segments.

    Args:
        x: 1D array of values (e.g., calibrated NIS, innovation magnitudes)

    Returns:
        tuple: (glr_peak, index)
            glr_peak: Maximum GLR statistic (higher = stronger evidence)
            index: Estimated change-point frame index (-1 if too short)
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return 0.0, -1
    total = x.sum()
    best, best_k, csum = 0.0, -1, 0.0
    for k in range(1, n):
        csum += x[k - 1]
        n1, n2 = k, n - k
        # GLR statistic for Gaussian mean-shift at position k
        stat = (csum * csum) / n1 + ((total - csum) ** 2) / n2 - (total * total) / n
        if stat > best:
            best, best_k = stat, k
    return float(best), int(best_k)


def glr_normalized_dispersion(x):
    """
    Compute the GLR change-point and its normalized location dispersion.

    Returns additional information about how confident the localization is:
    high dispersion means the change-point evidence is spread out (gradual change),
    low dispersion means it's concentrated (sharp edit).

    Args:
        x: 1D array of values

    Returns:
        dict with keys:
            glr_peak: Maximum GLR statistic
            glr_index: Estimated change-point frame
            glr_dispersion: Normalized location dispersion (0-1)
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 4:
        return {"glr_peak": 0.0, "glr_index": -1, "glr_dispersion": 0.0}

    total = x.sum()
    stats = np.zeros(n - 1)
    csum = 0.0
    for k in range(1, n):
        csum += x[k - 1]
        n1, n2 = k, n - k
        stats[k - 1] = (csum * csum) / n1 + ((total - csum) ** 2) / n2 - (total * total) / n

    best_k = int(np.argmax(stats))
    best = float(stats[best_k])

    # Normalized dispersion: how spread out is the evidence?
    if best > 1e-9:
        weights = stats / stats.sum()
        positions = np.arange(len(stats)) / len(stats)
        mean_pos = float(np.sum(weights * positions))
        dispersion = float(np.sqrt(np.sum(weights * (positions - mean_pos) ** 2)))
    else:
        dispersion = 0.0

    return {
        "glr_peak": best,
        "glr_index": best_k + 1,  # 1-indexed frame
        "glr_dispersion": dispersion,
    }
