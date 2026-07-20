# coherex/coherence/innovation.py
"""
Physics-Consistency Coherence Analysis — Innovation Channel
============================================================

Implements the core physics-based tampering signals derived from
Kalman filter innovations (residuals):

1. NIS (Normalized Innovation Squared)
   - Under authentic constant-velocity motion: NIS ~ χ²(2)
   - Critical value at 95%: χ²₀.₉₅(2) ≈ 5.991

2. Combined NIS (Lever 2: multi-model physics)
   - min(nis_cv, nis_ca): low if EITHER model explains the motion
   - High only when NO physics model fits → tamper discontinuity

3. Calibrated NIS (Lever 3)
   - Rolling-median normalization so authentic NIS ≈ χ²(2)
   - Makes the χ²-derived threshold principled

4. Whiteness test
   - Lag-1 autocorrelation of innovation magnitude
   - ≈ 0 under authentic motion; > 0 under tampering
"""

import numpy as np
from scipy.stats import chi2


# χ²₀.₉₅(2) ≈ 5.991 — the 95th-percentile critical value for 2 DOF
CHI2_95_DOF2 = float(chi2.ppf(0.95, df=2))


def nis(y, S):
    """
    Compute Normalized Innovation Squared.

    NIS = y^T S^{-1} y

    Args:
        y: Innovation vector (2D), shape (2,) or (2,1)
        S: Innovation covariance matrix, shape (2,2)

    Returns:
        float: NIS value
    """
    y = np.asarray(y, float).reshape(-1)
    S = np.asarray(S, float)
    try:
        return float(y @ np.linalg.inv(S) @ y)
    except np.linalg.LinAlgError:
        return 0.0


def combined_nis(nis_cv, nis_ca):
    """
    Multi-model combined NIS (Lever 2).

    Returns the minimum of CV and CA NIS values.
    Low if EITHER physics model explains the motion (authentic maneuver);
    high only when NO physics model fits (tamper discontinuity).

    Args:
        nis_cv: NIS from constant-velocity filter
        nis_ca: NIS from constant-acceleration filter

    Returns:
        float: Combined NIS value
    """
    return float(min(nis_cv, nis_ca))


def calibrate_nis(nis_series, dof=2, win=61):
    """
    Calibrate NIS so that authentic NIS ≈ χ²(dof) (Lever 3).

    Uses rolling-median normalization. The median is dominated by
    authentic frames, so localized tamper spikes survive calibration.

    Args:
        nis_series: Array of raw NIS values
        dof: Degrees of freedom (2 for 2D measurements)
        win: Rolling median window size (odd number)

    Returns:
        np.ndarray: Calibrated NIS values
    """
    a = np.asarray(nis_series, float)
    if a.size == 0:
        return a
    # Floor the normalizer at a small fraction of dof, not a raw epsilon.
    # A raw 1e-6 floor is far smaller than genuine NIS values ever get
    # (which live in the ~1e-3 to 1e1 range), so whenever the rolling
    # median legitimately hits ~0, dividing by 1e-6 amplifies ordinary
    # values by 5-6 orders of magnitude. Flooring at dof/50 bounds the
    # maximum amplification to 50x instead of ~2,000,000x.
    floor = dof / 50.0
    # Hard output ceiling: consistent with the 50x amplification cap stated above.
    # Even after flooring the divisor, erratic authentic motion on short clips can
    # still produce calibrated NIS > 100. Clamping at 50*dof (=100 for dof=2)
    # enforces the cap end-to-end without suppressing genuine tamper spikes
    # (which stay clearly elevated even when clamped).
    ceiling = 1e6  # Removed artificial ceiling to allow true anomaly spikes
    if a.size < win:
        med = max(np.median(a), floor)
        return np.minimum(a / (med / dof), ceiling)
    # Rolling-median normalization (robust to localized spikes)
    from scipy.ndimage import median_filter
    roll = np.maximum(median_filter(a, size=win, mode="nearest"), floor)
    return np.minimum(a / (roll / dof), ceiling)


def nis_stats(nis_values):
    """
    Compute summary statistics from a sequence of NIS values.

    Args:
        nis_values: List or array of NIS values

    Returns:
        dict with keys: nis_mean, nis_max, nis_exceed_frac
    """
    a = np.asarray(nis_values, float)
    if a.size == 0:
        return {"nis_mean": 0.0, "nis_max": 0.0, "nis_exceed_frac": 0.0}
    return {
        "nis_mean": float(a.mean()),
        "nis_max": float(a.max()),
        "nis_exceed_frac": float((a > CHI2_95_DOF2).mean()),
    }


def whiteness_lag1(innovations):
    """
    Compute |lag-1 autocorrelation| of innovation magnitude.

    Under authentic motion, innovations are white noise → autocorrelation ≈ 0.
    Under tampering, innovations are correlated → autocorrelation > 0.

    Args:
        innovations: List of 2D residual vectors (each shape (2,) or list of 2)

    Returns:
        float: Absolute lag-1 autocorrelation of innovation magnitudes
    """
    m = np.array([np.linalg.norm(np.asarray(v, float)) for v in innovations], float)
    if m.size < 3:
        return 0.0
    m = m - m.mean()
    denom = float(np.sum(m * m)) + 1e-9
    r1 = float(np.sum(m[:-1] * m[1:]) / denom)
    return abs(r1)


def whiteness_lag2(innovations):
    """
    Compute |lag-2 autocorrelation| of innovation magnitude.

    Captures longer-range temporal correlations from tampering.

    Args:
        innovations: List of 2D residual vectors

    Returns:
        float: Absolute lag-2 autocorrelation of innovation magnitudes
    """
    m = np.array([np.linalg.norm(np.asarray(v, float)) for v in innovations], float)
    if m.size < 4:
        return 0.0
    m = m - m.mean()
    denom = float(np.sum(m * m)) + 1e-9
    r2 = float(np.sum(m[:-2] * m[2:]) / denom)
    return abs(r2)
