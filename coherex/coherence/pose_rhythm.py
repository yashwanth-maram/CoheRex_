# coherex/coherence/pose_rhythm.py
"""
Pose-Based Gait Rhythm Analysis
=================================

Extracts gait-periodicity signals from pose keypoints to detect
temporal manipulation that disrupts walking rhythm:

1. Gait signal extraction (ankle-hip vertical oscillation, torso-normalized)
2. Rhythm residual via autocorrelation (period variability)
3. Spectral gait features (Upgrade 2a) — entropy, frequency stability, power

Tampering effects:
    - Duplication → raises spectral entropy (disrupted periodicity)
    - Speed-up → shifts dominant gait frequency
    - Deletion → breaks phase continuity
    - Splice → introduces foreign gait pattern
"""

import numpy as np

# COCO-17 keypoint indices used by YOLOv8-pose
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16


def gait_signal(keypoint_series):
    """
    Build a gait signal from a sequence of pose keypoints.

    The gait signal measures the torso-normalized vertical distance
    of ankles from the hip center — this oscillates during walking.

    Args:
        keypoint_series: List of (17,2) arrays (or None) for one track
                        across frames.

    Returns:
        np.ndarray: 1D gait signal (contains NaN where keypoints are missing)
    """
    sig = []
    for kp in keypoint_series:
        if kp is None or len(kp) <= R_ANK:
            sig.append(np.nan)
            continue
        kp = np.asarray(kp)
        hips = 0.5 * (kp[L_HIP] + kp[R_HIP])
        torso = np.linalg.norm(kp[L_SH] - kp[L_HIP]) + 1e-6
        val = (abs(kp[L_ANK][1] - hips[1]) + abs(kp[R_ANK][1] - hips[1])) / torso
        sig.append(val)
    return np.array(sig, float)


def rhythm_residual(sig, win=15):
    """
    Variability of local gait period estimated via autocorrelation.

    Spikes at stutter/phase/frequency breaks caused by tampering.

    Args:
        sig: 1D gait signal array
        win: Window size for local period estimation

    Returns:
        float: Coefficient of variation of local gait periods
               (higher = more disrupted rhythm)
    """
    g = np.asarray(sig, float)
    g = g[~np.isnan(g)]
    if g.size < 2 * win:
        return 0.0
    periods = []
    for s in range(0, len(g) - win, max(1, win // 2)):
        w = g[s:s + win] - np.mean(g[s:s + win])
        ac = np.correlate(w, w, mode="full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        for lag in range(2, len(ac) - 1):
            if ac[lag] > ac[lag - 1] and ac[lag] > ac[lag + 1]:
                periods.append(lag)
                break
    if len(periods) < 2:
        return 0.0
    return float(np.std(periods) / (np.mean(periods) + 1e-6))


def spectral_gait_features(gait_sig, fps=30.0):
    """
    Spectral analysis of gait signal for tampering detection (Upgrade 2a).

    Replaces the simpler period-variance heuristic with principled
    spectral features:
        - spec_entropy: Spectral entropy — high when periodicity is disrupted
        - dom_freq_stab: Dominant frequency stability between halves — high for speed tamper
        - dom_power: Dominant frequency power concentration — low when tampered

    Args:
        gait_sig: 1D gait signal array
        fps: Video frame rate

    Returns:
        dict with keys: spec_entropy, dom_freq_stab, dom_power
    """
    from scipy.signal import periodogram
    from scipy.stats import entropy

    g = np.asarray(gait_sig, float)
    g = g[~np.isnan(g)]
    if g.size < 16:
        return dict(spec_entropy=1.0, dom_freq_stab=0.0, dom_power=0.0)
    g = g - g.mean()

    def dom(seg):
        """Extract dominant frequency, power concentration, and spectral entropy."""
        f, p = periodogram(seg, fs=fps)
        if p.sum() <= 0:
            return 0.0, 0.0, 1.0
        pn = p / p.sum()
        return (
            float(f[np.argmax(p)]),
            float(p.max() / (p.sum() + 1e-9)),
            float(entropy(pn + 1e-12)),
        )

    h = len(g) // 2
    d1, _, _ = dom(g[:h])
    d2, _, _ = dom(g[h:])
    d_all, d_pow, s_ent = dom(g)

    return dict(
        spec_entropy=s_ent,
        dom_freq_stab=float(abs(d1 - d2) / (d_all + 1e-6)),  # ↑ = speed tamper
        dom_power=d_pow,
    )
