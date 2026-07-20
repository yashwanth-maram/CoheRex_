# coherex/meta/feature_extractor.py
"""
CoheRex 2.0 — Canonical Feature Set
=====================================

Converts raw pipeline outputs into the principled feature vector
defined in Section 3 of the paper. This is the SINGLE canonical
feature list — matching the paper exactly.

Feature channels (per track, aggregated to video level):

  NIS Channel (calibrated combined_nis, multi-scale extreme values):
    - nis_peak, nis_topk_mean, nis_exceed_frac, nis_longest_run

  Change-Point Channel (GLR):
    - glr_peak, glr_dispersion

  Whiteness Channel (innovation autocorrelation):
    - whiteness_lag1, whiteness_lag2

  Pose-Rhythm Channel (spectral gait analysis):
    - spec_entropy, dom_freq_stab, dom_power

  Scene Channel (track count dynamics):
    - tc_max_jump, tc_jump_frac
"""

# ─── Canonical Feature Names (must match training script) ──────────────────
# NOTE: Order must match the vector produced by build_feature_vector
FEATURE_NAMES = [
    # NIS channel (multi-scale extreme values on calibrated combined NIS)
    "nis_peak",
    "nis_topk_mean",
    "nis_exceed_frac",
    "nis_longest_run",
    "nis_relative_spike",
    "nis_mad_zscore",
    "nis_top10_vs_bot50",
    # Change-point channel (GLR)
    "glr_peak",
    "glr_dispersion",
    # Whiteness channel
    "whiteness_lag1",
    "whiteness_lag2",
    # Pose-rhythm channel (spectral)
    "spec_entropy",
    "dom_freq_stab",
    "dom_power",
    "tc_max_jump",
    "tc_jump_frac",
    "birth_burst",
    "death_burst",
    "_baseline_ssim_mean",
    "_baseline_flow_var",
    "_legacy_motion_mean",
    "_legacy_motion_min",
    "_legacy_max_mcv",
]

# Legacy feature names (for backward compatibility / baseline comparison)
LEGACY_FEATURE_NAMES = [
    "motion_min",
    "motion_mean",
    "min_integrity",
    "volatility",
    "anomaly_density",
    "perc_below_06",
    "max_mcv_log",
]


def build_feature_vector(
    nis_peak: float = 0.0,
    nis_topk_mean: float = 0.0,
    nis_exceed_frac: float = 0.0,
    nis_longest_run: int = 0,
    glr_peak: float = 0.0,
    glr_dispersion: float = 0.0,
    whiteness_lag1: float = 0.0,
    whiteness_lag2: float = 0.0,
    spec_entropy: float = 1.0,
    dom_freq_stab: float = 0.0,
    tc_max_jump: int = 0,
    tc_jump_frac: float = 0.0,
    birth_burst: int = 0,
    death_burst: int = 0,
    identity_switches: int = 0,
) -> list:
    """
    Build the canonical CoheRex 2.0 feature vector.

    All arguments correspond to the principled physics-based signals
    defined in the paper. No ad-hoc features.
    """
    dom_power = None
    return [
        float(nis_peak),
        float(nis_topk_mean),
        float(nis_exceed_frac),
        float(nis_longest_run),
        float(glr_peak),
        float(glr_dispersion),
        float(whiteness_lag1),
        float(whiteness_lag2),
        float(spec_entropy),
        float(dom_freq_stab),
        float(dom_power),
        float(tc_max_jump),
        float(tc_jump_frac),
        float(birth_burst),
        float(death_burst),
        float(identity_switches),
    ]


def build_feature_dict(**kwargs) -> dict:
    """
    Build a named feature dictionary from keyword arguments.
    Validates that all required features are present.
    """
    vec = build_feature_vector(**kwargs)
    return dict(zip(FEATURE_NAMES, vec))


def build_legacy_feature_vector(
    motion_mean: float = 1.0,
    motion_min: float = 1.0,
    min_integrity: float = 1.0,
    volatility: float = 0.0,
    anomaly_density: float = 0.0,
    perc_below_06: float = 0.0,
    max_mcv: float = 0.0,
) -> list:
    """
    Build the legacy feature vector (for baseline comparison).
    Preserved for backward compatibility with the original system.
    """
    import math
    return [
        float(motion_min),
        float(motion_mean),
        float(min_integrity),
        float(volatility),
        float(anomaly_density),
        float(perc_below_06),
        math.log1p(max(0.0, float(max_mcv))),  # max_mcv_log
    ]


def build_legacy_feature_dict(**kwargs) -> dict:
    """Build a named legacy feature dictionary."""
    vec = build_legacy_feature_vector(**kwargs)
    return dict(zip(LEGACY_FEATURE_NAMES, vec))
