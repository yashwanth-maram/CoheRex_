# coherex/config.py
"""
CoheRex 2.0 — Deterministic Configuration System

Single source of truth for ALL thresholds, weights, and parameters.
frozen=True ensures immutability — config cannot change mid-execution.

Usage:
    from coherex.config import CONFIG

    CONFIG.detection.confidence_threshold   # 0.4
    CONFIG.physics.chi2_threshold           # 5.991
    CONFIG.to_dict()                        # serialize for report inclusion
"""

from dataclasses import dataclass, asdict
from typing import Tuple


# ─── Detection ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectionConfig:
    """Object detection parameters."""
    confidence_threshold: float = 0.4
    person_class_id: int = 0


# ─── Tracking ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrackingConfig:
    """Multi-object tracking and identity parameters."""
    max_association_distance: float = 9999.0
    max_misses: int = 5
    max_dormant_frames: int = 30
    tamper_latch_frames: int = 30
    appearance_similarity_threshold: float = 0.3
    identity_area_range: Tuple[float, float] = (0.2, 3.0)
    identity_ratio_threshold: float = 2.0
    max_trajectory_history: int = 10


# ─── Motion Coherence (legacy MCV — auxiliary channel) ────────────────────

@dataclass(frozen=True)
class MotionConfig:
    """Motion coherence analysis parameters (MCV-based, auxiliary)."""
    mcv_window_size: int = 8          # Unified: was 3 in engine default, 8 in config
    mcv_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)   # speed, accel, angle
    mcv_noise_floor: float = 1.5      # Documented: prevents z-score explosion on tiny jitter
    max_expected_mcv: float = 1.8
    interpreter_low: float = 0.5      # MCV < low  → COHERENT
    interpreter_high: float = 1.8     # MCV ≥ high → INCONSISTENT


# ─── Physics (NIS / Kalman / Calibration) ─────────────────────────────────

@dataclass(frozen=True)
class PhysicsConfig:
    """Physics-based coherence analysis parameters (NIS, dual-filter)."""
    chi2_dof: int = 2                           # Degrees of freedom for 2D measurement
    chi2_threshold: float = 5.991               # χ²₀.₉₅(2) — 95th percentile critical value
    calibration_window: int = 61                # Rolling-median window for NIS calibration
    process_noise_cv: float = 1.5               # Process noise for constant-velocity filter
    process_noise_ca: float = 0.5               # Process noise for constant-acceleration filter
    measurement_noise: float = 10.0             # BUG FIX: Was 150.0, hiding tamper spikes
    multi_scale_windows: Tuple[int, ...] = (15, 45, 150)  # Multi-scale temporal windows
    extreme_topk_fraction: float = 0.1          # Top-k = top 10% for extreme-value features
    min_track_length_for_nis: int = 10          # Exclude short tracks from NIS
    nis_warmup_frames: int = 5                  # Discard first N NIS values (P convergence)


# ─── Pose Rhythm ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoseRhythmConfig:
    """Pose-based gait rhythm analysis parameters."""
    autocorr_window: int = 15           # Window for local period estimation
    min_gait_frames: int = 16           # Minimum frames for spectral analysis
    spectral_fps: float = 30.0          # Default FPS for spectral analysis


# ─── Change-point Detection ──────────────────────────────────────────────

@dataclass(frozen=True)
class ChangepointConfig:
    """Change-point detection parameters."""
    cusum_drift: float = 0.5            # CUSUM drift parameter
    glr_min_samples: int = 4            # Minimum samples for GLR test


# ─── Integrity Agents ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContinuityConfig:
    """Track lifecycle stability parameters."""
    miss_penalty_scale: float = 10.0
    dormant_penalty: float = 0.2
    reattach_penalty: float = 0.15
    max_reattach_penalty: float = 0.45


@dataclass(frozen=True)
class CrowdConfig:
    """Crowd-level directional coherence parameters."""
    min_speed_threshold: float = 0.5
    min_tracks: int = 2
    max_deviation: float = 1.2


@dataclass(frozen=True)
class IntegrityConfig:
    """Multi-agent integrity fusion parameters."""
    fusion_weights: Tuple[float, float, float] = (0.7, 0.3, 0.0)  # motion, continuity, crowd
    segment_window_frames: int = 150
    verdict_thresholds: Tuple[float, float] = (0.85, 0.60)
    # HIGH       > 0.85
    # MODERATE     0.60–0.85
    # COMPROMISED < 0.60
    continuity: ContinuityConfig = ContinuityConfig()
    crowd: CrowdConfig = CrowdConfig()


# ─── Root Config ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemConfig:
    """Root configuration — single source of truth."""
    version: str = "CoheRex-Integrity v2.0"
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    motion: MotionConfig = MotionConfig()
    physics: PhysicsConfig = PhysicsConfig()
    pose_rhythm: PoseRhythmConfig = PoseRhythmConfig()
    changepoint: ChangepointConfig = ChangepointConfig()
    integrity: IntegrityConfig = IntegrityConfig()

    def to_dict(self) -> dict:
        """Serialize full config for inclusion in forensic reports."""
        return asdict(self)


# ─── Global Default ──────────────────────────────────────────────────────

CONFIG = SystemConfig()
