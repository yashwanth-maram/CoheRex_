# coherex/integrity/motion_agent.py
"""
Motion Agent — NIS-Based Physics Coherence Scoring
====================================================

CoheRex 2.0: Uses Normalized Innovation Squared (NIS) from the
dual Kalman filters as the primary motion coherence signal.

Scoring:
    High score (→ 1.0) = low NIS, physics-consistent motion
    Low score  (→ 0.0) = high NIS, physics-violating motion

The undocumented "Motion-Triggered Latch" (which forced integrity to 0
for 10 frames) has been REMOVED as it was unreported and distorted
downstream features.

MCV (Motion Coherence Value) is retained as an auxiliary channel
but NIS is now the primary signal.
"""

import numpy as np
from coherex.config import CONFIG
from coherex.integrity.base_agent import IntegrityAgent
from coherex.coherence.mcv import MotionCoherenceEngine
from coherex.coherence.innovation import CHI2_95_DOF2


class MotionAgent(IntegrityAgent):
    """
    Wraps the physics-based motion coherence scoring.

    Primary signal: NIS from dual Kalman filters
    Auxiliary signal: MCV (motion coherence value)
    """

    def __init__(self, config=None):
        cfg = config or CONFIG
        self.mcv_engine = MotionCoherenceEngine(
            window_size=cfg.motion.mcv_window_size,
            weights=cfg.motion.mcv_weights,
            noise_floor=getattr(cfg.motion, "mcv_noise_floor", 1.0)
        )
        self.max_expected = cfg.motion.max_expected_mcv
        self.chi2_threshold = cfg.physics.chi2_threshold

    def evaluate(self, trajectory_state, track_confidence, track=None, current_frame=0):
        """
        Compute the motion coherence score for a single track.

        Uses NIS-based scoring when NIS history is available,
        falls back to MCV-based scoring for young tracks.

        Args:
            trajectory_state (TrajectoryState | None):
                The trajectory state object for the track.
            track_confidence (float):
                Detection confidence ∈ [0, 1].
            track (Track | None):
                The track object (for NIS access).
            current_frame (int):
                Current frame index.

        Returns:
            float: Coherence score ∈ [0.0, 1.0].
        """
        if trajectory_state is None:
            return 1.0

        # ── Primary: NIS-based scoring ──
        # Use the track's combined NIS if available
        if track is not None and hasattr(track, 'nis_combined_history') and len(track.nis_combined_history) > 0:
            # Use the latest combined NIS value
            latest_nis = track.nis_combined_history[-1]
            # Score = 1 - min(NIS / χ²₀.₉₅, 1.0)
            # NIS < threshold → coherent (score ≈ 1.0)
            # NIS >> threshold → anomalous (score → 0.0)
            coherence = 1.0 - min(latest_nis / self.chi2_threshold, 1.0)
            return max(0.0, coherence)

        # ── Fallback: MCV-based scoring (for young tracks without NIS) ──
        mcv = self.mcv_engine.compute(trajectory_state, track_confidence)
        normalized = min(abs(mcv) / self.max_expected, 1.0)
        coherence = 1.0 - normalized

        return max(0.0, coherence)

    def evaluate_nis(self, nis_value):
        """
        Pure NIS-based scoring (used by the evaluation pipeline).

        Args:
            nis_value: Combined NIS value (min of CV and CA)

        Returns:
            float: Coherence score ∈ [0.0, 1.0]
        """
        return max(0.0, 1.0 - min(nis_value / self.chi2_threshold, 1.0))
