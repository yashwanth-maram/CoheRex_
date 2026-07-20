# coherex/trajectory/state.py

import numpy as np
from collections import deque


class TrajectoryState:
    """
    Stores per-track temporal motion state.

    Contains two types of history:
    1. Capped deques (maxlen) — for MCV z-score computation (short-term)
    2. Unbounded lists — for NIS/innovation analysis (full-video)
    """

    def __init__(self, track_id, max_history=10):
        self.track_id = track_id

        # ── Short-term histories (capped, for MCV) ──
        self.positions = deque(maxlen=max_history)
        self.velocities = deque(maxlen=max_history)
        self.speeds = deque(maxlen=max_history)
        self.accelerations = deque(maxlen=max_history)
        self.angles = deque(maxlen=max_history)
        self.total_age = 0

        # ── Full-video histories (unbounded, for NIS/whiteness/CUSUM) ──
        self.nis_history = []           # Combined NIS values (min of CV, CA)
        self.nis_cv_history = []        # NIS from CV filter only
        self.nis_ca_history = []        # NIS from CA filter only
        self.innov_history = []         # Innovation vectors (2D) from CV filter

    def update(self, x, y, vx, vy):
        """
        Update trajectory state with new motion values.
        """
        self.total_age += 1

        # Position
        self.positions.append(np.array([x, y]))

        # Velocity
        vel = np.array([vx, vy])
        self.velocities.append(vel)

        # Speed
        speed = np.linalg.norm(vel)
        self.speeds.append(speed)

        # Acceleration
        if len(self.velocities) > 1:
            acc = self.velocities[-1] - self.velocities[-2]
            self.accelerations.append(np.linalg.norm(acc))
        else:
            self.accelerations.append(0.0)

        # Direction change (angle)
        if len(self.velocities) > 1:
            v1 = self.velocities[-2]
            v2 = self.velocities[-1]

            norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_prod > 0:
                cos_theta = np.clip(np.dot(v1, v2) / norm_prod, -1.0, 1.0)
                angle = np.arccos(cos_theta)
            else:
                angle = 0.0

            self.angles.append(angle)
        else:
            self.angles.append(0.0)

    def update_nis(self, nis_combined, nis_cv=None, nis_ca=None, innovation=None):
        """
        Update physics residual histories from Track's Kalman output.

        Args:
            nis_combined: min(nis_cv, nis_ca) — the combined NIS
            nis_cv: NIS from constant-velocity filter
            nis_ca: NIS from constant-acceleration filter
            innovation: 2D innovation vector (list or array)
        """
        self.nis_history.append(nis_combined)
        if nis_cv is not None:
            self.nis_cv_history.append(nis_cv)
        if nis_ca is not None:
            self.nis_ca_history.append(nis_ca)
        if innovation is not None:
            self.innov_history.append(innovation)
