import numpy as np
from .states import TrackState
from .kalman import create_kalman_filter, create_ca_kalman_filter


class Track:
    """
    Represents a single object track across time.
    Handles identity, lifecycle, and dual Kalman-based motion estimation.

    Physics channels:
        - CV (constant-velocity) Kalman filter  → kf_cv
        - CA (constant-acceleration) Kalman filter → kf_ca
        - NIS computed for both; combined_nis = min(nis_cv, nis_ca)
    """

    def __init__(self, track_id, init_position, frame_id, dt=1.0):
        self.track_id = track_id
        self.state = TrackState.NEW

        # Last known appearance proxy
        self.last_bbox_area = None
        self.last_bbox_ratio = None

        # Dual Kalman Filters
        self.kf_cv = create_kalman_filter(init_position[0], init_position[1], dt=dt)
        self.kf_ca = create_ca_kalman_filter(init_position[0], init_position[1], dt=dt)

        # Legacy alias for backward compatibility
        self.kf = self.kf_cv
        self.covariance = self.kf_cv.P

        # State variables
        self.x = init_position[0]
        self.y = init_position[1]
        self.vx = 0.0
        self.vy = 0.0

        # Lifecycle stats
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.reattach_count = 0  # times this track was re-attached after dormancy

        self.first_seen_frame = frame_id
        self.last_seen_frame = frame_id

        self.confidence = 0.1

        # Tampering event latch (frame-based)
        self.tamper_until_frame = -1
        
        self.color_hist = None
        
        self.pose_scale = None
        
        # Identity Structure History
        self.structure_history = []
        self.max_structure_history = 20

        # ── Physics residual histories (unbounded for full-video analysis) ──
        # NIS (Normalized Innovation Squared) per frame
        self.nis_cv_history = []       # NIS from constant-velocity filter
        self.nis_ca_history = []       # NIS from constant-acceleration filter
        self.nis_combined_history = []  # min(nis_cv, nis_ca) — the combined NIS
        # Raw innovation vectors
        self.innov_history = []        # innovation vectors (from CV filter)
        # Last innovation for immediate access
        self.last_innovation = None
        self.last_S = None

        # ── Pose keypoint history (for gait analysis) ──
        self.keypoint_history = []     # list of (17,2) arrays or None per frame

    # -----------------------------
    # Kalman motion methods
    # -----------------------------

    def predict(self):
        self.kf_cv.predict()
        self.kf_ca.predict()
        self._sync_from_kalman()

    def update(self, meas_x, meas_y):
        measurement = np.array([[meas_x], [meas_y]])

        # Update both filters
        self.kf_cv.update(measurement)
        self.kf_ca.update(measurement)

        # ── Compute and store NIS for both filters ──
        # CV filter NIS
        y_cv = self.kf_cv.y.copy()
        S_cv = self.kf_cv.S.copy()
        try:
            nis_cv = float(np.asarray(y_cv.T @ np.linalg.inv(S_cv) @ y_cv).squeeze())
        except np.linalg.LinAlgError:
            nis_cv = 0.0

        # CA filter NIS
        y_ca = self.kf_ca.y.copy()
        S_ca = self.kf_ca.S.copy()
        try:
            nis_ca = float(np.asarray(y_ca.T @ np.linalg.inv(S_ca) @ y_ca).squeeze())
        except np.linalg.LinAlgError:
            nis_ca = 0.0

        # Combined NIS — min of both (high only when NO model explains motion)
        # BUG FIX: The CA filter is TOO flexible and absorbs frame deletions/splices
        # as extreme "acceleration", which makes nis_ca artificially low during tampers.
        # By taking min(nis_cv, nis_ca), the CA filter masks the CV filter's valid tamper alarm!
        nis_combined = nis_cv

        from coherex.config import CONFIG
        if self.age > CONFIG.physics.nis_warmup_frames:
            self.nis_cv_history.append(nis_cv)
            self.nis_ca_history.append(nis_ca)
            self.nis_combined_history.append(nis_combined)
            
            # Store innovation from CV filter (primary)
            self.last_innovation = y_cv.copy()
            self.last_S = S_cv.copy()
            self.innov_history.append(y_cv.flatten().tolist())

        self._sync_from_kalman()

    def _sync_from_kalman(self):
        # Sync position/velocity from CV filter (primary for tracking)
        self.x, self.y, self.vx, self.vy = self.kf_cv.x.flatten()
        self.covariance = self.kf_cv.P

    # -----------------------------
    # Lifecycle methods
    # -----------------------------

    def mark_hit(self, frame_id):
        self.hits += 1
        self.misses = 0
        self.last_seen_frame = frame_id
        self.age += 1

        self.confidence = min(1.0, self.confidence + 0.1)

        if self.state == TrackState.NEW and self.hits >= 3:
            self.state = TrackState.ACTIVE
        elif self.state == TrackState.DORMANT:
            self.state = TrackState.ACTIVE

    def mark_miss(self):
        self.misses += 1
        self.age += 1

        if self.state == TrackState.ACTIVE:
            self.state = TrackState.DORMANT

    def should_terminate(self, max_misses):
        return self.misses > max_misses

    def terminate(self):
        self.state = TrackState.TERMINATED

    def update_appearance(self, bbox_or_hist):
        """
        Store lightweight appearance proxy from bbox or histogram.
        """
        if isinstance(bbox_or_hist, tuple) and len(bbox_or_hist) == 4:
            x1, y1, x2, y2 = bbox_or_hist
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                self.last_bbox_area = w * h
                self.last_bbox_ratio = w / h
        else:
            # Assume it's a color histogram
            self.color_hist = bbox_or_hist

    def update_pose_scale(self, scale):
        if scale is not None:
            self.pose_scale = scale

    def update_keypoints(self, keypoints):
        """Store keypoints for gait analysis. keypoints: (17,2) array or None."""
        self.keypoint_history.append(keypoints)

    def update_structure(self, structure):
        """
        Update body structure history.
        structure: BodyStructure object
        """
        if structure is None:
            return
            
        self.structure_history.append(structure)
        if len(self.structure_history) > self.max_structure_history:
            self.structure_history.pop(0)

    def latch_tampering(self, current_frame, window_frames):
        """
        Escalate identity break to a tampering event
        for a fixed temporal window.
        """
        self.tamper_until_frame = max(
            self.tamper_until_frame,
            current_frame + window_frames
        )