# coherex/trajectory/store.py

from .state import TrajectoryState
from coherex.config import CONFIG


class TrajectoryStore:
    """
    Maintains TrajectoryState objects for active tracks.

    In CoheRex 2.0, also syncs NIS/innovation data from Track objects
    into TrajectoryState for downstream coherence analysis.
    """

    def __init__(self, max_history=None, config=None):
        self.states = {}
        cfg = config or CONFIG
        self.max_history = (
            max_history if max_history is not None
            else cfg.tracking.max_trajectory_history
        )

    def update_from_tracks(self, tracks):
        """
        Update trajectory states from current active tracks.
        Also syncs NIS and innovation data from Track's dual Kalman filters.
        """
        for track in tracks:
            if track.track_id not in self.states:
                self.states[track.track_id] = TrajectoryState(
                    track.track_id, self.max_history
                )

            state = self.states[track.track_id]
            state.update(track.x, track.y, track.vx, track.vy)

            # Sync NIS data from track (if available and new)
            if (hasattr(track, 'nis_combined_history') and
                    len(track.nis_combined_history) > len(state.nis_history)):
                # Get the latest NIS values not yet synced
                n_existing = len(state.nis_history)
                for i in range(n_existing, len(track.nis_combined_history)):
                    nis_cv = (track.nis_cv_history[i]
                              if i < len(track.nis_cv_history) else None)
                    nis_ca = (track.nis_ca_history[i]
                              if i < len(track.nis_ca_history) else None)
                    innov = (track.innov_history[i]
                             if i < len(track.innov_history) else None)
                    state.update_nis(
                        nis_combined=track.nis_combined_history[i],
                        nis_cv=nis_cv,
                        nis_ca=nis_ca,
                        innovation=innov,
                    )

    def get_state(self, track_id):
        return self.states.get(track_id, None)
