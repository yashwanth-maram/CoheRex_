from .track import Track
from .states import TrackState
from .association import associate_tracks_to_detections
from .identity import identity_consistent, body_structure_consistent
from .appearance import extract_color_histogram, histogram_similarity
from .pose import compute_pose_scale, compute_body_structure
from coherex.config import CONFIG


class TrackManager:
    """
    Manages lifecycle of multiple tracks:
    NEW / ACTIVE ↔ DORMANT → TERMINATED
    """

    def __init__(
        self,
        max_misses=None,
        max_dormant_frames=None,
        reattach_distance=None,
        tamper_latch_frames=None,
        appearance_similarity_threshold=None,
        config=None,
        dt=1.0,
    ):
        self.tracks = []
        self.next_track_id = 0
        self.dt = dt

        cfg = config or CONFIG
        tc = cfg.tracking
        self.max_misses = (
            max_misses if max_misses is not None
            else tc.max_misses
        )
        self.max_dormant_frames = (
            max_dormant_frames if max_dormant_frames is not None
            else tc.max_dormant_frames
        )
        self.reattach_distance = (
            reattach_distance if reattach_distance is not None
            else tc.max_association_distance
        )
        self.tamper_latch_frames = (
            tamper_latch_frames if tamper_latch_frames is not None
            else tc.tamper_latch_frames
        )
        self.appearance_similarity_threshold = (
            appearance_similarity_threshold
            if appearance_similarity_threshold is not None
            else tc.appearance_similarity_threshold
        )

        # Stores bbox per detection index for current frame
        self.current_bboxes = {}

    # --------------------------------------------------
    # Main update call (called once per frame)
    # --------------------------------------------------

    def update(self, detections, bboxes, frame_id):
        """
        Args:
            detections (list[tuple]): (cx, cy)
            bboxes (list[tuple]): (x1, y1, x2, y2)
            frame_id (int)
        Returns:
            list[Track]: NEW + ACTIVE tracks
        """

        # Map detection index → bbox
        self.current_bboxes = {i: b for i, b in enumerate(bboxes)}

        # 1. Predict all tracks
        for track in self.tracks:
            if track.state != TrackState.TERMINATED:
                track.predict()

        # Split tracks
        active_tracks = [
            t for t in self.tracks
            if t.state in (TrackState.NEW, TrackState.ACTIVE)
        ]
        dormant_tracks = [
            t for t in self.tracks
            if t.state == TrackState.DORMANT
        ]

        # 2. Associate ACTIVE / NEW tracks
        matches, unmatched_active, unmatched_detections = associate_tracks_to_detections(
            active_tracks, detections, max_distance=self.reattach_distance
        )

        # Update matched ACTIVE/NEW tracks
        for t_idx, d_idx in matches:
            track = active_tracks[t_idx]
            cx, cy = detections[d_idx]
            bbox = self.current_bboxes[d_idx]

            track.update(cx, cy)
            track.mark_hit(frame_id)
            
            hist = extract_color_histogram(self.current_frame, bbox)
            track.update_appearance(hist)

        # Mark unmatched ACTIVE tracks as missed
        for idx in unmatched_active:
            active_tracks[idx].mark_miss()

        # 3. Try re-attaching to DORMANT tracks (WITH IDENTITY CHECK)
        if dormant_tracks and unmatched_detections:
            dormant_detections = [detections[i] for i in unmatched_detections]

            d_matches, _, _ = associate_tracks_to_detections(
                dormant_tracks,
                dormant_detections,
                max_distance=float('inf'), # BUG FIX: Allow infinite distance for forensic re-attachment
            )

            reattached = []

            for t_idx, d_idx in d_matches:
                track = dormant_tracks[t_idx]
                real_det_idx = unmatched_detections[d_idx]
                bbox = self.current_bboxes[real_det_idx]

                # --- Appearance extraction ---
                hist = extract_color_histogram(self.current_frame, bbox)

                # --- Appearance consistency check ---
                appearance_ok = True
                if track.color_hist is not None:
                    sim = histogram_similarity(track.color_hist, hist)
                    appearance_ok = sim > self.appearance_similarity_threshold

                # --- Final identity decision ---
                if not identity_consistent(track, bbox) or not appearance_ok:
                    track.identity_break = True
                    track.latch_tampering(
                        current_frame=frame_id,
                        window_frames=self.tamper_latch_frames
                    )
                    continue

                cx, cy = detections[real_det_idx]
                track.update(cx, cy)
                track.mark_hit(frame_id)
                track.update_appearance(hist)
                track.reattach_count += 1

                reattached.append(real_det_idx)

            # Remove reattached detections safely
            for idx in sorted(reattached, reverse=True):
                unmatched_detections.remove(idx)

        # 4. Create NEW tracks for remaining detections
        for det_idx in unmatched_detections:
            cx, cy = detections[det_idx]
            bbox = self.current_bboxes[det_idx]

            new_track = Track(self.next_track_id, (cx, cy), frame_id, dt=self.dt)
            # Initialize appearance
            hist = extract_color_histogram(self.current_frame, bbox)
            new_track.update_appearance(hist)

            self.tracks.append(new_track)
            self.next_track_id += 1

        # 5. Terminate stale dormant tracks
        for track in self.tracks:
            if track.state == TrackState.DORMANT:
                gap = frame_id - track.last_seen_frame
                if gap > self.max_dormant_frames or track.should_terminate(self.max_misses):
                    track.terminate()

        # Return visible tracks
        return [
            t for t in self.tracks
            if t.state in (TrackState.NEW, TrackState.ACTIVE)
        ]
