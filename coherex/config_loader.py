# coherex/config_loader.py
"""
YAML Configuration Override Loader

Loads a YAML file and merges specified overrides onto the frozen
default SystemConfig.  Only fields present in the YAML are replaced;
everything else keeps its default value.

Uses dataclasses.replace() to preserve frozen immutability.

Usage:
    from coherex.config_loader import load_config_from_yaml

    config = load_config_from_yaml("forensic.yaml")
"""

import yaml
from dataclasses import replace, fields
from coherex.config import (
    CONFIG,
    SystemConfig,
    DetectionConfig,
    TrackingConfig,
    MotionConfig,
    PhysicsConfig,
    PoseRhythmConfig,
    ChangepointConfig,
    IntegrityConfig,
    ContinuityConfig,
    CrowdConfig,
)

# Fields that must be stored as tuples (frozen dataclass requirement)
_TUPLE_FIELDS = {
    "mcv_weights",
    "fusion_weights",
    "verdict_thresholds",
    "identity_area_range",
    "multi_scale_windows",
}


def _coerce_tuples(data: dict) -> dict:
    """Convert YAML lists to tuples for fields that require it."""
    return {
        k: tuple(v) if k in _TUPLE_FIELDS and isinstance(v, list) else v
        for k, v in data.items()
    }


def load_config_from_yaml(path: str) -> SystemConfig:
    """
    Load a YAML config file and return a new SystemConfig with
    overrides applied on top of the defaults.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        SystemConfig: A new frozen config object with overrides applied.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        return CONFIG

    config = CONFIG

    # ── Detection ─────────────────────────────────────────────────────
    if "detection" in data:
        config = replace(
            config,
            detection=replace(
                config.detection,
                **_coerce_tuples(data["detection"])
            ),
        )

    # ── Tracking ──────────────────────────────────────────────────────
    if "tracking" in data:
        config = replace(
            config,
            tracking=replace(
                config.tracking,
                **_coerce_tuples(data["tracking"])
            ),
        )

    # ── Motion ────────────────────────────────────────────────────────
    if "motion" in data:
        config = replace(
            config,
            motion=replace(
                config.motion,
                **_coerce_tuples(data["motion"])
            ),
        )

    # ── Physics ───────────────────────────────────────────────────────
    if "physics" in data:
        config = replace(
            config,
            physics=replace(
                config.physics,
                **_coerce_tuples(data["physics"])
            ),
        )

    # ── Pose Rhythm ───────────────────────────────────────────────────
    if "pose_rhythm" in data:
        config = replace(
            config,
            pose_rhythm=replace(
                config.pose_rhythm,
                **_coerce_tuples(data["pose_rhythm"])
            ),
        )

    # ── Change-point ──────────────────────────────────────────────────
    if "changepoint" in data:
        config = replace(
            config,
            changepoint=replace(
                config.changepoint,
                **_coerce_tuples(data["changepoint"])
            ),
        )

    # ── Integrity (with nested continuity / crowd) ────────────────────
    if "integrity" in data:
        integrity_data = dict(data["integrity"])

        # Extract nested sub-sections before passing to replace()
        continuity_overrides = integrity_data.pop("continuity", None)
        crowd_overrides = integrity_data.pop("crowd", None)

        # Apply top-level integrity overrides
        new_integrity = replace(
            config.integrity,
            **_coerce_tuples(integrity_data)
        )

        # Apply nested continuity overrides
        if continuity_overrides:
            new_integrity = replace(
                new_integrity,
                continuity=replace(
                    new_integrity.continuity,
                    **_coerce_tuples(continuity_overrides)
                ),
            )

        # Apply nested crowd overrides
        if crowd_overrides:
            new_integrity = replace(
                new_integrity,
                crowd=replace(
                    new_integrity.crowd,
                    **_coerce_tuples(crowd_overrides)
                ),
            )

        config = replace(config, integrity=new_integrity)

    return config
