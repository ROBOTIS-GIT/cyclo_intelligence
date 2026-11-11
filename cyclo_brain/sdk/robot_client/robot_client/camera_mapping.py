#!/usr/bin/env python3
"""Backend-agnostic camera feature to robot source mapping helpers."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional


IMAGE_FEATURE_PREFIX = "observation.images."

_CAMERA_SEMANTIC_RE = re.compile(
    r"^cam_(?P<a>left|right|head|wrist)_(?P<b>left|right|head|wrist)$"
)


def resolve_camera_feature_sources(
    model_camera_keys: Iterable[str],
    robot_camera_names: Iterable[str],
) -> Dict[str, str]:
    """Map model camera keys to RobotClient camera names.

    Checkpoints may use full feature keys such as
    ``observation.images.rgb.cam_left_head``, while robot configs use shorter
    source names. The checkpoint key remains unchanged; this resolver only
    selects the robot source that feeds it. Generic side/part ordering variants
    are retained for legacy checkpoint metadata.
    """
    model_keys = _unique(model_camera_keys)
    camera_names = _unique(robot_camera_names)
    if not model_keys:
        return {}

    resolved: Dict[str, str] = {}
    source_to_model_key: Dict[str, str] = {}
    missing = []

    for model_key in model_keys:
        scored = []
        for camera_name in camera_names:
            score = _camera_match_score(model_key, camera_name)
            if score is not None:
                scored.append((score, camera_name))

        if not scored:
            missing.append(model_key)
            continue

        best_score = max(score for score, _ in scored)
        best_sources = sorted(
            camera_name for score, camera_name in scored if score == best_score
        )
        if len(best_sources) > 1:
            raise RuntimeError(
                "Ambiguous camera mapping for model key "
                f"{model_key}: sources={best_sources}"
            )

        source = best_sources[0]
        if source in source_to_model_key:
            raise RuntimeError(
                "Robot camera source "
                f"{source} matched multiple model keys: "
                f"{source_to_model_key[source]}, {model_key}"
            )

        resolved[model_key] = source
        source_to_model_key[source] = model_key

    if missing:
        raise RuntimeError(
            "Missing camera mappings for model camera keys: "
            f"{sorted(missing)}; robot has {camera_names}; matched {resolved}"
        )

    return resolved


def resolve_camera_mappings(
    robot_camera_names: Iterable[str],
    policy_camera_keys: Iterable[str],
) -> Dict[str, str]:
    """Compatibility adapter returning ``robot_source -> policy_key``."""
    policy_keys = _unique(policy_camera_keys)
    camera_names = _unique(robot_camera_names)
    if not policy_keys:
        return {camera_name: camera_name for camera_name in camera_names}
    feature_sources = resolve_camera_feature_sources(policy_keys, camera_names)
    return {source: feature for feature, source in feature_sources.items()}


def camera_key_variants(key: str) -> set[str]:
    """Return comparable forms of a model feature key or robot camera name."""
    body = _strip_image_feature_prefix(key)
    suffix = body.split(".")[-1]
    aliases = {key, body, suffix}

    semantic_names = {suffix}
    match = _CAMERA_SEMANTIC_RE.match(suffix)
    if match:
        first = match.group("a")
        second = match.group("b")
        side = first if first in {"left", "right"} else second
        part = first if first in {"head", "wrist"} else second
        if side in {"left", "right"} and part in {"head", "wrist"}:
            semantic_names.add(f"cam_{side}_{part}")
            semantic_names.add(f"cam_{part}_{side}")
            semantic_names.add(f"{side}_{part}")
            semantic_names.add(f"{part}_{side}")

    for name in semantic_names:
        aliases.add(name)
        aliases.add(f"rgb.{name}")
        aliases.add(f"{IMAGE_FEATURE_PREFIX}{name}")
        aliases.add(f"{IMAGE_FEATURE_PREFIX}rgb.{name}")

    return aliases


def _camera_match_score(model_key: str, camera_name: str) -> Optional[int]:
    if model_key == camera_name:
        return 100

    model_body = _strip_image_feature_prefix(model_key)
    camera_body = _strip_image_feature_prefix(camera_name)
    model_suffix = model_body.split(".")[-1]
    camera_suffix = camera_body.split(".")[-1]

    if model_body == camera_name:
        return 95
    if model_body == camera_body:
        return 94
    if model_suffix == camera_name:
        return 90
    if model_suffix == camera_suffix:
        return 89
    if camera_key_variants(model_key) & camera_key_variants(camera_name):
        return 70
    return None


def _strip_image_feature_prefix(key: str) -> str:
    return key[len(IMAGE_FEATURE_PREFIX):] if key.startswith(IMAGE_FEATURE_PREFIX) else key


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
