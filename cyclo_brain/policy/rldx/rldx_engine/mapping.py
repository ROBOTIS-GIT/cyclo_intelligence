#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""RLDX observation/action mapping helpers.

RLDX model modalities are model-semantic (``left_arm``, ``left_gripper``,
``base``). Cyclo robot configs publish commands in hardware-facing groups
(``arm_left``, ``arm_right``, ``mobile``). Keep that translation here so the
runtime server remains mostly lifecycle and transport code.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable

import numpy as np


LEFT_ARM_ALIASES = ("follower_left_arm", "follower_arm_left")
RIGHT_ARM_ALIASES = ("follower_right_arm", "follower_arm_right")

ACTION_KEY_TO_ROBOT_GROUP = {
    "left_arm": "arm_left",
    "left_gripper": "arm_left",
    "right_arm": "arm_right",
    "right_gripper": "arm_right",
    "base": "mobile",
    "odometry": "mobile",
}


def skipped_robot_action_keys() -> set[str]:
    raw = os.environ.get("RLDX_SKIP_ROBOT_ACTION_KEYS", "head,lift")
    return {key.strip() for key in raw.split(",") if key.strip()}



def aspect_area_resize_crop_sizes(
    height: int,
    width: int,
    *,
    max_area: int = 256**2,
    multiple: int = 32,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Match RLDX eval image geometry: preserve aspect, then center-crop.

    Returns ``((resize_h, resize_w), (crop_h, crop_w))``. The resize never
    upscales and the final crop dimensions are multiples of ``multiple``.
    """
    h = int(height)
    w = int(width)
    m = max(1, int(multiple))
    area = max(m * m, int(max_area))
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size: height={height}, width={width}")

    scale_max = min(1.0, math.sqrt(area / float(h * w)))
    short, long_ = (h, w) if h <= w else (w, h)
    short_r = max(m, int((short * scale_max) // m) * m)
    scale = short_r / float(short)
    long_r = int(long_ * scale)

    resize_h, resize_w = (short_r, long_r) if h <= w else (long_r, short_r)
    crop_h = resize_h - (resize_h % m)
    crop_w = resize_w - (resize_w % m)
    return (resize_h, resize_w), (crop_h, crop_w)


def modality_keys(config_entry: Any) -> list[str]:
    if config_entry is None:
        return []
    if isinstance(config_entry, dict):
        return list(config_entry.get("modality_keys") or [])
    return list(getattr(config_entry, "modality_keys", []) or [])


def delta_indices(config_entry: Any) -> list[int]:
    if config_entry is None:
        return []
    if isinstance(config_entry, dict):
        return [int(v) for v in (config_entry.get("delta_indices") or [])]
    return [int(v) for v in (getattr(config_entry, "delta_indices", []) or [])]


def stack_history_window(
    history: Iterable[np.ndarray],
    deltas: Iterable[int],
) -> np.ndarray:
    """Select a chronological observation window from a latest-ended history."""
    frames = [np.asarray(frame) for frame in history]
    if not frames:
        raise ValueError("history is empty")

    selected: list[np.ndarray] = []
    for delta in deltas:
        offset = int(delta)
        if offset > 0:
            raise ValueError(f"Observation delta index must be <= 0, got {offset}")
        index = len(frames) - 1 + offset
        selected.append(frames[max(0, index)])
    return np.stack(selected, axis=0)


def _first_array(mapping: dict[str, Any], keys: Iterable[str]) -> np.ndarray | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size:
            return arr
    return None


def _odom_base_vector(odom: dict | None) -> np.ndarray | None:
    if not odom:
        return None
    linear = np.asarray(odom.get("linear_velocity", []), dtype=np.float32).reshape(-1)
    angular = np.asarray(odom.get("angular_velocity", []), dtype=np.float32).reshape(-1)
    if linear.size < 2 or angular.size < 3:
        return None
    return np.array([linear[0], linear[1], angular[2]], dtype=np.float32)


def state_vector_from_robot(
    state_key: str,
    joints: dict[str, Any],
    odom: dict | None,
    *,
    allow_missing_base: bool = True,
) -> np.ndarray:
    """Return one RLDX state modality vector from RobotClient outputs."""
    key = state_key.strip()

    if key == "left_arm":
        arr = _first_array(joints, LEFT_ARM_ALIASES)
        if arr is not None and arr.size >= 7:
            return arr[:7]
    elif key == "arm_left":
        arr = _first_array(joints, LEFT_ARM_ALIASES)
        if arr is not None and arr.size >= 8:
            return arr[:8]
    elif key == "left_gripper":
        arr = _first_array(joints, ("follower_left_gripper", *LEFT_ARM_ALIASES))
        if arr is not None:
            return arr[-1:].astype(np.float32)
    elif key == "right_arm":
        arr = _first_array(joints, RIGHT_ARM_ALIASES)
        if arr is not None and arr.size >= 7:
            return arr[:7]
    elif key == "arm_right":
        arr = _first_array(joints, RIGHT_ARM_ALIASES)
        if arr is not None and arr.size >= 8:
            return arr[:8]
    elif key == "right_gripper":
        arr = _first_array(joints, ("follower_right_gripper", *RIGHT_ARM_ALIASES))
        if arr is not None:
            return arr[-1:].astype(np.float32)
    elif key in {"base", "mobile", "odometry"}:
        base = _odom_base_vector(odom)
        if base is not None:
            return base
        if allow_missing_base:
            return np.zeros(3, dtype=np.float32)
    else:
        arr = _first_array(joints, (f"follower_{key}", key))
        if arr is not None:
            return arr

    raise KeyError(f"Cannot build RLDX state modality {state_key!r}")


def build_state_observation(
    state_keys: Iterable[str],
    joints: dict[str, Any],
    odom: dict | None,
    state_t: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    t = max(1, int(state_t))
    for key in state_keys:
        vec = state_vector_from_robot(str(key), joints, odom)
        out[str(key)] = np.repeat(vec[None, None, :], t, axis=1).astype(np.float32)
    return out


def robot_action_keys_from_policy(
    policy_action_keys: Iterable[str],
    available_robot_action_keys: Iterable[str],
) -> list[str]:
    """Return robot action groups needed by the policy action modalities."""
    policy_keys = [str(key) for key in policy_action_keys]
    available = set(str(key) for key in available_robot_action_keys)
    skipped = skipped_robot_action_keys()
    out: list[str] = []
    for key in policy_keys:
        robot_key = ACTION_KEY_TO_ROBOT_GROUP.get(key, key)
        if robot_key in skipped:
            continue
        if robot_key not in available or robot_key in out:
            continue
        out.append(robot_key)
    if not out:
        raise ValueError(
            "No RLDX policy action modalities match robot action groups: "
            f"policy={policy_keys}, robot={sorted(available)}"
        )
    return out


def _action_value(actions: dict[str, Any], key: str) -> np.ndarray | None:
    for candidate in (f"action.{key}", key):
        value = actions.get(candidate)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim == 2:
            return arr
    return None


def _require_action(actions: dict[str, Any], key: str) -> np.ndarray:
    arr = _action_value(actions, key)
    if arr is None:
        raise KeyError(f"Missing RLDX action modality {key!r}")
    return arr


def robot_action_chunk_from_rldx(
    actions: dict[str, Any],
    robot_action_keys: Iterable[str],
) -> np.ndarray:
    """Flatten RLDX action dict into Cyclo robot action-key order."""
    chunks: list[np.ndarray] = []
    for key in robot_action_keys:
        key = str(key)
        exact = _action_value(actions, key)
        if exact is not None:
            chunks.append(exact)
            continue

        if key in {"arm_left", "left_arm"}:
            arm = _require_action(actions, "left_arm")
            gripper = _action_value(actions, "left_gripper")
            chunks.append(np.concatenate([arm, gripper], axis=1) if gripper is not None else arm)
        elif key in {"arm_right", "right_arm"}:
            arm = _require_action(actions, "right_arm")
            gripper = _action_value(actions, "right_gripper")
            chunks.append(np.concatenate([arm, gripper], axis=1) if gripper is not None else arm)
        elif key in {"mobile", "base", "odometry"}:
            chunks.append(_require_action(actions, "base"))
        else:
            chunks.append(_require_action(actions, key))

    if not chunks:
        raise ValueError("robot_action_keys is empty")
    horizon = chunks[0].shape[0]
    for chunk in chunks:
        if chunk.shape[0] != horizon:
            raise ValueError(
                f"RLDX action horizon mismatch: {chunk.shape[0]} != {horizon}"
            )
    return np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float64)


def policy_action_chunk_from_rldx(
    actions: dict[str, Any],
    policy_action_keys: Iterable[str],
) -> np.ndarray:
    """Flatten RLDX action dict in policy modality order for RTC prefixes."""
    chunks: list[np.ndarray] = []
    for key in policy_action_keys:
        chunks.append(_require_action(actions, str(key)))

    if not chunks:
        raise ValueError("policy_action_keys is empty")
    horizon = chunks[0].shape[0]
    for chunk in chunks:
        if chunk.shape[0] != horizon:
            raise ValueError(
                f"RLDX action horizon mismatch: {chunk.shape[0]} != {horizon}"
            )
    return np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float64)
