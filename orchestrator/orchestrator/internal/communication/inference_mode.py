#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Inference TaskInfo helpers shared by UI and BT command dispatch."""

from __future__ import annotations

import math


SIMULATION_MODE = "simulation"
ROBOT_MODE = "robot"
DEFAULT_CONTROL_HZ = 100
DEFAULT_INFERENCE_HZ = 15
DEFAULT_CHUNK_ALIGN_WINDOW_S = 0.3


def _positive_number(value, default, cast):
    try:
        parsed = cast(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed) or parsed <= 0:
        return default
    return parsed


def inference_timing_from_task_info(task_info) -> tuple[int, int, float]:
    """Return normalized LOAD-time action processing rates from TaskInfo."""
    control_hz = _positive_number(
        getattr(task_info, "control_hz", 0), DEFAULT_CONTROL_HZ, int
    )
    inference_hz = _positive_number(
        getattr(task_info, "inference_hz", 0), DEFAULT_INFERENCE_HZ, int
    )
    chunk_align_window_s = _positive_number(
        getattr(task_info, "chunk_align_window_s", 0.0),
        DEFAULT_CHUNK_ALIGN_WINDOW_S,
        float,
    )
    return control_hz, inference_hz, chunk_align_window_s


def inference_runtime_signature(
    policy_path: str,
    acceleration_mode: str,
    acceleration_engine_path: str,
    action_request_mode: str,
    control_hz: int,
    inference_hz: int,
    chunk_align_window_s: float,
    initial_pose_sync: bool = False,
    initial_pose_sync_duration_s: float = 5.0,
) -> tuple:
    """Return the LOAD-only values that determine policy runtime reuse."""
    return (
        policy_path,
        acceleration_mode,
        acceleration_engine_path,
        action_request_mode,
        control_hz,
        inference_hz,
        chunk_align_window_s,
        initial_pose_sync,
        initial_pose_sync_duration_s if initial_pose_sync else 0.0,
    )


def normalize_inference_mode(value) -> str:
    mode = str(value or "").strip().lower()
    if mode in {ROBOT_MODE, "robot_mode", "publish", "publish_to_robot"}:
        return ROBOT_MODE
    return SIMULATION_MODE


def inference_mode_from_task_info(task_info) -> str:
    """Return robot/simulation from TaskInfo fields and tags."""
    mode = getattr(task_info, "inference_mode", "")
    if mode:
        return normalize_inference_mode(mode)

    tags = getattr(task_info, "tags", []) or []
    for tag in tags:
        normalized = str(tag or "").strip().lower()
        if normalized in {"inference_mode:robot", "publish_to_robot:true"}:
            return ROBOT_MODE
        if normalized in {"inference_mode:simulation", "publish_to_robot:false"}:
            return SIMULATION_MODE

    return ROBOT_MODE if bool(getattr(task_info, "publish_to_robot", False)) else SIMULATION_MODE


def publish_to_robot_from_task_info(task_info) -> bool:
    """Return true only when a command explicitly asks for robot publish."""
    return inference_mode_from_task_info(task_info) == ROBOT_MODE
