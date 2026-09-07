#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Small inference-session state model for the Main process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SessionState:
    loaded: bool = False
    running: bool = False
    paused: bool = False
    robot_type: str = ""
    task_instruction: str = ""
    action_keys: List[str] = field(default_factory=list)
    model_path: str = ""
    policy_id: str = ""
    policy_parameters_json: str = "{}"
    publish_to_robot: bool = False
    runtime_id: str = ""
    action_request_mode: str = "async"
    acceleration_mode: str = "pytorch"
    acceleration_engine_path: str = ""
    control_hz: int = 0
    inference_hz: int = 0
    chunk_align_window_s: float = 0.0
    initial_pose_sync: bool = False
    initial_pose_sync_duration_s: float = 5.0
    error: str = ""

    def mark_loaded(
        self,
        robot_type: str,
        task_instruction: str,
        action_keys: list[str],
        *,
        model_path: str = "",
        policy_id: str = "",
        policy_parameters_json: str = "{}",
        publish_to_robot: bool = False,
        runtime_id: str = "",
        action_request_mode: str = "async",
        acceleration_mode: str = "pytorch",
        acceleration_engine_path: str = "",
        control_hz: int = 0,
        inference_hz: int = 0,
        chunk_align_window_s: float = 0.0,
        initial_pose_sync: bool = False,
        initial_pose_sync_duration_s: float = 5.0,
    ) -> None:
        self.loaded = True
        self.running = False
        self.paused = False
        self.robot_type = robot_type
        self.task_instruction = task_instruction
        self.action_keys = list(action_keys)
        self.model_path = str(model_path or "")
        self.policy_id = str(policy_id or "")
        self.policy_parameters_json = str(policy_parameters_json or "{}")
        self.publish_to_robot = bool(publish_to_robot)
        self.runtime_id = str(runtime_id or "")
        self.action_request_mode = str(action_request_mode or "async")
        self.acceleration_mode = str(acceleration_mode or "pytorch")
        self.acceleration_engine_path = str(acceleration_engine_path or "")
        self.control_hz = int(control_hz)
        self.inference_hz = int(inference_hz)
        self.chunk_align_window_s = float(chunk_align_window_s)
        self.initial_pose_sync = bool(initial_pose_sync)
        self.initial_pose_sync_duration_s = float(initial_pose_sync_duration_s)
        self.error = ""

    def mark_running(self) -> None:
        if not self.loaded:
            raise RuntimeError("LOAD first")
        self.running = True
        self.paused = False
        self.error = ""

    def mark_paused(self) -> None:
        if not self.running:
            raise RuntimeError("not running")
        self.paused = True

    def mark_resumed(self, task_instruction: str = "") -> None:
        if not self.running:
            raise RuntimeError("not running")
        if task_instruction:
            self.task_instruction = task_instruction
        self.paused = False

    def set_publish_to_robot(self, publish_to_robot: bool) -> None:
        self.publish_to_robot = bool(publish_to_robot)

    def mark_stopped(self) -> None:
        self.running = False
        self.paused = False
        self.error = ""

    def mark_error(self, message: str) -> None:
        self.running = False
        self.paused = False
        self.error = str(message or "policy runtime error")

    def mark_unloaded(self) -> None:
        self.loaded = False
        self.running = False
        self.paused = False
        self.robot_type = ""
        self.task_instruction = ""
        self.action_keys = []
        self.model_path = ""
        self.policy_id = ""
        self.policy_parameters_json = "{}"
        self.publish_to_robot = False
        self.runtime_id = ""
        self.action_request_mode = "async"
        self.acceleration_mode = "pytorch"
        self.acceleration_engine_path = ""
        self.control_hz = 0
        self.inference_hz = 0
        self.chunk_align_window_s = 0.0
        self.initial_pose_sync = False
        self.initial_pose_sync_duration_s = 5.0
        self.error = ""
