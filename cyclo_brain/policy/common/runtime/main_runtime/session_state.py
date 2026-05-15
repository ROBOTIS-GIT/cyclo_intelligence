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

    def mark_loaded(self, robot_type: str, task_instruction: str, action_keys: list[str]) -> None:
        self.loaded = True
        self.running = False
        self.paused = False
        self.robot_type = robot_type
        self.task_instruction = task_instruction
        self.action_keys = list(action_keys)

    def mark_running(self) -> None:
        if not self.loaded:
            raise RuntimeError("LOAD first")
        self.running = True
        self.paused = False

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

    def mark_stopped(self) -> None:
        self.running = False
        self.paused = False

    def mark_unloaded(self) -> None:
        self.loaded = False
        self.running = False
        self.paused = False
        self.robot_type = ""
        self.task_instruction = ""
        self.action_keys = []
