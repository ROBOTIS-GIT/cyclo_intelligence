#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Robot-facing control loop owned by the Main process."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


_parents = Path(__file__).resolve().parents
_default_acp = str(_parents[4] / "sdk" / "action_chunk_processing") if len(_parents) > 4 else ""
_ACTION_CHUNK_PATH = os.environ.get("ACTION_CHUNK_PROCESSING_SDK_PATH", _default_acp)
if os.path.exists(_ACTION_CHUNK_PATH) and _ACTION_CHUNK_PATH not in sys.path:
    sys.path.insert(0, _ACTION_CHUNK_PATH)

_default_rc = str(_parents[4] / "sdk" / "robot_client") if len(_parents) > 4 else ""
_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", _default_rc)
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)

from action_chunk_processing import ActionChunkProcessor  # noqa: E402
from robot_client import RobotClient  # noqa: E402


try:  # pragma: no cover - SDK exists only in runtime container here.
    from zenoh_ros2_sdk import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("main_runtime.control_loop")


class ControlLoop:
    """Ticks RobotClient command publishing and refills action buffers."""

    def __init__(
        self,
        requester,
        inference_hz: float = 15.0,
        control_hz: float = 100.0,
        chunk_align_window_s: float = 0.3,
        target_chunk_size: Optional[int] = 100,
        postprocess_actions: bool = True,
        alignment_mode: str = "l2",
        refill_margin_s: float = 0.2,
    ) -> None:
        self._requester = requester
        self._inference_hz = float(inference_hz)
        self._control_hz = float(control_hz)
        self._chunk_align_window_s = float(chunk_align_window_s)
        self._target_chunk_size = target_chunk_size
        self._postprocess_actions = bool(postprocess_actions)
        self._alignment_mode = alignment_mode
        self._refill_margin_s = float(refill_margin_s)

        self._lock = threading.RLock()
        self._robot: Optional[RobotClient] = None
        self._processor: Optional[ActionChunkProcessor] = None
        self._task_instruction = ""
        self._action_keys: list[str] = []
        self._running = False
        self._generation = 0
        self._shutdown = threading.Event()
        self._request_thread: Optional[threading.Thread] = None
        self._thread: Optional[threading.Thread] = None

    def configure(
        self,
        robot_type: str,
        task_instruction: str = "",
        action_keys: Optional[list[str]] = None,
    ) -> None:
        with self._lock:
            self.deconfigure()
            self._robot = RobotClient(robot_type, enable_command_publishers=True)
            self._processor = ActionChunkProcessor(
                inference_hz=self._inference_hz,
                control_hz=self._control_hz,
                chunk_align_window_s=self._chunk_align_window_s,
                postprocess=self._postprocess_actions,
                target_chunk_size=self._target_chunk_size,
                alignment_mode=self._alignment_mode,
            )
            self._task_instruction = task_instruction or ""
            self._action_keys = list(action_keys or self._robot.action_keys)
            self._generation += 1
            logger.info("configured RobotClient command path for %s", robot_type)

    def deconfigure(self) -> None:
        with self._lock:
            self._running = False
            self._task_instruction = ""
            self._action_keys = []
            self._processor = None
            self._generation += 1
            if self._robot is not None:
                self._robot.close()
                self._robot = None

    def start(self) -> None:
        with self._lock:
            self._running = True

    def pause(self) -> None:
        with self._lock:
            self._running = False

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._generation += 1

    def set_task_instruction(self, task_instruction: str) -> None:
        with self._lock:
            self._task_instruction = task_instruction or ""

    def run_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self) -> None:
        next_t = time.monotonic()
        while not self._shutdown.is_set():
            period = self._tick_period()
            self.tick()
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.deconfigure()

    def tick(self) -> None:
        with self._lock:
            if not self._running or self._robot is None or self._processor is None:
                return
            robot = self._robot
            processor = self._processor
            task_instruction = self._task_instruction
            action_keys = list(self._action_keys)
            generation = self._generation

            action = processor.pop_action()
            if action is not None:
                robot.publish_action(action, action_keys)

            refill_threshold = max(1, int(self._refill_margin_s * processor.output_hz))
            should_request = (
                processor.buffer_size < refill_threshold
                and (self._request_thread is None or not self._request_thread.is_alive())
            )

        if should_request:
            self._request_thread = threading.Thread(
                target=self._request_and_buffer,
                args=(task_instruction, generation),
                daemon=True,
            )
            self._request_thread.start()

    def _request_and_buffer(self, task_instruction: str, generation: int) -> None:
        response = self._requester.get_action(task_instruction)
        if not response.success:
            logger.warning("get_action failed: %s", response.message)
            return
        if response.chunk_size <= 0 or response.action_dim <= 0:
            logger.warning("get_action returned empty action list")
            return
        data = np.asarray(response.action_list, dtype=np.float64)
        if data.size != response.chunk_size * response.action_dim:
            logger.warning(
                "action list size mismatch: %d != %d * %d",
                data.size,
                response.chunk_size,
                response.action_dim,
            )
            return
        chunk = data.reshape(response.chunk_size, response.action_dim)
        with self._lock:
            if (
                generation == self._generation
                and self._running
                and self._processor is not None
            ):
                self._processor.push_actions(chunk)

    def _tick_period(self) -> float:
        with self._lock:
            if self._processor is None:
                hz = self._control_hz
            else:
                hz = self._processor.output_hz
        return 1.0 / max(1.0, hz)
