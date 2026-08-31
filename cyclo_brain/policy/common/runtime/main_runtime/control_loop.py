#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Robot-facing control loop owned by the Main process."""

from __future__ import annotations

import math
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

ACTION_REQUEST_MODE_ASYNC = "async"
ACTION_REQUEST_MODE_SYNC = "sync"
ACTION_REQUEST_MODES = {ACTION_REQUEST_MODE_ASYNC, ACTION_REQUEST_MODE_SYNC}


def positive_finite_or_default(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return float(default)
    return parsed


def normalize_action_request_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode == ACTION_REQUEST_MODE_SYNC:
        return ACTION_REQUEST_MODE_SYNC
    return ACTION_REQUEST_MODE_ASYNC


class ControlLoop:
    """Ticks RobotClient command publishing and refills action buffers."""

    def __init__(
        self,
        requester,
        inference_hz: float = 15.0,
        control_hz: float = 100.0,
        chunk_align_window_s: float = 0.3,
        target_chunk_size: Optional[int] = None,
        postprocess_actions: bool = True,
        alignment_mode: str = "l2",
        refill_margin_s: float = 0.2,
        latency_warmup_samples: int = 1,
        max_refill_latency_s: Optional[float] = 2.0,
        action_request_mode: str = ACTION_REQUEST_MODE_ASYNC,
    ) -> None:
        self._requester = requester
        self._default_inference_hz = positive_finite_or_default(inference_hz, 15.0)
        self._default_control_hz = positive_finite_or_default(control_hz, 100.0)
        self._default_chunk_align_window_s = positive_finite_or_default(
            chunk_align_window_s, 0.3
        )
        self._inference_hz = self._default_inference_hz
        self._control_hz = self._default_control_hz
        self._chunk_align_window_s = self._default_chunk_align_window_s
        self._target_chunk_size = target_chunk_size
        self._postprocess_actions = bool(postprocess_actions)
        self._alignment_mode = alignment_mode
        self._refill_margin_s = float(refill_margin_s)
        self._request_latency_ema_s: Optional[float] = None
        self._request_latency_alpha = 0.2
        self._latency_warmup_samples = max(0, int(latency_warmup_samples))
        self._latency_warmup_remaining = self._latency_warmup_samples
        self._max_refill_latency_s = (
            None
            if max_refill_latency_s is None or max_refill_latency_s <= 0.0
            else float(max_refill_latency_s)
        )
        self._default_action_request_mode = normalize_action_request_mode(
            action_request_mode
        )
        self._action_request_mode = self._default_action_request_mode

        self._lock = threading.RLock()
        self._robot: Optional[RobotClient] = None
        self._processor: Optional[ActionChunkProcessor] = None
        self._task_instruction = ""
        self._action_keys: list[str] = []
        self._publish_to_robot = False
        self._running = False
        self._initial_pose_sync_enabled = False
        self._initial_pose_sync_duration_s = 5.0
        self._initial_pose_sync_in_progress = False
        self._initial_pose_sync_completed = False
        self._initial_pose_sync_deadline: Optional[float] = None
        self._initial_pose_sync_hold_pending = False
        self._generation = 0
        self._shutdown = threading.Event()
        self._request_thread: Optional[threading.Thread] = None
        self._thread: Optional[threading.Thread] = None

    def configure(
        self,
        robot_type: str,
        task_instruction: str = "",
        action_keys: Optional[list[str]] = None,
        publish_to_robot: bool = False,
        action_request_mode: Optional[str] = None,
        control_hz: Optional[float] = None,
        inference_hz: Optional[float] = None,
        chunk_align_window_s: Optional[float] = None,
        initial_pose_sync: bool = False,
        initial_pose_sync_duration_s: float = 5.0,
    ) -> None:
        duration_s = float(initial_pose_sync_duration_s)
        if not math.isfinite(duration_s) or not 1.0 <= duration_s <= 60.0:
            raise ValueError(
                "initial_pose_sync_duration_s must be between 1.0 and 60.0"
            )
        with self._lock:
            self.deconfigure()
            self._control_hz = positive_finite_or_default(
                control_hz, self._default_control_hz
            )
            self._inference_hz = positive_finite_or_default(
                inference_hz, self._default_inference_hz
            )
            self._chunk_align_window_s = positive_finite_or_default(
                chunk_align_window_s, self._default_chunk_align_window_s
            )
            self._action_request_mode = normalize_action_request_mode(
                action_request_mode
                if action_request_mode is not None
                else self._default_action_request_mode
            )
            self._robot = RobotClient(
                robot_type,
                enable_command_publishers=True,
                enable_preview_publisher=True,
            )
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
            self._publish_to_robot = bool(publish_to_robot)
            self._initial_pose_sync_enabled = bool(initial_pose_sync)
            self._initial_pose_sync_duration_s = duration_s
            self._initial_pose_sync_in_progress = False
            self._initial_pose_sync_completed = False
            self._initial_pose_sync_deadline = None
            self._initial_pose_sync_hold_pending = False
            self._reset_request_latency_locked()
            self._generation += 1
            config_message = (
                "configured RobotClient command path for %s "
                "(publish_to_robot=%s action_request_mode=%s "
                "control_hz=%g inference_hz=%g chunk_align_window_s=%g "
                "initial_pose_sync=%s initial_pose_sync_duration_s=%g)"
                % (
                    robot_type,
                    self._publish_to_robot,
                    self._action_request_mode,
                    self._control_hz,
                    self._inference_hz,
                    self._chunk_align_window_s,
                    self._initial_pose_sync_enabled,
                    self._initial_pose_sync_duration_s,
                )
            )
            logger.info(config_message)
            print(f"[main-runtime] {config_message}", flush=True)

    def deconfigure(self) -> None:
        with self._lock:
            self._running = False
            self._task_instruction = ""
            self._action_keys = []
            self._publish_to_robot = False
            self._action_request_mode = self._default_action_request_mode
            self._inference_hz = self._default_inference_hz
            self._control_hz = self._default_control_hz
            self._chunk_align_window_s = self._default_chunk_align_window_s
            self._initial_pose_sync_enabled = False
            self._initial_pose_sync_duration_s = 5.0
            self._initial_pose_sync_in_progress = False
            self._initial_pose_sync_completed = False
            self._initial_pose_sync_deadline = None
            self._initial_pose_sync_hold_pending = False
            self._processor = None
            self._generation += 1
            if self._robot is not None:
                self._robot.close()
                self._robot = None
            self._reset_request_latency_locked()

    def start(self, publish_to_robot: Optional[bool] = None) -> bool:
        with self._lock:
            if self._initial_pose_sync_hold_pending:
                raise RuntimeError(
                    "initial pose sync hold is still pending - STOP again first"
                )
            if publish_to_robot is not None:
                self._set_publish_to_robot_locked(bool(publish_to_robot))
            should_sync = (
                self._initial_pose_sync_enabled
                and self._publish_to_robot
                and not self._initial_pose_sync_completed
            )
            if not should_sync:
                self._running = True
                return False
            if self._robot is None or self._processor is None:
                raise RuntimeError("LOAD first")
            robot = self._robot
            processor = self._processor
            task_instruction = self._task_instruction
            action_keys = list(self._action_keys)
            duration_s = self._initial_pose_sync_duration_s
            generation = self._generation
            self._running = False
            self._initial_pose_sync_in_progress = False
            self._initial_pose_sync_deadline = None

        started_at = time.monotonic()
        response = self._requester.get_action(task_instruction)
        latency_s = time.monotonic() - started_at
        self._record_request_latency(latency_s)
        chunk = self._decode_action_response(response)

        with self._lock:
            if (
                generation != self._generation
                or robot is not self._robot
                or processor is not self._processor
            ):
                raise RuntimeError("initial pose sync cancelled")
            processor.clear()
            try:
                robot.publish_initial_pose_sync(
                    chunk[0],
                    action_keys,
                    duration_s=duration_s,
                )
            except Exception:
                self._running = False
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_deadline = None
                raise
            self._initial_pose_sync_in_progress = True
            self._initial_pose_sync_deadline = time.monotonic() + duration_s
            self._running = True
            logger.info(
                "initial pose sync target published: duration=%.3fs; "
                "discarded source chunk=%d",
                duration_s,
                response.chunk_size,
            )
            return True

    def pause(self) -> bool:
        robot = None
        action_keys: list[str] = []
        with self._lock:
            should_hold = (
                (
                    self._initial_pose_sync_in_progress
                    or self._initial_pose_sync_hold_pending
                )
                and self._publish_to_robot
                and self._robot is not None
            )
            if should_hold:
                robot = self._robot
                action_keys = list(self._action_keys)
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._initial_pose_sync_deadline = None
            self._generation += 1
            if should_hold:
                self._initial_pose_sync_hold_pending = True
            else:
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_hold_pending = False
        if robot is not None:
            try:
                robot.publish_current_pose_hold(action_keys, duration_s=0.1)
                logger.info("initial pose sync interrupted; current pose hold published")
            except Exception as e:
                with self._lock:
                    if robot is self._robot:
                        self._initial_pose_sync_in_progress = True
                        self._initial_pose_sync_hold_pending = True
                logger.error("failed to hold current pose during sync pause: %s", e)
                return False
            with self._lock:
                if robot is not self._robot:
                    return False
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_hold_pending = False
        return True

    def stop(self) -> bool:
        return self.pause()

    def initial_pose_sync_hold_required(self) -> bool:
        with self._lock:
            return (
                self._initial_pose_sync_in_progress
                or self._initial_pose_sync_hold_pending
            )

    def set_publish_to_robot(self, publish_to_robot: bool) -> None:
        with self._lock:
            self._set_publish_to_robot_locked(bool(publish_to_robot))

    def _set_publish_to_robot_locked(self, publish_to_robot: bool) -> None:
        if self._publish_to_robot == publish_to_robot:
            return
        self._publish_to_robot = publish_to_robot
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
            publish_to_robot = self._publish_to_robot
            action_request_mode = self._action_request_mode

            if self._initial_pose_sync_in_progress:
                deadline = self._initial_pose_sync_deadline
                if deadline is not None and time.monotonic() < deadline:
                    if publish_to_robot:
                        idle = getattr(robot, "publish_idle_action", None)
                        if callable(idle):
                            try:
                                idle(action_keys)
                            except Exception as e:
                                logger.error(
                                    "failed to publish idle action during pose sync: %s",
                                    e,
                                )
                    return
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_completed = True
                self._initial_pose_sync_deadline = None
                logger.info(
                    "initial pose sync complete; requesting a fresh action chunk"
                )

            action = processor.pop_action()
            if action is not None:
                preview = getattr(robot, "publish_action_preview", None)
                if callable(preview):
                    try:
                        preview(action, action_keys)
                    except Exception as e:
                        logger.warning("failed to publish action preview: %s", e)
                if publish_to_robot:
                    try:
                        robot.publish_action(action, action_keys)
                    except Exception as e:
                        logger.error("failed to publish robot action: %s", e)
            elif publish_to_robot:
                idle = getattr(robot, "publish_idle_action", None)
                if callable(idle):
                    try:
                        idle(action_keys)
                    except Exception as e:
                        logger.error("failed to publish idle robot action: %s", e)

            should_request = self._should_request_actions(processor)

        if should_request:
            self._request_thread = threading.Thread(
                target=self._request_and_buffer,
                args=(task_instruction, generation, action_request_mode),
                daemon=True,
            )
            self._request_thread.start()

    def _request_and_buffer(
        self,
        task_instruction: str,
        generation: int,
        action_request_mode: str = ACTION_REQUEST_MODE_ASYNC,
    ) -> None:
        action_request_mode = normalize_action_request_mode(action_request_mode)
        started_at = time.monotonic()
        try:
            response = self._requester.get_action(task_instruction)
        except Exception as e:
            latency_s = time.monotonic() - started_at
            self._record_request_latency(latency_s)
            logger.warning("get_action raised: %s", e)
            return
        latency_s = time.monotonic() - started_at
        self._record_request_latency(latency_s)
        try:
            chunk = self._decode_action_response(response)
        except ValueError as e:
            logger.warning("get_action response rejected: %s", e)
            return
        with self._lock:
            if (
                generation == self._generation
                and self._running
                and self._processor is not None
            ):
                buffer_delay_s = self._processor.buffer_size / max(
                    1.0,
                    self._processor.output_hz,
                )

                scheduled_start_delay_s = (
                    None
                    if action_request_mode == ACTION_REQUEST_MODE_SYNC
                    else latency_s + buffer_delay_s
                )
                produced = self._processor.push_actions(
                    chunk,
                    scheduled_start_delay_s=scheduled_start_delay_s,
                    align=action_request_mode != ACTION_REQUEST_MODE_SYNC,
                )
                scheduled_start_text = (
                    "none"
                    if scheduled_start_delay_s is None
                    else f"{scheduled_start_delay_s:.3f}s"
                )
                logger.debug(
                    "buffered action chunk: source=%d produced=%d "
                    "mode=%s latency=%.3fs buffer_delay=%.3fs "
                    "scheduled_start=%s",
                    response.chunk_size,
                    produced,
                    action_request_mode,
                    latency_s,
                    buffer_delay_s,
                    scheduled_start_text,
                )

    @staticmethod
    def _decode_action_response(response) -> np.ndarray:
        if not response.success:
            raise ValueError(response.message or "get_action failed")
        if response.chunk_size <= 0 or response.action_dim <= 0:
            raise ValueError("get_action returned empty action list")
        data = np.asarray(response.action_list, dtype=np.float64)
        expected_size = response.chunk_size * response.action_dim
        if data.size != expected_size:
            raise ValueError(
                f"action list size mismatch: {data.size} != "
                f"{response.chunk_size} * {response.action_dim}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError("action list contains non-finite values")
        return data.reshape(response.chunk_size, response.action_dim)

    def _should_request_actions(self, processor: ActionChunkProcessor) -> bool:
        if self._request_thread is not None and self._request_thread.is_alive():
            return False
        if self._action_request_mode == ACTION_REQUEST_MODE_SYNC:
            return processor.buffer_size <= 0
        return processor.buffer_size < self._refill_threshold(processor)

    def _refill_threshold(self, processor: ActionChunkProcessor) -> int:
        threshold_s = max(0.0, self._refill_margin_s)
        if self._request_latency_ema_s is not None:
            threshold_s += max(0.0, self._request_latency_ema_s)
        return max(1, int(math.ceil(threshold_s * processor.output_hz)))

    def _record_request_latency(self, latency_s: float) -> None:
        latency_s = max(0.0, float(latency_s))
        with self._lock:
            if self._latency_warmup_remaining > 0:
                self._latency_warmup_remaining -= 1
                return
            if (
                self._max_refill_latency_s is not None
                and latency_s > self._max_refill_latency_s
            ):
                logger.debug(
                    "ignoring GET_ACTION latency sample %.3fs above %.3fs",
                    latency_s,
                    self._max_refill_latency_s,
                )
                return
            if self._request_latency_ema_s is None:
                self._request_latency_ema_s = latency_s
            else:
                alpha = self._request_latency_alpha
                self._request_latency_ema_s = (
                    alpha * latency_s
                    + (1.0 - alpha) * self._request_latency_ema_s
                )

    def _reset_request_latency_locked(self) -> None:
        self._request_latency_ema_s = None
        self._latency_warmup_remaining = self._latency_warmup_samples

    def _tick_period(self) -> float:
        with self._lock:
            if self._processor is None:
                hz = self._control_hz
            else:
                hz = self._processor.output_hz
        return 1.0 / max(1.0, hz)
