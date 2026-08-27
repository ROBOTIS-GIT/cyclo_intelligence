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
ACTION_POLICY_BASE = "base"
ACTION_POLICY_RLT = "rlt"
ACTION_POLICY_MODES = {ACTION_POLICY_BASE, ACTION_POLICY_RLT}


def normalize_action_request_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode == ACTION_REQUEST_MODE_SYNC:
        return ACTION_REQUEST_MODE_SYNC
    return ACTION_REQUEST_MODE_ASYNC


def normalize_action_policy_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode not in ACTION_POLICY_MODES:
        raise ValueError("action_policy_mode must be 'base' or 'rlt'")
    return mode


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
        self._inference_hz = float(inference_hz)
        self._control_hz = float(control_hz)
        self._chunk_align_window_s = float(chunk_align_window_s)
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
        self._mode_condition = threading.Condition(self._lock)
        self._robot: Optional[RobotClient] = None
        self._processor: Optional[ActionChunkProcessor] = None
        self._task_instruction = ""
        self._action_keys: list[str] = []
        self._publish_to_robot = False
        self._running = False
        self._generation = 0
        self._shutdown = threading.Event()
        self._request_thread: Optional[threading.Thread] = None
        self._thread: Optional[threading.Thread] = None
        self._rlt_enabled = False
        self._active_action_policy_mode = ACTION_POLICY_BASE
        self._pending_action_policy_mode: Optional[str] = None
        self._active_rlt_robot_override = False
        self._pending_rlt_robot_override: Optional[bool] = None

    def configure(
        self,
        robot_type: str,
        task_instruction: str = "",
        action_keys: Optional[list[str]] = None,
        publish_to_robot: bool = False,
        action_request_mode: Optional[str] = None,
        rlt_enabled: bool = False,
    ) -> None:
        with self._lock:
            self.deconfigure()
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
            self._rlt_enabled = bool(rlt_enabled)
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._reset_request_latency_locked()
            self._generation += 1
            logger.info(
                "configured RobotClient command path for %s "
                "(publish_to_robot=%s action_request_mode=%s)",
                robot_type,
                self._publish_to_robot,
                self._action_request_mode,
            )

    def deconfigure(self) -> None:
        with self._lock:
            self._running = False
            self._task_instruction = ""
            self._action_keys = []
            self._publish_to_robot = False
            self._action_request_mode = self._default_action_request_mode
            self._rlt_enabled = False
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._processor = None
            self._generation += 1
            if self._robot is not None:
                self._robot.close()
                self._robot = None
            self._reset_request_latency_locked()
            self._mode_condition.notify_all()

    def start(self, publish_to_robot: Optional[bool] = None) -> None:
        with self._lock:
            if publish_to_robot is not None:
                self._set_publish_to_robot_locked(bool(publish_to_robot))
            self._running = True

    def pause(self) -> None:
        with self._lock:
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._generation += 1
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._mode_condition.notify_all()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._generation += 1
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._mode_condition.notify_all()

    def set_action_policy(
        self,
        action_policy_mode: str,
        *,
        allow_robot_rlt: bool = False,
        timeout_s: float = 5.0,
    ) -> tuple[bool, str]:
        """Switch at the next drained-buffer boundary without reloading.

        A generation bump invalidates any old-mode async request already in
        flight. Actions already committed to the processor remain untouched.
        """
        try:
            target = normalize_action_policy_mode(action_policy_mode)
        except ValueError as error:
            return False, str(error)

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._mode_condition:
            if not self._running or self._processor is None:
                return False, "inference is not running"
            requested_robot_override = False
            if target == ACTION_POLICY_RLT:
                if not self._rlt_enabled:
                    return False, "RLT bundle was not preloaded"
                requested_robot_override = bool(allow_robot_rlt)
                if self._publish_to_robot and not requested_robot_override:
                    return False, (
                        "Real-robot RLT routing requires explicit "
                        "rlt_robot_override for the current bundle"
                    )
            if (
                self._active_action_policy_mode == target
                and self._pending_action_policy_mode is None
            ):
                if target == ACTION_POLICY_RLT and requested_robot_override:
                    self._active_rlt_robot_override = True
                return True, f"{target.upper()} action already active"

            self._pending_action_policy_mode = target
            self._pending_rlt_robot_override = requested_robot_override
            self._generation += 1
            self._mode_condition.notify_all()
            while (
                self._running
                and self._pending_action_policy_mode == target
                and self._active_action_policy_mode != target
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._mode_condition.wait(timeout=remaining)

            if (
                self._active_action_policy_mode == target
                and self._pending_action_policy_mode is None
            ):
                return True, f"{target.upper()} action active"
            if self._pending_action_policy_mode == target:
                self._pending_action_policy_mode = None
                self._pending_rlt_robot_override = None
                self._generation += 1
                self._mode_condition.notify_all()
            if not self._running:
                return False, "inference stopped before action switch"
            return False, f"Timed out waiting for {target.upper()} chunk boundary"

    def set_publish_to_robot(self, publish_to_robot: bool) -> None:
        with self._lock:
            self._set_publish_to_robot_locked(bool(publish_to_robot))

    def _set_publish_to_robot_locked(self, publish_to_robot: bool) -> None:
        if self._publish_to_robot == publish_to_robot:
            return
        self._publish_to_robot = publish_to_robot
        if self._processor is not None:
            self._processor.clear()
        if publish_to_robot:
            active_rlt_is_unsafe = (
                self._active_action_policy_mode == ACTION_POLICY_RLT
                and not self._active_rlt_robot_override
            )
            pending_rlt_is_unsafe = (
                self._pending_action_policy_mode == ACTION_POLICY_RLT
                and not bool(self._pending_rlt_robot_override)
            )
            if active_rlt_is_unsafe or pending_rlt_is_unsafe:
                logger.warning(
                    "falling back to base action while enabling robot publish: "
                    "RLT lacks explicit operator override"
                )
                self._active_action_policy_mode = ACTION_POLICY_BASE
                self._pending_action_policy_mode = None
                self._active_rlt_robot_override = False
                self._pending_rlt_robot_override = None
        self._generation += 1
        self._mode_condition.notify_all()

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
            publish_to_robot = self._publish_to_robot
            action_request_mode = self._action_request_mode

            if (
                publish_to_robot
                and self._active_action_policy_mode == ACTION_POLICY_RLT
                and not self._active_rlt_robot_override
            ):
                self._fallback_rlt_to_base_locked(
                    "blocked unauthorized RLT action before robot publish"
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

            self._commit_pending_action_policy_locked(processor)
            generation = self._generation
            action_policy_mode = self._active_action_policy_mode
            should_request = self._should_request_actions(processor)

        if should_request:
            self._request_thread = threading.Thread(
                target=self._request_and_buffer,
                args=(
                    task_instruction,
                    generation,
                    action_request_mode,
                    action_policy_mode,
                ),
                daemon=True,
            )
            self._request_thread.start()

    def _request_and_buffer(
        self,
        task_instruction: str,
        generation: int,
        action_request_mode: str = ACTION_REQUEST_MODE_ASYNC,
        action_policy_mode: str = ACTION_POLICY_BASE,
    ) -> None:
        action_request_mode = normalize_action_request_mode(action_request_mode)
        action_policy_mode = normalize_action_policy_mode(action_policy_mode)
        started_at = time.monotonic()
        try:
            response = self._requester.get_action(
                task_instruction,
                action_policy_mode=action_policy_mode,
            )
        except Exception as e:
            latency_s = time.monotonic() - started_at
            self._record_request_latency(latency_s)
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                f"get_action raised: {e}",
            )
            return
        latency_s = time.monotonic() - started_at
        self._record_request_latency(latency_s)
        if not bool(getattr(response, "success", False)):
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                f"get_action failed: {getattr(response, 'message', '')}",
            )
            return
        try:
            chunk_size = int(getattr(response, "chunk_size", 0))
            action_dim = int(getattr(response, "action_dim", 0))
        except (TypeError, ValueError) as error:
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                f"get_action returned invalid chunk shape: {error}",
            )
            return
        if chunk_size <= 0 or action_dim <= 0:
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                "get_action returned empty action list",
            )
            return
        try:
            data = np.asarray(
                getattr(response, "action_list", []),
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as error:
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                f"action list is not numeric: {error}",
            )
            return
        if not bool(np.isfinite(data).all()):
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                "action list contains NaN or Inf",
            )
            return
        if data.size != chunk_size * action_dim:
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                f"action list size mismatch: {data.size} != "
                f"{chunk_size} * {action_dim}",
            )
            return
        chunk = data.reshape(chunk_size, action_dim)
        if not bool(np.isfinite(chunk).all()):
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                "reshaped action chunk contains NaN or Inf",
            )
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
                    chunk_size,
                    produced,
                    f"{action_request_mode}/{action_policy_mode}",
                    latency_s,
                    buffer_delay_s,
                    scheduled_start_text,
                )

    def _handle_action_request_failure(
        self,
        generation: int,
        action_policy_mode: str,
        reason: str,
    ) -> None:
        logger.warning("%s", reason)
        if action_policy_mode != ACTION_POLICY_RLT:
            return
        with self._mode_condition:
            if (
                generation != self._generation
                or not self._running
                or self._active_action_policy_mode != ACTION_POLICY_RLT
            ):
                return
            self._fallback_rlt_to_base_locked(
                f"RLT inference failed; reverting to base action: {reason}"
            )

    def _fallback_rlt_to_base_locked(self, reason: str) -> None:
        if self._processor is not None:
            self._processor.clear()
        self._active_action_policy_mode = ACTION_POLICY_BASE
        self._pending_action_policy_mode = None
        self._active_rlt_robot_override = False
        self._pending_rlt_robot_override = None
        self._generation += 1
        logger.error("%s", reason)
        self._mode_condition.notify_all()

    def _commit_pending_action_policy_locked(
        self,
        processor: ActionChunkProcessor,
    ) -> None:
        target = self._pending_action_policy_mode
        if target is None or processor.buffer_size > 0:
            return
        if self._request_thread is not None and self._request_thread.is_alive():
            return
        self._active_action_policy_mode = target
        self._active_rlt_robot_override = (
            bool(self._pending_rlt_robot_override)
            if target == ACTION_POLICY_RLT
            else False
        )
        self._pending_action_policy_mode = None
        self._pending_rlt_robot_override = None
        logger.info("action policy switched at chunk boundary: %s", target)
        self._mode_condition.notify_all()

    def _should_request_actions(self, processor: ActionChunkProcessor) -> bool:
        if self._request_thread is not None and self._request_thread.is_alive():
            return False
        if self._pending_action_policy_mode is not None:
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
