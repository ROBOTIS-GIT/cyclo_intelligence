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

from .tt_rtc_timeline import TTActionTimeline
from rlt_recording import create_rlt_trace_publisher


try:  # pragma: no cover - SDK exists only in runtime container here.
    from zenoh_ros2_sdk import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("main_runtime.control_loop")

ACTION_REQUEST_MODE_ASYNC = "async"
ACTION_REQUEST_MODE_SYNC = "sync"
ACTION_REQUEST_MODE_TT_RTC = "tt_rtc"
ACTION_REQUEST_MODES = {
    ACTION_REQUEST_MODE_ASYNC,
    ACTION_REQUEST_MODE_SYNC,
    ACTION_REQUEST_MODE_TT_RTC,
}
ACTION_POLICY_BASE = "base"
ACTION_POLICY_RLT = "rlt"
ACTION_POLICY_MODES = {ACTION_POLICY_BASE, ACTION_POLICY_RLT}

TT_RTC_SOURCE_HZ = 15.0
TT_RTC_HORIZON = 16
TT_RTC_DELAY_STEPS = 6
TT_RTC_ACTION_DIM = 19
TT_RTC_RLT_CHUNK_SIZE = 10


def normalize_action_request_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode in ACTION_REQUEST_MODES:
        return mode
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
        self._rlt_trace = None
        self._active_action_policy_mode = ACTION_POLICY_BASE
        self._pending_action_policy_mode: Optional[str] = None
        self._action_policy_switch_error: Optional[str] = None
        self._active_rlt_robot_override = False
        self._pending_rlt_robot_override: Optional[bool] = None
        self._tt_rtc_failure_reason: Optional[str] = None
        self._tt_rtc_horizon = TT_RTC_HORIZON
        self._tt_rtc_action_dim = TT_RTC_ACTION_DIM
        self._tt_rtc_bootstrap_pending = True

    def configure(
        self,
        robot_type: str,
        task_instruction: str = "",
        action_keys: Optional[list[str]] = None,
        publish_to_robot: bool = False,
        action_request_mode: Optional[str] = None,
        rlt_enabled: bool = False,
        tt_rtc_horizon: int = TT_RTC_HORIZON,
        tt_rtc_action_dim: int = TT_RTC_ACTION_DIM,
        control_hz: float = 0.0,
    ) -> None:
        with self._lock:
            self.deconfigure()
            self._action_request_mode = normalize_action_request_mode(
                action_request_mode
                if action_request_mode is not None
                else self._default_action_request_mode
            )
            if self._action_request_mode == ACTION_REQUEST_MODE_TT_RTC:
                control_hz = float(control_hz or self._control_hz)
                if not math.isfinite(control_hz) or control_hz < TT_RTC_SOURCE_HZ:
                    raise ValueError("TT-RTC Control Hz must be finite and at least 15")
                if (tt_rtc_horizon, tt_rtc_action_dim) not in {(16, 19), (32, 16)}:
                    raise ValueError("Unsupported TT-RTC model action shape")
                if rlt_enabled and (tt_rtc_horizon, tt_rtc_action_dim) != (16, 19):
                    raise ValueError("Current RLT requires the 16x19 reference contract")
            self._tt_rtc_horizon = tt_rtc_horizon
            self._tt_rtc_action_dim = tt_rtc_action_dim
            self._robot = RobotClient(
                robot_type,
                enable_command_publishers=True,
                enable_preview_publisher=True,
            )
            tt_rtc_enabled = (
                self._action_request_mode == ACTION_REQUEST_MODE_TT_RTC
            )
            if tt_rtc_enabled:
                self._processor = TTActionTimeline(
                    source_hz=TT_RTC_SOURCE_HZ,
                    control_hz=control_hz,
                )
            else:
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
            if self._rlt_enabled:
                self._rlt_trace = create_rlt_trace_publisher()
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._action_policy_switch_error = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._tt_rtc_failure_reason = None
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
            self._tt_rtc_bootstrap_pending = True
            if self._rlt_trace is not None:
                self._record_rlt_event('buffer_cleared', reason='deconfigure')
                self._rlt_trace.close()
                self._rlt_trace = None
            self._running = False
            self._task_instruction = ""
            self._action_keys = []
            self._publish_to_robot = False
            self._action_request_mode = self._default_action_request_mode
            self._rlt_enabled = False
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._action_policy_switch_error = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._tt_rtc_failure_reason = None
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
            # A TT-RTC failure is latched until an explicit START/RESUME.  The
            # lifecycle command is the operator acknowledgement that permits a
            # fresh, bounded bootstrap request.
            self._tt_rtc_failure_reason = None
            if not self._running:
                self._tt_rtc_bootstrap_pending = True
            self._running = True

    def pause(self) -> None:
        with self._lock:
            self._record_rlt_event('buffer_cleared', reason='pause')
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._generation += 1
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._action_policy_switch_error = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._tt_rtc_failure_reason = None
            self._mode_condition.notify_all()

    def stop(self) -> None:
        with self._lock:
            self._record_rlt_event('buffer_cleared', reason='stop')
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._generation += 1
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._action_policy_switch_error = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._tt_rtc_failure_reason = None
            self._mode_condition.notify_all()

    @property
    def tt_rtc_failure_reason(self) -> Optional[str]:
        """Return the failure that latched TT-RTC in a safe paused state."""
        with self._lock:
            return self._tt_rtc_failure_reason

    def set_action_policy(
        self,
        action_policy_mode: str,
        *,
        allow_robot_rlt: bool = False,
        timeout_s: float = 5.0,
    ) -> tuple[bool, str]:
        """Switch policy routes without reloading the Engine.

        A generation bump invalidates any old-mode async request already in
        flight. Sync/async switch at a drained buffer; TT-RTC hands over by
        preserving the current prefix and atomically buffering the new route's
        postfix. Actions already committed to the processor remain untouched.
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
            self._action_policy_switch_error = None
            if self._action_request_mode != ACTION_REQUEST_MODE_TT_RTC:
                # Async/sync cannot merge an old-route response across a
                # switch. TT-RTC can: the old postfix safely bridges the
                # committed prefix, then the target route takes the following
                # refill. Keeping its generation prevents a switch requested
                # mid-inference from draining the queue idle.
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
            if self._action_policy_switch_error:
                message = self._action_policy_switch_error
                self._action_policy_switch_error = None
                return False, message
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
            if (
                action_request_mode == ACTION_REQUEST_MODE_TT_RTC
                and self._pending_action_policy_mode is not None
            ):
                # A TT-RTC route switch is itself a prefix-conditioned refill.
                # Keep executing the old route's committed prefix, but ask the
                # target route for the next postfix instead of draining idle.
                action_policy_mode = self._pending_action_policy_mode
            should_request = self._should_request_actions(processor)
            rtc_prefix_actions = None
            rtc_prefix_captured_at = None
            if (
                should_request
                and action_request_mode == ACTION_REQUEST_MODE_TT_RTC
            ):
                rtc_prefix_actions = processor.peek_actions()
                rtc_prefix_captured_at = time.monotonic()

        if should_request:
            self._request_thread = threading.Thread(
                target=self._request_and_buffer,
                args=(
                    task_instruction,
                    generation,
                    action_request_mode,
                    action_policy_mode,
                    rtc_prefix_actions,
                    rtc_prefix_captured_at,
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
        rtc_prefix_actions: Optional[np.ndarray] = None,
        rtc_prefix_captured_at: Optional[float] = None,
    ) -> None:
        action_request_mode = normalize_action_request_mode(action_request_mode)
        action_policy_mode = normalize_action_policy_mode(action_policy_mode)
        rtc_prefix = np.empty((0, self._tt_rtc_action_dim), dtype=np.float64)
        rtc_delay_steps = 0
        if action_request_mode == ACTION_REQUEST_MODE_TT_RTC:
            try:
                rtc_prefix = self._validate_tt_rtc_request_prefix(
                    rtc_prefix_actions, action_dim=self._tt_rtc_action_dim,
                )
            except ValueError as error:
                self._handle_tt_rtc_failure(
                    generation,
                    f"TT-RTC request prefix invalid; inference paused: {error}",
                )
                return
            rtc_delay_steps = int(rtc_prefix.shape[0])
        started_at = time.monotonic()
        recording_id = self._rlt_trace.recording_id if self._rlt_trace else ''
        if (
            action_request_mode == ACTION_REQUEST_MODE_TT_RTC
            and rtc_prefix_captured_at is None
        ):
            rtc_prefix_captured_at = started_at
        tt_rtc_timeout_s = None
        tt_rtc_bootstrap = False
        if action_request_mode == ACTION_REQUEST_MODE_TT_RTC:
            with self._lock:
                tt_rtc_bootstrap = (
                    self._tt_rtc_bootstrap_pending
                    and rtc_delay_steps == 0
                    and self._processor is not None
                    and self._processor.buffer_size == 0
                )
            if rtc_delay_steps == 0 and not tt_rtc_bootstrap:
                self._handle_tt_rtc_failure(
                    generation, "TT-RTC execution buffer exhausted; explicit START/RESUME required",
                )
                return
            tt_rtc_timeout_s = self._tt_rtc_remaining_timeout_s(
                prefix_captured_at=float(rtc_prefix_captured_at),
            )
            if tt_rtc_timeout_s <= 0.0:
                self._handle_tt_rtc_failure(
                    generation,
                    "TT-RTC deadline expired before the action request was sent",
                )
                return
        try:
            if action_request_mode == ACTION_REQUEST_MODE_TT_RTC:
                response = self._requester.get_action(
                    task_instruction,
                    action_policy_mode=action_policy_mode,
                    action_request_mode=action_request_mode,
                    rtc_delay_steps=rtc_delay_steps,
                    rtc_action_dim=self._tt_rtc_action_dim,
                    rtc_prefix_action_list=rtc_prefix.reshape(-1).tolist(),
                    timeout_s=tt_rtc_timeout_s,
                )
            else:
                response = self._requester.get_action(
                    task_instruction,
                    action_policy_mode=action_policy_mode,
                    action_request_mode=action_request_mode,
                )
        except Exception as e:
            latency_s = time.monotonic() - started_at
            self._record_request_latency(latency_s)
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                f"get_action raised: {e}",
                action_request_mode=action_request_mode,
            )
            return
        latency_s = time.monotonic() - started_at
        self._record_request_latency(latency_s)
        if not bool(getattr(response, "success", False)):
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                f"get_action failed: {getattr(response, 'message', '')}",
                action_request_mode=action_request_mode,
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
                action_request_mode=action_request_mode,
            )
            return
        if chunk_size <= 0 or action_dim <= 0:
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                "get_action returned empty action list",
                action_request_mode=action_request_mode,
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
                action_request_mode=action_request_mode,
            )
            return
        if not bool(np.isfinite(data).all()):
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                "action list contains NaN or Inf",
                action_request_mode=action_request_mode,
            )
            return
        if data.size != chunk_size * action_dim:
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                f"action list size mismatch: {data.size} != "
                f"{chunk_size} * {action_dim}",
                action_request_mode=action_request_mode,
            )
            return
        chunk = data.reshape(chunk_size, action_dim)
        if not bool(np.isfinite(chunk).all()):
            self._handle_action_request_failure(
                generation,
                action_policy_mode,
                "reshaped action chunk contains NaN or Inf",
                action_request_mode=action_request_mode,
            )
            return
        with self._lock:
            if (
                generation == self._generation
                and self._running
                and self._processor is not None
            ):
                if action_request_mode == ACTION_REQUEST_MODE_TT_RTC:
                    committed, commit_failure_reason = (
                        self._commit_tt_rtc_chunk_locked(
                            chunk=chunk,
                            captured_prefix=rtc_prefix,
                            delay_steps=rtc_delay_steps,
                            prefix_captured_at=float(rtc_prefix_captured_at),
                            action_policy_mode=action_policy_mode,
                        )
                    )
                    if committed:
                        self._tt_rtc_bootstrap_pending = False
                        self._record_rlt_event(
                            'buffer_accepted', recording_id=recording_id,
                            request_seq=getattr(response, 'seq_id', 0),
                            action_policy_mode=action_policy_mode,
                            delay_steps=rtc_delay_steps,
                            source_queue_size=self._processor.buffer_size,
                            action_request_mode='tt_rtc',
                        )
                        self._commit_tt_rtc_action_policy_locked(
                            action_policy_mode
                        )
                    else:
                        self._record_rlt_event(
                            'buffer_rejected', recording_id=recording_id,
                            request_seq=getattr(response, 'seq_id', 0),
                            reason=commit_failure_reason,
                        )
                        self._handle_tt_rtc_commit_failure_locked(
                            generation=generation,
                            action_policy_mode=action_policy_mode,
                            reason=commit_failure_reason,
                        )
                    return
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
                self._record_rlt_event(
                    'buffer_accepted', recording_id=recording_id,
                    request_seq=getattr(response, 'seq_id', 0),
                    action_policy_mode=action_policy_mode,
                    action_request_mode=action_request_mode,
                    generated_actions=chunk_size, queued_control_samples=produced,
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

    @staticmethod
    def _validate_tt_rtc_request_prefix(
        prefix_actions: Optional[np.ndarray],
        *, action_dim: int = TT_RTC_ACTION_DIM,
    ) -> np.ndarray:
        if prefix_actions is None:
            return np.empty((0, action_dim), dtype=np.float64)
        prefix = np.asarray(prefix_actions, dtype=np.float64)
        if prefix.size == 0:
            return np.empty((0, action_dim), dtype=np.float64)
        if prefix.ndim != 2:
            raise ValueError(
                f"prefix must be 2D (T, D); got shape {prefix.shape}"
            )
        if prefix.shape[1] != action_dim:
            raise ValueError(
                f"prefix action_dim must be {action_dim}; "
                f"got {prefix.shape[1]}"
            )
        if prefix.shape[0] > TT_RTC_DELAY_STEPS:
            raise ValueError(
                f"prefix may contain at most {TT_RTC_DELAY_STEPS} actions; "
                f"got {prefix.shape[0]}"
            )
        if not bool(np.isfinite(prefix).all()):
            raise ValueError("prefix contains NaN or Inf")
        return prefix.copy()

    def _commit_tt_rtc_chunk_locked(
        self,
        *,
        chunk: np.ndarray,
        captured_prefix: np.ndarray,
        delay_steps: int,
        prefix_captured_at: float,
        action_policy_mode: str,
    ) -> tuple[bool, str]:
        """Atomically validate and append one TT-RTC postfix.

        The already queued prefix remains the sole source of commands during
        inference.  A base GR00T response includes that prefix and contributes
        only ``chunk[d:]``; an RLT response is already the 10-action postfix.
        """
        processor = self._processor
        if processor is None:
            return False, "TT-RTC action processor is unavailable"
        if chunk.ndim != 2 or chunk.shape[1] != self._tt_rtc_action_dim:
            reason = (
                f"TT-RTC response expected action_dim={self._tt_rtc_action_dim}, "
                f"got shape={tuple(chunk.shape)}"
            )
            logger.warning(
                "TT-RTC response discarded: expected action_dim=%d, got %s",
                self._tt_rtc_action_dim,
                tuple(chunk.shape),
            )
            return False, reason
        remaining_prefix = processor.peek_actions()
        consumed_steps = self._tt_rtc_consumed_prefix_steps(
            captured_prefix,
            remaining_prefix,
        )
        if consumed_steps is None:
            reason = (
                "TT-RTC committed prefix changed while inference was in flight"
            )
            logger.warning(
                "TT-RTC response discarded: committed prefix changed while "
                "inference was in flight"
            )
            return False, reason

        if action_policy_mode == ACTION_POLICY_BASE:
            if chunk.shape[0] != self._tt_rtc_horizon:
                reason = (
                    f"TT-RTC VLA response expected horizon={self._tt_rtc_horizon}, "
                    f"got {chunk.shape[0]}"
                )
                logger.warning(
                    "TT-RTC base response discarded: expected H=%d, got %d",
                    self._tt_rtc_horizon,
                    chunk.shape[0],
                )
                return False, reason
            if delay_steps and not np.allclose(
                chunk[:delay_steps],
                captured_prefix,
                rtol=1e-6,
                atol=1e-6,
            ):
                reason = (
                    "TT-RTC VLA response did not preserve the committed prefix"
                )
                logger.warning(
                    "TT-RTC base response discarded: returned prefix does "
                    "not match the committed action prefix"
                )
                return False, reason
            postfix = chunk[delay_steps:]
        else:
            if chunk.shape[0] != TT_RTC_RLT_CHUNK_SIZE:
                reason = (
                    f"TT-RTC MLP response expected chunk={TT_RTC_RLT_CHUNK_SIZE}, "
                    f"got {chunk.shape[0]}"
                )
                logger.warning(
                    "TT-RTC RLT response discarded: expected C=%d, got %d",
                    TT_RTC_RLT_CHUNK_SIZE,
                    chunk.shape[0],
                )
                return False, reason
            postfix = chunk

        # Check the deadline immediately before enqueue so model execution,
        # response parsing, lock contention, and validation are all included.
        elapsed_since_capture_s = max(
            0.0,
            time.monotonic() - prefix_captured_at,
        )
        deadline_s = self._tt_rtc_deadline_budget_s()
        if elapsed_since_capture_s > deadline_s + 1e-9:
            reason = (
                "TT-RTC capture-to-enqueue deadline exceeded: "
                f"{elapsed_since_capture_s:.3f}s > {deadline_s:.3f}s "
                f"for d={delay_steps}"
            )
            logger.warning(
                "TT-RTC response discarded: capture-to-enqueue latency "
                "%.3fs exceeded normal request timeout %.3fs (prefix=%d)",
                elapsed_since_capture_s,
                deadline_s,
                delay_steps,
            )
            return False, reason

        produced = processor.push_actions(
            postfix,
            scheduled_start_delay_s=None,
            align=False,
        )
        logger.debug(
            "buffered TT-RTC postfix: policy=%s delay=%d source=%d "
            "produced=%d capture_to_enqueue=%.3fs consumed_prefix=%d "
            "remaining_prefix=%d",
            action_policy_mode,
            delay_steps,
            chunk.shape[0],
            produced,
            elapsed_since_capture_s,
            consumed_steps,
            remaining_prefix.shape[0],
        )
        return True, ""

    def _record_rlt_event(self, event, *, recording_id=None, **metadata):
        if self._rlt_trace is not None:
            self._rlt_trace.submit(dict(
                metadata, event=event, control_time_ns=time.time_ns(),
                timebase='unix_wall_clock_ns', generation=self._generation,
                execution_verified=False,
            ), recording_id=recording_id)

    def _commit_tt_rtc_action_policy_locked(self, action_policy_mode: str) -> None:
        """Commit a pending TT-RTC route only after its postfix is buffered."""
        if self._pending_action_policy_mode != action_policy_mode:
            return
        self._active_action_policy_mode = action_policy_mode
        self._active_rlt_robot_override = (
            bool(self._pending_rlt_robot_override)
            if action_policy_mode == ACTION_POLICY_RLT
            else False
        )
        self._pending_action_policy_mode = None
        self._pending_rlt_robot_override = None
        self._action_policy_switch_error = None
        logger.info(
            "TT-RTC action policy switched with prefix handoff: %s",
            action_policy_mode,
        )
        self._mode_condition.notify_all()

    def _handle_tt_rtc_commit_failure_locked(
        self,
        *,
        generation: int,
        action_policy_mode: str,
        reason: str,
    ) -> None:
        """Latch any rejected TT-RTC continuation in a safe paused state."""
        if generation != self._generation or not self._running:
            return
        route = "MLP" if action_policy_mode == ACTION_POLICY_RLT else "VLA"
        self._latch_tt_rtc_failure_locked(
            f"TT-RTC {route} response rejected; inference paused: {reason}"
        )

    def _tt_rtc_deadline_budget_s(self) -> float:
        """Bound unresponsive requests independently of the trained prefix length."""
        return float(getattr(self._requester, "get_action_timeout_s", 5.0))

    def _tt_rtc_remaining_timeout_s(
        self,
        *,
        prefix_captured_at: float,
    ) -> float:
        deadline = (
            float(prefix_captured_at)
            + self._tt_rtc_deadline_budget_s()
        )
        return max(0.0, deadline - time.monotonic())

    def _handle_tt_rtc_failure(self, generation: int, reason: str) -> None:
        with self._mode_condition:
            if generation != self._generation or not self._running:
                return
            self._latch_tt_rtc_failure_locked(reason)

    def _latch_tt_rtc_failure_locked(self, reason: str) -> None:
        """Stop command generation until an explicit lifecycle resume.

        Clearing the queue prevents an expired continuation from executing.
        Velocity-like robot modalities receive their existing idle command
        immediately because no subsequent control-loop tick runs while latched.
        """
        self._record_rlt_event('buffer_cleared', reason=reason)
        message = str(reason or "TT-RTC request failed; inference paused")
        self._running = False
        if self._processor is not None:
            self._processor.clear()
        self._tt_rtc_failure_reason = message
        self._action_policy_switch_error = message
        self._active_action_policy_mode = ACTION_POLICY_BASE
        self._pending_action_policy_mode = None
        self._active_rlt_robot_override = False
        self._pending_rlt_robot_override = None
        self._generation += 1

        if self._publish_to_robot and self._robot is not None:
            idle = getattr(self._robot, "publish_idle_action", None)
            if callable(idle):
                try:
                    idle(list(self._action_keys))
                except Exception as error:
                    logger.error(
                        "failed to publish idle action while latching TT-RTC: %s",
                        error,
                    )
        logger.error("%s", message)
        self._mode_condition.notify_all()

    @staticmethod
    def _tt_rtc_consumed_prefix_steps(
        captured_prefix: np.ndarray,
        remaining_prefix: np.ndarray,
    ) -> Optional[int]:
        """Return consumed steps when queue is the exact captured suffix."""
        captured = np.asarray(captured_prefix, dtype=np.float64)
        remaining = np.asarray(remaining_prefix, dtype=np.float64)
        if captured.ndim != 2 or remaining.ndim != 2:
            return None
        if remaining.shape[0] > captured.shape[0]:
            return None
        consumed = captured.shape[0] - remaining.shape[0]
        if remaining.shape[0] == 0:
            return consumed
        if remaining.shape[1] != captured.shape[1]:
            return None
        if not bool(np.array_equal(remaining, captured[consumed:])):
            return None
        return consumed

    def _handle_action_request_failure(
        self,
        generation: int,
        action_policy_mode: str,
        reason: str,
        *,
        action_request_mode: str = ACTION_REQUEST_MODE_ASYNC,
    ) -> None:
        logger.warning("%s", reason)
        if (
            normalize_action_request_mode(action_request_mode)
            == ACTION_REQUEST_MODE_TT_RTC
        ):
            self._handle_tt_rtc_failure(
                generation,
                f"TT-RTC action request failed; inference paused: {reason}",
            )
            return
        if action_policy_mode != ACTION_POLICY_RLT:
            return
        with self._mode_condition:
            if (
                generation != self._generation
                or not self._running
                or (
                    self._active_action_policy_mode != ACTION_POLICY_RLT
                    and self._pending_action_policy_mode != ACTION_POLICY_RLT
                )
            ):
                return
            self._fallback_rlt_to_base_locked(
                f"RLT inference failed; reverting to base action: {reason}",
                preserve_buffer=(
                    normalize_action_request_mode(action_request_mode)
                    == ACTION_REQUEST_MODE_TT_RTC
                ),
            )

    def _fallback_rlt_to_base_locked(
        self,
        reason: str,
        *,
        preserve_buffer: bool = False,
    ) -> None:
        if self._processor is not None and not preserve_buffer:
            self._processor.clear()
        if self._pending_action_policy_mode == ACTION_POLICY_RLT:
            self._action_policy_switch_error = reason
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
        if (
            target is not None
            and self._action_request_mode == ACTION_REQUEST_MODE_TT_RTC
        ):
            # TT-RTC commits the route together with the target postfix.  A
            # plain drained-buffer commit would introduce an idle bootstrap.
            return
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
        if (
            self._pending_action_policy_mode is not None
            and self._action_request_mode != ACTION_REQUEST_MODE_TT_RTC
        ):
            return False
        if self._action_request_mode == ACTION_REQUEST_MODE_SYNC:
            return processor.buffer_size <= 0
        if self._action_request_mode == ACTION_REQUEST_MODE_TT_RTC:
            if self._request_latency_ema_s is None:
                prefix_steps = TT_RTC_DELAY_STEPS
            else:
                prefix_steps = min(TT_RTC_DELAY_STEPS, max(1, math.ceil(
                    (self._request_latency_ema_s + max(0.0, self._refill_margin_s))
                    * TT_RTC_SOURCE_HZ,
                )))
            return processor.buffer_size <= prefix_steps
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
                self._action_request_mode != ACTION_REQUEST_MODE_TT_RTC
                and self._max_refill_latency_s is not None
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
