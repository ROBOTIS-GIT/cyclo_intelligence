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

from .confirmed_hold import hold_current_pose
from .tt_rtc_timeline import TTActionTimeline
from .arm_command_smoothing import ArmCommandSmoother, ArmSmoothingConfig, smoothing_enabled
from .position_deadband import PositionDeadband, left_arm_deadband_enabled
from .left_pose_hold import LeftPoseHold, HoldConfig, REFERENCE as LEFT_HOLD_REFERENCE, enabled_for_model
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
GRIPPER_COMMAND_THRESHOLD = 0.7
GRIPPER_COMMAND_VALUE = 0.9
TT_RTC_PREFIX_MODE_ENV = "CYCLO_TT_RTC_PREFIX_MODE"
TT_RTC_PREFIX_MODE_FIXED_FIRST_SIX = "fixed_first_6"
TT_RTC_PREFIX_MODE_FIXED_FIRST_EIGHT = "fixed_first_8"
TT_RTC_PREFIX_MODE_LATENCY_ADAPTIVE = "latency_adaptive"
TT_RTC_PREFIX_MODES = {
    TT_RTC_PREFIX_MODE_FIXED_FIRST_SIX,
    TT_RTC_PREFIX_MODE_FIXED_FIRST_EIGHT,
    TT_RTC_PREFIX_MODE_LATENCY_ADAPTIVE,
}
TT_RTC_FIXED_PREFIX_STEPS = {
    TT_RTC_PREFIX_MODE_FIXED_FIRST_SIX: 6,
    TT_RTC_PREFIX_MODE_FIXED_FIRST_EIGHT: 8,
}
TT_RTC_TEMPORAL_OVERLAP_STEPS_ENV = "CYCLO_TT_RTC_TEMPORAL_OVERLAP_STEPS"
TT_RTC_PROFILE_DELAY_RANGES = {
    (16, 19): (0, 6),
    (32, 16): (0, 8),
    (20, 16): (6, 8),
}


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


def normalize_tt_rtc_prefix_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if not mode:
        return TT_RTC_PREFIX_MODE_FIXED_FIRST_SIX
    if mode not in TT_RTC_PREFIX_MODES:
        supported = ", ".join(sorted(TT_RTC_PREFIX_MODES))
        raise ValueError(
            f"{TT_RTC_PREFIX_MODE_ENV} must be one of: {supported}"
        )
    return mode


def normalize_tt_rtc_temporal_overlap_steps(value: object) -> int:
    if value is None or str(value).strip() == "":
        return 0
    if isinstance(value, bool):
        raise ValueError(
            f"{TT_RTC_TEMPORAL_OVERLAP_STEPS_ENV} must be a non-negative integer"
        )
    try:
        steps = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{TT_RTC_TEMPORAL_OVERLAP_STEPS_ENV} must be a non-negative integer"
        ) from error
    if steps < 0 or str(value).strip() not in {str(steps), f"+{steps}"}:
        raise ValueError(
            f"{TT_RTC_TEMPORAL_OVERLAP_STEPS_ENV} must be a non-negative integer"
        )
    return steps


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
        tt_rtc_prefix_mode: Optional[str] = None,
        tt_rtc_temporal_overlap_steps: Optional[int] = None,
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
        self._tt_rtc_prefix_mode = normalize_tt_rtc_prefix_mode(
            os.environ.get(TT_RTC_PREFIX_MODE_ENV)
            if tt_rtc_prefix_mode is None
            else tt_rtc_prefix_mode
        )
        self._tt_rtc_temporal_overlap_steps = (
            normalize_tt_rtc_temporal_overlap_steps(
                os.environ.get(TT_RTC_TEMPORAL_OVERLAP_STEPS_ENV)
                if tt_rtc_temporal_overlap_steps is None
                else tt_rtc_temporal_overlap_steps
            )
        )

        self._lock = threading.RLock()
        self._hold_lock = threading.Lock()
        self._mode_condition = threading.Condition(self._lock)
        self._robot: Optional[RobotClient] = None
        self._processor: Optional[ActionChunkProcessor] = None
        self._task_instruction = ""
        self._action_keys: list[str] = []
        self._robot_type = ""
        self._arm_smoother = None
        self._arm_smoothing_last_time = None
        self._left_arm_deadband = None
        self._left_pose_hold = None
        self._publish_to_robot = False
        self._running = False
        self._initial_pose_sync_enabled = False
        self._initial_pose_sync_min_duration_s = 5.0
        self._initial_pose_sync_max_duration_s = 60.0
        self._initial_pose_sync_deferred_activation = False
        self._initial_pose_sync_effective_duration_s: Optional[float] = None
        self._initial_pose_sync_in_progress = False
        self._initial_pose_sync_completed = False
        self._initial_pose_sync_deadline: Optional[float] = None
        self._initial_pose_sync_hold_pending = False
        self._stream_handoff_pending = False
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
        self._tt_rtc_min_delay_steps = 0
        self._tt_rtc_max_delay_steps = TT_RTC_DELAY_STEPS
        self._tt_rtc_bootstrap_pending = True
        self._tt_rtc_rlt_vla_reference: Optional[np.ndarray] = None

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
        tt_rtc_min_delay_steps: Optional[int] = None,
        tt_rtc_max_delay_steps: Optional[int] = None,
        control_hz: float = 0.0,
        initial_pose_sync: bool = False,
        initial_pose_sync_duration_s: float = 5.0,
        defer_initial_pose_sync_activation: bool = False,
        model_path: str = "",
    ) -> None:
        minimum_duration_s = float(initial_pose_sync_duration_s)
        if (
            not math.isfinite(minimum_duration_s)
            or not 1.0 <= minimum_duration_s <= 60.0
        ):
            raise ValueError(
                "initial_pose_sync_duration_s must be between 1.0 and 60.0"
            )
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
                profile_delay_range = TT_RTC_PROFILE_DELAY_RANGES.get(
                    (tt_rtc_horizon, tt_rtc_action_dim)
                )
                if profile_delay_range is None:
                    raise ValueError("Unsupported TT-RTC model action shape")
                expected_min_delay, expected_max_delay = profile_delay_range
                resolved_min_delay = (
                    expected_min_delay
                    if tt_rtc_min_delay_steps is None
                    else tt_rtc_min_delay_steps
                )
                resolved_max_delay = (
                    expected_max_delay
                    if tt_rtc_max_delay_steps is None
                    else tt_rtc_max_delay_steps
                )
                if (
                    type(resolved_min_delay) is not int
                    or type(resolved_max_delay) is not int
                    or (resolved_min_delay, resolved_max_delay) not in (
                        (profile_delay_range, (0, 6))
                        if (tt_rtc_horizon, tt_rtc_action_dim) == (32, 16)
                        else (profile_delay_range,)
                    )
                ):
                    raise ValueError(
                        "TT-RTC delay range does not match the model action shape"
                    )
            else:
                resolved_min_delay = 0
                resolved_max_delay = TT_RTC_DELAY_STEPS
            self._tt_rtc_horizon = tt_rtc_horizon
            self._tt_rtc_action_dim = tt_rtc_action_dim
            self._tt_rtc_min_delay_steps = resolved_min_delay
            self._tt_rtc_max_delay_steps = resolved_max_delay
            if (
                self._action_request_mode == ACTION_REQUEST_MODE_TT_RTC
                and self._tt_rtc_prefix_mode in TT_RTC_FIXED_PREFIX_STEPS
                and not (
                    resolved_min_delay
                    <= TT_RTC_FIXED_PREFIX_STEPS[self._tt_rtc_prefix_mode]
                    <= resolved_max_delay
                )
            ):
                raise ValueError(
                    f"{self._tt_rtc_prefix_mode} requires a TT-RTC model "
                    "trained for that delay"
                )
            if (
                self._action_request_mode == ACTION_REQUEST_MODE_TT_RTC
                and self._tt_rtc_temporal_overlap_steps
                > tt_rtc_horizon - resolved_max_delay
            ):
                raise ValueError(
                    f"{TT_RTC_TEMPORAL_OVERLAP_STEPS_ENV} leaves no fresh "
                    "TT-RTC postfix"
                )
            self._robot = RobotClient(
                robot_type,
                enable_command_publishers=True,
                enable_preview_publisher=True,
                camera_names=[],  # Images are consumed by the Engine process.
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
            self._robot_type = robot_type
            self._task_instruction = task_instruction or ""
            self._action_keys = list(action_keys or self._robot.action_keys)
            self._configure_arm_smoothing_locked()
            self._configure_left_arm_deadband_locked()
            self._configure_left_pose_hold_locked(model_path, rlt_enabled)
            self._publish_to_robot = bool(publish_to_robot)
            self._initial_pose_sync_enabled = bool(initial_pose_sync)
            self._initial_pose_sync_min_duration_s = minimum_duration_s
            self._initial_pose_sync_deferred_activation = bool(
                defer_initial_pose_sync_activation
            )
            self._initial_pose_sync_effective_duration_s = None
            self._initial_pose_sync_in_progress = False
            self._initial_pose_sync_completed = False
            self._initial_pose_sync_deadline = None
            self._initial_pose_sync_hold_pending = False
            self._stream_handoff_pending = bool(publish_to_robot)
            self._rlt_enabled = bool(rlt_enabled)
            if self._rlt_enabled:
                self._rlt_trace = create_rlt_trace_publisher()
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._action_policy_switch_error = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._tt_rtc_failure_reason = None
            self._tt_rtc_rlt_vla_reference = None
            self._reset_request_latency_locked()
            self._generation += 1
            logger.info(
                "configured RobotClient command path for %s "
                "(publish_to_robot=%s action_request_mode=%s "
                "tt_rtc_prefix_mode=%s tt_rtc_temporal_overlap_steps=%d "
                "initial_pose_sync=%s "
                "initial_pose_sync_min_duration_s=%g)",
                robot_type,
                self._publish_to_robot,
                self._action_request_mode,
                self._tt_rtc_prefix_mode,
                self._tt_rtc_temporal_overlap_steps,
                self._initial_pose_sync_enabled,
                self._initial_pose_sync_min_duration_s,
            )

    def deconfigure(self) -> None:
        with self._lock:
            self._arm_smoother = None
            self._arm_smoothing_last_time = None
            self._left_arm_deadband = None
            self._left_pose_hold = None
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
            self._tt_rtc_horizon = TT_RTC_HORIZON
            self._tt_rtc_action_dim = TT_RTC_ACTION_DIM
            self._tt_rtc_min_delay_steps = 0
            self._tt_rtc_max_delay_steps = TT_RTC_DELAY_STEPS
            self._initial_pose_sync_enabled = False
            self._initial_pose_sync_min_duration_s = 5.0
            self._initial_pose_sync_deferred_activation = False
            self._initial_pose_sync_effective_duration_s = None
            self._initial_pose_sync_in_progress = False
            self._initial_pose_sync_completed = False
            self._initial_pose_sync_deadline = None
            self._initial_pose_sync_hold_pending = False
            self._stream_handoff_pending = False
            self._rlt_enabled = False
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._action_policy_switch_error = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            self._tt_rtc_failure_reason = None
            self._tt_rtc_rlt_vla_reference = None
            self._processor = None
            self._generation += 1
            if self._robot is not None:
                self._robot.close()
                self._robot = None
            self._reset_request_latency_locked()
            self._mode_condition.notify_all()

    def start(self, publish_to_robot: Optional[bool] = None, *,
              initial_pose_sync: Optional[bool] = None,
              initial_pose_sync_duration_s: Optional[float] = None) -> float:
        with self._lock:
            if self._initial_pose_sync_hold_pending:
                raise RuntimeError(
                    "initial pose sync hold is still pending - STOP again first"
                )
            if initial_pose_sync is not None:
                duration = float(initial_pose_sync_duration_s or self._initial_pose_sync_min_duration_s)
                if not math.isfinite(duration) or not 1.0 <= duration <= 60.0:
                    raise ValueError("initial_pose_sync_duration_s must be between 1.0 and 60.0")
                self._initial_pose_sync_enabled = bool(initial_pose_sync)
                self._initial_pose_sync_min_duration_s = duration
                if not self._initial_pose_sync_in_progress:
                    self._initial_pose_sync_completed = False
            if publish_to_robot is not None:
                self._set_publish_to_robot_locked(bool(publish_to_robot))
            if self._initial_pose_sync_in_progress:
                if (
                    self._initial_pose_sync_deferred_activation
                    and self._initial_pose_sync_deadline is None
                    and self._initial_pose_sync_effective_duration_s is not None
                ):
                    effective_duration_s = (
                        self._initial_pose_sync_effective_duration_s
                    )
                    self._initial_pose_sync_deadline = (
                        time.monotonic() + effective_duration_s
                    )
                    logger.info(
                        "deferred initial pose sync activated: effective=%.3fs",
                        effective_duration_s,
                    )
                    return effective_duration_s
                raise RuntimeError("initial pose sync is already in progress")
            # A TT-RTC failure is latched until an explicit START/RESUME.  The
            # lifecycle command is the operator acknowledgement that permits a
            # fresh, bounded bootstrap request.
            self._tt_rtc_failure_reason = None
            if not self._running:
                self._tt_rtc_bootstrap_pending = True
            should_sync = (
                self._initial_pose_sync_enabled
                and self._publish_to_robot
                and not self._initial_pose_sync_completed
            )
            if not should_sync:
                self._running = True
                return 0.0
            if self._robot is None or self._processor is None:
                raise RuntimeError("LOAD first")
            robot = self._robot
            processor = self._processor
            task_instruction = self._task_instruction
            action_keys = list(self._action_keys)
            request_mode = self._action_request_mode
            minimum_duration_s = self._initial_pose_sync_min_duration_s
            maximum_duration_s = self._initial_pose_sync_max_duration_s
            generation = self._generation
            self._running = False
            self._initial_pose_sync_in_progress = False
            self._initial_pose_sync_deadline = None

        started_at = time.monotonic()
        response = self._request_initial_pose_action(task_instruction, request_mode)
        self._record_request_latency(time.monotonic() - started_at)
        chunk = self._decode_action_response(response)

        with self._lock:
            if (
                generation != self._generation
                or robot is not self._robot
                or processor is not self._processor
            ):
                raise RuntimeError("initial pose sync cancelled")
            processor.clear()
            self._tt_rtc_rlt_vla_reference = None
            try:
                effective_duration_s = robot.publish_initial_pose_sync(
                    chunk[0],
                    action_keys,
                    minimum_duration_s=minimum_duration_s,
                    maximum_duration_s=maximum_duration_s,
                )
            except Exception:
                self._running = False
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_deadline = None
                raise
            self._initial_pose_sync_in_progress = True
            self._initial_pose_sync_effective_duration_s = effective_duration_s
            self._initial_pose_sync_deadline = (
                None
                if self._initial_pose_sync_deferred_activation
                else time.monotonic() + effective_duration_s
            )
            self._running = True
            logger.info(
                "initial pose sync target published: minimum=%.3fs "
                "effective=%.3fs; discarded source chunk=%d",
                minimum_duration_s,
                effective_duration_s,
                int(getattr(response, "chunk_size", len(chunk))),
            )
            return effective_duration_s

    def pause(self, *, expected_generation: Optional[int] = None) -> bool:
        # STOP cannot acknowledge while a concurrent automatic hold is pending.
        with self._hold_lock:
            return self._pause_serialized(expected_generation=expected_generation)

    def _pause_serialized(self, *, expected_generation: Optional[int] = None) -> bool:
        robot = None
        action_keys: list[str] = []
        with self._lock:
            if expected_generation is not None and expected_generation != self._generation:
                return False
            self._reset_arm_smoothing_locked()
            self._record_rlt_event('buffer_cleared', reason='pause')
            should_hold = (
                self._publish_to_robot
                and self._robot is not None
            )
            if should_hold:
                robot = self._robot
                action_keys = list(self._action_keys)
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._tt_rtc_rlt_vla_reference = None
            if self._publish_to_robot:
                # Any later START/RESUME must bridge from fresh follower
                # feedback.  Intervention may have moved the robot far away
                # from the policy trajectory while this buffer was paused.
                self._stream_handoff_pending = True
                if self._initial_pose_sync_enabled:
                    self._initial_pose_sync_completed = False
            self._initial_pose_sync_deadline = None
            self._initial_pose_sync_effective_duration_s = None
            self._generation += 1
            self._active_action_policy_mode = ACTION_POLICY_BASE
            self._pending_action_policy_mode = None
            self._action_policy_switch_error = None
            self._active_rlt_robot_override = False
            self._pending_rlt_robot_override = None
            if expected_generation is None:
                self._tt_rtc_failure_reason = None
            if should_hold:
                self._initial_pose_sync_hold_pending = True
            else:
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_hold_pending = False
            self._mode_condition.notify_all()
        if robot is not None:
            try:
                hold_current_pose(robot, action_keys)
                logger.info("policy output paused; current pose hold confirmed")
            except Exception as error:
                with self._lock:
                    if robot is self._robot:
                        self._initial_pose_sync_in_progress = True
                        self._initial_pose_sync_hold_pending = True
                logger.error("failed to hold current pose during sync pause: %s", error)
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

        TT-RTC switches in both directions preserve the committed execution
        prefix. An in-flight refill may finish as a bridge before requesting
        the new route. Commit the route only after buffering its response.
        Other request modes invalidate old-mode requests and drain the queue.
        """
        try:
            target = normalize_action_policy_mode(action_policy_mode)
        except ValueError as error:
            return False, str(error)

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._mode_condition:
            if self._left_pose_hold is not None and target != ACTION_POLICY_BASE:
                return False, "left pose hold is only supported with the base policy"
            if not self._running or self._processor is None:
                return False, "inference is not running"
            if self._initial_pose_sync_in_progress:
                return False, "action policy cannot change during initial pose sync"
            if self._pending_action_policy_mode is not None:
                return False, "action policy switch already in progress"
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
                self._running
                and self._active_action_policy_mode == target
                and self._pending_action_policy_mode is None
            ):
                return True, f"{target.upper()} action active"
            if self._action_policy_switch_error:
                message = self._action_policy_switch_error
                self._action_policy_switch_error = None
                return False, message
            if self._pending_action_policy_mode == target:
                if self._tt_rtc_base_switch_pending():
                    message = "MLP->VLA switch timed out; inference paused"
                    self._latch_tt_rtc_failure_locked(message)
                    return False, message
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
        self._reset_arm_smoothing_locked()
        if self._processor is not None:
            self._processor.clear()
        self._tt_rtc_rlt_vla_reference = None
        self._stream_handoff_pending = bool(publish_to_robot)
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
            now = time.monotonic()
            if next_t <= now:
                # An overrun must not trigger an immediate second command.
                # Keep queued source actions intact; resume normal cadence.
                next_t = now + period
            self._shutdown.wait(next_t - now)

    def shutdown(self) -> None:
        self.pause()
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

            if self._initial_pose_sync_in_progress:
                deadline = self._initial_pose_sync_deadline
                if deadline is None:
                    if publish_to_robot:
                        idle = getattr(robot, "publish_idle_action", None)
                        if callable(idle):
                            try:
                                idle(action_keys)
                            except Exception as error:
                                logger.error(
                                    "failed to publish idle action while waiting "
                                    "for pose-sync activation: %s",
                                    error,
                                )
                    return
                if deadline is not None and time.monotonic() < deadline:
                    if publish_to_robot:
                        idle = getattr(robot, "publish_idle_action", None)
                        if callable(idle):
                            try:
                                idle(action_keys)
                            except Exception as error:
                                logger.error(
                                    "failed to publish idle action during pose sync: %s",
                                    error,
                                )
                    return
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_completed = True
                self._initial_pose_sync_deadline = None
                self._initial_pose_sync_effective_duration_s = None
                logger.info(
                    "initial pose sync complete; requesting a fresh action chunk"
                )

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
                planned_action = np.asarray(action, dtype=np.float64).copy()
                try:
                    self._observe_left_pose_hold_locked(action)
                    if self._left_arm_deadband is not None:
                        action = self._left_arm_deadband.step(action)
                    action = self._smooth_arm_command_locked(action)
                except Exception as error:
                    message = f"arm command conditioning failed; inference paused: {error}"
                    self._stream_handoff_pending = True
                    self._latch_tt_rtc_failure_locked(message)
                    return
                action = self._apply_gripper_command_threshold_locked(action)
                command_started_ns = time.time_ns()
                command_started_monotonic_ns = time.monotonic_ns()
                publish_success = False
                publish_error = ""
                if publish_to_robot:
                    try:
                        robot.publish_action(action, action_keys)
                        publish_success = True
                    except Exception as e:
                        publish_error = f"{type(e).__name__}: {e}"
                        logger.error("failed to publish robot action: %s", e)
                        self._latch_tt_rtc_failure_locked(publish_error)
                        return
                # Viewer transport must not delay this tick's robot command.
                preview = getattr(robot, "publish_action_preview", None)
                if callable(preview):
                    try:
                        preview(action, action_keys)
                    except Exception as e:
                        logger.warning("failed to publish action preview: %s", e)
                self._record_rlt_event(
                    "action_commanded" if publish_to_robot else "action_previewed",
                    arrays={
                        "planned_action": planned_action.reshape(1, -1),
                        "commanded_action": np.asarray(
                            action, dtype=np.float64
                        ).reshape(1, -1),
                    },
                    control_monotonic_ns=command_started_monotonic_ns,
                    command_started_ns=command_started_ns,
                    command_completed_ns=time.time_ns(),
                    publish_to_robot=bool(publish_to_robot),
                    publish_success=bool(publish_success),
                    publish_error=publish_error,
                    action_policy_mode=self._active_action_policy_mode,
                    action_request_mode=action_request_mode,
                    action_keys=action_keys,
                    remaining_control_samples=processor.buffer_size,
                )
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
            should_request = self._should_request_actions(
                processor,
                action_policy_mode=action_policy_mode,
            )
            rtc_prefix_actions = None
            rtc_execution_prefix_actions = None
            rtc_captured_actions = None
            rtc_prefix_captured_at = None
            if (
                should_request
                and action_request_mode == ACTION_REQUEST_MODE_TT_RTC
            ):
                rtc_execution_prefix_actions = self._capture_tt_rtc_prefix(
                    processor
                )
                try:
                    rtc_prefix_actions = self._select_tt_rtc_vla_prefix(
                        rtc_execution_prefix_actions
                    )
                except RuntimeError as error:
                    self._latch_tt_rtc_failure_locked(
                        f"TT-RTC VLA prefix unavailable; inference paused: {error}"
                    )
                    should_request = False
                rtc_captured_actions = self._capture_tt_rtc_actions(
                    processor,
                    action_policy_mode=action_policy_mode,
                )
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
                    rtc_captured_actions,
                    rtc_execution_prefix_actions,
                ),
                daemon=True,
            )
            self._request_thread.start()

    def _request_initial_pose_action(self, task_instruction: str, mode: str):
        """Request a disposable base-policy chunk for initial pose sync."""
        mode = normalize_action_request_mode(mode)
        if mode != ACTION_REQUEST_MODE_TT_RTC:
            return self._requester.get_action(
                task_instruction,
                action_policy_mode=ACTION_POLICY_BASE,
                action_request_mode=mode,
            )
        timeout_s = self._tt_rtc_remaining_timeout_s(
            prefix_captured_at=time.monotonic()
        )
        if timeout_s <= 0.0:
            raise RuntimeError("TT-RTC deadline expired before initial pose request")
        return self._requester.get_action(
            task_instruction,
            action_policy_mode=ACTION_POLICY_BASE,
            action_request_mode=mode,
            rtc_delay_steps=0,
            rtc_action_dim=self._tt_rtc_action_dim,
            rtc_prefix_action_list=[],
            timeout_s=timeout_s,
        )

    @staticmethod
    def _decode_action_response(response) -> np.ndarray:
        if not bool(getattr(response, "success", False)):
            raise RuntimeError(
                str(getattr(response, "message", "") or "get_action failed")
            )
        try:
            chunk_size = int(getattr(response, "chunk_size", 0))
            action_dim = int(getattr(response, "action_dim", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"get_action returned invalid chunk shape: {error}") from error
        if chunk_size <= 0 or action_dim <= 0:
            raise ValueError("get_action returned empty action list")
        try:
            data = np.asarray(getattr(response, "action_list", []), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"action list is not numeric: {error}") from error
        if data.size != chunk_size * action_dim:
            raise ValueError(
                f"action list size mismatch: {data.size} != "
                f"{chunk_size} * {action_dim}"
            )
        if not bool(np.isfinite(data).all()):
            raise ValueError("action list contains NaN or Inf")
        return data.reshape(chunk_size, action_dim)

    def _request_and_buffer(
        self,
        task_instruction: str,
        generation: int,
        action_request_mode: str = ACTION_REQUEST_MODE_ASYNC,
        action_policy_mode: str = ACTION_POLICY_BASE,
        rtc_prefix_actions: Optional[np.ndarray] = None,
        rtc_prefix_captured_at: Optional[float] = None,
        rtc_captured_actions: Optional[np.ndarray] = None,
        rtc_execution_prefix_actions: Optional[np.ndarray] = None,
    ) -> None:
        action_request_mode = normalize_action_request_mode(action_request_mode)
        action_policy_mode = normalize_action_policy_mode(action_policy_mode)
        with self._lock:
            if generation != self._generation or not self._running:
                return
        rtc_prefix = np.empty((0, self._tt_rtc_action_dim), dtype=np.float64)
        rtc_captured_queue = np.empty(
            (0, self._tt_rtc_action_dim),
            dtype=np.float64,
        )
        rtc_delay_steps = 0
        if action_request_mode == ACTION_REQUEST_MODE_TT_RTC:
            try:
                rtc_prefix = self._validate_tt_rtc_request_prefix(
                    rtc_prefix_actions,
                    action_dim=self._tt_rtc_action_dim,
                    max_delay_steps=self._tt_rtc_max_delay_steps,
                )
            except ValueError as error:
                self._handle_tt_rtc_failure(
                    generation,
                    f"TT-RTC request prefix invalid; inference paused: {error}",
                )
                return
            rtc_delay_steps = int(rtc_prefix.shape[0])
            try:
                rtc_execution_prefix = self._validate_tt_rtc_request_prefix(
                    (
                        rtc_prefix
                        if rtc_execution_prefix_actions is None
                        else rtc_execution_prefix_actions
                    ),
                    action_dim=self._tt_rtc_action_dim,
                    max_delay_steps=self._tt_rtc_max_delay_steps,
                )
                if rtc_execution_prefix.shape[0] != rtc_delay_steps:
                    raise ValueError(
                        "VLA prefix and execution prefix must have equal length"
                    )
                rtc_captured_queue = self._validate_tt_rtc_captured_queue(
                    rtc_execution_prefix,
                    (
                        rtc_execution_prefix
                        if rtc_captured_actions is None
                        else rtc_captured_actions
                    ),
                    action_dim=self._tt_rtc_action_dim,
                    action_horizon=self._tt_rtc_horizon,
                )
            except ValueError as error:
                self._handle_tt_rtc_failure(
                    generation,
                    f"TT-RTC captured queue invalid; inference paused: {error}",
                )
                return
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
            actual_overlap_steps = (
                rtc_captured_queue.shape[0] - rtc_delay_steps
            )
            expected_overlap_steps = (
                self._tt_rtc_temporal_overlap_steps
                if action_policy_mode == ACTION_POLICY_BASE
                and not tt_rtc_bootstrap
                else 0
            )
            if actual_overlap_steps != expected_overlap_steps:
                self._handle_tt_rtc_failure(
                    generation,
                    "TT-RTC temporal overlap capture is incomplete: "
                    f"expected {expected_overlap_steps}, got "
                    f"{actual_overlap_steps}",
                )
                return
            if rtc_delay_steps == 0 and not tt_rtc_bootstrap:
                self._handle_tt_rtc_failure(
                    generation, "TT-RTC execution buffer exhausted; explicit START/RESUME required",
                )
                return
            if (
                not tt_rtc_bootstrap
                and self._tt_rtc_prefix_mode in TT_RTC_FIXED_PREFIX_STEPS
                and rtc_delay_steps
                != TT_RTC_FIXED_PREFIX_STEPS[self._tt_rtc_prefix_mode]
            ):
                expected_steps = TT_RTC_FIXED_PREFIX_STEPS[
                    self._tt_rtc_prefix_mode
                ]
                expected_text = {6: "six", 8: "eight"}.get(
                    expected_steps,
                    str(expected_steps),
                )
                self._handle_tt_rtc_failure(
                    generation,
                    f"TT-RTC {self._tt_rtc_prefix_mode} requires exactly "
                    f"{expected_text} committed source actions, got "
                    f"{rtc_delay_steps}",
                )
                return
            if (
                not tt_rtc_bootstrap
                and not self._tt_rtc_min_delay_steps
                <= rtc_delay_steps
                <= self._tt_rtc_max_delay_steps
            ):
                self._handle_tt_rtc_failure(
                    generation,
                    "TT-RTC delay is outside the loaded model's trained range: "
                    f"{rtc_delay_steps} not in "
                    f"{self._tt_rtc_min_delay_steps}..{self._tt_rtc_max_delay_steps}",
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
        rtc_reference_chunk = None
        if (
            action_request_mode == ACTION_REQUEST_MODE_TT_RTC
            and action_policy_mode == ACTION_POLICY_RLT
        ):
            try:
                reference_chunk_size = int(
                    getattr(response, "rtc_reference_chunk_size", 0)
                )
                reference_data = np.asarray(
                    getattr(response, "rtc_reference_action_list", []),
                    dtype=np.float64,
                )
            except (TypeError, ValueError, OverflowError) as error:
                self._handle_action_request_failure(
                    generation,
                    action_policy_mode,
                    f"VLA reference action is invalid: {error}",
                    action_request_mode=action_request_mode,
                )
                return
            expected_reference_size = reference_chunk_size * action_dim
            if (
                reference_chunk_size != TT_RTC_RLT_CHUNK_SIZE
                or reference_data.size != expected_reference_size
                or not bool(np.isfinite(reference_data).all())
            ):
                self._handle_action_request_failure(
                    generation,
                    action_policy_mode,
                    "TT-RTC RLT response requires a finite raw VLA "
                    f"reference of shape ({TT_RTC_RLT_CHUNK_SIZE}, {action_dim})",
                    action_request_mode=action_request_mode,
                )
                return
            rtc_reference_chunk = reference_data.reshape(
                reference_chunk_size,
                action_dim,
            )
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
                            captured_actions=rtc_captured_queue,
                            delay_steps=rtc_delay_steps,
                            prefix_captured_at=float(rtc_prefix_captured_at),
                            action_policy_mode=action_policy_mode,
                            vla_reference_chunk=rtc_reference_chunk,
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
                handoff_applied = False
                handoff_duration_s = 0.0
                if self._stream_handoff_pending:
                    try:
                        chunk, handoff_duration_s = (
                            self._prepend_policy_handoff_locked(chunk)
                        )
                        handoff_applied = True
                    except Exception as error:
                        self._latch_stream_handoff_failure_locked(str(error))
                        return
                scheduled_start_delay_s = (
                    None
                    if (
                        action_request_mode == ACTION_REQUEST_MODE_SYNC
                        or handoff_applied
                    )
                    else latency_s + buffer_delay_s
                )
                try:
                    produced = self._processor.push_actions(
                        chunk,
                        scheduled_start_delay_s=scheduled_start_delay_s,
                        align=(
                            action_request_mode != ACTION_REQUEST_MODE_SYNC
                            and not handoff_applied
                        ),
                        boundary_blend_source_steps=(
                            2 if action_policy_mode == ACTION_POLICY_RLT else None
                        ),
                    )
                except Exception as error:
                    if handoff_applied:
                        self._stream_handoff_pending = True
                    self._latch_stream_handoff_failure_locked(str(error))
                    return
                if handoff_applied:
                    logger.info(
                        "policy stream handoff ramp buffered: duration=%.3fs "
                        "samples=%d",
                        handoff_duration_s,
                        len(chunk),
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
        *,
        action_dim: int = TT_RTC_ACTION_DIM,
        max_delay_steps: int = TT_RTC_DELAY_STEPS,
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
        if prefix.shape[0] > max_delay_steps:
            raise ValueError(
                f"prefix may contain at most {max_delay_steps} actions; "
                f"got {prefix.shape[0]}"
            )
        if not bool(np.isfinite(prefix).all()):
            raise ValueError("prefix contains NaN or Inf")
        return prefix.copy()

    @staticmethod
    def _validate_tt_rtc_captured_queue(
        prefix_actions: np.ndarray,
        captured_actions: np.ndarray,
        *,
        action_dim: int,
        action_horizon: int,
    ) -> np.ndarray:
        """Validate the committed prefix plus optional future overlap."""

        prefix = np.asarray(prefix_actions, dtype=np.float64)
        captured = np.asarray(captured_actions, dtype=np.float64)
        if captured.ndim != 2:
            raise ValueError(
                f"captured actions must be 2D (T, D); got {captured.shape}"
            )
        if captured.shape[1] != action_dim:
            raise ValueError(
                f"captured action_dim must be {action_dim}; "
                f"got {captured.shape[1]}"
            )
        if captured.shape[0] > action_horizon:
            raise ValueError(
                f"captured queue may contain at most {action_horizon} actions; "
                f"got {captured.shape[0]}"
            )
        if captured.shape[0] < prefix.shape[0]:
            raise ValueError("captured queue is shorter than the committed prefix")
        if not bool(np.isfinite(captured).all()):
            raise ValueError("captured actions contain NaN or Inf")
        if prefix.shape[0] and not bool(
            np.array_equal(captured[: prefix.shape[0]], prefix)
        ):
            raise ValueError("captured queue does not start with the committed prefix")
        return captured.copy()

    def _commit_tt_rtc_chunk_locked(
        self,
        *,
        chunk: np.ndarray,
        captured_prefix: np.ndarray,
        captured_actions: np.ndarray,
        delay_steps: int,
        prefix_captured_at: float,
        action_policy_mode: str,
        vla_reference_chunk: Optional[np.ndarray] = None,
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
        remaining_actions = processor.peek_actions()
        consumed_steps = self._tt_rtc_consumed_prefix_steps(
            captured_actions,
            remaining_actions,
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
            if (
                vla_reference_chunk is None
                or vla_reference_chunk.shape
                != (TT_RTC_RLT_CHUNK_SIZE, self._tt_rtc_action_dim)
                or not bool(np.isfinite(vla_reference_chunk).all())
            ):
                return False, (
                    "TT-RTC MLP response did not include its 10-step raw "
                    "VLA reference"
                )
            postfix = chunk

        if (
            action_policy_mode == ACTION_POLICY_RLT
            and not self._stream_handoff_pending
        ):
            boundary_anchor = (
                remaining_actions[-1]
                if len(remaining_actions)
                else getattr(processor, "last_action", None)
            )
            postfix = self._smooth_rlt_chunk_boundary(
                postfix,
                boundary_anchor,
                source_steps=2,
            )

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

        overlap_steps = captured_actions.shape[0] - delay_steps
        if overlap_steps and action_policy_mode != ACTION_POLICY_BASE:
            return False, "TT-RTC temporal overlap is only valid for base GR00T"
        if overlap_steps and consumed_steps > delay_steps:
            return False, (
                "TT-RTC response arrived after the committed prefix; "
                "the temporal overlap already started executing"
            )
        if overlap_steps > postfix.shape[0]:
            return False, "TT-RTC overlap exceeds the generated postfix"

        handoff_applied = False
        handoff_duration_s = 0.0
        if self._stream_handoff_pending:
            try:
                postfix, handoff_duration_s = (
                    self._prepend_policy_handoff_locked(postfix)
                )
                handoff_applied = True
            except Exception as error:
                return False, f"policy handoff rejected: {error}"
        hold_candidate = None
        hold_events = []
        try:
            if self._left_pose_hold is not None and not handoff_applied:
                if overlap_steps or action_policy_mode != ACTION_POLICY_BASE:
                    return False, "left pose hold requires base TT-RTC without overlap"
                anchor = (remaining_actions[-1] if len(remaining_actions)
                          else getattr(processor, "last_action", None))
                postfix, hold_candidate, hold_events = self._left_pose_hold.plan(
                    postfix, anchor, time.monotonic(), TT_RTC_SOURCE_HZ,
                    allow_entry=bool(np.all(np.abs(
                        remaining_actions[:, self._left_pose_hold.indices]-LEFT_HOLD_REFERENCE
                    ) <= self._left_pose_hold.config.entry_rad)) if len(remaining_actions) else True)
            if overlap_steps:
                if handoff_applied:
                    return False, (
                        "policy handoff cannot replace an in-flight temporal overlap"
                    )
                old_overlap = captured_actions[delay_steps:]
                weights = self._tt_rtc_overlap_blend_weights(overlap_steps)
                blended_overlap = (
                    (1.0 - weights[:, None]) * old_overlap
                    + weights[:, None] * postfix[:overlap_steps]
                )
                remaining_committed = captured_actions[
                    consumed_steps:delay_steps
                ]
                replacement = np.concatenate(
                    (
                        remaining_committed,
                        blended_overlap,
                        postfix[overlap_steps:],
                    ),
                    axis=0,
                )
                replace = getattr(processor, "replace_queued_actions", None)
                if not callable(replace):
                    return False, (
                        "TT-RTC action processor cannot atomically replace overlap"
                    )
                produced = replace(remaining_actions, replacement)
            else:
                produced = processor.push_actions(
                    postfix,
                    scheduled_start_delay_s=None,
                    align=False,
                )
        except Exception as error:
            if handoff_applied:
                self._stream_handoff_pending = True
            return False, f"policy handoff enqueue failed: {error}"
        if hold_candidate is not None:
            self._left_pose_hold = hold_candidate
            for source_step, event in hold_events:
                measured_error = (None if hold_candidate.last_feedback is None else
                                  np.round(hold_candidate.last_feedback-LEFT_HOLD_REFERENCE, 6).tolist())
                logger.info(
                    "Left pose hold queued: event=%s postfix_step=%d remaining_prefix=%d "
                    "command_left_rad=%s measured_error_rad=%s",
                    event, source_step, len(remaining_actions),
                    np.round(postfix[source_step, hold_candidate.indices], 7).tolist(),
                    measured_error)
        if handoff_applied:
            logger.info(
                "TT-RTC policy handoff ramp buffered: duration=%.3fs "
                "source_samples=%d",
                handoff_duration_s,
                len(postfix),
            )
        if action_policy_mode == ACTION_POLICY_RLT:
            self._tt_rtc_rlt_vla_reference = vla_reference_chunk.copy()
        else:
            self._tt_rtc_rlt_vla_reference = None
        logger.info(
            "buffered TT-RTC postfix: policy=%s delay=%d source=%d "
            "produced=%d capture_to_enqueue=%.3fs consumed_prefix=%d "
            "remaining_prefix=%d overlap=%d timeline=%s",
            action_policy_mode,
            delay_steps,
            chunk.shape[0],
            produced,
            elapsed_since_capture_s,
            consumed_steps,
            remaining_actions.shape[0],
            overlap_steps,
            getattr(processor, "timing_stats", {}),
        )
        return True, ""

    def _configure_left_pose_hold_locked(self, model_path, rlt_enabled) -> None:
        self._left_pose_hold = None
        if not enabled_for_model(model_path):
            return
        if self._robot_type != "ffw_sg2_rev1" or "arm_left" not in self._action_keys:
            raise ValueError("left pose hold requires SG2 left arm")
        if (self._action_request_mode != ACTION_REQUEST_MODE_TT_RTC
                or self._tt_rtc_temporal_overlap_steps
                or self._arm_smoother is not None or self._left_arm_deadband is not None):
            raise ValueError("left pose hold requires TT-RTC, no overlap or other arm filters")
        if rlt_enabled:
            # RLT's physical-action decoder owns its bundle's left-arm hold.
            # Keep the base-only target-pose filter out of switchable sessions:
            # applying both would alter the Actor's committed action prefix.
            logger.info("Legacy left target-pose hold disabled for RLT session; "
                        "the RLT bundle owns MLP left-arm behavior")
            return
        layout = self._robot.get_action_joint_limits(self._action_keys)
        names = [item['name'] for item in layout]
        wanted = [f"arm_l_joint{i}" for i in range(1, 8)]
        if any(names.count(name) != 1 for name in wanted):
            raise ValueError("left pose hold requires seven unique named left joints")
        indices = [names.index(name) for name in wanted]
        config = HoldConfig.from_environment()
        if any(config.release_speed_rad_s > layout[i]['velocity'] for i in indices):
            raise ValueError("left pose hold release speed exceeds URDF velocity")
        self._left_pose_hold = LeftPoseHold(
            indices, [layout[i]['lower'] for i in indices],
            [layout[i]['upper'] for i in indices], config)
        logger.info("Model-scoped left pose hold enabled: model=%s config=%s", model_path, config)

    def _observe_left_pose_hold_locked(self, action) -> None:
        hold = self._left_pose_hold
        if hold is None:
            return
        now = time.monotonic()
        if hold.last_observed is not None and 0 <= now-hold.last_observed < .05:
            return
        # The SDK reads cached joint feedback and rejects missing/stale states;
        # this does not decode cameras or issue robot commands/services.
        feedback = (self._robot.get_current_action_positions(self._action_keys)
                    if self._publish_to_robot else action)
        hold.observe(feedback, now)

    def _configure_left_arm_deadband_locked(self) -> None:
        self._left_arm_deadband = None
        if (self._robot_type != "ffw_sg2_rev1" or "arm_left" not in self._action_keys
                or not left_arm_deadband_enabled()):
            return
        layout = self._robot.get_action_joint_limits(self._action_keys)
        wanted = {f"arm_l_joint{i}" for i in range(1, 8)}
        indices = [i for i, joint in enumerate(layout) if joint['name'] in wanted]
        if len(indices) != 7 or {layout[i]['name'] for i in indices} != wanted:
            raise ValueError("left arm deadband requires exactly seven named left arm joints")
        radius = float(os.environ.get("CYCLO_LEFT_ARM_DEADBAND_RAD", "0.003"))
        self._left_arm_deadband = PositionDeadband(
            indices, radius, lower=[layout[i]['lower'] for i in indices],
            upper=[layout[i]['upper'] for i in indices],
        )
        logger.info("Left arm position deadband enabled: radius=%.6frad joints=%s",
                    radius, [layout[i]['name'] for i in indices])

    def _configure_arm_smoothing_locked(self) -> None:
        self._arm_smoother = None
        self._arm_smoothing_last_time = None
        if self._robot_type != "ffw_sg2_rev1" or not smoothing_enabled():
            return
        layout = self._robot.get_action_joint_limits(self._action_keys)
        names = {f"arm_{side}_joint{i}" for side in ("l", "r") for i in range(1, 8)}
        indices = [i for i, item in enumerate(layout) if item['name'] in names]
        if not indices:
            return
        config = ArmSmoothingConfig.from_environment()
        if config.lookahead_s and not isinstance(self._processor, TTActionTimeline):
            raise ValueError("arm trajectory lookahead requires a TT-RTC timeline")
        if any(config.max_velocity_rad_s > layout[i]['velocity'] for i in indices):
            raise ValueError("arm smoothing velocity exceeds configured URDF velocity")
        self._arm_smoother = ArmCommandSmoother(
            indices, config=config,
            lower=[layout[i]['lower'] for i in indices],
            upper=[layout[i]['upper'] for i in indices],
        )
        logger.info("SG2 independent arm smoothing enabled: %s", config)

    def _reset_arm_smoothing_locked(self) -> None:
        if self._left_pose_hold is not None:
            self._left_pose_hold.clear()
        if self._left_arm_deadband is not None:
            self._left_arm_deadband.clear()
        if self._arm_smoother is not None:
            self._arm_smoother.clear()
        self._arm_smoothing_last_time = None

    def _smooth_arm_command_locked(self, action: np.ndarray) -> np.ndarray:
        smoother = self._arm_smoother
        if smoother is None:
            return action
        now = time.monotonic()
        nominal_dt = 1.0 / self._processor.output_hz
        if not smoother.initialized:
            # Physical execution must start from fresh measured feedback.
            # Preview starts from its first planned pose and sends no motion.
            anchor = (self._robot.get_current_action_positions(self._action_keys)
                      if self._publish_to_robot else action)
            smoother.reset(anchor)
            self._arm_smoothing_last_time = now
            command = np.asarray(action, dtype=np.float64).copy()
            command[smoother.indices] = np.asarray(anchor)[smoother.indices]
            return command
        elapsed = now - self._arm_smoothing_last_time
        if elapsed <= 0:
            raise RuntimeError("control clock did not advance")
        if elapsed > 0.2:
            raise RuntimeError("control command gap exceeded 200ms; fresh START/RESUME required")
        # Never catch up a late control tick with one large position jump.
        target = action
        if smoother.config.lookahead_s:
            future = self._processor.preview_action(smoother.config.lookahead_s)
            if future is None:
                raise RuntimeError("committed arm trajectory preview is unavailable")
            # Grippers and other modalities retain their current source time.
            target = np.asarray(action, dtype=np.float64).copy()
            target[smoother.indices] = future[smoother.indices]
        command = smoother.step(target, min(elapsed, nominal_dt))
        self._arm_smoothing_last_time = now
        return command

    def _apply_gripper_command_threshold_locked(
        self, action: np.ndarray,
    ) -> np.ndarray:
        """Snap SG2 grippers after interpolation, for both preview and command.

        This is a per-command rule, not a latch: values below 0.7 can reopen
        the gripper. Keep the model-rate TT-RTC prefix untouched.
        """
        if self._robot_type != "ffw_sg2_rev1":
            return action
        # SG2 action group widths follow ffw_sg2_rev1_config.yaml. Resolve
        # offsets in selected-key order, including single-arm policies.
        widths = {
            "arm_left": 8, "arm_right": 8, "head": 2,
            "lift": 1, "mobile": 3, "odometry": 3,
        }
        if any(key not in widths for key in self._action_keys):
            return action
        command = np.asarray(action, dtype=np.float64)
        if command.ndim != 1 or command.size != sum(
            widths[key] for key in self._action_keys
        ):
            return action
        result = command.copy()
        offset = 0
        for key in self._action_keys:
            if key in ("arm_left", "arm_right"):
                index = offset + 7
                value = result[index]
                if math.isfinite(value) and value >= GRIPPER_COMMAND_THRESHOLD:
                    result[index] = GRIPPER_COMMAND_VALUE
            offset += widths[key]
        return result

    @staticmethod
    def _smooth_rlt_chunk_boundary(
        chunk: np.ndarray,
        anchor: Optional[np.ndarray],
        source_steps: int = 2,
    ) -> np.ndarray:
        """Bridge only the MLP chunk boundary and rejoin it exactly."""

        if anchor is None or len(chunk) == 0:
            return chunk
        smoothed = chunk.copy()
        blend_steps = min(max(1, int(source_steps)), len(smoothed))
        blend_target = smoothed[blend_steps - 1].copy()
        for index in range(blend_steps):
            progress = (index + 1) / blend_steps
            weight = progress * progress * (3.0 - 2.0 * progress)
            smoothed[index] = (
                (1.0 - weight) * anchor + weight * blend_target
            )
        # A constant 0.9 gripper command is an explicit hold policy, not an
        # arm waypoint. Do not weaken it during the first boundary sample.
        if smoothed.shape[1] >= 16 and np.all(np.isclose(chunk[:, 15], 0.9, atol=1e-6, rtol=0)):
            smoothed[:, 15] = chunk[:, 15]
        return smoothed

    @staticmethod
    def _tt_rtc_overlap_blend_weights(overlap_steps: int) -> np.ndarray:
        """Cosine cross-fade from the established plan to the fresh plan."""

        if overlap_steps <= 0:
            return np.empty((0,), dtype=np.float64)
        phase = np.arange(1, overlap_steps + 1, dtype=np.float64)
        phase /= overlap_steps + 1
        return 0.5 - 0.5 * np.cos(np.pi * phase)

    def _prepend_policy_handoff_locked(
        self,
        chunk: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Prepend a velocity-bounded ramp from live feedback to chunk[0]."""
        if not self._stream_handoff_pending:
            return chunk, 0.0
        if self._robot is None:
            raise RuntimeError("RobotClient is unavailable")
        if chunk.ndim != 2 or len(chunk) == 0:
            raise ValueError("policy handoff requires a non-empty action chunk")
        ramp, duration_s = self._robot.build_policy_handoff_ramp(
            chunk[0],
            list(self._action_keys),
            sample_hz=self._inference_hz,
            maximum_duration_s=self._initial_pose_sync_max_duration_s,
        )
        combined = np.concatenate((ramp, chunk[1:]), axis=0)
        self._stream_handoff_pending = False
        return combined, duration_s

    def _latch_stream_handoff_failure_locked(self, reason: str) -> None:
        """Fail closed when a current-pose handoff cannot be constructed."""
        message = f"policy handoff rejected; inference paused: {reason}"
        self._stream_handoff_pending = True
        self._latch_tt_rtc_failure_locked(message)

    def _record_rlt_event(
        self,
        event,
        *,
        recording_id=None,
        arrays=None,
        **metadata,
    ):
        if self._rlt_trace is not None:
            self._rlt_trace.submit(dict(
                metadata, event=event, control_time_ns=time.time_ns(),
                timebase='unix_wall_clock_ns', generation=self._generation,
                execution_verified=False,
            ), arrays, recording_id=recording_id)

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
            "TT-RTC action policy switch completed: %s",
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
        self._reset_arm_smoothing_locked()
        self._record_rlt_event('buffer_cleared', reason=reason)
        message = str(reason or "TT-RTC request failed; inference paused")
        self._running = False
        if self._processor is not None:
            self._processor.clear()
        self._tt_rtc_rlt_vla_reference = None
        self._tt_rtc_failure_reason = message
        self._action_policy_switch_error = message
        self._active_action_policy_mode = ACTION_POLICY_BASE
        self._pending_action_policy_mode = None
        self._active_rlt_robot_override = False
        self._pending_rlt_robot_override = None
        self._generation += 1

        if self._publish_to_robot and self._robot is not None:
            self._initial_pose_sync_hold_pending = True
            threading.Thread(
                target=self.pause, kwargs={"expected_generation": self._generation}, daemon=True,
            ).start()
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

    def _tt_rtc_base_switch_pending(self) -> bool:
        """An outstanding MLP->VLA switch must fail closed on timeout."""
        return (
            self._action_request_mode == ACTION_REQUEST_MODE_TT_RTC
            and self._active_action_policy_mode == ACTION_POLICY_RLT
            and self._pending_action_policy_mode == ACTION_POLICY_BASE
        )

    def _should_request_actions(
        self,
        processor: ActionChunkProcessor,
        action_policy_mode: Optional[str] = None,
    ) -> bool:
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
            return processor.buffer_size <= self._tt_rtc_refill_threshold(
                action_policy_mode=action_policy_mode,
            )
        return processor.buffer_size < self._refill_threshold(processor)

    def _tt_rtc_prefix_steps(self) -> int:
        """Return the clean committed-prefix length for the next request."""

        fixed_steps = TT_RTC_FIXED_PREFIX_STEPS.get(self._tt_rtc_prefix_mode)
        if fixed_steps is not None:
            return fixed_steps
        if self._request_latency_ema_s is None:
            return self._tt_rtc_max_delay_steps
        return min(self._tt_rtc_max_delay_steps, max(
            max(1, self._tt_rtc_min_delay_steps), math.ceil(
            (
                self._request_latency_ema_s
                + max(0.0, self._refill_margin_s)
            )
            * TT_RTC_SOURCE_HZ,
        )))

    def _tt_rtc_overlap_steps_for_policy(
        self,
        action_policy_mode: Optional[str],
    ) -> int:
        mode = action_policy_mode or self._active_action_policy_mode
        if (
            mode != ACTION_POLICY_BASE
            or self._stream_handoff_pending
        ):
            return 0
        return self._tt_rtc_temporal_overlap_steps

    def _tt_rtc_refill_threshold(
        self,
        action_policy_mode: Optional[str] = None,
    ) -> int:
        """Return committed prefix plus same-time future overlap."""

        return (
            self._tt_rtc_prefix_steps()
            + self._tt_rtc_overlap_steps_for_policy(action_policy_mode)
        )

    def _capture_tt_rtc_prefix(self, processor) -> np.ndarray:
        """Copy only the clean committed prefix from the execution queue."""

        actions = processor.peek_actions(count=self._tt_rtc_prefix_steps())
        if actions.size == 0:
            # A freshly cleared TTActionTimeline has not learned its action
            # dimension yet.  This is expected immediately after the
            # disposable initial-pose request, so preserve the model contract
            # instead of exposing the timeline's internal (0, 0) sentinel.
            return np.empty((0, self._tt_rtc_action_dim), dtype=np.float64)
        return actions

    def _select_tt_rtc_vla_prefix(
        self,
        execution_prefix: np.ndarray,
    ) -> np.ndarray:
        """Use the commands still queued for execution as the next VLA prefix.

        In RLT mode those commands are the unconsumed tail of the 10-step MLP
        output M.  The raw VLA reference R=VLA[d:d+10] remains an MLP input and
        trace artifact; it must not replace the commands the robot will execute.
        """

        return np.asarray(execution_prefix, dtype=np.float64).copy()

    def _capture_tt_rtc_actions(
        self,
        processor,
        *,
        action_policy_mode: Optional[str] = None,
    ) -> np.ndarray:
        """Copy the committed prefix and optional uncommitted overlap."""

        actions = processor.peek_actions(
            count=self._tt_rtc_refill_threshold(
                action_policy_mode=action_policy_mode,
            )
        )
        if actions.size == 0:
            return np.empty((0, self._tt_rtc_action_dim), dtype=np.float64)
        return actions

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
