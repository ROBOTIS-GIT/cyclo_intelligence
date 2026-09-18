#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Robot-facing control loop owned by the central Policy Runtime."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

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
from inference_context.execution import ExecutionContext  # noqa: E402
from .execution_feedback import ExecutionFeedback  # noqa: E402
from .step_schedule import StepSchedule  # noqa: E402
from inference_context.contract import ExecutionContract  # noqa: E402
from inference_context.timing import action_latency  # noqa: E402


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
        fault_callback: Optional[Callable[[str, bool], None]] = None,
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
        self._input_execution_contract = ExecutionContract()
        self._fault_callback = fault_callback

        self._lock = threading.RLock()
        self._robot: Optional[RobotClient] = None
        self._processor: Optional[ActionChunkProcessor] = None
        self._feedback = None
        self._step_schedule = None
        self._initial_action_timeout_s = None
        self._observation_warmup_timeout_s = None
        self._preparation_needed = False
        self._preparation_active = False
        self._feedback_io_lock = threading.Lock()
        self._feedback_dirty = False
        self._feedback_thread = None
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
        self._request_reserved = False
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
        execution_context: ExecutionContext | None = None,
        execution_contract: ExecutionContract | None = None,
    ) -> None:
        duration_s = float(initial_pose_sync_duration_s)
        if not math.isfinite(duration_s) or not 1.0 <= duration_s <= 60.0:
            raise ValueError(
                "initial_pose_sync_duration_s must be between 1.0 and 60.0"
            )
        contract = execution_contract or ExecutionContract()
        if contract.is_step and execution_context is None:
            raise ValueError("step execution requires a contextual Worker session")
        if contract.observation_warmup_timeout_s is not None and execution_context is None:
            raise ValueError("observation warmup requires a contextual Worker session")
        if contract.is_step and not publish_to_robot:
            raise ValueError("step execution requires command receipts; preview-only is unsupported")
        if contract.feedback_schema == 2 and (execution_context is None or execution_context.feedback_schema != 2):
            raise ValueError("input pipeline requires feedback schema 2")
        if contract.request_after not in {"prediction_success", "plan_accepted"} and not publish_to_robot:
            raise ValueError("execution prerequisite requires command receipts; preview-only is unsupported")
        with self._lock:
            self.deconfigure()
            self._input_execution_contract = contract
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
                subscribe_images=False,
                subscribe_state=bool(publish_to_robot),
                subscribe_sensors=False,
            )
            processing = dict(
                inference_hz=self._inference_hz,
                control_hz=self._control_hz,
                chunk_align_window_s=self._chunk_align_window_s,
                postprocess=self._postprocess_actions,
                target_chunk_size=self._target_chunk_size,
                alignment_mode=self._alignment_mode,
            )
            self._step_schedule = StepSchedule(self._inference_hz) if contract.is_step else None
            self._initial_action_timeout_s = contract.initial_action_timeout_s
            self._observation_warmup_timeout_s = contract.observation_warmup_timeout_s
            if self._observation_warmup_timeout_s is not None:
                base_timeout = self._initial_action_timeout_s or self._requester.get_action_timeout_s
                self._initial_action_timeout_s = base_timeout + self._observation_warmup_timeout_s
            self._preparation_needed = self._initial_action_timeout_s is not None
            self._preparation_active = False
            if self._step_schedule:
                # A public select_action result is already one model step.
                processing.update(postprocess=False, target_chunk_size=None)
            self._feedback = (
                ExecutionFeedback(execution_context, pending_command_count=contract.pending_command_count, **processing)
                if execution_context is not None else None
            )
            self._processor = self._feedback.buffer if self._feedback else ActionChunkProcessor(**processing)
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
            print(f"[policy-runtime] {config_message}", flush=True)

    def deconfigure(self) -> None:
        with self._lock:
            self._running = False
            self._task_instruction = ""
            self._action_keys = []
            self._publish_to_robot = False
            self._action_request_mode = self._default_action_request_mode
            self._input_execution_contract = ExecutionContract()
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
            self._feedback = None
            self._step_schedule = None
            self._initial_action_timeout_s = None
            self._observation_warmup_timeout_s = None
            self._preparation_needed = False
            self._preparation_active = False
            self._feedback_dirty = False
            self._generation += 1
            if self._robot is not None:
                self._robot.close()
                self._robot = None
            self._reset_request_latency_locked()

    def start(self, publish_to_robot: Optional[bool] = None) -> bool:
        with self._lock:
            if self._initial_action_timeout_s is not None and self.prediction_pending():
                raise RuntimeError("previous prediction is still finishing; retry START after it completes")
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
                if self._feedback:
                    self._feedback.phase = "running"
                return False
            if self._robot is None or self._processor is None:
                raise RuntimeError("LOAD first")
            generation = self._generation
            self._running = False
            self._initial_pose_sync_in_progress = False
            self._initial_pose_sync_deadline = None
            if self._feedback:
                self._feedback.phase = "syncing"
            if self._initial_action_timeout_s is not None:
                self._request_reserved = True
                self._preparation_active = True
                self._request_thread = threading.Thread(
                    target=self._prepare_pose_sync, args=(generation,), daemon=True,
                )
                try:
                    self._request_thread.start()
                except Exception:
                    self._request_reserved = False
                    self._preparation_active = False
                    raise
                return True
        return self._start_pose_sync(generation)

    def preparing(self) -> bool:
        with self._lock:
            return self._preparation_active or (
                self._running and self._preparation_needed and not self._initial_pose_sync_in_progress
            )

    def prediction_pending(self) -> bool:
        with self._lock:
            return self._request_reserved or (
                self._request_thread is not None and self._request_thread.is_alive()
            )

    def _prepare_pose_sync(self, generation):
        try:
            self._start_pose_sync(generation)
        except Exception as exc:
            self._report_fault(f"initial prediction failed: {exc}", generation=generation)
        finally:
            with self._lock:
                self._request_reserved = False
                if generation == self._generation:
                    self._preparation_active = False

    def _start_pose_sync(self, generation):
        with self._lock:
            if generation != self._generation:
                raise RuntimeError("initial pose sync cancelled")
            robot, processor = self._robot, self._processor
            task_instruction = self._task_instruction
            action_keys = list(self._action_keys)
            duration_s = self._initial_pose_sync_duration_s
            timeout_s = self._initial_action_timeout_s

        started_at = time.monotonic()
        try:
            response = self._get_action_with_feedback(task_instruction, generation, timeout_s=timeout_s)
            latency_s = time.monotonic() - started_at
            if self._observation_warmup_timeout_s is None:
                self._record_request_latency(latency_s)
            chunk = self._decode_action_response(response)
            if self._observation_warmup_timeout_s is not None:
                latency_s = action_latency(response.capabilities_json, latency_s)
                self._record_request_latency(latency_s)
            if self._step_schedule and len(chunk) != 1:
                raise ValueError("step pose sync requires exactly one action")
        except Exception:
            with self._lock:
                if self._feedback and generation == self._generation:
                    self._clear_plan_locked("initial pose request failed", "error")
            self._schedule_feedback_update()
            raise

        with self._lock:
            if (
                generation != self._generation
                or robot is not self._robot
                or processor is not self._processor
            ):
                raise RuntimeError("initial pose sync cancelled")
            self._clear_plan_locked("initial pose sync target", "syncing")
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
                if self._feedback:
                    self._clear_plan_locked("initial pose publication failed", "error")
                    self._schedule_feedback_update()
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
        return self._pause("paused", "pause")

    def _pause(self, phase, reason) -> bool:
        robot = None
        action_keys: list[str] = []
        with self._lock:
            should_hold = (
                (
                    self._initial_pose_sync_in_progress
                    or self._initial_pose_sync_hold_pending
                    or self._feedback is not None
                )
                and self._publish_to_robot
                and self._robot is not None
            )
            if should_hold:
                robot = self._robot
                action_keys = list(self._action_keys)
            self._running = False
            self._preparation_active = False
            self._clear_plan_locked(reason, "error" if should_hold else phase)
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
                self._schedule_feedback_update()
                return False
            with self._lock:
                if robot is not self._robot:
                    return False
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_hold_pending = False
                if self._feedback:
                    self._feedback.phase = phase
        self._schedule_feedback_update()
        return True

    def stop(self) -> bool:
        return self._pause("stopped", "stop")

    def set_requester(self, requester) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("cannot switch policy worker while running")
            self._requester = requester

    def set_fault_callback(
        self,
        callback: Optional[Callable[[str, bool], None]],
    ) -> None:
        with self._lock:
            self._fault_callback = callback

    def configuration_snapshot(self) -> dict:
        """Return the normalized LOAD-time settings currently in use."""
        with self._lock:
            return {
                "action_request_mode": self._action_request_mode,
                "control_hz": int(self._control_hz),
                "inference_hz": int(self._inference_hz),
                "chunk_align_window_s": float(self._chunk_align_window_s),
                "initial_pose_sync": self._initial_pose_sync_enabled,
                "initial_pose_sync_duration_s": self._initial_pose_sync_duration_s,
            }

    def emergency_stop(self, reason: str, *, expected_generation=None) -> bool | None:
        """Clear queued commands and hold the latest real-robot joint pose."""
        robot = None
        action_keys: list[str] = []
        with self._lock:
            if expected_generation is not None and expected_generation != self._generation:
                return None
            self._running = False
            self._preparation_active = False
            self._clear_plan_locked(reason[:1024], "error")
            self._generation += 1
            self._initial_pose_sync_deadline = None
            should_hold = (
                self._publish_to_robot
                and self._robot is not None
            )
            if should_hold:
                robot = self._robot
                action_keys = list(self._action_keys)
                self._initial_pose_sync_hold_pending = True
            else:
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_hold_pending = False
        logger.error("policy runtime safety stop: %s", reason)
        self._schedule_feedback_update()
        if robot is None:
            return True
        try:
            robot.publish_current_pose_hold(action_keys, duration_s=0.1)
        except Exception as e:
            logger.error("policy runtime current-pose hold failed: %s", e)
            return False
        with self._lock:
            if robot is not self._robot:
                return False
            self._initial_pose_sync_in_progress = False
            self._initial_pose_sync_hold_pending = False
        return True

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
        if self._step_schedule and not publish_to_robot:
            raise ValueError("step execution requires command receipts; preview-only is unsupported")
        if self._publish_to_robot == publish_to_robot:
            return
        if self._robot is not None:
            set_state_subscription = getattr(
                self._robot,
                "set_state_subscription",
                None,
            )
            if callable(set_state_subscription):
                set_state_subscription(publish_to_robot)
        self._publish_to_robot = publish_to_robot
        self._clear_plan_locked("publish mode changed", "running" if self._running else "ready")
        self._generation += 1

    def set_task_instruction(self, task_instruction: str) -> None:
        with self._lock:
            instruction = task_instruction or ""
            if self._feedback and instruction != self._task_instruction:
                if self._initial_pose_sync_in_progress or self._initial_pose_sync_hold_pending:
                    raise RuntimeError("stop pose sync before changing a contextual instruction")
                self._clear_plan_locked("instruction changed", self._feedback.phase)
                self._generation += 1
            self._task_instruction = instruction

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
        fault = self._tick_once()
        if fault is not None:
            reason, generation = fault
            self._report_fault(reason, generation=generation)

    def _tick_once(self):
        fault = None
        request_thread = None
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
                                if self._feedback:
                                    self._running = False
                                    return f"failed to publish sync idle action: {e}", generation
                    return
                self._initial_pose_sync_in_progress = False
                self._initial_pose_sync_completed = True
                self._initial_pose_sync_deadline = None
                if self._feedback:
                    self._clear_plan_locked("initial pose sync complete", "running")
                logger.info(
                    "initial pose sync complete; requesting a fresh action chunk"
                )

            command = None
            if self._feedback:
                command = processor.take(repeat_last=True) if self._step_schedule else processor.take()
                action = None if command is None else np.asarray(command.values, dtype=command.dtype)
            else:
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
                        if self._feedback:
                            expired = self._step_schedule and self._step_schedule.velocity_expired(
                                command.prediction_id, time.monotonic(),
                            )
                            options = {"zero_twist": True} if expired else {}
                            emitted = robot.publish_action_with_receipt(action, action_keys, **options)
                            processor.finish(command.command_id, status="published", emitted_values=emitted,
                                             reason="step period elapsed" if expired else "")
                            if self._step_schedule:
                                self._step_schedule.published(command.prediction_id, time.monotonic())
                        else:
                            robot.publish_action(action, action_keys)
                    except Exception as e:
                        logger.error("failed to publish robot action: %s", e)
                        if self._feedback:
                            processor.finish(command.command_id, status="failed", reason=str(e)[:1024])
                            self._running = False
                            fault = f"failed to publish robot action: {e}"
                elif self._feedback:
                    processor.finish(command.command_id, status="discarded", reason="preview-only")
            elif publish_to_robot:
                idle = getattr(robot, "publish_idle_action", None)
                if callable(idle):
                    try:
                        idle(action_keys)
                    except Exception as e:
                        logger.error("failed to publish idle robot action: %s", e)
                        if self._feedback:
                            self._running = False
                            fault = f"failed to publish idle robot action: {e}"

            try:
                should_request = fault is None and self._should_request_actions(processor)
            except RuntimeError as exc:
                should_request = False
                fault = str(exc)
            if should_request:
                self._request_reserved = True
                request_thread = threading.Thread(
                    target=self._run_reserved_request,
                    args=(task_instruction, generation, action_request_mode), daemon=True,
                )
                self._request_thread = request_thread

        if fault is not None:
            return fault, generation
        if request_thread is not None:
            try:
                request_thread.start()
            except Exception as e:
                with self._lock:
                    self._request_reserved = False
                    self._request_thread = None
                return f"could not start inference request: {e}", generation

    def _run_reserved_request(self, *args):
        try:
            self._request_and_buffer(*args)
        finally:
            with self._lock:
                self._request_reserved = False

    def _request_and_buffer(
        self,
        task_instruction: str,
        generation: int,
        action_request_mode: str = ACTION_REQUEST_MODE_ASYNC,
    ) -> None:
        action_request_mode = normalize_action_request_mode(action_request_mode)
        started_at = time.monotonic()
        try:
            with self._lock:
                timeout_s = self._initial_action_timeout_s if self._preparation_needed else None
            response = self._get_action_with_feedback(task_instruction, generation, timeout_s=timeout_s)
        except Exception as e:
            logger.warning("get_action raised: %s", e)
            self._report_fault(f"get_action raised: {e}", generation=generation)
            return
        latency_s = time.monotonic() - started_at
        with self._lock:
            if generation != self._generation:
                return
        if self._observation_warmup_timeout_s is None:
            self._record_request_latency(latency_s)
        try:
            chunk = self._decode_action_response(response)
            if self._observation_warmup_timeout_s is not None:
                latency_s = action_latency(response.capabilities_json, latency_s)
                self._record_request_latency(latency_s)
        except ValueError as e:
            logger.warning("get_action response rejected: %s", e)
            self._report_fault(f"get_action failed: {e}", generation=generation)
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
                if self._feedback:
                    try:
                        if self._step_schedule:
                            self._step_schedule.accepted(response.seq_id, chunk)
                        produced = self._processor.enqueue(
                            response.seq_id, chunk, scheduled_start_delay_s,
                            align=self._step_schedule is None and action_request_mode != ACTION_REQUEST_MODE_SYNC,
                        ).command_count
                        contract = self._input_execution_contract
                        if contract.feedback_schema == 2:
                            self._processor.request_ready(contract.request_after, contract.request_after_count)
                    except Exception as e:
                        self._running = False
                        fault = str(e)
                    else:
                        fault = None
                        self._preparation_needed = False
                else:
                    fault = None
                    produced = self._processor.push_actions(
                        chunk,
                        scheduled_start_delay_s=scheduled_start_delay_s,
                        align=action_request_mode != ACTION_REQUEST_MODE_SYNC,
                    )
                if fault is not None:
                    produced = 0
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
            else:
                fault = None
        if fault is not None:
            self._report_fault(f"action buffering failed: {fault}", generation=generation)

    def _report_fault(self, reason: str, *, generation=None) -> None:
        hold_ok = self.emergency_stop(reason, expected_generation=generation)
        if hold_ok is None:
            return
        if self._fault_callback is not None:
            try:
                self._fault_callback(reason, hold_ok)
            except Exception as e:
                logger.error("policy runtime fault callback failed: %s", e)

    def _clear_plan_locked(self, reason, phase):
        self._preparation_needed = self._initial_action_timeout_s is not None
        if self._step_schedule:
            self._step_schedule.reset()
        if self._feedback:
            self._feedback.reset(reason, phase)
        elif self._processor is not None:
            self._processor.clear()

    def _get_action_with_feedback(self, instruction, generation, *, timeout_s=None):
        # All feedback projection/serialization and Worker waits run outside tick.
        with self._feedback_io_lock:
            with self._lock:
                if generation != self._generation:
                    raise RuntimeError("inference generation cancelled")
                feedback, requester = self._feedback, self._requester
                capture = feedback.capture() if feedback else None
            context = feedback.project(capture) if feedback else None
            options = {} if timeout_s is None else {"timeout_s": timeout_s}
            response = (
                requester.get_action(instruction, context=context, **options)
                if context is not None else requester.get_action(instruction, **options)
            )
            with self._lock:
                if feedback is not None and feedback is self._feedback and response.success:
                    feedback.acknowledge(context)
            return response

    def _schedule_feedback_update(self):
        with self._lock:
            if self._feedback is None:
                return
            self._feedback_dirty = True
            if self._feedback_thread is not None:
                return
            self._feedback_thread = threading.Thread(target=self._flush_feedback, daemon=True)
            try:
                self._feedback_thread.start()
            except Exception as e:
                self._feedback_thread = None
                logger.warning("could not start execution feedback update: %s", e)

    def _flush_feedback(self):
        while True:
            with self._feedback_io_lock:
                with self._lock:
                    if not self._feedback_dirty or self._feedback is None:
                        self._feedback_thread = None
                        return
                    self._feedback_dirty = False
                    feedback, requester = self._feedback, self._requester
                try:
                    with self._lock:
                        if feedback is not self._feedback:
                            continue
                        capture = feedback.capture()
                    context = feedback.project(capture)
                    response = requester.update_context(context)
                    if not response.success:
                        raise RuntimeError(response.message)
                    with self._lock:
                        if feedback is self._feedback:
                            feedback.acknowledge(context)
                except Exception as e:
                    # Do not turn an unacknowledged update into a successful ACK,
                    # or wait for a Worker ACK before carrying out a safety hold.
                    logger.warning("execution feedback update failed: %s", e)

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
        if self._request_reserved:
            return False
        if self._request_thread is not None and self._request_thread.is_alive():
            return False
        contract = self._input_execution_contract
        if contract.feedback_schema == 2 and not processor.request_ready(contract.request_after, contract.request_after_count):
            return False
        if self._step_schedule:
            return self._step_schedule.can_request(time.monotonic(), processor.buffer_size)
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
            if self._processor is None or self._step_schedule is not None:
                hz = self._control_hz
            else:
                hz = self._processor.output_hz
        return 1.0 / max(1.0, hz)
