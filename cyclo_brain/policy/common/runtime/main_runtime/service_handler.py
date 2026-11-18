#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""External InferenceCommand handler for the Main process."""

from __future__ import annotations

from typing import List, Optional
import json
import math
import logging
from engine_process.protocol import POLICY_UPDATE_COMMANDS, CMD_ASYNC_RL_DATASETS, CMD_ASYNC_RL_ON
import os
import threading
import time
from pathlib import Path
from inference_chunks import act_execution_steps, requested_chunk_size
from checkpoint_instructions import checkpoint_subtask_choices

from .confirmed_hold import HOLD_CAPABILITY


CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = 0, 1, 2, 3, 4, 5
CMD_UPDATE_INSTRUCTION = 6
# Read-only capability query; request/response wire fields are unchanged.
CMD_CAPABILITIES = 7
CMD_FAULT_STOP = 8
CMD_SET_ACTION_POLICY = 20


class ServiceHandler:
    def __init__(self, session, requester, control_loop, response_factory):
        self._session = session
        self._requester = requester
        self._control_loop = control_loop
        self._response_factory = response_factory
        self._lifecycle_lock = threading.RLock()
        self._fault_latched = threading.Event()
        self._per_step_actions = False
        self._playground_session = False
        self._subtask_choices = []
        self._loaded_model_path = ''
        self._instruction_lock = threading.RLock()

    def handle(self, request):
        # A reply confirms the lifecycle operation has settled. A late START
        # must never overtake STOP/PAUSE and republish after hold acknowledgment.
        if int(request.command) == CMD_FAULT_STOP:
            # Set before waiting for an in-flight START to finish. A delayed
            # START delivered after STOP must also be rejected, not re-enable
            # publishing. Only a Main process restart clears this fault latch.
            logging.getLogger(__name__).warning("Supervision FAULT_STOP received: pid=%s already_latched=%s", os.getpid(), self._fault_latched.is_set())
            self._fault_latched.set()
        with self._lifecycle_lock:
            return self._handle_serialized(request)

    def _handle_serialized(self, request):
        cmd = int(request.command)
        try:
            if cmd == CMD_CAPABILITIES:
                return self._make_response(True, HOLD_CAPABILITY)
            if self._fault_latched.is_set() and cmd in (
                CMD_LOAD, CMD_START, CMD_RESUME, CMD_UPDATE_INSTRUCTION, CMD_SET_ACTION_POLICY,
                *POLICY_UPDATE_COMMANDS,
            ):
                logging.getLogger(__name__).warning("Supervision latch rejected command=%s pid=%s", cmd, os.getpid())
                return self._make_response(False, "supervision fault latched; restart policy runtime")
            if cmd == CMD_FAULT_STOP:
                return self._stop()
            if cmd in POLICY_UPDATE_COMMANDS:
                if not self._session.loaded:
                    return self._make_response(False, "LOAD first")
                response = self._requester.policy_update(
                    cmd, str(getattr(request, "rlt_bundle_path", "") or ""),
                    **({'dataset_paths': list(getattr(request, 'rlt_dataset_paths', []) or [])}
                       if cmd == CMD_ASYNC_RL_DATASETS else {}),
                    **({'max_updates': int(getattr(request, 'rlt_max_updates', 0) or 0)}
                       if cmd == CMD_ASYNC_RL_ON else {}),
                )
                return self._make_response(response.success, response.message)
            if cmd == CMD_SET_ACTION_POLICY:
                return self._set_action_policy(request)
            if cmd == CMD_LOAD:
                return self._load(request)
            if cmd == CMD_START:
                return self._start(request)
            if cmd == CMD_PAUSE:
                return self._pause()
            if cmd == CMD_RESUME:
                return self._resume(request)
            if cmd == CMD_STOP:
                return self._stop()
            if cmd == CMD_UNLOAD:
                return self._unload()
            if cmd == CMD_UPDATE_INSTRUCTION:
                return self._update_instruction(request)
            return self._make_response(False, f"Unknown command: {cmd}")
        except Exception as e:
            return self._make_response(False, str(e))

    def _load(self, request):
        if self._session.loaded:
            return self._make_response(False, "policy already loaded - UNLOAD first")
        if not request.model_path:
            return self._make_response(False, "model_path is required")
        if not request.robot_type:
            return self._make_response(False, "robot_type is required")

        duration = float(getattr(request, "initial_pose_sync_duration_s", 0.0) or 5.0)
        if not math.isfinite(duration) or not 1.0 <= duration <= 60.0:
            raise ValueError("initial_pose_sync_duration_s must be between 1.0 and 60.0")

        root = Path(request.model_path)
        if not (root / "config.json").is_file() and (root / "pretrained_model/config.json").is_file():
            root = root / "pretrained_model"
        config_path = root / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        subtask_choices = checkpoint_subtask_choices(request.model_path)
        if subtask_choices and not any(
            (request.task_instruction or "").strip() == prompt.strip()
            for choices in subtask_choices for prompt in choices
        ):
            raise ValueError("Select a trained subtask instruction before LOAD")
        chunk_size = requested_chunk_size(getattr(request, "action_chunk_size", 0))
        tt_rtc_enabled = getattr(request, "action_request_mode", "async") == "tt_rtc"
        rlt_enabled = bool(getattr(request, "rlt_enabled", False))
        if tt_rtc_enabled or rlt_enabled:
            if os.environ.get("POLICY_BACKEND", "groot") != "groot":
                raise ValueError("TT-RTC/RLT requires the GR00T backend")
            if getattr(request, "vla_rail_enabled", False):
                raise ValueError("TT-RTC/RLT and VLA-RAIL cannot be enabled together")
        if tt_rtc_enabled:
            control_hz = float(getattr(request, "control_hz", 0) or 0)
            if not math.isfinite(control_hz) or (control_hz != 0 and control_hz < 15):
                raise ValueError("TT-RTC Control Hz must be at least 15")
            if chunk_size:
                raise ValueError("TT-RTC requires the checkpoint's original action horizon")
        per_step_actions = config.get("type") == "act" and act_execution_steps(config, chunk_size) == 1
        if per_step_actions:
            contract_path = root / "inference_contract.json"
            contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
            fps = contract.get("fps")
            if fps is not None and float(getattr(request, "inference_hz", 0)) != float(fps):
                raise ValueError(f"This ACT checkpoint requires Inference Hz={fps}")

        response = self._requester.load_policy(request)
        if not response.success:
            return self._make_response(False, response.message)

        action_keys = list(response.action_keys)
        try:
            if not action_keys:
                raise ValueError("Model returned no action keys")
            self._control_loop.configure(
                robot_type=request.robot_type,
                task_instruction=request.task_instruction or "",
                action_keys=action_keys,
                publish_to_robot=bool(getattr(request, "publish_to_robot", False)),
                action_request_mode=getattr(request, "action_request_mode", "async"),
                control_hz=getattr(request, "control_hz", 0),
                initial_pose_sync=bool(getattr(request, "initial_pose_sync", False)),
                initial_pose_sync_duration_s=(
                    float(getattr(request, "initial_pose_sync_duration_s", 0.0)) or 5.0
                ),
                **({"per_step_actions": True} if per_step_actions else {}),
                **({
                    "vla_rail_enabled": bool(getattr(request, "vla_rail_enabled", False)),
                    "inference_hz": getattr(request, "inference_hz", 0),
                    "chunk_align_window_s": getattr(request, "chunk_align_window_s", 0.0),
                } if not (tt_rtc_enabled or rlt_enabled) else {}),
                **({
                    "model_path": request.model_path,
                    "rlt_enabled": rlt_enabled,
                    "tt_rtc_horizon": int(getattr(response, "chunk_size", 0) or 16),
                    "tt_rtc_action_dim": int(getattr(response, "action_dim", 0) or 19),
                    "tt_rtc_min_delay_steps": int(getattr(response, "rtc_min_delay_steps", 0)) if getattr(response, "rtc_max_delay_steps", 0) else None,
                    "tt_rtc_max_delay_steps": int(getattr(response, "rtc_max_delay_steps", 0)) if getattr(response, "rtc_max_delay_steps", 0) else None,
                    "defer_initial_pose_sync_activation": bool(getattr(request, "initial_pose_sync_defer_activation", False)),
                } if tt_rtc_enabled or rlt_enabled else {}),
            )
        except Exception:
            try:
                self._control_loop.deconfigure()
            finally:
                self._requester.unload_policy()
            raise
        self._session.mark_loaded(
            robot_type=request.robot_type,
            task_instruction=request.task_instruction or "",
            action_keys=action_keys,
            publish_to_robot=bool(getattr(request, "publish_to_robot", False)),
        )
        self._playground_session = tt_rtc_enabled or rlt_enabled
        self._per_step_actions = per_step_actions
        self._subtask_choices = subtask_choices
        self._loaded_model_path = request.model_path
        return self._make_response(True, response.message or "loaded", action_keys)

    def _start_control_loop(self, request):
        if self._playground_session:
            self._require_loaded_publish_mode(request)
        options = {}
        if getattr(request, 'initial_pose_sync', False) or (
                request.command in (CMD_START, CMD_RESUME) and hasattr(request, 'initial_pose_sync')):
            options = {
                'initial_pose_sync': bool(getattr(request, 'initial_pose_sync', False)),
                'initial_pose_sync_duration_s': float(getattr(request, 'initial_pose_sync_duration_s', 0.0)) or 5.0,
            }
        return self._control_loop.start(
            publish_to_robot=bool(getattr(request, 'publish_to_robot', False)), **options
        )

    def _start(self, request):
        if self._playground_session:
            self._require_loaded_publish_mode(request)
        if not self._session.loaded:
            raise RuntimeError("LOAD first")
        self._reset_policy_if_needed()
        instruction = str(getattr(request, "task_instruction", "") or "").strip()
        if instruction:
            self._session.task_instruction = instruction
            self._control_loop.set_task_instruction(instruction)
        syncing = self._start_control_loop(request)
        self._session.mark_running()
        return self._make_response(True, "syncing" if syncing else "running", initial_pose_sync_duration_s=float(syncing) if self._playground_session else 0.0)

    def _pause(self):
        if not self._session.running:
            raise RuntimeError("not running")
        hold_ok = self._control_loop.pause()
        if not hold_ok:
            return self._make_response(False, "current-pose hold failed; retry STOP")
        self._session.mark_paused()
        return self._make_response(True, "paused")

    def _resume(self, request):
        if self._playground_session:
            self._require_loaded_publish_mode(request)
        if not self._session.loaded:
            raise RuntimeError("LOAD first")
        if not self._session.running:
            # LOAD prepares an idle session without a speculative START/STOP.
            # Use the same start/hold checks for its first explicit RESUME.
            return self._start(request)
        self._reset_policy_if_needed()
        task_instruction = request.task_instruction or self._session.task_instruction
        self._control_loop.set_task_instruction(task_instruction)
        syncing = self._start_control_loop(request)
        self._session.mark_resumed(task_instruction)
        return self._make_response(True, "syncing" if syncing else "resumed", initial_pose_sync_duration_s=float(syncing) if self._playground_session else 0.0)

    def _stop(self):
        hold_ok = self._control_loop.stop()
        if not hold_ok:
            return self._make_response(False, "current-pose hold failed; retry STOP")
        self._session.mark_stopped()
        return self._make_response(True, "stopped")

    def _unload(self):
        if self._control_loop.initial_pose_sync_hold_required():
            return self._make_response(
                False,
                "current-pose hold is pending; retry STOP before UNLOAD",
            )
        self._control_loop.deconfigure()
        response = self._requester.unload_policy()
        self._session.mark_unloaded()
        self._per_step_actions = False
        self._playground_session = False
        self._subtask_choices = []
        self._loaded_model_path = ''
        if not response.success:
            return self._make_response(False, response.message)
        return self._make_response(True, response.message or "unloaded")

    def _reset_policy_if_needed(self):
        if self._per_step_actions:
            if not self._control_loop.pause():
                raise RuntimeError("Current-pose hold failed before ACT reset")
            response = self._requester.reset_policy()
            if not response.success:
                raise RuntimeError(response.message or "policy reset failed")

    def _update_instruction(self, request):
        # UI and automatic completion detectors use the same existing command.
        with self._instruction_lock:
            return self._update_instruction_locked(request)

    def subtask_state(self):
        """Confirmed state for UI observers of manual or external updates."""
        with self._instruction_lock:
            instruction = self._session.task_instruction
            index = next((i for i, prompts in enumerate(self._subtask_choices)
                          if instruction.strip() in [p.strip() for p in prompts]), -1)
            return {"available": bool(self._session.loaded and self._subtask_choices),
                    "model_path": self._loaded_model_path, "instruction": instruction,
                    "subtask_index": index, "running": self._session.running,
                    "paused": self._session.paused, "timestamp": time.time()}

    def _update_instruction_locked(self, request):
        if not self._session.loaded:
            return self._make_response(False, "LOAD first")
        if not self._session.running:
            return self._make_response(False, "not running - START first")
        new_instruction = (request.task_instruction or "").strip()
        if not new_instruction:
            return self._make_response(False, "task_instruction must be non-empty")
        if self._subtask_choices:
            if self._session.paused:
                raise RuntimeError("RESUME inference before completing a subtask")
            current = self._session.task_instruction.strip()
            if new_instruction == current:
                return self._make_response(True, "subtask instruction already applied")
            current_stage = next((i for i, prompts in enumerate(self._subtask_choices)
                                  if any(current == prompt.strip() for prompt in prompts)), -1)
            if current_stage < 0 or current_stage + 1 >= len(self._subtask_choices):
                raise ValueError("No next subtask; stop inference after the final subtask")
            if not any(new_instruction == prompt.strip() for prompt in self._subtask_choices[current_stage + 1]):
                raise ValueError("Completion must select the next trained subtask instruction")
            self._control_loop.transition_subtask(new_instruction)
        else:
            self._control_loop.set_task_instruction(new_instruction)
        self._session.task_instruction = new_instruction
        return self._make_response(True, f'instruction updated: "{new_instruction}"')

    def _set_action_policy(self, request):
        if not self._session.loaded:
            return self._make_response(False, "LOAD first")
        if not self._session.running:
            return self._make_response(False, "not running - START first")
        mode = str(getattr(request, "action_policy_mode", "") or "").strip().lower()
        success, message = self._control_loop.set_action_policy(
            mode,
            allow_robot_rlt=bool(
                getattr(request, "rlt_robot_override", False)
            ),
            timeout_s=5.0,
        )
        return self._make_response(success, message)

    def _require_loaded_publish_mode(self, request) -> bool:
        """Prevent START/RESUME from bypassing Engine load-time safety checks."""

        requested = bool(getattr(request, "publish_to_robot", False))
        loaded = bool(getattr(self._session, "publish_to_robot", False))
        if requested != loaded:
            raise RuntimeError(
                "Deploy Target differs from the loaded policy; STOP, UNLOAD, "
                "then LOAD again with the requested simulation/robot target"
            )
        return loaded

    def _make_response(
        self,
        success: bool,
        message: str = "",
        action_keys: Optional[List[str]] = None,
        initial_pose_sync_duration_s: float = 0.0,
    ):
        return self._response_factory(
            success=bool(success),
            message=str(message),
            action_keys=list(action_keys) if action_keys else [],
            initial_pose_sync_duration_s=float(initial_pose_sync_duration_s),
        )
