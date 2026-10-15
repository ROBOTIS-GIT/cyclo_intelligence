#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""External InferenceCommand handler for the Main process."""

from __future__ import annotations

from typing import List, Optional
import threading

from .confirmed_hold import HOLD_CAPABILITY


CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = 0, 1, 2, 3, 4, 5
CMD_UPDATE_INSTRUCTION = 6
# Read-only capability query; request/response wire fields are unchanged.
CMD_CAPABILITIES = 7
CMD_FAULT_STOP = 8


class ServiceHandler:
    def __init__(self, session, requester, control_loop, response_factory):
        self._session = session
        self._requester = requester
        self._control_loop = control_loop
        self._response_factory = response_factory
        self._lifecycle_lock = threading.RLock()
        self._fault_latched = threading.Event()

    def handle(self, request):
        # A reply confirms the lifecycle operation has settled. A late START
        # must never overtake STOP/PAUSE and republish after hold acknowledgment.
        if int(request.command) == CMD_FAULT_STOP:
            # Set before waiting for an in-flight START to finish. A delayed
            # START delivered after STOP must also be rejected, not re-enable
            # publishing. Only a Main process restart clears this fault latch.
            self._fault_latched.set()
        with self._lifecycle_lock:
            return self._handle_serialized(request)

    def _handle_serialized(self, request):
        cmd = int(request.command)
        try:
            if cmd == CMD_CAPABILITIES:
                return self._make_response(True, HOLD_CAPABILITY)
            if self._fault_latched.is_set() and cmd in (
                CMD_LOAD, CMD_START, CMD_RESUME, CMD_UPDATE_INSTRUCTION,
            ):
                return self._make_response(False, "supervision fault latched; restart policy runtime")
            if cmd == CMD_FAULT_STOP:
                return self._stop()
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

        response = self._requester.load_policy(request)
        if not response.success:
            return self._make_response(False, response.message)

        action_keys = list(response.action_keys)
        try:
            self._control_loop.configure(
                robot_type=request.robot_type,
                task_instruction=request.task_instruction or "",
                action_keys=action_keys,
                publish_to_robot=bool(getattr(request, "publish_to_robot", False)),
                vla_rail_enabled=bool(getattr(request, "vla_rail_enabled", False)),
                action_request_mode=getattr(request, "action_request_mode", "async"),
                control_hz=getattr(request, "control_hz", 0),
                inference_hz=getattr(request, "inference_hz", 0),
                chunk_align_window_s=getattr(request, "chunk_align_window_s", 0.0),
                initial_pose_sync=bool(getattr(request, "initial_pose_sync", False)),
                initial_pose_sync_duration_s=(
                    float(getattr(request, "initial_pose_sync_duration_s", 0.0)) or 5.0
                ),
            )
        except Exception:
            self._control_loop.deconfigure()
            self._requester.unload_policy()
            raise
        self._session.mark_loaded(
            robot_type=request.robot_type,
            task_instruction=request.task_instruction or "",
            action_keys=action_keys,
        )
        return self._make_response(True, response.message or "loaded", action_keys)

    def _start_control_loop(self, request):
        options = {}
        if getattr(request, 'initial_pose_sync', False) or (
                request.command == CMD_RESUME and hasattr(request, 'initial_pose_sync')):
            options = {
                'initial_pose_sync': bool(getattr(request, 'initial_pose_sync', False)),
                'initial_pose_sync_duration_s': float(getattr(request, 'initial_pose_sync_duration_s', 0.0)) or 5.0,
            }
        return self._control_loop.start(
            publish_to_robot=bool(getattr(request, 'publish_to_robot', False)), **options
        )

    def _start(self, request):
        if not self._session.loaded:
            raise RuntimeError("LOAD first")
        instruction = str(getattr(request, "task_instruction", "") or "").strip()
        if instruction:
            self._session.task_instruction = instruction
            self._control_loop.set_task_instruction(instruction)
        syncing = self._start_control_loop(request)
        self._session.mark_running()
        return self._make_response(True, "syncing" if syncing else "running")

    def _pause(self):
        if not self._session.running:
            raise RuntimeError("not running")
        hold_ok = self._control_loop.pause()
        if not hold_ok:
            return self._make_response(False, "current-pose hold failed; retry STOP")
        self._session.mark_paused()
        return self._make_response(True, "paused")

    def _resume(self, request):
        if not self._session.loaded:
            raise RuntimeError("LOAD first")
        if not self._session.running:
            # LOAD prepares an idle session without a speculative START/STOP.
            # Use the same start/hold checks for its first explicit RESUME.
            return self._start(request)
        task_instruction = request.task_instruction or self._session.task_instruction
        self._control_loop.set_task_instruction(task_instruction)
        syncing = self._start_control_loop(request)
        self._session.mark_resumed(task_instruction)
        return self._make_response(True, "syncing" if syncing else "resumed")

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
        if not response.success:
            return self._make_response(False, response.message)
        return self._make_response(True, response.message or "unloaded")

    def _update_instruction(self, request):
        if not self._session.loaded:
            return self._make_response(False, "LOAD first")
        if not self._session.running:
            return self._make_response(False, "not running - START first")
        new_instruction = (request.task_instruction or "").strip()
        if not new_instruction:
            return self._make_response(False, "task_instruction must be non-empty")
        self._session.task_instruction = new_instruction
        self._control_loop.set_task_instruction(new_instruction)
        return self._make_response(True, f'instruction updated: "{new_instruction}"')

    def _make_response(
        self,
        success: bool,
        message: str = "",
        action_keys: Optional[List[str]] = None,
    ):
        return self._response_factory(
            success=bool(success),
            message=str(message),
            action_keys=list(action_keys) if action_keys else [],
        )
