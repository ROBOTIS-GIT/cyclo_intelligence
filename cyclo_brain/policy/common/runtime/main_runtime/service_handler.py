#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""External InferenceCommand handler for the Main process."""

from __future__ import annotations

from typing import List, Optional


CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = 0, 1, 2, 3, 4, 5
CMD_UPDATE_INSTRUCTION = 6


class ServiceHandler:
    def __init__(self, session, requester, control_loop, response_factory):
        self._session = session
        self._requester = requester
        self._control_loop = control_loop
        self._response_factory = response_factory

    def handle(self, request):
        cmd = int(request.command)
        try:
            if cmd == CMD_LOAD:
                return self._load(request)
            if cmd == CMD_START:
                return self._start()
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
        self._session.mark_loaded(
            robot_type=request.robot_type,
            task_instruction=request.task_instruction or "",
            action_keys=action_keys,
        )
        self._control_loop.configure(
            robot_type=request.robot_type,
            task_instruction=self._session.task_instruction,
            action_keys=action_keys,
        )
        return self._make_response(True, response.message or "loaded", action_keys)

    def _start(self):
        self._session.mark_running()
        self._control_loop.start()
        return self._make_response(True, "running")

    def _pause(self):
        self._session.mark_paused()
        self._control_loop.pause()
        return self._make_response(True, "paused")

    def _resume(self, request):
        self._session.mark_resumed(request.task_instruction or "")
        self._control_loop.set_task_instruction(self._session.task_instruction)
        self._control_loop.start()
        return self._make_response(True, "resumed")

    def _stop(self):
        self._session.mark_stopped()
        self._control_loop.stop()
        return self._make_response(True, "stopped")

    def _unload(self):
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
