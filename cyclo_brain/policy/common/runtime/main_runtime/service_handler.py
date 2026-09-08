#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""External InferenceCommand handler for the central Policy Runtime."""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from pathlib import Path
from typing import List, Optional

_LOCAL_POLICY_COMMON = Path(__file__).resolve().parents[2]
if (_LOCAL_POLICY_COMMON / "catalog").is_dir():
    sys.path.insert(0, str(_LOCAL_POLICY_COMMON))

from catalog import (
    normalize_policy_parameters,
    resolve_policy,
    resolve_policy_id,
    resolve_runtime,
)


CMD_LOAD, CMD_START, CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_UNLOAD = 0, 1, 2, 3, 4, 5
CMD_UPDATE_INSTRUCTION = 6
CMD_STATUS = 7
logger = logging.getLogger("main_runtime.service_handler")


class ServiceHandler:
    def __init__(
        self,
        session,
        requester,
        control_loop,
        response_factory,
        *,
        catalog=None,
        backend="",
        worker_registry=None,
    ):
        self._session = session
        self._requester = requester
        self._control_loop = control_loop
        self._response_factory = response_factory
        self._catalog = catalog
        self._backend = str(backend or "")
        self._worker_registry = worker_registry
        self._active_requester = requester
        self._loading_runtime_id = ""
        self._worker_mutations: dict[str, str] = {}
        self._lock = threading.RLock()

    def handle(self, request, *, backend_override: str = ""):
        with self._lock:
            cmd = int(request.command)
            try:
                if cmd == CMD_LOAD:
                    return self._load(request, backend_override=backend_override)
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
                if cmd == CMD_STATUS:
                    return self._status()
                return self._make_response(False, f"Unknown command: {cmd}")
            except Exception as e:
                return self._make_response(False, str(e))

    def _load(self, request, *, backend_override: str = ""):
        if self._session.loaded:
            return self._make_response(False, "policy already loaded - UNLOAD first")
        if not request.model_path:
            return self._make_response(False, "model_path is required")
        if not request.robot_type:
            return self._make_response(False, "robot_type is required")

        runtime_id = str(backend_override or self._backend)
        if self._catalog is not None:
            requested_policy_id = str(getattr(request, "policy_id", "") or "")
            if not runtime_id:
                if requested_policy_id:
                    runtime, _model = resolve_policy(
                        self._catalog,
                        requested_policy_id,
                    )
                    runtime_id = runtime["id"]
                else:
                    runtime_id = self._runtime_from_model_path(request.model_path)
            policy_id = resolve_policy_id(
                self._catalog,
                runtime_id,
                requested_policy_id,
                request.model_path,
            )
            if not requested_policy_id:
                logger.warning(
                    "policy_id was empty; resolved legacy checkpoint as %s",
                    policy_id,
                )
            parameters_json = normalize_policy_parameters(
                self._catalog,
                policy_id,
                getattr(request, "policy_parameters_json", ""),
            )
            action_request_mode = str(
                getattr(request, "action_request_mode", "") or "async"
            ).strip().lower()
            if action_request_mode not in {"async", "sync"}:
                raise RuntimeError(
                    f"unsupported action request mode {action_request_mode!r}"
                )
            supported_modes = resolve_runtime(
                self._catalog, runtime_id
            )["capabilities"]["action_request_modes"]
            if action_request_mode not in supported_modes:
                raise RuntimeError(
                    f"policy runtime {runtime_id!r} does not support "
                    f"action request mode {action_request_mode!r}"
                )
            request.action_request_mode = action_request_mode
            request.policy_id = policy_id
            request.policy_parameters_json = parameters_json

        if not runtime_id and self._worker_registry is not None:
            raise RuntimeError("policy runtime could not be determined")
        if runtime_id in self._worker_mutations:
            raise RuntimeError(f"{runtime_id} worker is being changed; retry LOAD")
        if self._worker_registry is not None:
            self._loading_runtime_id = runtime_id
            try:
                self._active_requester = self._worker_registry.requester(
                    runtime_id,
                    str(getattr(request, "policy_id", "") or ""),
                )
            except Exception:
                self._loading_runtime_id = ""
                raise
            self._control_loop.set_requester(self._active_requester)

        requester = self._active_requester or self._requester
        if requester is None:
            self._loading_runtime_id = ""
            raise RuntimeError("policy worker requester is unavailable")
        try:
            response = requester.load_policy(request)
        finally:
            self._loading_runtime_id = ""
        if not response.success:
            return self._make_response(False, response.message)

        action_keys = list(response.action_keys)
        publish_to_robot = bool(getattr(request, "publish_to_robot", False))
        acceleration_mode = str(
            getattr(request, "acceleration_mode", "") or "pytorch"
        ).strip()
        acceleration_engine_path = str(
            getattr(request, "acceleration_engine_path", "") or ""
        ).strip()
        if acceleration_mode == "pytorch":
            acceleration_engine_path = ""
        try:
            self._control_loop.configure(
                robot_type=request.robot_type,
                task_instruction=request.task_instruction or "",
                action_keys=action_keys,
                publish_to_robot=publish_to_robot,
                action_request_mode=getattr(request, "action_request_mode", "async"),
                control_hz=getattr(request, "control_hz", 0),
                inference_hz=getattr(request, "inference_hz", 0),
                chunk_align_window_s=getattr(request, "chunk_align_window_s", 0.0),
                initial_pose_sync=bool(getattr(request, "initial_pose_sync", False)),
                initial_pose_sync_duration_s=(
                    float(getattr(request, "initial_pose_sync_duration_s", 0.0)) or 5.0
                ),
            )
            applied_config = self._control_loop.configuration_snapshot()
        except Exception as configure_error:
            self._control_loop.deconfigure()
            rollback_error = ""
            try:
                rollback = requester.unload_policy()
                if not rollback.success:
                    rollback_error = rollback.message or "worker unload failed"
            except Exception as unload_error:
                rollback_error = str(unload_error)

            if rollback_error:
                message = (
                    f"Policy Runtime configuration failed: {configure_error}; "
                    f"worker rollback failed: {rollback_error}"
                )
                self._session.mark_loaded(
                    robot_type=request.robot_type,
                    task_instruction=request.task_instruction or "",
                    action_keys=action_keys,
                    model_path=request.model_path,
                    policy_id=getattr(request, "policy_id", ""),
                    policy_parameters_json=getattr(
                        request, "policy_parameters_json", "{}"
                    ),
                    publish_to_robot=False,
                    runtime_id=runtime_id,
                    action_request_mode=getattr(
                        request, "action_request_mode", "async"
                    ),
                    acceleration_mode=acceleration_mode,
                    acceleration_engine_path=acceleration_engine_path,
                    control_hz=getattr(request, "control_hz", 0),
                    inference_hz=getattr(request, "inference_hz", 0),
                    chunk_align_window_s=getattr(
                        request, "chunk_align_window_s", 0.0
                    ),
                    initial_pose_sync=bool(
                        getattr(request, "initial_pose_sync", False)
                    ),
                    initial_pose_sync_duration_s=float(
                        getattr(request, "initial_pose_sync_duration_s", 0.0)
                        or 5.0
                    ),
                )
                self._session.mark_error(message)
                raise RuntimeError(message) from configure_error

            if self._worker_registry is not None:
                self._active_requester = None
            raise RuntimeError(
                f"Policy Runtime configuration failed: {configure_error}"
            ) from configure_error

        self._session.mark_loaded(
            robot_type=request.robot_type,
            task_instruction=request.task_instruction or "",
            action_keys=action_keys,
            model_path=request.model_path,
            policy_id=getattr(request, "policy_id", ""),
            policy_parameters_json=getattr(
                request, "policy_parameters_json", "{}"
            ),
            publish_to_robot=publish_to_robot,
            runtime_id=runtime_id,
            action_request_mode=applied_config["action_request_mode"],
            acceleration_mode=acceleration_mode,
            acceleration_engine_path=acceleration_engine_path,
            control_hz=applied_config["control_hz"],
            inference_hz=applied_config["inference_hz"],
            chunk_align_window_s=applied_config["chunk_align_window_s"],
            initial_pose_sync=applied_config["initial_pose_sync"],
            initial_pose_sync_duration_s=applied_config[
                "initial_pose_sync_duration_s"
            ],
        )
        return self._make_response(True, response.message or "loaded", action_keys)

    def _start(self, request):
        if not self._session.loaded:
            raise RuntimeError("LOAD first")
        syncing = self._control_loop.start(
            publish_to_robot=bool(getattr(request, "publish_to_robot", False))
        )
        self._session.mark_running()
        self._session.set_publish_to_robot(
            bool(getattr(request, "publish_to_robot", False))
        )
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
        if not self._session.running:
            raise RuntimeError("not running")
        task_instruction = request.task_instruction or self._session.task_instruction
        self._control_loop.set_task_instruction(task_instruction)
        syncing = self._control_loop.start(
            publish_to_robot=bool(getattr(request, "publish_to_robot", False))
        )
        self._session.mark_resumed(task_instruction)
        self._session.set_publish_to_robot(
            bool(getattr(request, "publish_to_robot", False))
        )
        return self._make_response(True, "syncing" if syncing else "resumed")

    def _stop(self):
        hold_ok = self._control_loop.stop()
        if not hold_ok:
            return self._make_response(False, "current-pose hold failed; retry STOP")
        self._session.mark_stopped()
        return self._make_response(True, "stopped")

    def _unload(self):
        if self._session.running and not self._session.paused:
            return self._make_response(
                False,
                "policy is running; STOP or PAUSE before UNLOAD",
            )
        if self._control_loop.initial_pose_sync_hold_required():
            return self._make_response(
                False,
                "current-pose hold is pending; retry STOP before UNLOAD",
            )
        requester = self._active_requester or self._requester
        if requester is None:
            return self._make_response(False, "policy worker requester is unavailable")
        response = requester.unload_policy()
        if not response.success:
            return self._make_response(False, response.message)
        self._control_loop.deconfigure()
        self._session.mark_unloaded()
        self._active_requester = None if self._worker_registry is not None else requester
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

    def _status(self):
        return self._make_response(True, self._runtime_state())

    def _runtime_state(self) -> str:
        if self._session.error:
            return "error"
        if not self._session.loaded:
            return "unloaded"
        if self._control_loop.initial_pose_sync_hold_required():
            return "syncing"
        if self._session.paused:
            return "paused"
        if self._session.running:
            return "running"
        return "loaded"

    def fail_safe(self, reason: str) -> bool:
        """Stop command publication after a worker/orchestrator failure."""
        with self._lock:
            if not self._session.loaded:
                return True
            hold_ok = self._control_loop.emergency_stop(reason)
            self._session.mark_error(reason)
            return hold_ok

    def on_control_fault(self, reason: str, _hold_ok: bool) -> None:
        with self._lock:
            if self._session.loaded:
                self._session.mark_error(reason)

    def runtime_snapshot(self, *, blocking: bool = True) -> dict:
        if not self._lock.acquire(blocking=blocking):
            raise RuntimeError(
                "Policy Runtime is busy handling a lifecycle request; "
                "retry after it finishes"
            )
        try:
            return {
                "runtime_state": self._runtime_state(),
                "runtime_id": self._session.runtime_id,
                "policy_id": self._session.policy_id,
                "model_path": self._session.model_path,
                "publish_to_robot": self._session.publish_to_robot,
                "hold_pending": self._control_loop.initial_pose_sync_hold_required(),
                "loading_runtime_id": self._loading_runtime_id,
                "mutating_runtime_ids": sorted(self._worker_mutations),
                "error": self._session.error,
            }
        finally:
            self._lock.release()

    def can_mutate_worker(self, runtime_id: str) -> tuple[bool, str]:
        try:
            snapshot = self.runtime_snapshot(blocking=False)
        except RuntimeError as exc:
            return False, str(exc)
        if snapshot["loading_runtime_id"] == runtime_id:
            return False, "worker is loading a policy"
        if snapshot["runtime_id"] != runtime_id:
            return True, "worker is not used by the active session"
        if snapshot["hold_pending"]:
            return False, "current-pose hold is pending"
        if snapshot["runtime_state"] != "unloaded":
            return False, "UNLOAD the active policy before changing its worker"
        return True, "worker is idle"

    def begin_worker_mutation(self, runtime_id: str) -> tuple[bool, str, str]:
        # Never queue a container operation behind a potentially minutes-long LOAD.
        if not self._lock.acquire(blocking=False):
            return (
                False,
                "Policy Runtime is busy handling a lifecycle request; retry after it finishes",
                "",
            )
        try:
            if runtime_id in self._worker_mutations:
                return False, "another worker operation is already in progress", ""
            allowed, reason = self.can_mutate_worker(runtime_id)
            if not allowed:
                return False, reason, ""
            token = uuid.uuid4().hex
            self._worker_mutations[runtime_id] = token
            return True, "worker mutation reserved", token
        finally:
            self._lock.release()

    def release_undelivered_worker_mutation(self, request: dict, response: dict) -> None:
        if (
            isinstance(request, dict)
            and request.get("operation") == "begin_worker_mutation"
            and response.get("ok")
            and response.get("allowed")
            and response.get("token")
        ):
            self.end_worker_mutation(
                str(request.get("runtime_id", "")), str(response["token"])
            )

    def end_worker_mutation(self, runtime_id: str, token: str) -> bool:
        with self._lock:
            if self._worker_mutations.get(runtime_id) != token:
                return False
            del self._worker_mutations[runtime_id]
            return True

    def _runtime_from_model_path(self, model_path: str) -> str:
        normalized = str(Path(model_path))
        matches = []
        for runtime in self._catalog.get("runtimes", []):
            root = str(Path(runtime["checkpoint_root"]))
            if normalized == root or normalized.startswith(root + "/"):
                matches.append(runtime["id"])
        if len(matches) == 1:
            logger.warning(
                "policy_id was empty; inferred runtime %s from model_path",
                matches[0],
            )
            return matches[0]
        raise RuntimeError(
            "a namespaced policy_id is required for /policy/inference_command"
        )

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
            runtime_state=self._runtime_state(),
            loaded_model_path=str(self._session.model_path or ""),
            loaded_policy_id=str(self._session.policy_id or ""),
            loaded_policy_parameters_json=str(
                self._session.policy_parameters_json or "{}"
            ),
            publish_to_robot=bool(self._session.publish_to_robot),
            loaded_action_request_mode=str(self._session.action_request_mode),
            loaded_acceleration_mode=str(self._session.acceleration_mode),
            loaded_acceleration_engine_path=str(
                self._session.acceleration_engine_path
            ),
            loaded_control_hz=int(self._session.control_hz),
            loaded_inference_hz=int(self._session.inference_hz),
            loaded_chunk_align_window_s=float(
                self._session.chunk_align_window_s
            ),
            loaded_initial_pose_sync=bool(self._session.initial_pose_sync),
            loaded_initial_pose_sync_duration_s=float(
                self._session.initial_pose_sync_duration_s
            ),
            runtime_error=str(self._session.error or ""),
        )
