#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Engine process worker.

This class is deliberately small: it receives EngineCommand requests, invokes
the concrete backend ``InferenceEngine``, and returns action lists. It does not
know about the control loop, command publishing, or external lifecycle service.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

_ZENOH_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk")
if os.path.exists(_ZENOH_SDK_PATH) and _ZENOH_SDK_PATH not in sys.path:
    sys.path.insert(0, _ZENOH_SDK_PATH)

from .protocol import (
    CMD_DESCRIBE,
    CMD_GET_ACTION,
    CMD_LOAD_POLICY,
    CMD_STATUS,
    CMD_UNLOAD_POLICY,
    CMD_UPDATE_CONTEXT,
    ENGINE_PROTOCOL_VERSION,
    ENGINE_COMMAND_REQUEST_DEF,
    ENGINE_COMMAND_RESPONSE_DEF,
    EngineCommandRequest,
    EngineCommandResponse,
    flatten_action_list,
    request_from_message,
    response_to_message_kwargs,
)
from inference_context.execution import ExecutionContext
from inference_context.contract import ExecutionContract, LoadedExecution
from inference_context.timing import encode_observation_wait


try:  # pragma: no cover - exercised only in container runtime.
    from zenoh_ros2_sdk import ROS2Publisher, ROS2ServiceServer, get_logger
except Exception:  # pragma: no cover - local unit tests do not ship SDK.
    ROS2Publisher = None  # type: ignore[assignment]
    ROS2ServiceServer = None  # type: ignore[assignment]

    class _FallbackLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass

    def get_logger(_name: str):  # type: ignore[override]
        return _FallbackLogger()


logger = get_logger("engine_process")


class ContextRejected(ValueError):
    """Invalid/stale caller facts, not a failure of the currently loaded model."""


def _parse_context(raw):
    try:
        return ExecutionContext.from_json(raw)
    except ValueError as exc:
        raise ContextRejected(str(exc)) from exc


class EngineWorker:
    """Internal service handler hosted by the Engine process."""

    def __init__(
        self,
        engine: Any,
        *,
        runtime_id: str = "policy",
        supported_policy_ids: Optional[list[str]] = None,
        capabilities: Optional[dict[str, Any]] = None,
    ):
        self._engine = engine
        self._runtime_id = str(runtime_id or "policy")
        self._supported_policy_ids = sorted(set(supported_policy_ids or []))
        self._capabilities = dict(capabilities or {})
        self._instance_id = uuid.uuid4().hex
        self._engine_state = "unloaded"
        self._service = None
        self._status_service = None
        self._heartbeat_publisher = None
        self._heartbeat_thread = None
        self._shutdown = threading.Event()
        self._command_lock = threading.Lock()
        self._execution_context = None
        self._engine_execution_context = None
        self._last_context_action = None
        self._failed_context_generation = None

    def handle(self, request: Any) -> EngineCommandResponse:
        req = (
            request
            if isinstance(request, EngineCommandRequest)
            else request_from_message(request)
        )
        try:
            if req.command == CMD_DESCRIBE:
                return self._metadata_response(req.seq_id, "worker description")
            if req.command == CMD_STATUS:
                return self._metadata_response(req.seq_id, self._engine_state)
            with self._command_lock:
                if req.command == CMD_LOAD_POLICY:
                    return self._load_policy(req)
                if req.command == CMD_GET_ACTION:
                    return self._get_action(req)
                if req.command == CMD_UNLOAD_POLICY:
                    return self._unload_policy(req)
                if req.command == CMD_UPDATE_CONTEXT:
                    if not req.execution_context_json:
                        raise ContextRejected("UPDATE_CONTEXT requires an execution context")
                    self._accept_context(req.execution_context_json)
                    return EngineCommandResponse(success=True, seq_id=req.seq_id, message="context updated")
            return EngineCommandResponse(
                success=False,
                seq_id=req.seq_id,
                message=f"unknown engine command: {req.command}",
            )
        except ContextRejected as e:
            return EngineCommandResponse(success=False, seq_id=req.seq_id, message=str(e))
        except Exception as e:
            self._engine_state = "error"
            logger.error("Engine command failed: %s", e, exc_info=True)
            return EngineCommandResponse(
                success=False,
                seq_id=req.seq_id,
                message=str(e),
            )

    def _load_policy(self, request: EngineCommandRequest) -> EngineCommandResponse:
        if (
            request.policy_id
            and self._supported_policy_ids
            and request.policy_id not in self._supported_policy_ids
        ):
            return EngineCommandResponse(
                success=False,
                seq_id=request.seq_id,
                message=(
                    f"policy {request.policy_id!r} is not supported by "
                    f"runtime {self._runtime_id!r}"
                ),
            )
        context = _parse_context(request.execution_context_json)
        if context is not None and not callable(getattr(self._engine, "update_execution_context", None)):
            raise ContextRejected("engine does not implement execution context")
        self._engine_state = "loading"
        self._execution_context = None
        self._engine_execution_context = None
        self._last_context_action = None
        self._failed_context_generation = None
        result = self._engine.load_policy(request)
        contract = result.get("execution_contract", ExecutionContract())
        if not isinstance(contract, ExecutionContract):
            raise ValueError("engine returned an invalid execution contract")
        requires_context = result.get("requires_execution_context", contract.requires_context)
        if type(requires_context) is not bool:
            raise ValueError("engine returned an invalid context requirement")
        if requires_context and "execution_contract" not in result:
            raise ValueError("contextual LOAD requires an explicit execution contract")
        if result.get("success") and (requires_context or contract.requires_context) and context is None:
            context = ExecutionContext(uuid.uuid4().hex, 0, 0, "ready", feedback_schema=contract.feedback_schema)
        # Explicit caller-supplied contexts predate LOAD negotiation. Preserve
        # that path unless the adapter opts into a resolved execution contract.
        loaded = (LoadedExecution(contract, context)
                  if result.get("success") and "execution_contract" in result else LoadedExecution())
        if result.get("success") and context is not None:
            self._accept_parsed_context(context, new_session=True)
        self._engine_state = "loaded" if result.get("success") else "error"
        return EngineCommandResponse(
            success=bool(result.get("success", False)),
            seq_id=request.seq_id,
            message=str(result.get("message", "")),
            action_keys=list(result.get("action_keys", []) or []),
            capabilities_json=loaded.to_json(),
        )

    def _get_action(self, request: EngineCommandRequest) -> EngineCommandResponse:
        parsed_context = _parse_context(request.execution_context_json)
        context = self._accept_parsed_context(parsed_context, validate_only=True)
        identity = (context.session_id, context.generation, request.seq_id) if context else None
        if identity is not None and identity[:2] == self._failed_context_generation:
            raise ContextRejected("previous inference failed; reset execution generation before retrying")
        if identity is not None and self._last_context_action is not None:
            previous, signature, response = self._last_context_action
            if identity == previous:
                if signature != (request.task_instruction, request.execution_context_json):
                    raise ContextRejected("request ID reused with different inference input")
                return response
            if identity[:2] == previous[:2] and identity[2] < previous[2]:
                raise ContextRejected("stale inference request")
        if context is not None and context.phase not in {"running", "syncing"}:
            raise ContextRejected(f"cannot infer while execution phase is {context.phase}")
        context = self._accept_parsed_context(parsed_context)
        self._engine_state = "running"
        # Until inference succeeds the model may have partially advanced its
        # internal state. Do not retry that generation after an exception/failure.
        if identity is not None:
            self._failed_context_generation = identity[:2]
        result = self._engine.get_action_chunk(
            SimpleNamespace(task_instruction=request.task_instruction, execution_context=context,
                            prediction_id=request.seq_id)
        )
        if not result.get("success"):
            self._engine_state = "error"
            return EngineCommandResponse(
                success=False,
                seq_id=request.seq_id,
                message=str(result.get("message", "get_action failed")),
            )

        response = EngineCommandResponse(
            success=True,
            seq_id=request.seq_id,
            message=str(result.get("message", "")),
            chunk_size=int(result.get("chunk_size", 0)),
            action_dim=int(result.get("action_dim", 0)),
            action_list=flatten_action_list(result.get("action_chunk", [])),
            capabilities_json=(encode_observation_wait(result["observation_wait_s"])
                               if "observation_wait_s" in result else "{}"),
        )
        if context is not None and (
            response.chunk_size <= 0 or response.action_dim <= 0
            or len(response.action_list) != response.chunk_size * response.action_dim
            or not all(math.isfinite(v) for v in response.action_list)
        ):
            raise ValueError("contextual inference returned an invalid action chunk")
        self._engine_state = "loaded"
        if identity is not None:
            self._failed_context_generation = None
            self._last_context_action = (
                identity, (request.task_instruction, request.execution_context_json), response,
            )
        return response

    def _accept_context(self, raw: str, *, new_session: bool = False, validate_only: bool = False):
        return self._accept_parsed_context(
            _parse_context(raw), new_session=new_session, validate_only=validate_only,
        )

    def _accept_parsed_context(self, context, *, new_session=False, validate_only=False):
        previous = self._execution_context
        if context is None:
            if previous is not None:
                raise ContextRejected("active contextual session requires execution context")
            return None
        if previous is None and not new_session:
            raise ContextRejected("contextual sessions must be established by LOAD")
        if previous is not None:
            if context.session_id != previous.session_id:
                raise ContextRejected("different execution session requires LOAD")
            if context.after_event_id > previous.latest_event_id:
                raise ContextRejected("execution feedback skips unacknowledged events")
            if context.latest_event_id < previous.latest_event_id:
                raise ContextRejected("execution feedback cursor moved backwards")
            if (context.generation, context.revision) < (previous.generation, previous.revision):
                raise ContextRejected("stale execution context")
            if (context.generation, context.revision) == (previous.generation, previous.revision):
                if context != previous:
                    raise ContextRejected("context revision reused with different facts")
                return context if validate_only else self._engine_execution_context
        receiver = getattr(self._engine, "update_execution_context", None)
        if not callable(receiver):
            raise ContextRejected("engine does not implement execution context")
        if validate_only:
            return context
        if (context.session_id, context.generation) == self._failed_context_generation:
            raise ContextRejected("previous context/inference failed; reset execution generation")
        delivered = context
        if previous is not None and context.after_event_id < previous.latest_event_id:
            # An ACK may have been lost after the hook succeeded. Never deliver
            # already-consumed terminal events a second time to a stateful model.
            cursor = previous.latest_event_id
            delivered = replace(
                context, after_event_id=cursor,
                actions=tuple(a for a in context.actions if a.event_id is None or a.event_id > cursor),
                planning=tuple(p for p in context.planning if p.event_id > cursor),
                resets=tuple(r for r in context.resets if r.event_id > cursor),
            )
        try:
            receiver(delivered)
        except Exception:
            self._failed_context_generation = (context.session_id, context.generation)
            raise
        self._execution_context = context
        self._engine_execution_context = delivered
        if previous is None or context.generation != previous.generation:
            self._last_context_action = None
        return delivered

    def _unload_policy(self, request: EngineCommandRequest) -> EngineCommandResponse:
        self._engine.cleanup()
        self._execution_context = None
        self._engine_execution_context = None
        self._last_context_action = None
        self._failed_context_generation = None
        self._engine_state = "unloaded"
        return EngineCommandResponse(
            success=True,
            seq_id=request.seq_id,
            message="unloaded",
        )

    def _metadata_response(self, seq_id: int, message: str) -> EngineCommandResponse:
        return EngineCommandResponse(
            success=True,
            seq_id=seq_id,
            message=message,
            protocol_version=ENGINE_PROTOCOL_VERSION,
            runtime_id=self._runtime_id,
            worker_instance_id=self._instance_id,
            supported_policy_ids=list(self._supported_policy_ids),
            capabilities_json=json.dumps(
                self._capabilities,
                sort_keys=True,
                separators=(",", ":"),
            ),
            engine_state=self._engine_state,
        )

    def make_ros_callback(self, *, metadata_only: bool = False):
        """Return a ROS service callback that wraps ``handle``."""

        def _callback(request):
            if metadata_only and request.command not in (CMD_DESCRIBE, CMD_STATUS):
                response = EngineCommandResponse(
                    success=False,
                    seq_id=request.seq_id,
                    message="engine_status only accepts DESCRIBE and STATUS",
                )
            else:
                response = self.handle(request)
            ResponseClass = self._service.response_msg_class
            return ResponseClass(**response_to_message_kwargs(response))

        return _callback

    def start_service(
        self,
        service_name: str,
        router_ip: str,
        router_port: int,
        domain_id: int,
        node_name: str,
        namespace: str = "/",
    ) -> None:
        """Host the internal EngineCommand service and block until shutdown."""
        if ROS2ServiceServer is None:
            raise RuntimeError("zenoh_ros2_sdk.ROS2ServiceServer is unavailable")
        self._service = ROS2ServiceServer(
            service_name=service_name,
            srv_type="interfaces/srv/EngineCommand",
            callback=self.make_ros_callback(),
            request_definition=ENGINE_COMMAND_REQUEST_DEF,
            response_definition=ENGINE_COMMAND_RESPONSE_DEF,
            router_ip=router_ip,
            router_port=router_port,
            domain_id=domain_id,
            node_name=node_name,
            namespace=namespace,
        )
        # Zenoh serializes callbacks on one queryable. Metadata must not queue
        # behind model loading or a long GET_ACTION on the command service.
        self._status_service = ROS2ServiceServer(
            service_name=f"/{self._runtime_id}/engine_status",
            srv_type="interfaces/srv/EngineCommand",
            callback=self.make_ros_callback(metadata_only=True),
            request_definition=ENGINE_COMMAND_REQUEST_DEF,
            response_definition=ENGINE_COMMAND_RESPONSE_DEF,
            router_ip=router_ip,
            router_port=router_port,
            domain_id=domain_id,
            node_name=f"{node_name}_status",
            namespace=namespace,
        )
        self._start_heartbeat(
            router_ip=router_ip,
            router_port=router_port,
            domain_id=domain_id,
            namespace=namespace,
        )
        self._write_ready_marker()
        logger.info("EngineCommand service up at %s", service_name)
        logger.info("ZENOH_SUB_READY")
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=1.0)

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
        if self._heartbeat_publisher is not None:
            try:
                self._heartbeat_publisher.close()
            except Exception:
                pass
            self._heartbeat_publisher = None
        if self._status_service is not None:
            try:
                self._status_service.close()
            except Exception:
                pass
            self._status_service = None
        if self._service is not None:
            try:
                self._service.close()
            except Exception:
                pass
            self._service = None
        try:
            self._engine.cleanup()
        except Exception as e:
            logger.warning("engine cleanup raised: %s", e, exc_info=True)
        self._remove_ready_marker()

    def request_shutdown(self, *_args) -> None:
        self._shutdown.set()

    def _start_heartbeat(
        self,
        *,
        router_ip: str,
        router_port: int,
        domain_id: int,
        namespace: str,
    ) -> None:
        if ROS2Publisher is None:
            return
        self._heartbeat_publisher = ROS2Publisher(
            topic=f"/{self._runtime_id}/worker_heartbeat",
            msg_type="std_msgs/msg/String",
            msg_definition="string data\n",
            router_ip=router_ip,
            router_port=router_port,
            domain_id=domain_id,
            node_name=f"{self._runtime_id}_worker_heartbeat",
            namespace=namespace,
        )
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"{self._runtime_id}-worker-heartbeat",
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        interval_s = max(0.1, float(os.environ.get("WORKER_HEARTBEAT_PERIOD_S", "0.5")))
        while not self._shutdown.is_set():
            payload = {
                "protocol_version": ENGINE_PROTOCOL_VERSION,
                "runtime_id": self._runtime_id,
                "worker_instance_id": self._instance_id,
                "engine_state": self._engine_state,
                "timestamp_ns": time.time_ns(),
            }
            try:
                self._heartbeat_publisher.publish(
                    data=json.dumps(payload, sort_keys=True, separators=(",", ":"))
                )
            except Exception as e:
                logger.warning("worker heartbeat publish failed: %s", e)
            self._shutdown.wait(interval_s)

    @staticmethod
    def _ready_marker_path() -> Path:
        return Path(os.environ.get("ENGINE_READY_MARKER", "/run/cyclo/engine-process.ready"))

    def _write_ready_marker(self) -> None:
        marker = self._ready_marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(".tmp")
        tmp.write_text(self._instance_id + "\n", encoding="utf-8")
        tmp.replace(marker)

    def _remove_ready_marker(self) -> None:
        try:
            self._ready_marker_path().unlink()
        except FileNotFoundError:
            pass


def resolve_engine() -> Any:
    """Load the concrete backend engine from POLICY_ENGINE_MODULE."""
    backend = os.environ.get("POLICY_BACKEND", "").strip()
    if not backend:
        raise RuntimeError("POLICY_BACKEND env var is required")
    module_name = os.environ.get("POLICY_ENGINE_MODULE", f"{backend}_engine")
    factory_name = os.environ.get("POLICY_ENGINE_FACTORY", "create_engine")
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    return factory()


def _worker_metadata(backend: str) -> tuple[list[str], dict[str, Any]]:
    manifest_path = Path(
        os.environ.get("POLICY_MANIFEST_PATH", "/app/policy_manifest.yaml")
    )
    try:
        # Source checkouts keep catalog beside runtime; images install it on
        # /policy_runtime, already on the Engine process's Python path.
        common = Path(__file__).resolve().parents[2]
        if (common / "catalog").is_dir() and str(common) not in sys.path:
            sys.path.insert(0, str(common))
        from catalog import load_runtime_catalog, resolve_runtime

        runtime = resolve_runtime(load_runtime_catalog(manifest_path, backend), backend)
        return (
            [model["policy_id"] for model in runtime["models"]],
            dict(runtime["capabilities"]),
        )
    except Exception as e:
        raise RuntimeError(f"invalid worker manifest {manifest_path}: {e}") from e


def main() -> None:  # pragma: no cover - container entrypoint.
    backend = os.environ.get("POLICY_BACKEND", "").strip() or "policy"
    policy_ids, capabilities = _worker_metadata(backend)
    worker = EngineWorker(
        resolve_engine(),
        runtime_id=backend,
        supported_policy_ids=policy_ids,
        capabilities=capabilities,
    )
    signal.signal(signal.SIGTERM, worker.request_shutdown)
    signal.signal(signal.SIGINT, worker.request_shutdown)
    try:
        worker.start_service(
            service_name=f"/{backend}/engine_command",
            router_ip=os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1"),
            router_port=int(os.environ.get("ZENOH_ROUTER_PORT", "7447")),
            domain_id=int(os.environ.get("ROS_DOMAIN_ID", "30")),
            node_name=f"{backend}_engine_process",
        )
    finally:
        worker.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
