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
import os
import signal
import sys
import threading
import time
import uuid
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
    ENGINE_PROTOCOL_VERSION,
    ENGINE_COMMAND_REQUEST_DEF,
    ENGINE_COMMAND_RESPONSE_DEF,
    EngineCommandRequest,
    EngineCommandResponse,
    flatten_action_list,
    request_from_message,
    response_to_message_kwargs,
)


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
        self._heartbeat_publisher = None
        self._heartbeat_thread = None
        self._shutdown = threading.Event()
        self._command_lock = threading.Lock()

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
            return EngineCommandResponse(
                success=False,
                seq_id=req.seq_id,
                message=f"unknown engine command: {req.command}",
            )
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
        self._engine_state = "loading"
        result = self._engine.load_policy(request)
        self._engine_state = "loaded" if result.get("success") else "error"
        return EngineCommandResponse(
            success=bool(result.get("success", False)),
            seq_id=request.seq_id,
            message=str(result.get("message", "")),
            action_keys=list(result.get("action_keys", []) or []),
        )

    def _get_action(self, request: EngineCommandRequest) -> EngineCommandResponse:
        self._engine_state = "running"
        result = self._engine.get_action_chunk(
            SimpleNamespace(task_instruction=request.task_instruction)
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
        )
        self._engine_state = "loaded"
        return response

    def _unload_policy(self, request: EngineCommandRequest) -> EngineCommandResponse:
        self._engine.cleanup()
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

    def make_ros_callback(self):
        """Return a ROS service callback that wraps ``handle``."""

        def _callback(request):
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
    if not manifest_path.is_file():
        logger.warning("policy manifest unavailable at %s", manifest_path)
        return [], {}
    try:
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        runtime = manifest.get("runtime", {})
        if runtime.get("id") != backend:
            raise ValueError(
                f"manifest runtime {runtime.get('id')!r} does not match {backend!r}"
            )
        policy_ids = [
            f"{backend}:{model['id']}"
            for model in manifest.get("models", [])
            if isinstance(model, dict) and model.get("id")
        ]
        return policy_ids, dict(runtime.get("capabilities", {}) or {})
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
