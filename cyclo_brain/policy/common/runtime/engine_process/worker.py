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
import os
import sys
import threading
from types import SimpleNamespace
from typing import Any, Optional

_ZENOH_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk")
if os.path.exists(_ZENOH_SDK_PATH) and _ZENOH_SDK_PATH not in sys.path:
    sys.path.insert(0, _ZENOH_SDK_PATH)

from .protocol import (
    CMD_GET_ACTION,
    CMD_LOAD_POLICY,
    CMD_UNLOAD_POLICY,
    ENGINE_COMMAND_REQUEST_DEF,
    ENGINE_COMMAND_RESPONSE_DEF,
    EngineCommandRequest,
    EngineCommandResponse,
    flatten_action_list,
    request_from_message,
    response_to_message_kwargs,
)


try:  # pragma: no cover - exercised only in container runtime.
    from zenoh_ros2_sdk import ROS2ServiceServer, get_logger
except Exception:  # pragma: no cover - local unit tests do not ship SDK.
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

    def __init__(self, engine: Any):
        self._engine = engine
        self._service = None
        self._shutdown = threading.Event()

    def handle(self, request: Any) -> EngineCommandResponse:
        req = (
            request
            if isinstance(request, EngineCommandRequest)
            else request_from_message(request)
        )
        try:
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
            logger.error("Engine command failed: %s", e, exc_info=True)
            return EngineCommandResponse(
                success=False,
                seq_id=req.seq_id,
                message=str(e),
            )

    def _load_policy(self, request: EngineCommandRequest) -> EngineCommandResponse:
        result = self._engine.load_policy(request)
        return EngineCommandResponse(
            success=bool(result.get("success", False)),
            seq_id=request.seq_id,
            message=str(result.get("message", "")),
            action_keys=list(result.get("action_keys", []) or []),
        )

    def _get_action(self, request: EngineCommandRequest) -> EngineCommandResponse:
        result = self._engine.get_action_chunk(
            SimpleNamespace(task_instruction=request.task_instruction)
        )
        if not result.get("success"):
            return EngineCommandResponse(
                success=False,
                seq_id=request.seq_id,
                message=str(result.get("message", "get_action failed")),
            )

        return EngineCommandResponse(
            success=True,
            seq_id=request.seq_id,
            message=str(result.get("message", "")),
            chunk_size=int(result.get("chunk_size", 0)),
            action_dim=int(result.get("action_dim", 0)),
            action_list=flatten_action_list(result.get("action_chunk", [])),
        )

    def _unload_policy(self, request: EngineCommandRequest) -> EngineCommandResponse:
        self._engine.cleanup()
        return EngineCommandResponse(
            success=True,
            seq_id=request.seq_id,
            message="unloaded",
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
        logger.info("EngineCommand service up at %s", service_name)
        logger.info("ZENOH_SUB_READY")
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=1.0)

    def shutdown(self) -> None:
        self._shutdown.set()
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


def main() -> None:  # pragma: no cover - container entrypoint.
    backend = os.environ.get("POLICY_BACKEND", "").strip() or "policy"
    worker = EngineWorker(resolve_engine())
    try:
        worker.start_service(
            service_name=f"/{backend}/engine_command",
            router_ip=os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1"),
            router_port=int(os.environ.get("ZENOH_ROUTER_PORT", "7447")),
            domain_id=int(os.environ.get("ROS_DOMAIN_ID", "30")),
            node_name=f"{backend}_engine_process",
        )
    except KeyboardInterrupt:
        logger.info("shutdown via SIGINT")
    finally:
        worker.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
