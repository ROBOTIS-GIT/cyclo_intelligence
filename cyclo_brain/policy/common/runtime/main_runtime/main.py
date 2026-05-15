#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Main process entrypoint.

Hosts the external ``/<backend>/inference_command`` service and a local control
loop. Heavy policy imports and sensor reads stay isolated in the Engine process.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


_ZENOH_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk")
if os.path.exists(_ZENOH_SDK_PATH) and _ZENOH_SDK_PATH not in sys.path:
    sys.path.insert(0, _ZENOH_SDK_PATH)

_parents = Path(__file__).resolve().parents
_default_rc = str(_parents[4] / "sdk" / "robot_client") if len(_parents) > 4 else ""
_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", _default_rc)
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)

from zenoh_ros2_sdk import ROS2ServiceServer, get_logger  # noqa: E402
from robot_client.messages import (  # noqa: E402
    INFERENCE_COMMAND_REQUEST_DEF,
    INFERENCE_COMMAND_RESPONSE_DEF,
)

from .control_loop import ControlLoop  # noqa: E402
from .inference_requester import InferenceRequester  # noqa: E402
from .service_handler import ServiceHandler  # noqa: E402
from .session_state import SessionState  # noqa: E402
from .zenoh_client import ZenohEngineCommandClient  # noqa: E402


logger = get_logger("main_runtime")


class MainRuntime:
    def __init__(
        self,
        backend: str,
        router_ip: str,
        router_port: int,
        domain_id: int,
        namespace: str = "/",
    ) -> None:
        self._backend = backend
        self._router_ip = router_ip
        self._router_port = router_port
        self._domain_id = domain_id
        self._namespace = namespace
        self._node_name = f"{backend}_main_process"

        engine_client = ZenohEngineCommandClient(
            service_name=f"/{backend}/engine_command",
            router_ip=router_ip,
            router_port=router_port,
            domain_id=domain_id,
            node_name=f"{backend}_engine_client",
            namespace=namespace,
        )
        self._requester = InferenceRequester(
            engine_client,
            get_action_timeout_s=float(os.environ.get("GET_ACTION_TIMEOUT_S", "5.0")),
            load_policy_timeout_s=float(os.environ.get("LOAD_POLICY_TIMEOUT_S", "300.0")),
        )
        self._session = SessionState()
        self._control_loop = ControlLoop(
            self._requester,
            inference_hz=float(os.environ.get("INFERENCE_HZ", "15.0")),
            control_hz=float(os.environ.get("CONTROL_HZ", "100.0")),
            chunk_align_window_s=float(os.environ.get("CHUNK_ALIGN_WINDOW_S", "0.3")),
            target_chunk_size=self._target_chunk_size_from_env(),
            postprocess_actions=self._bool_env("POSTPROCESS_ACTIONS", True),
            alignment_mode=os.environ.get("ACTION_ALIGNMENT_MODE", "l2"),
        )
        self._engine_client = engine_client
        self._command_srv = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        self._control_loop.run_background()

        def _response_factory(**kwargs):
            ResponseClass = self._command_srv.response_msg_class
            return ResponseClass(**kwargs)

        handler = ServiceHandler(
            self._session,
            self._requester,
            self._control_loop,
            _response_factory,
        )
        self._command_srv = ROS2ServiceServer(
            service_name=f"/{self._backend}/inference_command",
            srv_type="interfaces/srv/InferenceCommand",
            callback=handler.handle,
            request_definition=INFERENCE_COMMAND_REQUEST_DEF,
            response_definition=INFERENCE_COMMAND_RESPONSE_DEF,
            router_ip=self._router_ip,
            router_port=self._router_port,
            domain_id=self._domain_id,
            node_name=self._node_name,
            namespace=self._namespace,
        )
        logger.info("InferenceCommand service up at /%s/inference_command", self._backend)
        logger.info("ZENOH_SUB_READY")
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=1.0)

    def shutdown(self) -> None:
        self._shutdown.set()
        self._control_loop.shutdown()
        if self._command_srv is not None:
            try:
                self._command_srv.close()
            except Exception:
                pass
            self._command_srv = None
        self._engine_client.close()

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _target_chunk_size_from_env() -> int | None:
        raw = os.environ.get("TARGET_CHUNK_SIZE", "100").strip().lower()
        if raw in {"", "none", "off", "0"}:
            return None
        return int(raw)


def main() -> None:  # pragma: no cover - container entrypoint.
    backend = os.environ.get("POLICY_BACKEND", "").strip()
    if not backend:
        raise RuntimeError("POLICY_BACKEND env var is required")
    runtime = MainRuntime(
        backend=backend,
        router_ip=os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1"),
        router_port=int(os.environ.get("ZENOH_ROUTER_PORT", "7447")),
        domain_id=int(os.environ.get("ROS_DOMAIN_ID", "30")),
    )
    try:
        runtime.start()
    except KeyboardInterrupt:
        logger.info("shutdown via SIGINT")
    finally:
        runtime.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
