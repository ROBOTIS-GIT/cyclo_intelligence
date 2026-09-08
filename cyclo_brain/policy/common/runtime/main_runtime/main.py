#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Central Cyclo Policy Runtime.

The runtime owns one global policy session and the robot-facing action loop.
Model frameworks remain isolated in Engine-only worker containers.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any


_ZENOH_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "/opt/cyclo/sdk/zenoh_ros2_sdk")
if os.path.exists(_ZENOH_SDK_PATH) and _ZENOH_SDK_PATH not in sys.path:
    sys.path.insert(0, _ZENOH_SDK_PATH)

_parents = Path(__file__).resolve().parents
_default_sdk_root = _parents[4] / "sdk" if len(_parents) > 4 else Path("/opt/cyclo/sdk")
for path in (
    os.environ.get("ROBOT_CLIENT_SDK_PATH", str(_default_sdk_root / "robot_client")),
    os.environ.get(
        "ACTION_CHUNK_PROCESSING_SDK_PATH",
        str(_default_sdk_root / "action_chunk_processing"),
    ),
):
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

_POLICY_COMMON_PATH = os.environ.get("POLICY_COMMON_PATH", str(_parents[2]))
if os.path.exists(_POLICY_COMMON_PATH) and _POLICY_COMMON_PATH not in sys.path:
    sys.path.insert(0, _POLICY_COMMON_PATH)

from catalog import load_catalog  # noqa: E402
from robot_client.messages import (  # noqa: E402
    INFERENCE_COMMAND_REQUEST_DEF,
    INFERENCE_COMMAND_RESPONSE_DEF,
)
from zenoh_ros2_sdk import ROS2ServiceServer, ROS2Subscriber, get_logger  # noqa: E402

from .control_loop import ControlLoop  # noqa: E402
from .runtime_control import RuntimeControlServer  # noqa: E402
from .service_handler import ServiceHandler  # noqa: E402
from .session_state import SessionState  # noqa: E402
from .worker_registry import (  # noqa: E402
    WorkerRegistry,
    runtime_health_failure_reason,
)


logger = get_logger("policy_runtime")


class PolicyRuntime:
    def __init__(
        self,
        router_ip: str,
        router_port: int,
        domain_id: int,
        namespace: str = "/",
    ) -> None:
        self._router_ip = router_ip
        self._router_port = int(router_port)
        self._domain_id = int(domain_id)
        self._namespace = namespace
        policy_root = Path(os.environ.get("CYCLO_POLICY_ROOT", "/opt/cyclo/policy"))
        self._catalog = load_catalog(policy_root)
        self._workers = WorkerRegistry(
            self._catalog,
            router_ip=router_ip,
            router_port=router_port,
            domain_id=domain_id,
            namespace=namespace,
        )
        self._session = SessionState()
        self._control_loop = ControlLoop(
            None,
            inference_hz=float(os.environ.get("INFERENCE_HZ", "15.0")),
            control_hz=float(os.environ.get("CONTROL_HZ", "100.0")),
            chunk_align_window_s=float(os.environ.get("CHUNK_ALIGN_WINDOW_S", "0.3")),
            target_chunk_size=self._target_chunk_size_from_env(),
            postprocess_actions=self._bool_env("POSTPROCESS_ACTIONS", True),
            alignment_mode=os.environ.get("ACTION_ALIGNMENT_MODE", "l2"),
            refill_margin_s=float(os.environ.get("REFILL_MARGIN_S", "0.2")),
            latency_warmup_samples=int(
                os.environ.get("REFILL_LATENCY_WARMUP_SAMPLES", "1")
            ),
            max_refill_latency_s=self._optional_float_env(
                "REFILL_LATENCY_SAMPLE_MAX_S", "2.0"
            ),
            action_request_mode=os.environ.get("ACTION_REQUEST_MODE", "async"),
        )
        self._response_class: dict[str, Any] = {}
        self._handler = ServiceHandler(
            self._session,
            None,
            self._control_loop,
            lambda **kwargs: self._response_class["class"](**kwargs),
            catalog=self._catalog,
            worker_registry=self._workers,
        )
        self._control_loop.set_fault_callback(self._handler.on_control_fault)
        self._services: list[Any] = []
        self._subscribers: list[Any] = []
        self._control_server = RuntimeControlServer(
            self._handle_control_request,
            os.environ.get(
                "POLICY_RUNTIME_CONTROL_SOCKET",
                "/run/cyclo/policy-runtime.sock",
            ),
            on_response_failure=self._handler.release_undelivered_worker_mutation,
        )
        self._shutdown = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._orchestrator_last_seen = time.monotonic()
        self._active_since: float | None = None

    def start(self) -> None:
        self._control_loop.run_background()
        self._start_services()
        self._start_heartbeat_subscribers()
        self._control_server.start()
        self._write_ready_marker()
        self._monitor_thread = threading.Thread(
            target=self._monitor_health,
            daemon=True,
            name="policy-runtime-health",
        )
        self._monitor_thread.start()
        logger.info("Policy Runtime ready at /policy/inference_command")
        while not self._shutdown.is_set():
            self._shutdown.wait(timeout=1.0)

    def shutdown(self) -> None:
        snapshot = self._handler.runtime_snapshot()
        if snapshot["runtime_state"] in {"running", "syncing", "error"}:
            for _ in range(3):
                if self._handler.fail_safe("policy runtime shutting down"):
                    break
                time.sleep(0.1)
        self._shutdown.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        self._control_server.close()
        for subscriber in self._subscribers:
            try:
                subscriber.close()
            except Exception:
                pass
        self._subscribers.clear()
        for service in self._services:
            try:
                service.close()
            except Exception:
                pass
        self._services.clear()
        self._control_loop.shutdown()
        self._workers.close()
        self._remove_ready_marker()

    def request_shutdown(self, *_args) -> None:
        self._shutdown.set()

    def _start_services(self) -> None:
        central = self._make_service("/policy/inference_command")
        self._response_class["class"] = central.response_msg_class
        self._services.append(central)
        for runtime_id in self._workers.runtime_ids:
            self._services.append(
                self._make_service(
                    f"/{runtime_id}/inference_command",
                    backend_override=runtime_id,
                )
            )
        logger.info(
            "legacy inference aliases enabled for: %s",
            ", ".join(self._workers.runtime_ids),
        )

    def _make_service(
        self,
        service_name: str,
        *,
        backend_override: str = "",
    ) -> Any:
        def callback(request):
            self._orchestrator_last_seen = time.monotonic()
            return self._handler.handle(request, backend_override=backend_override)

        return ROS2ServiceServer(
            service_name=service_name,
            srv_type="interfaces/srv/InferenceCommand",
            callback=callback,
            request_definition=INFERENCE_COMMAND_REQUEST_DEF,
            response_definition=INFERENCE_COMMAND_RESPONSE_DEF,
            router_ip=self._router_ip,
            router_port=self._router_port,
            domain_id=self._domain_id,
            node_name="cyclo_policy_runtime",
            namespace=self._namespace,
        )

    def _start_heartbeat_subscribers(self) -> None:
        self._subscribers.append(
            ROS2Subscriber(
                topic="/heartbeat",
                msg_type="std_msgs/msg/Empty",
                callback=lambda _msg: self._record_orchestrator_heartbeat(),
                router_ip=self._router_ip,
                router_port=self._router_port,
                domain_id=self._domain_id,
                node_name="policy_runtime_orchestrator_watchdog",
                namespace=self._namespace,
            )
        )
        for runtime_id in self._workers.runtime_ids:
            self._subscribers.append(
                ROS2Subscriber(
                    topic=f"/{runtime_id}/worker_heartbeat",
                    msg_type="std_msgs/msg/String",
                    msg_definition="string data\n",
                    callback=lambda msg, rid=runtime_id: self._workers.record_heartbeat(
                        rid,
                        str(getattr(msg, "data", "")),
                    ),
                    router_ip=self._router_ip,
                    router_port=self._router_port,
                    domain_id=self._domain_id,
                    node_name=f"policy_runtime_{runtime_id}_watchdog",
                    namespace=self._namespace,
                )
            )

    def _record_orchestrator_heartbeat(self) -> None:
        self._orchestrator_last_seen = time.monotonic()

    def _monitor_health(self) -> None:
        worker_timeout = max(
            0.5,
            float(os.environ.get("WORKER_HEARTBEAT_TIMEOUT_S", "2.0")),
        )
        orchestrator_timeout = max(
            1.0,
            float(os.environ.get("ORCHESTRATOR_HEARTBEAT_TIMEOUT_S", "3.0")),
        )
        while not self._shutdown.wait(0.25):
            snapshot = self._handler.runtime_snapshot()
            active = snapshot["runtime_state"] in {"running", "syncing"}
            if not active:
                self._active_since = None
                continue
            now = time.monotonic()
            if self._active_since is None:
                self._active_since = now
            runtime_id = snapshot["runtime_id"]
            age = self._workers.heartbeat_age(runtime_id)
            orchestrator_age = now - self._orchestrator_last_seen
            reason = runtime_health_failure_reason(
                runtime_id,
                active_for_s=now - self._active_since,
                worker_heartbeat_age_s=age,
                worker_instance_changed=self._workers.worker_instance_changed(
                    runtime_id
                ),
                orchestrator_heartbeat_age_s=orchestrator_age,
                worker_timeout_s=worker_timeout,
                orchestrator_timeout_s=orchestrator_timeout,
            )
            if reason:
                self._handler.fail_safe(reason)

    def _handle_control_request(self, request: dict) -> dict:
        operation = str(request.get("operation", "status"))
        if operation == "status":
            return {"ok": True, **self._handler.runtime_snapshot(blocking=False)}
        if operation == "can_mutate_worker":
            runtime_id = str(request.get("runtime_id", ""))
            self._require_runtime_id(runtime_id)
            allowed, reason = self._handler.can_mutate_worker(runtime_id)
            return {"ok": True, "allowed": allowed, "reason": reason}
        if operation == "begin_worker_mutation":
            runtime_id = str(request.get("runtime_id", ""))
            self._require_runtime_id(runtime_id)
            allowed, reason, token = self._handler.begin_worker_mutation(runtime_id)
            return {
                "ok": True,
                "allowed": allowed,
                "reason": reason,
                "token": token,
            }
        if operation == "end_worker_mutation":
            runtime_id = str(request.get("runtime_id", ""))
            self._require_runtime_id(runtime_id)
            released = self._handler.end_worker_mutation(
                runtime_id,
                str(request.get("token", "")),
            )
            if released:
                return {"ok": True}
            return {"ok": False, "error": "invalid worker mutation token"}
        if operation == "worker_status":
            runtime_id = str(request.get("runtime_id", ""))
            self._require_runtime_id(runtime_id)
            return {"ok": True, **self._workers.describe(runtime_id)}
        return {"ok": False, "error": f"unknown operation: {operation}"}

    def _require_runtime_id(self, runtime_id: str) -> None:
        if runtime_id not in self._workers.runtime_ids:
            raise ValueError(f"unknown policy runtime: {runtime_id!r}")

    @staticmethod
    def _ready_marker_path() -> Path:
        return Path(
            os.environ.get(
                "POLICY_RUNTIME_READY_MARKER",
                "/run/cyclo/policy-runtime.ready",
            )
        )

    def _write_ready_marker(self) -> None:
        marker = self._ready_marker_path()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ready\n", encoding="utf-8")

    def _remove_ready_marker(self) -> None:
        try:
            self._ready_marker_path().unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _target_chunk_size_from_env() -> int | None:
        raw = os.environ.get("TARGET_CHUNK_SIZE", "none").strip().lower()
        if raw in {"", "none", "off", "0"}:
            return None
        return int(raw)

    @staticmethod
    def _optional_float_env(name: str, default: str) -> float | None:
        raw = os.environ.get(name, default).strip().lower()
        if raw in {"", "none", "off", "0"}:
            return None
        return float(raw)


MainRuntime = PolicyRuntime


def main() -> None:  # pragma: no cover - container entrypoint.
    runtime = PolicyRuntime(
        router_ip=os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1"),
        router_port=int(os.environ.get("ZENOH_ROUTER_PORT", "7447")),
        domain_id=int(os.environ.get("ROS_DOMAIN_ID", "30")),
    )
    signal.signal(signal.SIGTERM, runtime.request_shutdown)
    signal.signal(signal.SIGINT, runtime.request_shutdown)
    try:
        runtime.start()
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
