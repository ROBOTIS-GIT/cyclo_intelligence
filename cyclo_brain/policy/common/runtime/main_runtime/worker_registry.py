#!/usr/bin/env python3

"""Worker discovery, compatibility checks, and requester lifecycle."""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from typing import Any, Callable

from catalog import resolve_runtime
from engine_process.protocol import ENGINE_PROTOCOL_VERSION

from .inference_requester import DEFAULT_LOAD_POLICY_TIMEOUT_S, InferenceRequester
from .zenoh_client import ZenohEngineCommandClient

logger = logging.getLogger(__name__)


class WorkerCompatibilityError(RuntimeError):
    pass


def _protocol_major(version: str) -> str:
    return str(version or "").split(".", 1)[0]


def runtime_health_failure_reason(
    runtime_id: str,
    *,
    active_for_s: float,
    worker_heartbeat_age_s: float | None,
    worker_instance_changed: bool,
    orchestrator_heartbeat_age_s: float,
    worker_timeout_s: float,
    orchestrator_timeout_s: float,
) -> str:
    """Return the fail-safe reason for one active-session health sample."""
    if worker_instance_changed:
        return f"{runtime_id} worker restarted"
    if worker_heartbeat_age_s is None:
        if active_for_s > worker_timeout_s:
            return f"{runtime_id} worker heartbeat was not received"
        return ""
    if worker_heartbeat_age_s > worker_timeout_s:
        return (
            f"{runtime_id} worker heartbeat stale for "
            f"{worker_heartbeat_age_s:.2f}s"
        )
    if orchestrator_heartbeat_age_s > orchestrator_timeout_s:
        return (
            "orchestrator heartbeat stale for "
            f"{orchestrator_heartbeat_age_s:.2f}s"
        )
    return ""


class WorkerRegistry:
    _STATUS_REFRESH_S = 1.0
    _STATUS_MAX_AGE_S = 3.0
    _STATUS_IDLE_S = 10.0

    def __init__(
        self,
        catalog: dict[str, Any],
        *,
        router_ip: str,
        router_port: int,
        domain_id: int,
        namespace: str = "/",
        client_factory: Callable[..., Any] = ZenohEngineCommandClient,
    ) -> None:
        self._catalog = catalog
        self._router_ip = router_ip
        self._router_port = int(router_port)
        self._domain_id = int(domain_id)
        self._namespace = namespace
        self._client_factory = client_factory
        self._requesters: dict[str, InferenceRequester] = {}
        self._status_requesters: dict[str, InferenceRequester] = {}
        self._requester_locks: dict[tuple[str, str], threading.Lock] = {}
        self._heartbeat_at: dict[str, float] = {}
        self._heartbeat_payload: dict[str, dict[str, Any]] = {}
        self._validated_instance: dict[str, str] = {}
        self._describe_wait_since: dict[str, float] = {}
        self._lock = threading.RLock()
        self._status_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._status_last_requested: dict[str, float] = {}
        self._status_threads: dict[str, threading.Thread] = {}
        self._status_shutdown = threading.Event()
        self._closed = False

    @property
    def runtime_ids(self) -> list[str]:
        return [runtime["id"] for runtime in self._catalog.get("runtimes", [])]

    def requester(self, runtime_id: str, policy_id: str) -> InferenceRequester:
        requester = self._get_or_create(runtime_id)
        descriptor = requester.describe(
            timeout_s=float(os.environ.get("WORKER_DESCRIBE_TIMEOUT_S", "2.0"))
        )
        self._validate_descriptor(runtime_id, descriptor, policy_id=policy_id)
        with self._lock:
            self._validated_instance[runtime_id] = descriptor.worker_instance_id
        return requester

    def describe(self, runtime_id: str) -> dict[str, Any]:
        resolve_runtime(self._catalog, runtime_id)
        requester = self._get_or_create_status_requester(runtime_id)
        try:
            response = requester.describe(
                timeout_s=float(os.environ.get("WORKER_DESCRIBE_TIMEOUT_S", "2.0"))
            )
        except TimeoutError as exc:
            logger.debug("Worker %s readiness probe: %s", runtime_id, exc)
            return self._describe_timeout_status(runtime_id)
        with self._lock:
            self._describe_wait_since.pop(runtime_id, None)
        self._validate_descriptor(runtime_id, response)
        return {
            "protocol_version": response.protocol_version,
            "runtime_id": response.runtime_id,
            "worker_instance_id": response.worker_instance_id,
            "supported_policy_ids": list(response.supported_policy_ids),
            "capabilities_json": response.capabilities_json,
            "engine_state": response.engine_state,
            "heartbeat_age_s": self.heartbeat_age(runtime_id),
        }

    def status_snapshot(self, runtime_id: str) -> dict[str, Any]:
        """Nonblocking shared readiness; never use this cache to authorize LOAD."""
        resolve_runtime(self._catalog, runtime_id)
        now = time.monotonic()
        with self._lock:
            if self._closed:
                return {"readiness": "error", "message": "Policy Runtime is shutting down"}
            self._status_last_requested[runtime_id] = now
            if runtime_id not in self._status_threads:
                thread = threading.Thread(target=self._poll_status, args=(runtime_id,), daemon=True,
                                          name=f"worker-status-{runtime_id}")
                self._status_threads[runtime_id] = thread
                try:
                    thread.start()
                except Exception:
                    self._status_threads.pop(runtime_id, None)
                    raise
            cached = self._status_cache.get(runtime_id)
            heartbeat_at = self._heartbeat_at.get(runtime_id)
            payload = self._heartbeat_payload.get(runtime_id, {})
            if cached is None or now - cached[0] > self._STATUS_MAX_AGE_S:
                return {"readiness": "waiting", "message": "Checking model worker response..."}
            status = copy.deepcopy(cached[1])
            if "worker_instance_id" not in status:
                return status
            observed = payload.get("worker_instance_id")
            if heartbeat_at is not None and observed != status["worker_instance_id"]:
                return {"readiness": "waiting", "message": (
                    "Worker restarted; checking compatibility..." if observed
                    else "Waiting for an identified worker heartbeat..."
                )}
            age = None if heartbeat_at is None else max(0., now - heartbeat_at)
            status["heartbeat_age_s"] = age
            # Descriptor compatibility is cached briefly; live state and age
            # come from the same worker instance's most recent heartbeat.
            if observed == status["worker_instance_id"] and isinstance(payload.get("engine_state"), str):
                status["engine_state"] = payload["engine_state"]
            return status

    def _poll_status(self, runtime_id: str) -> None:
        try:
            while not self._status_shutdown.is_set():
                with self._lock:
                    idle = time.monotonic() - self._status_last_requested.get(runtime_id, 0.)
                    if self._closed or idle >= self._STATUS_IDLE_S:
                        return
                try:
                    status = self.describe(runtime_id)
                except Exception as exc:
                    status = {"readiness": "error", "message": str(exc)}
                with self._lock:
                    if self._closed:
                        return
                    now = time.monotonic()
                    self._status_cache[runtime_id] = (now, status)
                    if now - self._status_last_requested.get(runtime_id, 0.) >= self._STATUS_IDLE_S:
                        return
                if self._status_shutdown.wait(self._STATUS_REFRESH_S):
                    return
        finally:
            with self._lock:
                if self._status_threads.get(runtime_id) is threading.current_thread():
                    self._status_threads.pop(runtime_id, None)

    def _describe_timeout_status(self, runtime_id: str) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            started = self._describe_wait_since.setdefault(runtime_id, now)
            payload = self._heartbeat_payload.get(runtime_id, {})
            heartbeat_at = self._heartbeat_at.get(runtime_id)
        fresh = heartbeat_at is not None and now - heartbeat_at <= float(
            os.environ.get("WORKER_HEARTBEAT_TIMEOUT_S", "2.0")
        )
        state = payload.get("engine_state")
        loading = fresh and state == "loading"
        limit = float(os.environ.get(
            "LOAD_POLICY_TIMEOUT_S" if loading else "WORKER_READY_TIMEOUT_S",
            str(DEFAULT_LOAD_POLICY_TIMEOUT_S) if loading else "120.0",
        ))
        # Only readiness probes get a bounded grace period; action failures
        # and LOAD responses remain on their existing error/safety paths.
        failed = now - started >= limit or (state in ("loaded", "running", "error"))
        return {
            "readiness": "error" if failed else "waiting",
            "message": (
                "Model worker is not responding. Check worker logs and the Zenoh connection."
                if failed else (
                    "Model is loading..." if loading else "Waiting for model worker response..."
                )
            ),
        }

    def record_heartbeat(self, runtime_id: str, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("runtime_id") != runtime_id:
            return
        if _protocol_major(payload.get("protocol_version", "")) != _protocol_major(
            ENGINE_PROTOCOL_VERSION
        ):
            return
        with self._lock:
            self._heartbeat_at[runtime_id] = time.monotonic()
            self._heartbeat_payload[runtime_id] = payload

    def heartbeat_age(self, runtime_id: str) -> float | None:
        with self._lock:
            timestamp = self._heartbeat_at.get(runtime_id)
        return None if timestamp is None else max(0.0, time.monotonic() - timestamp)

    def worker_instance_changed(self, runtime_id: str) -> bool:
        with self._lock:
            expected = self._validated_instance.get(runtime_id)
            observed = self._heartbeat_payload.get(runtime_id, {}).get(
                "worker_instance_id"
            )
        return bool(expected and observed and expected != observed)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._status_shutdown.set()
            threads = list(self._status_threads.values())
            requesters = [
                *self._requesters.values(),
                *self._status_requesters.values(),
            ]
            self._requesters.clear()
            self._status_requesters.clear()
            self._status_cache.clear()
            self._status_last_requested.clear()
        for thread in threads:
            thread.join(timeout=3.0)
        for requester in requesters:
            close = getattr(requester, "close", None)
            if callable(close):
                close()
            else:
                client = getattr(requester, "_client", None)
                if client is not None and hasattr(client, "close"):
                    client.close()

    def _get_or_create(self, runtime_id: str) -> InferenceRequester:
        return self._get_requester(runtime_id, purpose="inference")

    def _get_or_create_status_requester(
        self,
        runtime_id: str,
    ) -> InferenceRequester:
        return self._get_requester(runtime_id, purpose="status")

    def _get_requester(self, runtime_id: str, *, purpose: str) -> InferenceRequester:
        resolve_runtime(self._catalog, runtime_id)
        requesters = self._status_requesters if purpose == "status" else self._requesters
        with self._lock:
            if self._closed:
                raise RuntimeError("Worker registry is closed")
            existing = requesters.get(runtime_id)
            if existing is not None:
                return existing
            creation_lock = self._requester_locks.setdefault((runtime_id, purpose), threading.Lock())
        # SDK/session creation may wait for middleware. It cannot hold the lock
        # used by heartbeat callbacks, cached status reads or another runtime.
        with creation_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Worker registry is closed")
                existing = requesters.get(runtime_id)
                if existing is not None:
                    return existing
            requester = self._make_requester(runtime_id, purpose=purpose)
            with self._lock:
                if not self._closed:
                    requesters[runtime_id] = requester
                    return requester
            requester.close()
            raise RuntimeError("Worker registry closed during client creation")

    def _make_requester(
        self,
        runtime_id: str,
        *,
        purpose: str,
    ) -> InferenceRequester:
        client = self._client_factory(
            service_name=f"/{runtime_id}/engine_{'status' if purpose == 'status' else 'command'}",
            router_ip=self._router_ip,
            router_port=self._router_port,
            domain_id=self._domain_id,
            node_name=f"policy_runtime_{runtime_id}_{purpose}_client",
            namespace=self._namespace,
        )
        return InferenceRequester(
            client,
            get_action_timeout_s=float(
                os.environ.get("GET_ACTION_TIMEOUT_S", "5.0")
            ),
            load_policy_timeout_s=float(
                os.environ.get(
                    "LOAD_POLICY_TIMEOUT_S",
                    str(DEFAULT_LOAD_POLICY_TIMEOUT_S),
                )
            ),
        )

    def _validate_descriptor(
        self,
        runtime_id: str,
        descriptor: Any,
        *,
        policy_id: str = "",
    ) -> None:
        if not descriptor.success:
            raise WorkerCompatibilityError(
                descriptor.message or f"{runtime_id} worker is unavailable"
            )
        if _protocol_major(descriptor.protocol_version) != _protocol_major(
            ENGINE_PROTOCOL_VERSION
        ):
            raise WorkerCompatibilityError(
                f"{runtime_id} worker protocol {descriptor.protocol_version!r} is "
                f"incompatible with {ENGINE_PROTOCOL_VERSION!r}"
            )
        if descriptor.runtime_id != runtime_id:
            raise WorkerCompatibilityError(
                f"worker runtime mismatch: expected {runtime_id!r}, "
                f"got {descriptor.runtime_id!r}"
            )
        if not isinstance(descriptor.worker_instance_id, str) or not descriptor.worker_instance_id.strip():
            raise WorkerCompatibilityError("worker descriptor requires a non-empty instance ID")
        runtime = resolve_runtime(self._catalog, runtime_id)
        expected_policy_ids = sorted(
            model["policy_id"] for model in runtime.get("models", [])
        )
        actual_policy_ids = sorted(set(descriptor.supported_policy_ids))
        if actual_policy_ids != expected_policy_ids:
            raise WorkerCompatibilityError(
                f"worker {runtime_id!r} policy catalog mismatch: expected "
                f"{expected_policy_ids!r}, got {actual_policy_ids!r}"
            )
        try:
            actual_capabilities = json.loads(descriptor.capabilities_json or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise WorkerCompatibilityError(
                f"worker {runtime_id!r} returned invalid capabilities JSON"
            ) from exc
        expected_capabilities = runtime.get("capabilities", {})
        if not isinstance(actual_capabilities, dict):
            raise WorkerCompatibilityError(
                f"worker {runtime_id!r} capabilities must be a JSON object"
            )
        if actual_capabilities != expected_capabilities:
            raise WorkerCompatibilityError(
                f"worker {runtime_id!r} capability mismatch: expected "
                f"{expected_capabilities!r}, got {actual_capabilities!r}"
            )
        if policy_id and policy_id not in actual_policy_ids:
            raise WorkerCompatibilityError(
                f"worker {runtime_id!r} does not support {policy_id!r}"
            )
