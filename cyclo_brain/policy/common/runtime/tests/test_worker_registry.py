#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = RUNTIME_ROOT.parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))
if str(POLICY_ROOT / "common") not in sys.path:
    sys.path.insert(0, str(POLICY_ROOT / "common"))

from catalog import load_catalog  # noqa: E402
from engine_process.protocol import (  # noqa: E402
    CMD_DESCRIBE,
    ENGINE_PROTOCOL_VERSION,
    EngineCommandResponse,
)
from main_runtime.worker_registry import (  # noqa: E402
    WorkerCompatibilityError,
    WorkerRegistry,
    runtime_health_failure_reason,
)


class FakeClient:
    def __init__(self, descriptor, **kwargs):
        self.descriptor = descriptor
        self.kwargs = kwargs
        self.closed = False

    def call(self, request, timeout_s):
        assert request.command == CMD_DESCRIBE
        self.descriptor.seq_id = request.seq_id
        return self.descriptor

    def close(self):
        self.closed = True


class WorkerRegistryTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog(POLICY_ROOT)

    def _registry(self, descriptor):
        clients = []

        def factory(**kwargs):
            client = FakeClient(descriptor, **kwargs)
            clients.append(client)
            return client

        return WorkerRegistry(
            self.catalog,
            router_ip="127.0.0.1",
            router_port=7447,
            domain_id=30,
            client_factory=factory,
        ), clients

    def _descriptor(self, runtime_id="lerobot", **overrides):
        runtime = next(
            value
            for value in self.catalog["runtimes"]
            if value["id"] == runtime_id
        )
        values = {
            "success": True,
            "protocol_version": ENGINE_PROTOCOL_VERSION,
            "runtime_id": runtime_id,
            "worker_instance_id": "worker-a",
            "supported_policy_ids": [
                model["policy_id"] for model in runtime["models"]
            ],
            "capabilities_json": json.dumps(
                runtime["capabilities"], sort_keys=True
            ),
            "engine_state": "unloaded",
        }
        values.update(overrides)
        return EngineCommandResponse(**values)

    def test_requester_requires_matching_runtime_protocol_and_policy(self):
        descriptor = self._descriptor()
        registry, clients = self._registry(descriptor)

        requester = registry.requester("lerobot", "lerobot:act")

        self.assertIsNotNone(requester)
        self.assertEqual(clients[0].kwargs["service_name"], "/lerobot/engine_command")
        registry.close()
        self.assertTrue(clients[0].closed)

    def test_requester_rejects_incompatible_protocol(self):
        descriptor = self._descriptor(protocol_version="2.0")
        registry, _clients = self._registry(descriptor)

        with self.assertRaisesRegex(WorkerCompatibilityError, "incompatible"):
            registry.requester("lerobot", "lerobot:act")

    def test_requester_rejects_policy_missing_from_worker(self):
        descriptor = self._descriptor(
            supported_policy_ids=["lerobot:diffusion"]
        )
        registry, _clients = self._registry(descriptor)

        with self.assertRaisesRegex(WorkerCompatibilityError, "catalog mismatch"):
            registry.requester("lerobot", "lerobot:act")

    def test_describe_rejects_capabilities_that_do_not_match_catalog(self):
        descriptor = self._descriptor(capabilities_json="{}")
        registry, _clients = self._registry(descriptor)

        with self.assertRaisesRegex(WorkerCompatibilityError, "capability mismatch"):
            registry.describe("lerobot")

    def test_describe_rejects_invalid_capabilities_json(self):
        descriptor = self._descriptor(capabilities_json="not-json")
        registry, _clients = self._registry(descriptor)

        with self.assertRaisesRegex(WorkerCompatibilityError, "invalid capabilities"):
            registry.describe("lerobot")

    def test_explicit_describe_failure_is_not_a_waiting_status(self):
        registry, _ = self._registry(self._descriptor(success=False, message="Worker failed"))
        with self.assertRaisesRegex(WorkerCompatibilityError, "Worker failed"):
            registry.describe("lerobot")

    def test_describe_uses_a_dedicated_client_separate_from_inference(self):
        descriptor = self._descriptor()
        registry, clients = self._registry(descriptor)
        registry.requester("lerobot", "lerobot:act")

        registry.describe("lerobot")

        self.assertEqual(len(clients), 2)
        self.assertFalse(clients[0].closed)
        self.assertFalse(clients[1].closed)
        registry.close()
        self.assertTrue(clients[0].closed)
        self.assertTrue(clients[1].closed)

    def test_readiness_timeout_is_bounded_and_recovers(self):
        registry, _ = self._registry(self._descriptor())
        requester = registry._get_or_create_status_requester("lerobot")
        with patch.object(requester, "describe", side_effect=TimeoutError("seq=425")):
            with patch.dict("os.environ", {"WORKER_READY_TIMEOUT_S": "120"}):
                with patch("main_runtime.worker_registry.time.monotonic", return_value=10):
                    waiting = registry.describe("lerobot")
                self.assertEqual(waiting["readiness"], "waiting")
                self.assertNotIn("seq=", waiting["message"])
                with patch("main_runtime.worker_registry.time.monotonic", return_value=130):
                    self.assertEqual(registry.describe("lerobot")["readiness"], "error")
        self.assertEqual(registry.describe("lerobot")["runtime_id"], "lerobot")
        self.assertNotIn("lerobot", registry._describe_wait_since)

    def test_loading_requires_fresh_heartbeat_and_has_a_deadline(self):
        registry, _ = self._registry(self._descriptor())
        requester = registry._get_or_create_status_requester("lerobot")
        with patch.object(requester, "describe", side_effect=TimeoutError()):
            with patch.dict("os.environ", {"LOAD_POLICY_TIMEOUT_S": "300", "WORKER_READY_TIMEOUT_S": "120"}):
                for now, heartbeat_at, expected in (
                    (10, 10, "waiting"), (200, 200, "waiting"),
                    (200, 10, "error"), (310, 310, "error"),
                ):
                    registry._heartbeat_payload["lerobot"] = {"engine_state": "loading"}
                    registry._heartbeat_at["lerobot"] = heartbeat_at
                    with patch("main_runtime.worker_registry.time.monotonic", return_value=now):
                        result = registry.describe("lerobot")
                    self.assertEqual(result["readiness"], expected)
                    if expected == "waiting":
                        self.assertEqual(result["message"], "Model is loading...")

    def test_running_worker_timeout_is_not_reported_as_loading(self):
        registry, _ = self._registry(self._descriptor())
        registry._heartbeat_payload["lerobot"] = {"engine_state": "running"}
        requester = registry._get_or_create_status_requester("lerobot")
        with patch.object(requester, "describe", side_effect=TimeoutError()):
            self.assertEqual(registry.describe("lerobot")["readiness"], "error")

    def test_heartbeat_detects_validated_worker_restart(self):
        descriptor = self._descriptor()
        registry, _clients = self._registry(descriptor)
        registry.requester("lerobot", "lerobot:act")
        registry.record_heartbeat(
            "lerobot",
            json.dumps(
                {
                    "protocol_version": ENGINE_PROTOCOL_VERSION,
                    "runtime_id": "lerobot",
                    "worker_instance_id": "worker-b",
                }
            ),
        )

        self.assertIsNotNone(registry.heartbeat_age("lerobot"))
        self.assertTrue(registry.worker_instance_changed("lerobot"))

    def test_runtime_health_rejects_worker_restart(self):
        reason = runtime_health_failure_reason(
            "lerobot",
            active_for_s=0.1,
            worker_heartbeat_age_s=0.1,
            worker_instance_changed=True,
            orchestrator_heartbeat_age_s=0.1,
            worker_timeout_s=2.0,
            orchestrator_timeout_s=3.0,
        )

        self.assertEqual(reason, "lerobot worker restarted")

    def test_runtime_health_allows_heartbeat_startup_grace(self):
        reason = runtime_health_failure_reason(
            "groot",
            active_for_s=1.9,
            worker_heartbeat_age_s=None,
            worker_instance_changed=False,
            orchestrator_heartbeat_age_s=0.1,
            worker_timeout_s=2.0,
            orchestrator_timeout_s=3.0,
        )

        self.assertEqual(reason, "")

    def test_runtime_health_rejects_missing_and_stale_heartbeats(self):
        missing = runtime_health_failure_reason(
            "groot",
            active_for_s=2.1,
            worker_heartbeat_age_s=None,
            worker_instance_changed=False,
            orchestrator_heartbeat_age_s=0.1,
            worker_timeout_s=2.0,
            orchestrator_timeout_s=3.0,
        )
        stale = runtime_health_failure_reason(
            "groot",
            active_for_s=5.0,
            worker_heartbeat_age_s=2.5,
            worker_instance_changed=False,
            orchestrator_heartbeat_age_s=0.1,
            worker_timeout_s=2.0,
            orchestrator_timeout_s=3.0,
        )

        self.assertEqual(missing, "groot worker heartbeat was not received")
        self.assertEqual(stale, "groot worker heartbeat stale for 2.50s")

    def test_runtime_health_rejects_orchestrator_heartbeat_timeout(self):
        reason = runtime_health_failure_reason(
            "lerobot",
            active_for_s=5.0,
            worker_heartbeat_age_s=0.1,
            worker_instance_changed=False,
            orchestrator_heartbeat_age_s=3.5,
            worker_timeout_s=2.0,
            orchestrator_timeout_s=3.0,
        )

        self.assertEqual(reason, "orchestrator heartbeat stale for 3.50s")


if __name__ == "__main__":
    unittest.main()
