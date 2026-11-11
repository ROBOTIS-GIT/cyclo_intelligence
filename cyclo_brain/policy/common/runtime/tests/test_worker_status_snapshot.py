"""Shared Worker probes cannot occupy the Unix lifecycle control handler."""

import json
import socket
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from .test_preparation_health import runtime_module
from . import test_worker_registry as registry_fixtures
from main_runtime.runtime_control import RuntimeControlServer


def request(path, payload, timeout=.5):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(json.dumps(payload).encode())
        client.shutdown(socket.SHUT_WR)
        return json.loads(client.recv(65536))


@pytest.fixture
def registry_setup():
    helper = registry_fixtures.WorkerRegistryTests()
    helper.setUp()
    descriptor = helper._descriptor()
    registry, clients = helper._registry(descriptor)
    registry._STATUS_REFRESH_S = 3600.
    try:
        yield registry, descriptor, clients
    finally:
        registry.close()


def wait_snapshot(registry, runtime="lerobot", predicate=lambda value: "runtime_id" in value):
    deadline = time.monotonic() + 2.
    while time.monotonic() < deadline:
        result = registry.status_snapshot(runtime)
        if predicate(result):
            return result
        time.sleep(.005)
    pytest.fail(f"status snapshot did not meet condition: {result}")


def heartbeat(registry, instance="worker-a", state="loaded"):
    registry.record_heartbeat("lerobot", json.dumps({
        "protocol_version": registry_fixtures.ENGINE_PROTOCOL_VERSION,
        "runtime_id": "lerobot", "worker_instance_id": instance, "engine_state": state,
    }))


@pytest.mark.parametrize("delay_stage", ["probe", "client_creation"])
def test_slow_worker_probe_does_not_block_unix_mutation_check(runtime_module, tmp_path, delay_stage):
    helper = registry_fixtures.WorkerRegistryTests()
    helper.setUp()
    registry, _ = helper._registry(helper._descriptor())
    if delay_stage == "probe":
        requester = registry._get_or_create_status_requester("lerobot")
        original = requester.describe
    else:
        original = registry._client_factory
    entered, release = threading.Event(), threading.Event()
    probes, responses, failures = [], [], []

    def slow(**kwargs):
        probes.append(True)
        entered.set()
        assert release.wait(3)
        return original(**kwargs)

    if delay_stage == "probe":
        requester.describe = slow
    else:
        registry._client_factory = slow
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    runtime._shutdown = threading.Event()
    runtime._workers = registry
    runtime._handler = SimpleNamespace(can_mutate_worker=lambda runtime_id: (True, ""))
    path = tmp_path / "runtime.sock"
    server = RuntimeControlServer(runtime._handle_control_request, str(path))

    def first_request():
        try:
            responses.append(request(path, {"operation": "worker_status", "runtime_id": "lerobot"}, timeout=1.))
        except Exception as exc:
            failures.append(exc)

    server.start()
    ui = threading.Thread(target=first_request)
    ui.start()
    try:
        assert entered.wait(1.)
        result = request(path, {"operation": "can_mutate_worker", "runtime_id": "lerobot"})
        assert result["ok"] and result["allowed"]
        assert not release.is_set()
        for _ in range(10):
            state = request(path, {"operation": "worker_status", "runtime_id": "lerobot"})
            assert state["readiness"] == "waiting"
        ui.join(1.)
        assert not ui.is_alive() and not failures
        assert responses[0]["readiness"] == "waiting"
        assert len(probes) == 1
    finally:
        release.set()
        ui.join(2.)
        server.close()
        registry.close()


def test_many_readers_share_descriptor_without_refreshing_its_age(registry_setup):
    registry, _, _ = registry_setup
    heartbeat(registry)
    requester = registry._get_or_create_status_requester("lerobot")
    with mock.patch.object(requester, "describe", wraps=requester.describe) as probe:
        state = wait_snapshot(registry)
        assert state["engine_state"] == "loaded"
        stamp = registry._status_cache["lerobot"][0]
        state["supported_policy_ids"].clear()
        for _ in range(100):
            result = registry.status_snapshot("lerobot")
            assert result["supported_policy_ids"]
        assert probe.call_count == 1
        assert registry._status_cache["lerobot"][0] == stamp
        with registry._lock:
            cached = registry._status_cache["lerobot"][1]
            registry._status_cache["lerobot"] = (stamp - 10., cached)
        assert registry.status_snapshot("lerobot")["readiness"] == "waiting"
        assert probe.call_count == 1


def test_cached_status_updates_heartbeat_age_and_does_not_accept_a_new_instance(registry_setup):
    registry, _, _ = registry_setup
    registry.requester("lerobot", "lerobot:act")
    heartbeat(registry)
    wait_snapshot(registry)
    with registry._lock:
        registry._heartbeat_at["lerobot"] = time.monotonic() - 5.
    assert registry.status_snapshot("lerobot")["heartbeat_age_s"] >= 5.
    heartbeat(registry, state="running")
    assert registry.status_snapshot("lerobot")["engine_state"] == "running"
    heartbeat(registry, instance="worker-b")
    result = registry.status_snapshot("lerobot")
    assert result["readiness"] == "waiting" and "restarted" in result["message"]
    assert registry.worker_instance_changed("lerobot")


def test_cached_readiness_does_not_bypass_fresh_load_compatibility_check(registry_setup):
    registry, descriptor, clients = registry_setup
    heartbeat(registry)
    wait_snapshot(registry)
    descriptor.supported_policy_ids = []
    with pytest.raises(registry_fixtures.WorkerCompatibilityError, match="catalog mismatch"):
        registry.requester("lerobot", "lerobot:act")
    assert {client.kwargs["service_name"] for client in clients} == {
        "/lerobot/engine_status", "/lerobot/engine_command",
    }


def test_restart_during_probe_does_not_publish_old_descriptor_as_current(registry_setup):
    registry, descriptor, _ = registry_setup
    registry._STATUS_REFRESH_S = .02
    heartbeat(registry)
    requester = registry._get_or_create_status_requester("lerobot")
    entered, release = threading.Event(), threading.Event()
    original = requester.describe

    def old_probe(**kwargs):
        entered.set()
        assert release.wait(2.)
        return original(**kwargs)

    with mock.patch.object(requester, "describe", side_effect=old_probe):
        registry.status_snapshot("lerobot")
        try:
            assert entered.wait(1.)
            heartbeat(registry, instance="worker-b")
        finally:
            release.set()
        result = wait_snapshot(registry, predicate=lambda value: "restarted" in value.get("message", ""))
        assert result["readiness"] == "waiting"
    descriptor.worker_instance_id = "worker-b"
    result = wait_snapshot(registry)
    assert result["worker_instance_id"] == "worker-b"


def test_background_failure_replaces_ready_state_and_can_recover(registry_setup):
    registry, descriptor, _ = registry_setup
    registry._STATUS_REFRESH_S = .02
    heartbeat(registry)
    wait_snapshot(registry)
    descriptor.capabilities_json = "{}"
    failed = wait_snapshot(registry, predicate=lambda value: value.get("readiness") == "error")
    assert "capability mismatch" in failed["message"]
    runtime = next(r for r in registry._catalog["runtimes"] if r["id"] == "lerobot")
    descriptor.capabilities_json = json.dumps(runtime["capabilities"])
    assert "runtime_id" in wait_snapshot(registry)


def test_idle_polling_stops_and_close_prevents_new_clients(registry_setup):
    registry, _, clients = registry_setup
    registry._STATUS_REFRESH_S = .02
    registry._STATUS_IDLE_S = .04
    wait_snapshot(registry)
    thread = registry._status_threads["lerobot"]
    thread.join(1.)
    assert not thread.is_alive()
    assert "lerobot" not in registry._status_threads
    registry.status_snapshot("lerobot")
    second = registry._status_threads["lerobot"]
    assert second is not thread
    registry.close()
    assert not second.is_alive()
    assert all(client.closed for client in clients)
    assert not registry._status_cache and not registry._status_threads
    assert registry.status_snapshot("lerobot")["readiness"] == "error"
    with pytest.raises(RuntimeError, match="closed"):
        registry.requester("lerobot", "lerobot:act")


def test_missing_worker_identity_is_not_cacheable_readiness(registry_setup):
    registry, descriptor, _ = registry_setup
    descriptor.worker_instance_id = ""
    status = wait_snapshot(registry, predicate=lambda value: value.get("readiness") == "error")
    assert "instance ID" in status["message"]


def test_unidentified_heartbeat_cannot_freshen_a_cached_worker(registry_setup):
    registry, _, _ = registry_setup
    heartbeat(registry)
    wait_snapshot(registry)
    registry.record_heartbeat("lerobot", json.dumps({
        "protocol_version": registry_fixtures.ENGINE_PROTOCOL_VERSION,
        "runtime_id": "lerobot", "engine_state": "running",
    }))
    assert registry.status_snapshot("lerobot")["readiness"] == "waiting"


def test_client_creation_is_shared_without_blocking_heartbeat_or_inference_client(registry_setup):
    registry, _, _ = registry_setup
    original = registry._client_factory
    entered, release = threading.Event(), threading.Event()
    creations, results, errors = [], [], []

    def factory(**kwargs):
        creations.append(kwargs["service_name"])
        if kwargs["service_name"].endswith("engine_status"):
            entered.set()
            assert release.wait(2.)
        return original(**kwargs)

    def create():
        try:
            results.append(registry._get_or_create_status_requester("lerobot"))
        except Exception as exc:
            errors.append(exc)

    registry._client_factory = factory
    callers = [threading.Thread(target=create) for _ in range(5)]
    for thread in callers:
        thread.start()
    try:
        assert entered.wait(1.)
        heartbeat(registry)
        assert registry.heartbeat_age("lerobot") is not None
        registry._get_or_create("lerobot")
        assert creations.count("/lerobot/engine_status") == 1
        assert creations.count("/lerobot/engine_command") == 1
        assert not release.is_set()
    finally:
        release.set()
        for thread in callers:
            thread.join(2.)
            assert not thread.is_alive()
    assert not errors and len(results) == 5
    assert all(value is results[0] for value in results)


def test_close_during_client_creation_discards_and_closes_the_late_client(registry_setup):
    registry, _, clients = registry_setup
    original = registry._client_factory
    entered, release = threading.Event(), threading.Event()
    errors = []

    def factory(**kwargs):
        entered.set()
        assert release.wait(2.)
        return original(**kwargs)

    def create():
        try:
            registry._get_or_create("lerobot")
        except RuntimeError as exc:
            errors.append(str(exc))

    registry._client_factory = factory
    caller = threading.Thread(target=create)
    caller.start()
    try:
        assert entered.wait(1.)
        registry.close()
        assert not release.is_set()
    finally:
        release.set()
        caller.join(2.)
    assert not caller.is_alive()
    assert errors == ["Worker registry closed during client creation"]
    assert len(clients) == 1 and clients[0].closed
    assert not registry._requesters


def test_slow_runtime_does_not_delay_another_runtime_descriptor():
    helper = registry_fixtures.WorkerRegistryTests()
    helper.setUp()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def factory(**kwargs):
        runtime_id = kwargs["service_name"].split("/")[1]

        def call(req, timeout_s):
            calls.append(runtime_id)
            if runtime_id == "lerobot":
                entered.set()
                assert release.wait(2.)
            descriptor = helper._descriptor(runtime_id)
            descriptor.seq_id = req.seq_id
            return descriptor

        return SimpleNamespace(call=call, close=lambda: None)

    registry = registry_fixtures.WorkerRegistry(helper.catalog, router_ip="127.0.0.1", router_port=7447,
                                               domain_id=30, client_factory=factory)
    registry._STATUS_REFRESH_S = 3600.
    try:
        registry.status_snapshot("lerobot")
        assert entered.wait(1.)
        assert wait_snapshot(registry, runtime="groot")["runtime_id"] == "groot"
        assert not release.is_set()
        assert calls.count("groot") == calls.count("lerobot") == 1
    finally:
        release.set()
        registry.close()
