"""Task Engine readiness is discovery only: no ROS service is ever invoked."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))
from supervisor_api.bt_readiness import BTReadiness, DEFAULT_SERVICES, ReadinessUnavailable, create_router


@pytest.fixture
def ros(monkeypatch):
    clients = {name: Mock() for name in (*DEFAULT_SERVICES, '/bt/set_paused')}
    for client in clients.values(): client.service_is_ready.return_value = False
    context = Mock()
    context.ok.return_value = True
    context_factory = Mock(return_value=context)
    node = Mock()
    node.create_client.side_effect = lambda service_type, name: clients[name]
    node_factory = Mock(return_value=node)
    monkeypatch.setitem(sys.modules, 'rclpy.context', SimpleNamespace(Context=context_factory))
    monkeypatch.setitem(sys.modules, 'rclpy.node', SimpleNamespace(Node=node_factory))
    monkeypatch.setitem(sys.modules, 'interfaces.srv', SimpleNamespace(GetNodeCatalog=object(), LoadAndRunTree=object()))
    monkeypatch.setitem(sys.modules, 'std_srvs.srv', SimpleNamespace(SetBool=object()))
    return SimpleNamespace(clients=clients, context=context, context_factory=context_factory,
                           node=node, node_factory=node_factory)


def http(probe):
    app = FastAPI()
    app.include_router(create_router(probe))
    return TestClient(app)


def test_lazy_probe_waits_for_every_required_service_and_reuses_node(ros):
    probe = BTReadiness()
    ros.context_factory.assert_not_called()
    try:
        initial = probe.status()
        assert initial['ready'] is False
        assert initial['services'] == dict.fromkeys(DEFAULT_SERVICES, False)
        assert initial['missing_services'] == list(DEFAULT_SERVICES)
        for name in DEFAULT_SERVICES[:-1]: ros.clients[name].service_is_ready.return_value = True
        waiting = probe.status()
        assert waiting['ready'] is False
        assert waiting['missing_services'] == ['/bt/set_running']
        assert '/bt/set_running' in waiting['reason']
        ros.clients['/bt/set_running'].service_is_ready.return_value = True
        assert probe.status() == {'ready': True, 'services': dict.fromkeys(DEFAULT_SERVICES, True),
                                  'missing_services': [], 'reason': None}
        ros.context_factory.assert_called_once()
        ros.context.init.assert_called_once_with(args=[])
        ros.node_factory.assert_called_once_with('cyclo_bt_readiness', context=ros.context)
        assert ros.node.create_client.call_count == 3
        ros.clients['/bt/set_paused'].service_is_ready.assert_not_called()
        for client in ros.clients.values():
            client.call.assert_not_called()
            client.call_async.assert_not_called()
            client.wait_for_service.assert_not_called()
    finally:
        probe.close()


def test_readiness_detects_a_service_disappearing_after_it_was_ready(ros):
    probe = BTReadiness()
    try:
        for client in ros.clients.values(): client.service_is_ready.return_value = True
        assert probe.status()['ready'] is True
        ros.clients['/bt/nodes/catalog'].service_is_ready.return_value = False
        assert probe.status()['ready'] is False
    finally:
        probe.close()


def test_concurrent_polls_share_one_context_and_node(ros):
    probe = BTReadiness()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: probe.status(), range(16)))
        assert len(results) == 16 and all(result['ready'] is False for result in results)
        ros.context_factory.assert_called_once()
        ros.node_factory.assert_called_once()
    finally:
        probe.close()


def test_internal_ros_failure_discards_clients_and_retry_recovers(ros):
    probe = BTReadiness()
    try:
        ros.clients['/bt/load_and_run'].service_is_ready.side_effect = RuntimeError('DDS disconnected')
        response = http(probe).get('/bt/readiness')
        assert response.status_code == 503
        assert response.json()['ready'] is False
        assert response.json()['services'] == dict.fromkeys(DEFAULT_SERVICES, False)
        assert 'DDS disconnected' in response.json()['reason']
        ros.node.destroy_node.assert_called_once()
        ros.context.try_shutdown.assert_called_once()
        ros.clients['/bt/load_and_run'].service_is_ready.side_effect = None
        for client in ros.clients.values(): client.service_is_ready.return_value = True
        assert probe.status()['ready'] is True
        assert ros.node_factory.call_count == 2
    finally:
        probe.close()


def test_missing_ros_dependencies_fail_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, 'rclpy.context', None)
    probe = BTReadiness()
    response = http(probe).get('/bt/readiness')
    assert response.status_code == 503
    assert response.json()['ready'] is False
    assert response.json()['services'] == dict.fromkeys(DEFAULT_SERVICES, False)
    assert 'ROS readiness check failed' in response.json()['reason']
    probe.close()


def test_partial_initialization_is_cleaned_up(ros):
    probe = BTReadiness()
    ros.node.create_client.side_effect = RuntimeError('Client initialization failed')
    with pytest.raises(ReadinessUnavailable, match='Client initialization failed'):
        probe.status()
    ros.node.destroy_node.assert_called_once()
    ros.context.try_shutdown.assert_called_once()
    probe.close()
    ros.node.destroy_node.assert_called_once()


def test_node_creation_failure_closes_context(ros):
    probe = BTReadiness()
    ros.node_factory.side_effect = RuntimeError('Node initialization failed')
    with pytest.raises(ReadinessUnavailable, match='Node initialization failed'):
        probe.status()
    ros.context.try_shutdown.assert_called_once()
    ros.node.destroy_node.assert_not_called()
    probe.close()


def test_lost_ros_context_fails_closed_and_cleans_up(ros):
    probe = BTReadiness()
    probe.status()
    ros.context.ok.return_value = False
    response = http(probe).get('/bt/readiness')
    assert response.status_code == 503
    assert response.json()['ready'] is False
    assert 'ROS context is unavailable' in response.json()['reason']
    ros.node.destroy_node.assert_called_once()
    probe.close()


def test_shutdown_is_idempotent_and_does_not_reopen_ros(ros):
    probe = BTReadiness()
    probe.status()
    probe.close(); probe.close()
    response = http(probe).get('/bt/readiness')
    assert response.status_code == 503 and response.json()['ready'] is False
    assert 'shutting down' in response.json()['reason']
    ros.node.destroy_node.assert_called_once()
    ros.context.try_shutdown.assert_called_once()
    ros.context_factory.assert_called_once()


def test_unused_probe_shutdown_does_not_initialize_ros(ros):
    BTReadiness().close()
    ros.context_factory.assert_not_called()
    ros.node_factory.assert_not_called()


def test_explicit_optional_pause_service_is_supported(ros):
    probe = BTReadiness((*DEFAULT_SERVICES, '/bt/set_paused'))
    try:
        for name in DEFAULT_SERVICES: ros.clients[name].service_is_ready.return_value = True
        assert probe.status()['missing_services'] == ['/bt/set_paused']
        ros.clients['/bt/set_paused'].service_is_ready.return_value = True
        assert probe.status()['ready'] is True
    finally:
        probe.close()


@pytest.mark.parametrize('services', [(), ('/unrelated/service',)])
def test_invalid_service_sets_cannot_create_a_probe(ros, services):
    with pytest.raises(ValueError, match='Unsupported'):
        BTReadiness(services)
    ros.context_factory.assert_not_called()


def test_http_readiness_is_read_only_and_missing_services_are_not_server_errors(ros):
    probe = BTReadiness()
    try:
        client = http(probe)
        assert client.post('/bt/readiness').status_code == 405
        ros.context_factory.assert_not_called()
        response = client.get('/bt/readiness')
        assert response.status_code == 200 and response.json()['ready'] is False
        for name in DEFAULT_SERVICES: ros.clients[name].service_is_ready.return_value = True
        response = client.get('/bt/readiness')
        assert response.status_code == 200 and response.json()['ready'] is True
    finally:
        probe.close()
