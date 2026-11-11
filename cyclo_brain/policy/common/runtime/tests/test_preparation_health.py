"""Initial preparation stays under the same heartbeat/shutdown safety monitor."""

import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from .test_control_loop import ControlLoop  # Establish the lightweight RobotClient stub.


@pytest.fixture
def runtime_module():
    path = Path(__file__).resolve().parents[1] / "main_runtime/main.py"
    spec = importlib.util.spec_from_file_location("main_runtime.preparation_health_main", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        "robot_client.messages": SimpleNamespace(INFERENCE_COMMAND_REQUEST_DEF="", INFERENCE_COMMAND_RESPONSE_DEF="",
                                                ROBOT_POSE_COMMAND_REQUEST_DEF="", ROBOT_POSE_COMMAND_RESPONSE_DEF=""),
        "zenoh_ros2_sdk": SimpleNamespace(ROS2Publisher=object, ROS2ServiceServer=object, ROS2Subscriber=object, get_logger=logging.getLogger),
    }):
        spec.loader.exec_module(module)
    return module


def test_heartbeat_subscriptions_supply_offline_message_definitions(runtime_module):
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    runtime._subscribers = []
    runtime._workers = SimpleNamespace(runtime_ids=("lerobot", "groot"))
    runtime._router_ip, runtime._router_port = "127.0.0.1", 7447
    runtime._domain_id, runtime._namespace = 30, ""
    with mock.patch.object(runtime_module, "ROS2Subscriber") as subscriber:
        runtime._start_heartbeat_subscribers()
    definitions = {call.kwargs["msg_type"]: call.kwargs.get("msg_definition")
                   for call in subscriber.call_args_list}
    assert len(runtime._subscribers) == 3
    # An empty string means "look up the definition" in the SDK. A comment-only
    # definition expresses Empty without needing a downloaded message cache.
    empty = definitions["std_msgs/msg/Empty"]
    assert empty and all(not line.strip() or line.lstrip().startswith("#")
                         for line in empty.splitlines())
    assert definitions["std_msgs/msg/String"].strip() == "string data"


@pytest.mark.parametrize("phase", ["preparing", "running", "syncing"])
def test_heartbeat_loss_is_not_ignored_during_preparation(runtime_module, phase):
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    runtime._shutdown = mock.Mock()
    runtime._shutdown.wait.side_effect = [False, True]
    runtime._handler = mock.Mock()
    runtime._handler.runtime_snapshot.return_value = {"runtime_state": phase, "runtime_id": "lerobot"}
    runtime._handler.pose_snapshot.return_value = {}
    runtime._pose_status_publisher = mock.Mock()
    runtime._active_since = 0.
    runtime._orchestrator_last_seen = 10.
    runtime._workers = mock.Mock()
    runtime._workers.heartbeat_age.return_value = 20.
    runtime._workers.worker_instance_changed.return_value = False
    with mock.patch.object(runtime_module.time, "monotonic", return_value=10.):
        runtime._monitor_health()
    runtime._handler.fail_safe.assert_called_once()


def test_shutdown_stops_preparation_before_closing_the_control_loop(runtime_module):
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    calls = mock.Mock()
    runtime._handler = calls.handler
    runtime._handler.runtime_snapshot.return_value = {"runtime_state": "preparing"}
    runtime._handler.fail_safe.return_value = True
    runtime._shutdown = mock.Mock()
    runtime._monitor_thread = None
    runtime._control_server = mock.Mock()
    runtime._subscribers, runtime._services = [], []
    runtime._control_loop = calls.control
    runtime._workers = mock.Mock()
    runtime._remove_ready_marker = mock.Mock()
    runtime._saved_pose = mock.Mock()
    runtime._pose_status_publisher = mock.Mock()
    runtime.shutdown()
    assert calls.mock_calls.index(mock.call.handler.fail_safe("policy runtime shutting down")) < calls.mock_calls.index(mock.call.control.shutdown())


def test_shutdown_keeps_robot_io_until_hold_succeeds(runtime_module):
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    calls = mock.Mock()
    runtime._handler = calls.handler
    runtime._handler.runtime_snapshot.return_value = {"runtime_state": "paused", "hold_pending": True}
    runtime._handler.fail_safe.side_effect = [False, False, False, True]
    runtime._shutdown = mock.Mock()
    runtime._monitor_thread = None
    runtime._control_server = calls.server
    runtime._subscribers, runtime._services = [calls.subscriber], []
    runtime._control_loop = calls.control
    runtime._workers = calls.workers
    runtime._saved_pose = calls.pose
    runtime._pose_status_publisher = calls.pose_publisher
    runtime._remove_ready_marker = calls.remove_marker
    with mock.patch.object(runtime_module.time, "sleep"):
        runtime.shutdown()
    events = calls.mock_calls
    holds = [i for i, call in enumerate(events) if call == mock.call.handler.fail_safe("policy runtime shutting down")]
    assert len(holds) == 4
    assert events.index(mock.call.remove_marker()) < holds[0]
    assert events.index(mock.call.subscriber.close()) > holds[-1]
    assert events.index(mock.call.control.shutdown()) > holds[-1]


def test_shutdown_refuses_worker_mutation(runtime_module):
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    runtime._shutdown = mock.Mock()
    runtime._shutdown.is_set.return_value = True
    runtime._handler = mock.Mock()
    response = runtime._handle_control_request({"operation": "begin_worker_mutation", "runtime_id": "lerobot"})
    assert response["ok"] is False
    runtime._handler.begin_worker_mutation.assert_not_called()


def test_pose_commands_do_not_refresh_orchestrator_heartbeat(runtime_module):
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    runtime._orchestrator_last_seen = 1.
    runtime._handler = mock.Mock()
    request = SimpleNamespace(command=0, robot_type="test")
    runtime._handle_pose_request(request)
    assert runtime._orchestrator_last_seen == 1.
    runtime._handler.handle_pose.assert_called_once_with(request)


def test_pose_return_is_monitored_without_a_loaded_model(runtime_module):
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    runtime._shutdown = mock.Mock()
    runtime._shutdown.wait.side_effect = [False, True]
    runtime._handler = mock.Mock()
    runtime._handler.runtime_snapshot.return_value = {"runtime_state": "unloaded", "pose_returning": True}
    runtime._handler.pose_snapshot.return_value = {"returning": True}
    runtime._pose_status_publisher = mock.Mock()
    runtime._orchestrator_last_seen = 0.
    with mock.patch.object(runtime_module.time, "monotonic", return_value=10.):
        runtime._monitor_health()
    runtime._handler.fail_safe.assert_called_once_with("orchestrator heartbeat lost during pose return")
    runtime._pose_status_publisher.publish.assert_called_once_with(data='{"returning": true}')


def test_ready_marker_identifies_this_process(runtime_module, tmp_path, monkeypatch):
    marker = tmp_path / "ready"
    monkeypatch.setenv("POLICY_RUNTIME_READY_MARKER", str(marker))
    runtime = runtime_module.PolicyRuntime.__new__(runtime_module.PolicyRuntime)
    runtime._write_ready_marker()
    assert marker.read_text().strip() == str(runtime_module.os.getpid())
