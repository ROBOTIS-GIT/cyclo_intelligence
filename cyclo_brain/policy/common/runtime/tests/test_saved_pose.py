"""Manual pose control must remain fail-closed without a model or UI polling."""

from types import SimpleNamespace
from unittest.mock import Mock, patch
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

from . import test_service_handler as service_tests
from main_runtime.saved_pose import SavedPoseManager


@pytest.fixture
def manager():
    robot = Mock()
    robot.get_named_joint_positions.return_value = {"arm": 0.25, "lift": 0.1}
    manager = SavedPoseManager(Mock(return_value=robot))
    manager._profiles["test"] = {
        "names": ["arm", "lift"], "units": {"arm": "rad", "lift": "m"},
    }
    manager.save("test")
    return manager


def test_save_read_and_return_are_session_scoped(manager):
    status = manager.status("test")
    assert status["positions"] == [0.25, 0.1]
    assert status["units"] == ["rad", "m"]
    manager.restore("test")
    manager._robot.publish_named_pose.assert_called_once_with(
        {"arm": 0.25, "lift": 0.1}, duration_s=5.0
    )
    manager.poll()
    assert manager.returning  # Reaching the target does not skip the trajectory duration.
    with patch("main_runtime.saved_pose.time.monotonic", return_value=manager._started + 6):
        manager.poll()
    assert not manager.returning
    manager.close()
    assert not manager._poses


@pytest.mark.parametrize("positions", [{"arm": 0.1}, {"arm": float("nan"), "lift": 0.1},
                                      {"arm": float("inf"), "lift": 0.1}])
def test_invalid_snapshot_does_not_overwrite_saved_pose(manager, positions):
    manager._robot.get_named_joint_positions.return_value = positions
    with pytest.raises(ValueError):
        manager.save("test")
    assert manager._load("test") == {"arm": 0.25, "lift": 0.1}


def test_missing_state_blocks_return_before_publication(manager):
    manager._robot.get_named_joint_positions.side_effect = RuntimeError("stale state")
    with pytest.raises(RuntimeError, match="stale"):
        manager.restore("test")
    manager._robot.publish_named_pose.assert_not_called()
    assert not manager.returning


def test_partial_publish_and_failed_hold_remain_pending(manager):
    manager._robot.publish_named_pose.side_effect = RuntimeError("publish failed")
    manager._robot.publish_current_pose_hold.side_effect = RuntimeError("stale state")
    with pytest.raises(RuntimeError, match="publish failed"):
        manager.restore("test")
    assert manager.returning
    assert "hold failed" in manager.error
    manager._robot.publish_current_pose_hold.assert_called()
    manager._robot.publish_idle_action.assert_not_called()
    manager.poll()
    assert manager.returning
    manager._robot.publish_current_pose_hold.side_effect = None
    manager.poll()
    assert not manager.returning
    assert "publish failed" in manager.error


def test_return_publishes_once_and_expires_without_reaching_target(manager):
    manager.restore("test")
    manager._robot.get_named_joint_positions.return_value = {"arm": 0., "lift": 0.}
    for elapsed in (0., 1., 4.99, 5., 36.):
        with patch("main_runtime.saved_pose.time.monotonic", return_value=manager._started + elapsed):
            manager.poll()
        assert manager.returning == (elapsed < 5.)
    manager._robot.publish_named_pose.assert_called_once_with(
        {"arm": 0.25, "lift": 0.1}, duration_s=5.0,
    )
    manager._robot.publish_current_pose_hold.assert_not_called()
    assert manager.error == ""


def test_configured_duration_controls_publication_and_expiry(manager):
    manager.set_duration("test", 8.5)
    assert manager.status("test")["duration_s"] == 8.5
    manager.restore("test")
    manager._robot.publish_named_pose.assert_called_once_with(
        {"arm": 0.25, "lift": 0.1}, duration_s=8.5,
    )
    with pytest.raises(RuntimeError, match="while returning"):
        manager.set_duration("test", 2)
    for elapsed in (5., 8.49, 8.5):
        with patch("main_runtime.saved_pose.time.monotonic", return_value=manager._started + elapsed):
            manager.poll()
        assert manager.returning == (elapsed < 8.5)


@pytest.mark.parametrize("duration", [0, -1, 0.9, 60.1, float("nan"), float("inf")])
def test_invalid_duration_keeps_previous_setting(manager, duration):
    manager.set_duration("test", 7)
    with pytest.raises(ValueError, match="between 1 and 60"):
        manager.set_duration("test", duration)
    assert manager.duration("test") == 7
    manager._robot.publish_named_pose.assert_not_called()


def test_duration_is_per_robot_and_resets_with_session(manager):
    manager.set_duration("test", 7)
    assert manager.duration("other") == 5
    manager.save("test")
    assert manager.status("test")["duration_s"] == 7
    manager.close()
    assert manager.duration("test") == 5


def test_return_stale_state_still_attempts_hold(manager):
    manager.restore("test")
    manager._robot.get_named_joint_positions.side_effect = RuntimeError("stale state")
    manager.poll()
    assert not manager.returning
    assert "stale state" in manager.error
    manager._robot.publish_current_pose_hold.assert_called_once()


@pytest.fixture
def handler(manager):
    handler, session, loop = service_tests.ServiceHandlerPublishModeTests()._handler(backend="lerobot")
    handler._pose_manager = manager
    handler._pose_response_factory = lambda **fields: SimpleNamespace(**fields)
    return handler


@pytest.fixture
def paused_handler(handler):
    handler._session.mark_loaded(
        "test", "pick", ["arm", "lift"], model_path="/models/policy",
        policy_id="lerobot:act", runtime_id="lerobot", publish_to_robot=True,
        control_hz=100, inference_hz=15, initial_pose_sync=True,
    )
    assert handler.handle(SimpleNamespace(command=service_tests.CMD_START, publish_to_robot=True)).success
    assert handler.handle(SimpleNamespace(command=service_tests.CMD_PAUSE)).success
    return handler


@pytest.mark.parametrize("finish", ["duration", "stop"])
def test_return_preserves_paused_session_and_allows_resume(paused_handler, manager, finish):
    handler = paused_handler
    before = deepcopy(handler._session)
    loop = handler._control_loop
    starts = list(loop.starts)
    with patch.object(loop, "stop", wraps=loop.stop) as stop:
        result = handler.handle_pose(SimpleNamespace(command=2, robot_type="test"))
        assert result.success and result.returning
        assert handler._session == before
        status = handler.handle(SimpleNamespace(command=service_tests.CMD_STATUS))
        assert status.runtime_state == "paused"
        assert status.loaded_model_path == before.model_path
        assert status.publish_to_robot

        resume = SimpleNamespace(command=service_tests.CMD_RESUME, task_instruction="", publish_to_robot=True)
        assert not handler.handle(resume).success
        assert loop.starts == starts
        if finish == "duration":
            with patch("main_runtime.saved_pose.time.monotonic", return_value=manager._started + 6):
                manager.poll()
        else:
            assert handler.handle_pose(SimpleNamespace(command=3, robot_type="test")).success
        assert not manager.returning
        assert handler._session == before
        assert handler.handle(resume).success
        assert handler._session.loaded and handler._session.running and not handler._session.paused
        stop.assert_not_called()
    assert loop.starts == starts + [True]
    assert not loop.deconfigure_count
    assert not handler._requester.unload_count
    assert handler._requester.loaded_with is None


def test_failed_return_keeps_paused_session_and_hold_interlock(paused_handler, manager):
    handler = paused_handler
    before = deepcopy(handler._session)
    manager._robot.publish_named_pose.side_effect = RuntimeError("publish failed")
    manager._robot.publish_current_pose_hold.side_effect = RuntimeError("hold failed")
    result = handler.handle_pose(SimpleNamespace(command=2, robot_type="test"))
    assert not result.success and result.returning
    assert handler._session == before
    resume = SimpleNamespace(command=service_tests.CMD_RESUME, task_instruction="", publish_to_robot=True)
    assert not handler.handle(resume).success
    assert handler._session == before
    manager._robot.publish_current_pose_hold.side_effect = None
    assert handler.handle_pose(SimpleNamespace(command=3, robot_type="test")).success
    assert handler._session == before
    assert handler.handle(resume).success


@pytest.mark.parametrize("state", ["unloaded", "loaded", "running"])
def test_return_does_not_change_inference_lifecycle(handler, manager, state):
    if state != "unloaded":
        handler._session.mark_loaded("test", "pick", ["arm", "lift"])
    if state == "running":
        handler._session.mark_running()
    before = deepcopy(handler._session)
    with patch.object(handler._control_loop, "stop") as stop:
        result = handler.handle_pose(SimpleNamespace(command=2, robot_type="test"))
        assert result.success == (state != "running")
        assert handler._session == before
        stop.assert_not_called()
    if state == "running":
        manager._robot.publish_named_pose.assert_not_called()


def test_return_blocks_inference_and_worker_maintenance(handler, manager):
    from main_runtime.service_handler import CMD_LOAD, CMD_START, CMD_RESUME
    manager.restore("test")
    for cmd in (CMD_LOAD, CMD_START, CMD_RESUME):
        result = handler.handle(SimpleNamespace(command=cmd))
        assert not result.success
        assert "pose return" in result.message
    for runtime in ("lerobot", "groot"):
        assert not handler.can_mutate_worker(runtime)[0]


def test_duration_service_updates_shared_status_without_motion(handler, manager):
    result = handler.handle_pose(SimpleNamespace(command=4, robot_type="test", duration_s=9))
    assert result.success and result.duration_s == 9
    assert handler.pose_snapshot()["duration_s"] == 9
    manager._robot.publish_named_pose.assert_not_called()
    result = handler.handle_pose(SimpleNamespace(command=2, robot_type="test", duration_s=5))
    assert not result.success and "duration changed" in result.message
    manager._robot.publish_named_pose.assert_not_called()
    result = handler.handle_pose(SimpleNamespace(command=2, robot_type="test", duration_s=9))
    assert result.success and result.returning
    assert not handler.handle_pose(SimpleNamespace(command=4, robot_type="test", duration_s=8)).success


def test_duration_update_rejected_during_inference(handler, manager):
    with patch.object(handler, "_runtime_state", return_value="running"):
        result = handler.handle_pose(SimpleNamespace(command=4, robot_type="test", duration_s=9))
    assert not result.success
    assert manager.duration("test") == 5


def test_return_rejects_hold_pending_and_worker_mutation(handler, manager):
    request = SimpleNamespace(command=2, robot_type="test")
    handler._control_loop.hold_pending = True
    assert not handler.handle_pose(request).success
    handler._control_loop.hold_pending = False
    handler._worker_mutations["lerobot"] = "reservation"
    result = handler.handle_pose(request)
    assert not result.success
    assert "maintenance" in result.message
    manager._robot.publish_named_pose.assert_not_called()


def test_pose_stop_retains_pending_until_hold_succeeds(handler, manager):
    manager.restore("test")
    manager._robot.publish_current_pose_hold.side_effect = RuntimeError("stale state")
    result = handler.handle_pose(SimpleNamespace(command=3, robot_type="test"))
    assert not result.success and result.returning
    with patch("main_runtime.saved_pose.time.monotonic", return_value=manager._started + 36):
        manager.poll()
    assert manager.returning
    assert not handler.can_mutate_worker("lerobot")[0]
    manager._robot.publish_current_pose_hold.side_effect = None
    result = handler.handle_pose(SimpleNamespace(command=3, robot_type="test"))
    assert result.success and not result.returning


def test_shutdown_rejects_pose_commands(handler):
    handler.begin_shutdown()
    result = handler.handle_pose(SimpleNamespace(command=2, robot_type="test"))
    assert not result.success
    assert "shutting down" in result.message


def test_motion_guard_failure_never_publishes(handler, manager):
    handler._pose_motion_guard = Mock(side_effect=RuntimeError("heartbeat stale"))
    result = handler.handle_pose(SimpleNamespace(command=2, robot_type="test"))
    assert not result.success and "heartbeat stale" in result.message
    manager._robot.publish_named_pose.assert_not_called()


def test_fail_safe_reports_reason_even_without_loaded_policy(handler, manager):
    manager.restore("test")
    assert handler.fail_safe("heartbeat lost")
    assert not manager.returning
    assert manager.error == "heartbeat lost"


@pytest.mark.parametrize("robot_type", [
    "ffw_sg2_rev1", "ffw_sh5_rev1", "ffw_bg2_rev4", "f1", "f2", "omx_f", "omy_f3m",
])
def test_pose_joint_names_and_units_come_from_robot_configuration(robot_type):
    root = Path(__file__).resolve().parents[5]
    schema_path = root / "shared/shared/robot_configs/schema.py"
    spec = importlib.util.spec_from_file_location("pose_robot_schema", schema_path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    with patch.dict(sys.modules, {"robot_client.robot_client": SimpleNamespace(robot_schema=schema)}):
        manager = SavedPoseManager(Mock())
        profile = manager.profile(robot_type)
    assert profile["names"]
    assert set(profile["names"]) == set(profile["units"])
    assert set(profile["units"].values()) <= {"rad", "m"}
    section = schema.load_robot_section(robot_type)
    joints = {joint.get("name"): joint for joint in
              ET.parse(schema.get_urdf_path(section)).getroot().findall("joint")}
    for name in profile["names"]:
        joint = joints[name]
        assert joint.get("type") in {"revolute", "continuous", "prismatic"}
        assert profile["units"][name] == ("m" if joint.get("type") == "prismatic" else "rad")

    # A finite measured pose is retained without clamping to URDF limits.
    positions = {name: float(joints[name].find("limit").get("upper")) + 0.01
                 if joints[name].find("limit") is not None and
                 joints[name].find("limit").get("upper") is not None else 0.
                 for name in profile["names"]}
    robot = Mock()
    robot.get_named_joint_positions.return_value = positions
    manager._factory = Mock(return_value=robot)
    manager.save(robot_type)
    assert manager._load(robot_type) == positions
    manager.restore(robot_type)
    robot.publish_named_pose.assert_called_once_with(positions, duration_s=5.)
