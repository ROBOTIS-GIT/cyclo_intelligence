from types import SimpleNamespace

import numpy as np
import pytest

from vitacformer_engine.constants import ACTION_KEYS, CAMERA_NAME, JOINT_NAMES
from vitacformer_engine.sensor_client import ViTacFormerSensorClient, resample_history


def test_missing_robot_config_preserves_load_error(monkeypatch):
    def missing_config(_robot_type):
        raise FileNotFoundError("missing SH5 configuration")

    monkeypatch.setattr(
        "vitacformer_engine.sensor_client.robot_schema.load_robot_section", missing_config,
    )
    with pytest.raises(FileNotFoundError, match="missing SH5 configuration"):
        ViTacFormerSensorClient("ffw_sh5_rev1")


def test_upstream_config_subscribes_to_head_camera_and_both_hands():
    robot = ViTacFormerSensorClient("ffw_sh5_rev1")
    try:
        assert robot.camera_names == [CAMERA_NAME]
        assert set(robot._config["tactile_modalities"]) == {
            "left_hand_pressure", "right_hand_pressure",
        }
        assert robot.action_dimension(ACTION_KEYS) == 54
        assert [name for key in ACTION_KEYS for name in robot._action_groups[key]["joint_names"]] == list(JOINT_NAMES)
        assert robot._command_publishers == {}
    finally:
        robot.close()


def test_history_selects_past_frames_without_interpolating_future_values():
    samples = [(10.0, [1]), (10.15, [2]), (10.20, [3]), (10.30, [4])]
    result = resample_history(samples, 3, 10.0, "state", now=10.31)
    np.testing.assert_array_equal(result[:, 0], [1, 3, 4])


@pytest.mark.parametrize("samples,now,match", [
    ([], 10.0, "not ready"),
    ([(10.0, [1]), (10.1, [2])], 10.1, "warmup incomplete"),
    ([(10.0, [1]), (10.5, [2])], 10.5, "resampling gap"),
    ([(10.0, [1]), (10.0, [2])], 10.0, "monotonic"),
    ([(9.5, [1]), (10.0, [2])], 11.0, "stale"),
])
def test_history_rejects_missing_stale_or_invalid_frames(samples, now, match):
    with pytest.raises(RuntimeError, match=match):
        resample_history(samples, 6, 10.0, "state", now=now)


def test_joint_callback_reorders_named_superset_and_rejects_missing_joint(monkeypatch):
    robot = ViTacFormerSensorClient("ffw_sh5_rev1")
    monkeypatch.setattr("vitacformer_engine.sensor_client.time.monotonic", lambda: 10.0)
    try:
        names = list(reversed(JOINT_NAMES)) + ["extra"]
        msg = SimpleNamespace(name=names, position=list(range(55)), velocity=[], effort=[])
        robot._update_joint("follower_upper_body", msg)
        result = robot.get_joint_position_history(JOINT_NAMES, 1, 10.0)
        np.testing.assert_array_equal(result[0], list(reversed(range(54))))
        msg.name = ["extra"]
        msg.position = [0]
        robot._update_joint("follower_upper_body", msg)
        with pytest.raises(RuntimeError, match="Invalid"):
            robot.get_joint_position_history(JOINT_NAMES, 1, 10.0)
    finally:
        robot.close()


def test_tactile_calibration_keeps_exactly_first_twenty_valid_frames():
    robot = ViTacFormerSensorClient("ffw_sh5_rev1")
    try:
        for value in range(30):
            msg = SimpleNamespace(sensors=[
                SimpleNamespace(pressure_values=bytes([value] * 9)) for _ in range(5)
            ])
            robot._update_sensor("left_hand_pressure", msg)
        samples = robot.wait_for_tactile_samples("left_hand_pressure", 20, 0.1)
        assert samples.shape == (20, 5, 3, 3)
        np.testing.assert_array_equal(samples[:, 0, 0, 0], np.arange(20))
        with pytest.raises(ValueError, match="five"):
            robot._hand_pressure_taxels(SimpleNamespace(sensors=[]))
    finally:
        robot.close()
