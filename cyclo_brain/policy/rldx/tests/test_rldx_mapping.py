import sys
from pathlib import Path

import numpy as np


POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT))

from rldx_engine.mapping import (  # noqa: E402
    aspect_area_resize_crop_sizes,
    build_state_observation,
    policy_action_chunk_from_rldx,
    robot_action_keys_from_policy,
    robot_action_chunk_from_rldx,
    stack_history_window,
)


def test_robot_action_chunk_combines_rldx_arm_and_gripper_modalities():
    actions = {
        "action.left_arm": np.ones((1, 16, 7), dtype=np.float32),
        "action.left_gripper": np.full((1, 16, 1), 2, dtype=np.float32),
        "action.right_arm": np.full((1, 16, 7), 3, dtype=np.float32),
        "action.right_gripper": np.full((1, 16, 1), 4, dtype=np.float32),
        "action.head": np.full((1, 16, 2), 5, dtype=np.float32),
        "action.lift": np.full((1, 16, 1), 6, dtype=np.float32),
        "action.base": np.full((1, 16, 3), 7, dtype=np.float32),
    }

    chunk = robot_action_chunk_from_rldx(
        actions,
        ["arm_left", "arm_right", "mobile"],
    )

    assert chunk.shape == (16, 19)
    np.testing.assert_array_equal(chunk[:, :7], 1)
    np.testing.assert_array_equal(chunk[:, 7:8], 2)
    np.testing.assert_array_equal(chunk[:, 8:15], 3)
    np.testing.assert_array_equal(chunk[:, 15:16], 4)
    np.testing.assert_array_equal(chunk[:, 16:19], 7)


def test_policy_action_chunk_preserves_rldx_modality_order_for_rtc_prefix():
    actions = {
        "action.arm_left": np.ones((1, 16, 8), dtype=np.float32),
        "action.arm_right": np.full((1, 16, 8), 2, dtype=np.float32),
        "action.lift": np.full((1, 16, 1), 3, dtype=np.float32),
    }

    chunk = policy_action_chunk_from_rldx(
        actions,
        ["arm_left", "arm_right", "lift"],
    )

    assert chunk.shape == (16, 17)
    np.testing.assert_array_equal(chunk[:, :8], 1)
    np.testing.assert_array_equal(chunk[:, 8:16], 2)
    np.testing.assert_array_equal(chunk[:, 16:17], 3)


def test_robot_action_keys_from_policy_uses_only_model_outputs(monkeypatch):
    monkeypatch.setenv("RLDX_SKIP_ROBOT_ACTION_KEYS", "head,lift")
    robot_keys = ["arm_left", "arm_right", "head", "lift", "mobile"]

    assert robot_action_keys_from_policy(
        ["arm_left", "arm_right", "lift"],
        robot_keys,
    ) == ["arm_left", "arm_right"]

    assert robot_action_keys_from_policy(
        ["left_arm", "left_gripper", "right_arm", "right_gripper", "head", "lift", "base"],
        robot_keys,
    ) == ["arm_left", "arm_right", "mobile"]


def test_robot_action_keys_can_publish_lift_when_skip_env_empty(monkeypatch):
    monkeypatch.setenv("RLDX_SKIP_ROBOT_ACTION_KEYS", "")

    assert robot_action_keys_from_policy(
        ["arm_left", "arm_right", "lift"],
        ["arm_left", "arm_right", "lift"],
    ) == ["arm_left", "arm_right", "lift"]



def test_build_state_observation_splits_combined_robot_arm_groups():
    joints = {
        "follower_arm_left": np.arange(8, dtype=np.float32),
        "follower_arm_right": np.arange(10, 18, dtype=np.float32),
        "follower_head": np.array([20, 21], dtype=np.float32),
        "follower_lift": np.array([30], dtype=np.float32),
    }
    odom = {
        "linear_velocity": np.array([1.0, 2.0, 0.0], dtype=np.float32),
        "angular_velocity": np.array([0.0, 0.0, 3.0], dtype=np.float32),
    }

    obs = build_state_observation(
        ["left_arm", "left_gripper", "right_arm", "right_gripper", "head", "lift", "base"],
        joints,
        odom,
        state_t=1,
    )

    assert obs["left_arm"].shape == (1, 1, 7)
    np.testing.assert_array_equal(obs["left_arm"][0, 0], np.arange(7))
    np.testing.assert_array_equal(obs["left_gripper"][0, 0], [7])
    np.testing.assert_array_equal(obs["right_arm"][0, 0], np.arange(10, 17))
    np.testing.assert_array_equal(obs["right_gripper"][0, 0], [17])
    np.testing.assert_array_equal(obs["head"][0, 0], [20, 21])
    np.testing.assert_array_equal(obs["lift"][0, 0], [30])
    np.testing.assert_array_equal(obs["base"][0, 0], [1.0, 2.0, 3.0])


def test_build_state_observation_accepts_hardware_facing_arm_names():
    joints = {
        "follower_arm_left": np.arange(8, dtype=np.float32),
        "follower_arm_right": np.arange(10, 18, dtype=np.float32),
        "follower_lift": np.array([30], dtype=np.float32),
    }

    obs = build_state_observation(
        ["arm_left", "arm_right", "lift"],
        joints,
        odom=None,
        state_t=1,
    )

    assert obs["arm_left"].shape == (1, 1, 8)
    np.testing.assert_array_equal(obs["arm_left"][0, 0], np.arange(8))
    np.testing.assert_array_equal(obs["arm_right"][0, 0], np.arange(10, 18))
    np.testing.assert_array_equal(obs["lift"][0, 0], [30])


def test_aspect_area_resize_crop_sizes_match_rldx_eval_geometry():
    assert aspect_area_resize_crop_sizes(720, 1280) == ((192, 341), (192, 320))
    assert aspect_area_resize_crop_sizes(480, 640) == ((192, 256), (192, 256))
    assert aspect_area_resize_crop_sizes(256, 256) == ((256, 256), (256, 256))


def test_stack_history_window_uses_delta_indices_with_oldest_padding():
    frames = [np.full((2, 2, 1), value, dtype=np.uint8) for value in range(7)]

    warmup = stack_history_window(frames[-2:], [-6, -4, -2, 0])
    np.testing.assert_array_equal(warmup[:, 0, 0, 0], [5, 5, 5, 6])

    window = stack_history_window(frames, [-6, -4, -2, 0])
    np.testing.assert_array_equal(window[:, 0, 0, 0], [0, 2, 4, 6])
