"""Contract tests without allocating weights or contacting a robot."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "policy_validation", Path(__file__).resolve().parents[1] / "lerobot_engine/policy_validation.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


@pytest.mark.parametrize("config", [{"model_type": "gr00t_n1_7"}, {"type": "act"}])
def test_lerobot_groot_selection_rejects_raw_or_other_checkpoints(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="requires a LeRobot checkpoint"):
        validation.validate_requested_policy(tmp_path, "lerobot:groot")


def test_legacy_pi0_selection_does_not_restrict_checkpoint_type(tmp_path):
    (tmp_path / "config.json").write_text('{"type": "pi0_fast"}')
    validation.validate_requested_policy(tmp_path, "lerobot:pi0")


def wall_config(state=20, action=20, **kwargs):
    return {
        "type": "wall_x",
        "input_features": {"observation.state": {"shape": [state]}},
        "output_features": {"action": {"shape": [action]}},
        **kwargs,
    }


@pytest.mark.parametrize("state,action", [(1, 1), (16, 20), (20, 16), (20, 20)])
def test_wall_x_accepts_independent_dimensions(state, action, tmp_path):
    validation.validate_checkpoint(wall_config(state, action), tmp_path)


@pytest.mark.parametrize("state,action", [(21, 20), (20, 21), (22, 22), (0, 20)])
def test_wall_x_rejects_unsupported_dimensions(state, action, tmp_path):
    with pytest.raises(ValueError):
        validation.validate_checkpoint(wall_config(state, action), tmp_path)


@pytest.mark.parametrize("key,value", [("max_state_dim", 32), ("max_action_dim", 16)])
def test_wall_x_rejects_config_that_disagrees_with_fixed_core(key, value, tmp_path):
    with pytest.raises(ValueError, match="requires"):
        validation.validate_checkpoint(wall_config(**{key: value}), tmp_path)


@pytest.mark.parametrize("policy", ["eo1", "evo1", "wall_x", "groot"])
def test_new_policy_history_is_not_silently_duplicated(policy, tmp_path):
    with pytest.raises(ValueError, match="history"):
        validation.validate_checkpoint({"type": policy, "n_obs_steps": 2}, tmp_path)


@pytest.mark.parametrize("state,action,valid", [(16, 20, True), (20, 16, True), (22, 20, False), (20, 19, False)])
def test_wall_x_robot_layout(state, action, valid):
    config = SimpleNamespace(**wall_config(16 if state == 16 else 20, 16 if action == 16 else 20))
    robot = Mock()
    robot.get_joint_names.return_value = list(range(state))
    robot._action_groups = {"arm": {"msg_type": "trajectory_msgs/msg/JointTrajectory", "joint_names": list(range(action))}}
    if valid:
        validation.validate_wall_x_robot(config, robot, ["arm"], ["arm"])
    else:
        with pytest.raises(ValueError, match="dimension"):
            validation.validate_wall_x_robot(config, robot, ["arm"], ["arm"])
    robot.publish_action.assert_not_called()


@pytest.fixture
def groot_checkpoint(tmp_path):
    config = {"type": "groot", "base_model_path": str(tmp_path), "embodiment_tag": "test_robot"}
    pack = {"embodiment_tag": "test_robot", "stats": {"observation.state": {"min": [0], "max": [1]}}}
    pre = {"steps": [
        {"registry_name": "groot_n1_7_pack_inputs_v1", "config": pack},
        {"registry_name": "groot_n1_7_vlm_encode_v1", "config": {}},
    ]}
    post = {"steps": [{"registry_name": "groot_action_unpack_unnormalize_v2", "config": {
        "stats": {"action": {"min": [0], "max": [1]}}
    }}]}

    def save():
        for name, value in (("policy_preprocessor", pre), ("policy_postprocessor", post)):
            (tmp_path / f"{name}.json").write_text(json.dumps(value))
    save()
    return config, pre, post, save


def test_groot_accepts_saved_le_robot_pipeline(groot_checkpoint, tmp_path):
    config, _, _, _ = groot_checkpoint
    validation.validate_checkpoint(config, tmp_path)


@pytest.mark.parametrize("key", ["state_horizon", "video_horizon"])
def test_groot_checks_history_in_saved_processor(groot_checkpoint, tmp_path, key):
    config, pre, _, save = groot_checkpoint
    pre["steps"][0]["config"][key] = 2
    save()
    with pytest.raises(ValueError, match="history"):
        validation.validate_checkpoint(config, tmp_path)


@pytest.mark.parametrize("name", ["policy_preprocessor", "policy_postprocessor"])
def test_groot_missing_pipeline_is_not_reconstructed(groot_checkpoint, tmp_path, name):
    config, _, _, _ = groot_checkpoint
    (tmp_path / f"{name}.json").unlink()
    with pytest.raises(ValueError, match="saved processor"):
        validation.validate_checkpoint(config, tmp_path)


def test_groot_missing_base_is_reported_before_hub_validation(groot_checkpoint, tmp_path):
    config, _, _, _ = groot_checkpoint
    config["base_model_path"] = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="base_model_path is missing"):
        validation.validate_checkpoint(config, tmp_path)


@pytest.mark.parametrize("part", ["embodiment", "state_stats", "action_stats", "state_file"])
def test_groot_missing_metadata_cannot_fall_back(groot_checkpoint, tmp_path, part):
    config, pre, post, save = groot_checkpoint
    if part == "embodiment":
        pre["steps"][0]["config"]["embodiment_tag"] = "other"
    elif part == "state_stats":
        pre["steps"][0]["config"]["stats"] = None
    elif part == "action_stats":
        post["steps"][0]["config"]["stats"] = None
    else:
        pre["steps"][0]["state_file"] = "missing.safetensors"
    save()
    with pytest.raises(ValueError):
        validation.validate_checkpoint(config, tmp_path)
