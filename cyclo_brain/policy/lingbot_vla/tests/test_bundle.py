from copy import deepcopy
import json

import pytest
import yaml

from lingbot_vla_engine.bundle import CheckpointBundle, validate_assets
from scripts.export_checkpoint import export_checkpoint


def test_bundle_requires_all_saved_assets(bundle):
    result = CheckpointBundle(bundle)
    assert result.horizon == 2
    assert result.weights.parent.parent.parent == bundle
    (bundle / "norm_stats.json").unlink()
    with pytest.raises(FileNotFoundError):
        CheckpointBundle(bundle)


@pytest.mark.parametrize("mutation,match", [
    ("missing_channel", "every source channel"),
    ("duplicate_channel", "capacity|every source channel"),
    ("different_relative_order", "Relative action/state"),
    ("missing_stats", "Missing statistics"),
    ("wrong_stat_dim", "Invalid statistics"),
    ("stale_server_path", "portable"),
    ("implicit_transform", "geometric conversion"),
    ("different_camera", "Camera mapping"),
    ("capacity", "exceeds canonical capacity"),
    ("implicit_quaternion", "relative_type"),
])
def test_bad_bundle_is_rejected_before_model_import(bundle, assets, mutation, match):
    training, robot, stats, metadata = deepcopy(assets)
    action = robot["actions"][0]["action.arm.position"]
    if mutation == "missing_channel":
        action["origin_keys"][0]["action"]["end"] = 1
        stats["norm_stats"]["action.arm.position"] = {"mean": [0.], "std": [1.]}
    elif mutation == "duplicate_channel":
        action["origin_keys"] *= 2
        stats["norm_stats"]["action.arm.position"] = {"mean": [0.] * 4, "std": [1.] * 4}
    elif mutation == "different_relative_order":
        metadata["action_names"].reverse()
    elif mutation == "missing_stats":
        del stats["norm_stats"]["action.arm.position"]
    elif mutation == "wrong_stat_dim":
        stats["norm_stats"]["action.arm.position"]["std"] = [1.]
    elif mutation == "stale_server_path":
        training["model"]["tokenizer_path"] = "/nonexistent/training-server/Qwen3-VL"
    elif mutation == "implicit_transform":
        action["convert_from_state"] = True
    elif mutation == "different_camera":
        metadata["cameras"][0] = "observation.images.other"
    elif mutation == "capacity":
        training["data"]["joints"] = ["{'arm.position': 1}"]
    elif mutation == "implicit_quaternion":
        del action["relative_type"]
    for name, value in zip(("lingbotvla_cli.yaml", "robot_config.yaml", "norm_stats.json", "cyclo_input_metadata.json"),
                           (training, robot, stats, metadata)):
        (bundle / name).write_text(yaml.safe_dump(value) if name.endswith("yaml") else json.dumps(value))
    with pytest.raises(ValueError, match=match):
        CheckpointBundle(bundle)


def test_export_roundtrip_preserves_source(bundle, tmp_path):
    original = {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    output = tmp_path / "export"
    kwargs = dict(weights=bundle / "checkpoints/export/hf_ckpt", training_config=bundle / "lingbotvla_cli.yaml",
                  robot_config=bundle / "robot_config.yaml", norm_stats=bundle / "norm_stats.json",
                  metadata=bundle / "cyclo_input_metadata.json", output=output,
                  base_model="Qwen/Qwen3-VL-4B-Instruct")
    export_checkpoint(**kwargs)
    assert CheckpointBundle(output).metadata == CheckpointBundle(bundle).metadata
    assert original == {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overwrite"):
        export_checkpoint(**kwargs)


def test_missing_weights_is_not_a_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="exported"):
        CheckpointBundle(tmp_path)


def test_state_and_action_dimensions_can_differ(assets):
    training, robot, stats, metadata = assets
    metadata["state_names"].append("j3")
    robot["states"][0]["observation.state.arm.position"]["origin_keys"][0]["observation.state"]["end"] = 3
    robot["actions"][0]["action.arm.position"]["subtract_state"] = False
    stats["norm_stats"]["observation.state.arm.position"] = {"mean": [0.] * 3, "std": [1.] * 3}
    assert validate_assets(training, robot, stats, metadata) == 2
