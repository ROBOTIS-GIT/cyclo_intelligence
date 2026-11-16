import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def assets():
    training = {
        "model": {"tokenizer_path": "Qwen/Qwen3-VL-4B-Instruct"},
        "train": {"chunk_size": 2, "max_state_dim": 4, "max_action_dim": 4},
        "data": {"joints": ["{'arm.position': 4}"], "norm_type": ["{'arm.position': 'meanstd'}"],
                 "cameras": ["camera_top", "camera_wrist_left", "camera_wrist_right"]},
    }
    cameras = ["observation.images.rgb.head", "observation.images.rgb.left", "observation.images.rgb.right"]
    robot = {
        "norm_stats": "norm_stats.json",
        "states": [{"observation.state.arm.position": {"origin_keys": [{"observation.state": {"start": 0, "end": 2}}]}}],
        "actions": [{"action.arm.position": {"origin_keys": [{"action": {"start": 0, "end": 2}}], "subtract_state": True, "relative_type": None}}],
        "images": [{f"observation.images.{target}": {"origin_keys": source}}
                   for target, source in zip(training["data"]["cameras"], cameras)],
    }
    stats = {"norm_stats": {key: {"mean": [0., 0.], "std": [1., 1.]}
                           for key in ("observation.state.arm.position", "action.arm.position")}}
    metadata = {"robot_type": "test_robot", "state_key": "observation.state", "action_key": "action",
                "state_names": ["j2", "j1"], "action_names": ["j2", "j1"], "cameras": cameras}
    return training, robot, stats, metadata


@pytest.fixture
def bundle(tmp_path, assets):
    root = tmp_path / "model"
    weights = root / "checkpoints/export/hf_ckpt"
    weights.mkdir(parents=True)
    (weights / "model.safetensors").write_bytes(b"mocked weights; not a real model")
    for name, value in zip(("lingbotvla_cli.yaml", "robot_config.yaml", "norm_stats.json", "cyclo_input_metadata.json"), assets):
        (root / name).write_text(yaml.safe_dump(value) if name.endswith("yaml") else json.dumps(value))
    return root
