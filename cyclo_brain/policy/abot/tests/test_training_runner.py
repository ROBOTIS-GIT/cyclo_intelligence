from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_training.runner import prediction, validate_config
from abot_training.checkpoint import file_hash
from test_training_data import prepared


def run_config():
    return {"version": 1, "model_id": "abot-m0", "policy_id": "abot:m0",
            "profile": {}, "dataset": {}, "assets": {}, "source": {"image_id": "test"},
            "output_dir": "/output", "parameters": {
                "steps": 2, "batch_size": 1, "gradient_accumulation_steps": 1,
                "save_interval": 1, "save_total_limit": 1, "learning_rate": 1e-4,
                "optimizer": "sgd", "seed": 42, "image_size": [224, 224], "horizon": 16,
                "repeated_diffusion_steps": 1, "gradient_checkpointing": True, "max_grad_norm": 1.0}}


def test_explicit_training_config_and_no_mutation():
    config = run_config()
    before = deepcopy(config)
    result = validate_config(config)
    result["parameters"]["steps"] = 9
    assert config == before


@pytest.mark.parametrize("key,value", [
    ("steps", 0), ("batch_size", True), ("save_interval", -1), ("save_total_limit", 0),
    ("learning_rate", float("nan")), ("max_grad_norm", float("inf")),
    ("optimizer", "automatic"), ("seed", -1), ("gradient_checkpointing", "false"),
])
def test_invalid_training_config_rejected(key, value):
    config = run_config()
    config["parameters"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_no_ambiguous_selection_or_unqualified_distributed_run(monkeypatch):
    config = run_config()
    config.update(channel_selection={}, channel_selection_yaml="file.yaml")
    with pytest.raises(ValueError, match="not both"):
        validate_config(config)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single-process"):
        validate_config(run_config())


def test_checkpoint_argument_does_not_silently_request_resume():
    from abot_training.runner import main

    with pytest.raises(ValueError, match="optimizer resume is unsupported"):
        main("unused.json", "train", Path("checkpoint-1"))


def test_saving_probe_preserves_random_stream_and_training_mode():
    torch = pytest.importorskip("torch")

    class Probe(torch.nn.Module):
        def predict_action(self, batch):
            random.random()
            np.random.rand()
            return {"normalized_actions": torch.rand(1, 2, 3).numpy()}

    model = Probe().train()
    before_py, before_np, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    a, b = prediction(model, {}, 42), prediction(model, {}, 42)
    np.testing.assert_array_equal(a, b)
    assert model.training and random.getstate() == before_py
    np.testing.assert_array_equal(np.random.get_state()[1], before_np[1])
    assert torch.equal(before_torch, torch.get_rng_state())


def test_training_lock_and_changed_input_failure(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("check_training_lock", root / "training/check_lock.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_lock(root / "training")
    shutil.copytree(root / "training", tmp_path / "training")
    shutil.copyfile(root / "requirements-inference.txt", tmp_path / "requirements-inference.txt")
    (tmp_path / "requirements-inference.txt").write_text("torch==0.0.0\n")
    with pytest.raises(ValueError, match="lock input changed"):
        module.check_lock(tmp_path / "training")


def test_preflight_preserves_selection_and_rejects_untracked_or_changed_assets(prepared, tmp_path, monkeypatch):
    from abot_training import runner

    root, profile, _ = prepared
    monkeypatch.setattr(runner, "load_profile", lambda *args, **kwargs: deepcopy(profile))
    config = run_config()
    config["profile"] = {"id": "test-abot", "sha256": "fixture"}
    config["dataset"] = {"path": str(root)}
    config["parameters"]["horizon"] = 2
    config["channel_selection"] = {"state_names": ["s2"], "action_names": ["a2", "a1"]}
    for key, identity in profile["assets"].items():
        directory = tmp_path / key
        directory.mkdir()
        if key == "base":
            (directory / "config.yaml").write_text(
                "framework:\n  name: ABot_M0\n  action_model: {}\n  qwenvl: {}\n")
            (directory / "checkpoints").mkdir()
            (directory / "checkpoints/ABot_M0_Pretrain.pt").write_bytes(b"fixture")
        else:
            (directory / "config.json").write_text("{}")
        config["assets"][key] = {**identity, "path": str(directory), "sha256": {
            str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*") if p.is_file()}}
    _, data, model_config, _, _ = runner.prepare(config)
    assert data.selected["state_names"] == ["s2"]
    assert data.selected["action_names"] == ["a2", "a1"]
    assert model_config["datasets"]["vla_data"]["action_mode"] == "abs"
    extra = tmp_path / "vlm/extra.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="inventory differs"):
        runner.prepare(config)
    extra.unlink()
    (tmp_path / "vlm/config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.prepare(config)
