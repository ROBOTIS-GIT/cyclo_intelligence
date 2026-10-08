"""Run-scoped training contract checks that need no GPU or model download."""

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("rldx_training_run", SCRIPTS / "train_run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config(tmp_path):
    dataset = tmp_path / ("a" * 64)
    dataset.mkdir(exist_ok=True)
    return {
        "version": 1, "model_id": "rldx-rldx1", "policy_id": "rldx:rldx1",
        "dataset_path": str(dataset), "run_root": str(tmp_path / "run"),
        "dataset_identity": "a" * 64,
        "base_model": {"repo_id": "RLWRLD/RLDX-1-PT-IMG", "revision": "b" * 40},
        "steps": 10, "batch_size": 1, "gradient_accumulation_steps": 1,
        "save_steps": 10,
    }


def test_run_requires_pinned_base_and_verified_dataset_path(tmp_path):
    value = config(tmp_path)
    assert module.validate_config(value) is value
    value["base_model"]["revision"] = "main"
    with pytest.raises(ValueError, match="pinned"):
        module.validate_config(value)
    value = config(tmp_path)
    value["dataset_identity"] = "not-a-digest"
    with pytest.raises(ValueError, match="path"):
        module.validate_config(value)


def test_checkpoint_retention_is_independent_of_save_interval(tmp_path):
    value = config(tmp_path)
    value["save_steps"] = 2
    value["save_total_limit"] = 3
    assert module.validate_config(value) is value
    value["save_total_limit"] = 0
    with pytest.raises(ValueError, match="save_total_limit"):
        module.validate_config(value)


def test_run_passes_retention_limit_to_trainer_and_records_it(tmp_path, monkeypatch):
    import json

    value = config(tmp_path)
    value["save_steps"] = 2
    value["save_total_limit"] = 3
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(value))
    checkpoint = tmp_path / "run/outputs/rldx/rldx/checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "cyclo_input_metadata.json").write_text(json.dumps({
        "state_names": ["joint"], "action_names": ["joint"],
        "cameras": {"front": "observation.images.front"},
    }))
    receipt = tmp_path / "training_environment.json"
    receipt.write_text('{"pixi_lock_sha256": "pinned"}\n')
    monkeypatch.setattr(module, "_environment_receipt", lambda: receipt)
    monkeypatch.setattr(module, "_prepared", lambda _: Path(value["dataset_path"]))
    monkeypatch.setattr(module, "snapshot_download", lambda **_: None)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(kwargs))
    result = module.run(config_path, "train")
    assert result["step"] == 10
    assert calls[0]["env"]["SAVE_STEPS"] == "2"
    assert calls[0]["env"]["SAVE_TOTAL_LIMIT"] == "3"
    assert json.loads((checkpoint / "train_config.json").read_text())["save_total_limit"] == 3
    assert (checkpoint / "training_environment.json").read_text() == receipt.read_text()


def test_missing_environment_receipt_rejects_before_training(tmp_path, monkeypatch):
    import json

    value = config(tmp_path)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(value))
    monkeypatch.setattr(module, "_prepared", lambda _: Path(value["dataset_path"]))
    monkeypatch.setattr(module, "_environment_receipt", lambda: tmp_path / "missing.json")
    monkeypatch.setattr(module, "snapshot_download", lambda **_: pytest.fail("download started"))
    with pytest.raises(ValueError, match="receipt is missing"):
        module.run(config_path, "train")


def test_checkpoint_selection_uses_numeric_step_and_exact_layout(tmp_path):
    for step in (2, 10):
        (tmp_path / "rldx" / "rldx" / f"checkpoint-{step}").mkdir(parents=True)
    (tmp_path / "checkpoint-999").mkdir()
    assert module._selected_checkpoint(tmp_path).name == "checkpoint-10"


def test_trainer_requires_bench_prepared_v21_metadata(tmp_path):
    import json

    value = config(tmp_path)
    root = Path(value["dataset_path"])
    (root / "meta").mkdir()
    names = ["joint_a", "joint_b"]
    (root / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v3.0",
        "features": {"observation.state": {"names": names}, "action": {"names": names}},
    }))
    (root / "cyclo_input_metadata.json").write_text(json.dumps({
        "state_names": names, "action_names": names,
        "cameras": {"front": "observation.images.front"},
    }))
    for name in ("modality.json", "stats.json", "tasks.jsonl", "episodes.jsonl"):
        (root / "meta" / name).write_text("{}\n")
    with pytest.raises(ValueError, match="Bench-prepared"):
        module._prepared(value)
    info = json.loads((root / "meta/info.json").read_text())
    info["codebase_version"] = "v2.1"
    (root / "meta/info.json").write_text(json.dumps(info))
    assert module._prepared(value) == root
    assert not (tmp_path / "run/dataset-rldx-v21").exists()
