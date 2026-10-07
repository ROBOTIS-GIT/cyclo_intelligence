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
