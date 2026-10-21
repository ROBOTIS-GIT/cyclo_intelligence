import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("checkpoint_language", Path(__file__).with_name("checkpoint_instructions.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def checkpoint(tmp_path):
    model = tmp_path / "model/groot/example"
    model.mkdir(parents=True)
    return tmp_path, model


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read(checkpoint):
    root, model = checkpoint
    return module.read_checkpoint_instructions(str(model), "groot", root)


def test_exact_phase_protocol_and_separate_overall_instruction(checkpoint):
    _, model = checkpoint
    prompts = [["1. Align 3–5 cm."], ["2. Release, then wait."]]
    write(model / "inference_protocol.json", {"stage_instruction_choices": prompts})
    meta = model / "training_dataset_metadata/meta"
    meta.mkdir(parents=True)
    (meta / "tasks.jsonl").write_text('\n'.join(json.dumps(row) for row in [
        {"task_index": 0, "task": "Insert and release."},
        {"task_index": 1, "task": prompts[0][0]},
        {"task_index": 2, "task": "Overall task: Insert and release. Current subtask 1/2: Align."},
    ]))
    result = read(checkpoint)
    assert result["conditioning"] == "subtask"
    assert result["task_instructions"] == ["Insert and release."]
    assert result["subtask_instruction_choices"] == prompts
    assert result["default_instruction"] == prompts[0][0]


def test_instruction_revision_overrides_older_vocabulary(checkpoint):
    _, model = checkpoint
    write(model / "instruction_revision.json", {"raw_instruction": "New three to five centimeters.", "model_instruction": "normalized text"})
    result = read(checkpoint)
    assert result["conditioning"] == "task"
    assert result["default_instruction"] == "New three to five centimeters."
    assert result["subtask_instruction_choices"] == []


def test_v21_and_v30_metadata_and_processor_language_key(checkpoint):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    _, model = checkpoint
    write(model / "processor_config.json", {"processor_kwargs": {"modality_configs": {"new_embodiment": {"language": {"modality_keys": ["sub_task"]}}}}})
    meta = model / "training_dataset_metadata/meta"
    meta.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"subtask_index": 1, "subtask": "Release."},
        {"subtask_index": 0, "subtask": "Align 3–5 cm."},
    ]), meta / "subtasks.parquet")
    # Native LeRobot v3 uses the pandas index for the task text.
    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "__index_level_0__": "Overall task."}]), meta / "tasks.parquet")
    result = read(checkpoint)
    assert result["task_instructions"] == ["Overall task."]
    assert result["subtask_instruction_choices"] == [["Align 3–5 cm."], ["Release."]]
    assert result["default_instruction"] == "Align 3–5 cm."


def test_training_config_can_reference_local_dataset(checkpoint):
    root, model = checkpoint
    dataset = root / "dataset/example"
    write(model / "train_config.json", {"dataset": {"root": str(dataset), "repo_id": "org/example"}})
    write(model / "inference_instructions.json", {"conditioning": "task"})
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"Pick up."}\n')
    result = read(checkpoint)
    assert result["task_instructions"] == ["Pick up."]
    assert result["dataset"] == {"repo_id": "org/example"}


def test_unknown_checkpoint_does_not_invent_language(checkpoint):
    result = read(checkpoint)
    assert not result["available"]
    assert result["default_instruction"] == ""


def test_invalid_oversize_and_symlink_metadata_are_ignored(checkpoint, tmp_path):
    root, model = checkpoint
    (model / "inference_protocol.json").write_text("{")
    (model / "inference_instructions.json").write_text(" " * (module.MAX_BYTES + 1))
    outside = root.parent / "outside-checkpoint-instruction.json"
    write(outside, {"raw_instruction": "Must not leak."})
    (model / "instruction_revision.json").symlink_to(outside)
    assert not read(checkpoint)["available"]


def test_rejects_outside_model_paths_and_symlinks(checkpoint):
    root, model = checkpoint
    with pytest.raises(ValueError):
        module.read_checkpoint_instructions(str(root / "../outside"), "groot", root)
    link = model.parent / "escape"
    link.symlink_to(root.parent)
    with pytest.raises(ValueError):
        module.read_checkpoint_instructions(str(link), "groot", root)


def test_pretrained_model_inherits_checkpoint_metadata(checkpoint):
    root, model = checkpoint
    child = model / "pretrained_model"
    child.mkdir()
    write(model / "inference_instructions.json", {"task_instructions": ["Overall."], "subtask_instruction_choices": [["Align."], ["Release."]]})
    result = module.read_checkpoint_instructions(str(child), "groot", root)
    assert result["default_instruction"] == "Align."
