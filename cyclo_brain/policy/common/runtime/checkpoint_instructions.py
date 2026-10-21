"""Read checkpoint language metadata without loading a policy or training frames.

Portable checkpoints can carry ``inference_instructions.json`` with
``task_instructions``, ``subtask_instruction_choices`` (ordered lists of exact
prompts), and ``conditioning`` (task/subtask). Alternatively, place the native
LeRobot meta files in ``training_dataset_metadata/meta`` beside the checkpoint.
Explicit inference protocols/instruction revisions take precedence over the
dataset vocabulary: annotations alone do not establish a policy's language input.
"""

from __future__ import annotations

import json
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024
MAX_ITEMS = 512


def checkpoint_subtask_choices(model_path: str) -> list[list[str]]:
    """Enable transitions only for a known GR00T multi-phase checkpoint."""
    try:
        profile = read_checkpoint_instructions(model_path, "groot")
    except (ValueError, OSError):
        return []
    choices = profile["subtask_instruction_choices"]
    return choices if profile["conditioning"] == "subtask" and len(choices) > 1 else []


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _text(path: Path, root: Path) -> str:
    if not _inside(path, root) or not path.is_file() or path.stat().st_size > MAX_BYTES:
        return ""
    return path.read_text(encoding="utf-8")


def _json(path: Path, root: Path) -> dict:
    try:
        value = json.loads(_text(path, root))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _strings(values) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(v for v in values[:MAX_ITEMS] if isinstance(v, str) and v.strip()))


def _choices(values) -> list[list[str]]:
    if not isinstance(values, list):
        return []
    return [choices for v in values[:MAX_ITEMS] if (choices := _strings(v))]


def _vocabulary(path: Path, root: Path, field: str) -> list[str]:
    """Read only small metadata tables; preserve punctuation and index order."""
    try:
        if not _inside(path, root) or not path.is_file() or path.stat().st_size > MAX_BYTES:
            return []
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in _text(path, root).splitlines() if line.strip()]
        else:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows > MAX_ITEMS:
                return []
            columns = [c for c in [field, f"{field}_index", "__index_level_0__"]
                       if c in parquet.schema.names]
            rows = parquet.read(columns=columns).to_pylist()
        if len(rows) > MAX_ITEMS or not all(isinstance(row, dict) for row in rows):
            return []
        rows.sort(key=lambda r: int(r.get(f"{field}_index", 0)))
        return _strings([r.get(field, r.get("__index_level_0__")) for r in rows])
    except (OSError, ValueError, TypeError, ImportError):
        return []


def read_checkpoint_instructions(model_path: str, backend: str, workspace="/workspace") -> dict:
    workspace = Path(workspace)
    root = workspace / "model" / backend
    model = Path(model_path)
    if not model.is_absolute() or not _inside(model, root):
        raise ValueError(f"model_path must be under {root}")
    if not model.is_dir():
        raise FileNotFoundError("Checkpoint directory does not exist")
    folders = [model]
    if (model / "pretrained_model").is_dir():
        folders.insert(0, model / "pretrained_model")
    if model.name == "pretrained_model" or model.name.startswith("checkpoint-") or model.name == "best_validation":
        if model.parent != root and _inside(model.parent, root):
            folders.append(model.parent)

    sources = []

    def document(name):
        for folder in folders:
            path = folder / name
            value = _json(path, root)
            if value:
                sources.append(str(path))
                return value
        return {}

    profile = document("inference_instructions.json")
    protocol = document("inference_protocol.json")
    revision = document("instruction_revision.json")
    run = document("provenance/run_status.json")
    train = document("train_config.json")
    bundled_source = document("training_dataset_metadata/source.json")
    dataset = profile.get("dataset") or run.get("source_dataset") or train.get("dataset") or bundled_source or {}
    if not isinstance(dataset, dict):
        dataset = {}
    tasks = _strings(profile.get("task_instructions"))
    stages = _choices(profile.get("subtask_instruction_choices")) or _choices(protocol.get("stage_instruction_choices"))
    conditioning = profile.get("conditioning", "unknown")
    if stages:
        conditioning = "subtask"
    single_prompt = "" if stages else revision.get("raw_instruction", "")
    if not single_prompt and not stages:
        for folder in folders:
            for path in sorted(folder.glob("inference_instruction*.txt"))[:MAX_ITEMS]:
                try:
                    value = _text(path, root).strip()
                except (OSError, UnicodeError):
                    continue
                if value:
                    single_prompt = value
                    sources.append(str(path))
                    break
            if single_prompt:
                break

    metadata_folders = [folder / "training_dataset_metadata" for folder in folders] + folders
    dataset_root = dataset.get("root")
    if isinstance(dataset_root, str) and dataset_root:
        candidate = Path(dataset_root)
        if not candidate.is_absolute():
            candidate = model / candidate
        if _inside(candidate, workspace):
            metadata_folders.append(candidate)
    # GR00T exports contain Python-tagged YAML objects. BaseLoader treats them
    # as data; it neither imports nor executes checkpoint code.
    for folder in folders:
        path = folder / "experiment_cfg/config.yaml"
        try:
            import yaml

            cfg = yaml.load(_text(path, root), Loader=yaml.BaseLoader) or {}
            datasets = cfg.get("data", {}).get("datasets", [])
            if isinstance(datasets, dict):
                datasets = [datasets]
            for entry in datasets[:MAX_ITEMS]:
                for value in entry.get("dataset_paths", [])[:MAX_ITEMS]:
                    candidate = Path(value)
                    if candidate.is_absolute() and _inside(candidate, workspace):
                        metadata_folders.append(candidate)
        except (OSError, ValueError, TypeError, AttributeError, ImportError):
            pass
    processor = document("processor_config.json")
    if conditioning == "unknown":
        language = processor.get("processor_kwargs", {}).get("modality_configs", {}).get("new_embodiment", {}).get("language", {})
        keys = language.get("modality_keys", [])
        if any("sub_task" in k or "subtask" in k for k in keys):
            conditioning = "subtask"
        elif keys:
            conditioning = "task"
    dataset_tasks, dataset_subtasks = [], []
    for folder in dict.fromkeys(metadata_folders):
        for meta in [folder / "meta", folder]:
            for filename, field, target in [("tasks.jsonl", "task", dataset_tasks),
                                             ("tasks.parquet", "task", dataset_tasks),
                                             ("subtasks.parquet", "subtask", dataset_subtasks)]:
                path = meta / filename
                values = _vocabulary(path, workspace, field)
                if values:
                    target.extend(values)
                    sources.append(str(path))
    if not stages and not single_prompt:
        stages = [[text] for text in _strings(dataset_subtasks)]
    phase_prompts = {s for choices in stages for s in choices} | set(dataset_subtasks)
    if not tasks:
        tasks = _strings([text for text in dataset_tasks if text not in phase_prompts
                          and not (text.startswith("Overall task:") and "Current subtask" in text)])
    if single_prompt:
        # A revised full-task prompt replaces older dataset phase vocabulary.
        stages = []
        conditioning = "task"
        if not tasks:
            tasks = [single_prompt]
    default = (stages[0][0] if stages and conditioning == "subtask"
               else single_prompt or (tasks[0] if tasks else ""))
    return {
        "model_path": str(model),
        "available": bool(tasks or stages or default),
        "task_instructions": tasks,
        "subtask_instruction_choices": stages,
        "conditioning": conditioning,
        "default_instruction": default,
        "instruction_choices": ([single_prompt] if single_prompt else
                                [text for choices in stages for text in choices]
                                if conditioning == "subtask" else tasks),
        "dataset": {key: dataset[key] for key in ["repo_id", "revision"] if key in dataset},
        "sources": list(dict.fromkeys(sources)),
    }
