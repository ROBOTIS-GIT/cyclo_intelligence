"""Run-scoped RLDX training and checkpoint verification on a prepared Dataset.

The caller supplies a pinned JSON configuration.  This script never chooses a
robot layout from its name, and it never mutates the source Dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from huggingface_hub import snapshot_download



SHA = re.compile(r"^[0-9a-f]{40}$")
CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_config(value: dict) -> dict:
    if set(value) != {
        "version", "model_id", "policy_id", "dataset_path", "run_root",
        "base_model", "steps", "batch_size", "gradient_accumulation_steps",
        "save_steps", "dataset_identity",
    } or (value["version"], value["model_id"], value["policy_id"]) != (
        1, "rldx-rldx1", "rldx:rldx1",
    ):
        raise ValueError("RLDX Run configuration identity is invalid")
    run_root = Path(value["run_root"])
    dataset_path = Path(value["dataset_path"])
    if (not run_root.is_absolute() or run_root == Path("/")
            or not dataset_path.is_absolute() or not dataset_path.is_dir()
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}",
                                str(value["dataset_identity"]))
            ):
        raise ValueError("RLDX Dataset or Run path is invalid")
    base = value["base_model"]
    if (not isinstance(base, dict) or set(base) != {"repo_id", "revision"}
            or base["repo_id"] != "RLWRLD/RLDX-1-PT-IMG"
            or not SHA.fullmatch(str(base["revision"]))):
        raise ValueError("RLDX base model must use a pinned PT-IMG revision")
    for name in ("steps", "batch_size", "gradient_accumulation_steps", "save_steps"):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError(f"RLDX {name} must be a positive integer")
    if value["save_steps"] > value["steps"]:
        raise ValueError("RLDX save_steps cannot exceed steps")
    return value


def _prepared(value: dict) -> Path:
    target = Path(value["dataset_path"])
    if target.is_symlink() or (target / "INCOMPLETE").exists():
        raise ValueError("RLDX prepared Dataset is unsafe or incomplete")
    metadata = _load(target / "cyclo_input_metadata.json")
    info = _load(target / "meta/info.json")
    if str(info.get("codebase_version") or "").lstrip("v") != "2.1":
        raise ValueError("RLDX requires a Bench-prepared LeRobot v2.1 Dataset")
    for name in ("modality.json", "stats.json", "tasks.jsonl", "episodes.jsonl"):
        if not (target / "meta" / name).is_file():
            raise ValueError(f"RLDX prepared Dataset has no meta/{name}")
    for key, feature in (("state_names", "observation.state"),
                         ("action_names", "action")):
        if metadata.get(key) != info["features"][feature].get("names"):
            raise ValueError(f"RLDX prepared {key} differs from meta/info.json")
    if not metadata.get("cameras"):
        raise ValueError("RLDX prepared Dataset has no cameras")
    return target


def _selected_checkpoint(root: Path) -> Path:
    candidates = []
    for path in root.glob("rldx/rldx/checkpoint-*"):
        match = CHECKPOINT.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError("RLDX trainer saved no checkpoint")
    return max(candidates)[1]


def run(config_path: Path, phase: str) -> dict:
    value = validate_config(_load(config_path))
    root = Path(value["run_root"])
    prepared = _prepared(value)
    if phase not in {"validate", "train"}:
        raise ValueError("RLDX phase must be validate or train")
    base = value["base_model"]
    local_base = root / "base-model" / base["revision"]
    local_base.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=base["repo_id"], revision=base["revision"],
                      local_dir=local_base)
    output_root = (root / "input-validation" / "output"
                   if phase == "validate" else root / "outputs")
    output = output_root / "rldx"
    steps = 1 if phase == "validate" else value["steps"]
    batch_size = 1 if phase == "validate" else value["batch_size"]
    accumulation = 1 if phase == "validate" else value["gradient_accumulation_steps"]
    save_steps = 1 if phase == "validate" else value["save_steps"]
    environment = {
        **os.environ,
        "RLDX_REPO": "/rldx", "RLDX_DATASET_PATH": str(prepared),
        "RLDX_OUTPUT": str(output), "RLDX_BASE_MODEL": str(local_base),
        "STEPS": str(steps), "BATCH_SIZE": str(batch_size),
        "GRAD_ACCUM": str(accumulation),
        "SAVE_STEPS": str(save_steps),
    }
    subprocess.run(["bash", "/opt/cyclo-training/train.sh"],
                   env=environment, check=True)
    checkpoint = _selected_checkpoint(output_root)
    metadata = _load(checkpoint / "cyclo_input_metadata.json")
    source = {"repo_id": base["repo_id"], "revision": base["revision"]}
    mapping = {
        "version": 1, "state_names": metadata["state_names"],
        "action_names": metadata["action_names"],
        "state_dim": len(metadata["state_names"]),
        "action_dim": len(metadata["action_names"]),
        "dataset": {"identity": value["dataset_identity"]},
    }
    config = {
        **value, "base_model": source,
        "resolved": {"state_names": mapping["state_names"],
                     "action_names": mapping["action_names"],
                     "state_dim": mapping["state_dim"],
                     "action_dim": mapping["action_dim"],
                     "cameras": metadata["cameras"],
                     "action_representation": "absolute"},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    for name, content in (("train_config.json", config),
                          ("cyclo_io_mapping.json", mapping)):
        (checkpoint / name).write_text(json.dumps(content, sort_keys=True, indent=2) + "\n")
    subprocess.run([sys.executable, "/opt/cyclo-training/checkpoint_smoke.py",
                    str(checkpoint), str(prepared)], check=True)
    return {"status": "validated" if phase == "validate" else "succeeded",
            "checkpoint": str(checkpoint),
            "step": int(CHECKPOINT.fullmatch(checkpoint.name).group(1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("validate", "train"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.phase), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
