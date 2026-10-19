"""Synchronous, checksummed native checkpoints. Optimizer resume is not supported."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .profile import inside, read_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_checkpoint(root, *, step, model, processor, metadata):
    import torch

    if type(step) is not int or step < 1:
        raise ValueError("Checkpoint step must be a positive optimizer-step count")
    required = {"profile.json", "train_config.json", "model_config.json", "dataset_statistics.json",
                "cyclo_io_mapping.json", "initialization.json"}
    if set(metadata) != required:
        raise ValueError("Checkpoint metadata set differs from the native contract")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"checkpoint-{step}"
    if target.exists():
        raise FileExistsError(target)
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    try:
        (pending / "checkpoints").mkdir()
        weights = f"checkpoints/steps_{step}_pytorch_model.pt"
        torch.save(model.state_dict(), pending / weights)
        processor.save_pretrained(pending / "processor")
        if not (pending / "processor").is_dir() or not any((pending / "processor").iterdir()):
            raise ValueError("Processor did not save its assets")
        for name, value in metadata.items():
            write_json(pending / name, value)
        files = {}
        for file in sorted(pending.rglob("*")):
            if file.is_symlink():
                raise ValueError("Checkpoint may not contain symlinks")
            if file.is_file():
                files[file.relative_to(pending).as_posix()] = file_hash(file)
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        write_json(pending / "checkpoint.json", {
            "version": 1, "policy_id": "abot:m0", "optimizer_step": step,
            "weights": weights, "resume_supported": False, "sha256": files,
        })
        with (pending / "checkpoint.json").open("rb") as stream:
            os.fsync(stream.fileno())
        pending.rename(target)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return target


def verify_checkpoint(path):
    path = Path(path).resolve(strict=True)
    receipt = read_json(path / "checkpoint.json")
    if (receipt.get("version") != 1 or receipt.get("policy_id") != "abot:m0"
            or type(receipt.get("optimizer_step")) is not int or receipt["optimizer_step"] < 1
            or receipt.get("resume_supported") is not False):
        raise ValueError("Invalid native checkpoint identity")
    hashes = receipt["sha256"]
    required = {"profile.json", "train_config.json", "model_config.json", "dataset_statistics.json",
                "cyclo_io_mapping.json", "initialization.json", receipt["weights"]}
    if not required <= hashes.keys() or not any(p.startswith("processor/") for p in hashes):
        raise ValueError("Checkpoint is missing required artifacts")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    if actual != set(hashes) | {"checkpoint.json"} or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Checkpoint file inventory differs")
    for name, expected in hashes.items():
        if file_hash(inside(path, name)) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return receipt


def retain_checkpoints(root, *, save_total_limit):
    if type(save_total_limit) is not int or save_total_limit < 1:
        raise ValueError("save_total_limit must be a positive checkpoint count")
    checkpoints = []
    for path in Path(root).glob("checkpoint-*"):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Unexpected checkpoint entry; refusing retention")
        receipt = verify_checkpoint(path)
        if path.name != f"checkpoint-{receipt['optimizer_step']}":
            raise ValueError("Checkpoint directory/step mismatch")
        checkpoints.append((receipt["optimizer_step"], path))
    for _, path in sorted(checkpoints)[:-save_total_limit]:
        shutil.rmtree(path)
