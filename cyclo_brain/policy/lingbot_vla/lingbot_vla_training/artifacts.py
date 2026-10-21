"""Pinned asset verification and atomic, immutable checkpoint completion."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil


ASSET_ROLES = {"pretrained", "processor", "moge", "morgbd", "video", "video_config"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_assets(manifest):
    if manifest.get("version") != 1 or set(manifest.get("assets", {})) != ASSET_ROLES:
        raise ValueError("All six pinned pretrained/processor/teacher asset roles are required")
    paths = {}
    for role, asset in manifest["assets"].items():
        if not asset.get("repo_id") or not re.fullmatch(r"[a-f0-9]{40}", asset.get("revision", "")):
            raise ValueError(f"{role}: immutable repository revision is required")
        root = Path(asset["path"]).resolve(strict=True)
        files = asset.get("files", [])
        seen = set()
        for entry in files:
            name = entry["path"]
            file = root / name
            if (name in seen or Path(name).is_absolute() or ".." in Path(name).parts
                or not file.resolve(strict=True).is_relative_to(root) or not file.is_file()
                or not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", ""))):
                raise ValueError(f"Invalid {role} asset entry: {name}")
            if sha256(file) != entry["sha256"]:
                raise ValueError(f"{role} asset checksum mismatch: {name}")
            seen.add(name)
        if not seen:
            raise ValueError(f"Empty asset inventory: {role}")
        if role == "pretrained":
            index_path = root / "model.safetensors.index.json"
            if index_path.name not in seen:
                raise ValueError("Pretrained index must be pinned")
            index = json.loads(index_path.read_text())
            shards = set(index["weight_map"].values())
            if shards != {name for name in seen if name.endswith(".safetensors")}:
                raise ValueError("Pretrained index and shard inventory disagree")
        if role == "processor" and not {"config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"} <= seen:
            raise ValueError("Incomplete processor/config inventory")
        paths[role] = root if role in {"pretrained", "processor"} else root / next(iter(seen))
        if role not in {"pretrained", "processor"} and len(seen) != 1:
            raise ValueError(f"{role} must identify exactly one file")
    return paths


def complete_checkpoint(staging, destination, step):
    staging, destination = Path(staging), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    required = {"train_config.json", "cyclo_io_mapping.json", "norm_stats.json", "config.json", "probe.safetensors"}
    if not all((staging / name).is_file() for name in required) or not list(staging.glob("model*.safetensors")):
        raise ValueError("Checkpoint is missing required assets")
    files = {}
    for file in sorted(staging.rglob("*")):
        if file.is_symlink():
            raise ValueError("Checkpoint must not depend on symlinks")
        if file.is_file():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
            files[str(file.relative_to(staging))] = sha256(file)
    write_json(staging / "complete.json", {"version": 1, "optimizer_step": step, "files": files,
                                           "resume_supported": False})
    staging.rename(destination)
    fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify_checkpoint(path):
    path = Path(path).resolve(strict=True)
    receipt = json.loads((path / "complete.json").read_text())
    actual = {str(file.relative_to(path)) for file in path.rglob("*") if file.is_file() and file.name != "complete.json"}
    if actual != set(receipt["files"]):
        raise ValueError("Checkpoint inventory changed")
    for name, digest in receipt["files"].items():
        file = path / name
        if not file.resolve(strict=True).is_relative_to(path) or file.is_symlink() or sha256(file) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return receipt


def prune_checkpoints(root, keep):
    if type(keep) is not int or keep < 1:
        raise ValueError("save_total_limit must be at least one")
    completed = []
    for path in Path(root).iterdir():
        if not path.is_symlink() and re.fullmatch(r"step-\d+", path.name) and (path / "complete.json").is_file():
            receipt = json.loads((path / "complete.json").read_text())
            if receipt["optimizer_step"] != int(path.name.removeprefix("step-")):
                raise ValueError("Checkpoint step/name mismatch")
            completed.append(path)
    completed.sort(key=lambda path: int(path.name.removeprefix("step-")))
    for path in completed[:-keep]:
        shutil.rmtree(path)
