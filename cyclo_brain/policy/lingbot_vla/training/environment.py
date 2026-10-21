"""Build-time checks and receipts for the isolated LingBot training environment."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_tree_sha256(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Source symlinks are unsupported: {relative}")
        if path.is_file():
            digest.update(relative.as_posix().encode() + b"\0")
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def verify_inputs(policy, upstream):
    contract = tomllib.loads((policy / "environment.toml").read_text())
    for name, expected in contract["policy_hashes"].items():
        if sha256(policy / name) != expected:
            raise ValueError(f"Training lock input changed: {name}; regenerate and review the lock")
    if source_tree_sha256(upstream) != contract["upstream_tree_sha256"]:
        raise ValueError("LingBot upstream source differs from the qualified lock input")
    return contract


def fetch_source(asset, destination):
    """Install only a checksum-verified source tree, without resolving dependencies."""
    if destination.exists():
        raise ValueError(f"Refusing to overwrite source destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        tmp = Path(directory)
        archive = tmp / "source.tar.gz"
        with urllib.request.urlopen(asset["url"], timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(archive) != asset["sha256"]:
            raise ValueError("Source archive checksum mismatch")
        with tarfile.open(archive) as source:
            members = []
            subtree = asset.get("source_subdir")
            for member in source.getmembers():
                path = Path(member.name)
                if (
                    path.is_absolute() or ".." in path.parts
                    or not path.parts or path.parts[0] != asset["archive_root"]
                ):
                    raise ValueError(f"Unsupported source archive entry: {member.name}")
                retained_root_file = (
                    len(path.parts) == 2 and path.parts[1] in asset.get("root_files", [])
                )
                if subtree and not retained_root_file and (len(path.parts) < 2 or path.parts[1] != subtree):
                    continue
                if path.suffix in asset.get("exclude_suffixes", []):
                    continue
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported source archive entry: {member.name}")
                members.append(member)
            if not members:
                raise ValueError("Source archive contains no selected files")
            source.extractall(tmp / "unpacked", members=members, filter="data")
        (tmp / "unpacked" / asset["archive_root"]).rename(destination)


def smoke():
    """Imports only: no checkpoint download, teacher weights, CUDA allocation or ROS."""
    modules = [
        "torch", "torchvision", "torchaudio", "torchcodec", "torchdata",
        "transformers", "peft", "datasets", "av", "cv2", "numpydantic",
        "utils3d", "moge.model.v2", "mdm.model.v2",
        "lerobot.datasets.lerobot_dataset", "lingbotvla.utils.normalize",
        "lingbotvla.models.vla.vision_models.dino_video.teacher",
        "lingbotvla.models.vla.vision_models.dino_video.lumos_dinov3.models.navit_eval",
    ]
    for name in modules:
        importlib.import_module(name)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lingbotvla.utils.normalize import NormStats

    stats = NormStats(mean=np.zeros(2), std=np.ones(2))
    assert stats.mean.shape == (2,)
    table = pa.table({"sample": [0, 1], "state": [[1., 2.], [3., 4.]]})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.parquet"
        pq.write_table(table, path)
        assert pq.read_table(path).equals(table)
    return {
        "imports": modules, "pyarrow_version": pa.__version__,
        "parquet_roundtrip": True, "normalization_schema": True,
        "teacher_weights_loaded": False, "gpu_execution": False,
    }


def write_receipt(policy, upstream, output, arch, max_jobs):
    contract = verify_inputs(policy, upstream)
    if arch not in contract["flash_attention"]["supported_build_archs"] or max_jobs < 1:
        raise ValueError("Unsupported FlashAttention build architecture or MAX_JOBS")
    subprocess.run(["uv", "pip", "check", "--python", sys.executable], check=True)
    import torch

    if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.8":
        raise ValueError("Training requires Torch 2.8.0 built for CUDA 12.8")
    flash = importlib.import_module("flash_attn")
    if flash.__version__ != contract["flash_attention"]["version"]:
        raise ValueError("Installed FlashAttention differs from the source contract")
    validation = smoke()
    validation["flash_attention_import"] = flash.__version__
    packages = dict(sorted(
        (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
    ))
    receipt = {
        "schema_version": 1, "cyclo_commit": os.environ.get("CYCLO_COMMIT", "uncommitted"),
        "environment_sha256": sha256(policy / "environment.toml"),
        "inputs": contract, "python": sys.executable, "python_version": platform.python_version(),
        "torch_cuda": torch.version.cuda, "packages": packages,
        "flash_attention_build": {
            "source": contract["flash_attention"], "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "FLASH_ATTN_CUDA_ARCHS": arch, "MAX_JOBS": max_jobs, "no_build_isolation": True,
        },
        "validation": validation, "training_qualified": False,
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "fetch", "smoke", "receipt", "tree-hash"))
    parser.add_argument("--policy", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--upstream", type=Path, default=Path("/lingbot-vla-v2"))
    parser.add_argument("--source-destination", type=Path, default=Path("/opt/lerobot"))
    parser.add_argument("--output", type=Path, default=Path("/opt/lingbot-build/build_receipt.json"))
    parser.add_argument("--flash-arch", default="120")
    parser.add_argument("--max-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.command == "tree-hash":
        print(source_tree_sha256(args.upstream))
    elif args.command == "smoke":
        print(json.dumps(smoke(), indent=2))
    elif args.command == "receipt":
        write_receipt(args.policy, args.upstream, args.output, args.flash_arch, args.max_jobs)
    else:
        contract = verify_inputs(args.policy, args.upstream)
        if args.command == "fetch":
            fetch_source(contract["lerobot_dataset_source"], args.source_destination)
        else:
            print("Training sources and dependency lock match.")


if __name__ == "__main__":
    main()
