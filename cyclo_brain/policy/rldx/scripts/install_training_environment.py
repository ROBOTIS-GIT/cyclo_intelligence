"""Install the pinned RLDX training environment from upstream's Pixi lock."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib


POLICY = Path(__file__).resolve().parents[1] / "training" / "environment.toml"
ARCHES = {"80", "90", "100", "120"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepend_path(path: Path, name: str) -> str:
    return ":".join(filter(None, (str(path), os.environ.get(name))))


def validate(repo: Path, policy_path: Path, archs: str, max_jobs: int) -> dict:
    with policy_path.open("rb") as file:
        policy = tomllib.load(file)
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported RLDX training environment policy")
    for name in ("pixi.toml", "pixi.lock"):
        expected = policy[f"{name.replace('.', '_')}_sha256"]
        if sha256(repo / name) != expected:
            raise ValueError(f"RLDX {name} does not match the pinned training environment")
    if not archs or any(arch not in ARCHES for arch in archs.split(";")):
        raise ValueError("Unsupported FlashAttention CUDA architecture")
    if max_jobs < 1:
        raise ValueError("MAX_JOBS must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", policy["flash_attention_commit"]):
        raise ValueError("FlashAttention must be pinned to a Git commit")
    return policy


def verify_python(python: Path, policy: dict) -> dict:
    script = """
import importlib.metadata as m
import json
import torch
import torchvision
import pyarrow
import pyarrow.parquet as pq
import io
import pytz, dateutil, tqdm, filelock
import flash_attn
import rldx
assert torch.__version__ == '2.8.0+cu128', torch.__version__
assert torchvision.__version__ == '0.23.0+cu128', torchvision.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert pyarrow.__version__ == EXPECTED_PYARROW, pyarrow.__version__
assert m.version('flash-attn') == EXPECTED_FLASH, m.version('flash-attn')
buffer = io.BytesIO()
pq.write_table(pyarrow.table({'frame': [1, 2]}), buffer)
buffer.seek(0)
assert pq.read_table(buffer)['frame'].to_pylist() == [1, 2]
print(json.dumps({
    'python': __import__('sys').executable,
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'cuda': torch.version.cuda,
    'pyarrow': pyarrow.__version__,
    'flash_attention': m.version('flash-attn'),
    'common_imports': ['pytz', 'dateutil', 'tqdm', 'filelock'],
}, sort_keys=True))
""".replace("EXPECTED_PYARROW", repr(policy["pyarrow_version"])).replace(
        "EXPECTED_FLASH", repr(policy["flash_attention_version"])
    )
    result = subprocess.run([str(python), "-c", script], capture_output=True,
                            text=True, check=True)
    return json.loads(result.stdout)


def install(repo: Path, policy_path: Path, archs: str, max_jobs: int) -> dict:
    policy = validate(repo, policy_path, archs, max_jobs)
    version = subprocess.check_output(["pixi", "--version"], text=True).strip()
    if version != f"pixi {policy['pixi_version']}":
        raise ValueError(f"Expected pixi {policy['pixi_version']}, got {version}")
    environment = repo / ".pixi" / "envs" / "rldx"
    if environment.exists():
        raise ValueError("Refusing to replace an existing RLDX training environment")
    subprocess.run(["pixi", "install", "--locked", "-e", "rldx"], cwd=repo,
                   env={**os.environ, "CONDA_OVERRIDE_CUDA": "12.8"}, check=True)
    python = environment / "bin" / "python"
    build_env = {
        **os.environ,
        "PATH": f"{environment / 'bin'}:{os.environ['PATH']}",
        "CUDA_HOME": str(environment),
        "CPATH": _prepend_path(environment / "targets/x86_64-linux/include", "CPATH"),
        "LIBRARY_PATH": _prepend_path(environment / "targets/x86_64-linux/lib", "LIBRARY_PATH"),
        "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
        "FLASH_ATTN_CUDA_ARCHS": archs,
        "TORCH_CUDA_ARCH_LIST": ";".join(
            f"{int(arch) // 10}.{int(arch) % 10}" for arch in archs.split(";")),
        "MAX_JOBS": str(max_jobs),
    }
    source = f"git+{policy['flash_attention_repository']}@{policy['flash_attention_commit']}"
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                    "--no-build-isolation", source], cwd=repo, env=build_env, check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                    "--no-build-isolation", "-e", str(repo)], cwd=repo,
                   env=build_env, check=True)
    versions = verify_python(python, policy)
    validate(repo, policy_path, archs, max_jobs)
    receipt = {
        "upstream_revision": policy["upstream_revision"],
        "pixi_version": policy["pixi_version"],
        "pixi_toml_sha256": policy["pixi_toml_sha256"],
        "pixi_lock_sha256": policy["pixi_lock_sha256"],
        "cyclo_environment_sha256": sha256(policy_path),
        "flash_attention_source": source,
        "flash_attention_build": {
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "FLASH_ATTN_CUDA_ARCHS": archs,
            "TORCH_CUDA_ARCH_LIST": build_env["TORCH_CUDA_ARCH_LIST"],
            "CUDA_HOME": str(environment),
            "CPATH": build_env["CPATH"],
            "LIBRARY_PATH": build_env["LIBRARY_PATH"],
            "MAX_JOBS": str(max_jobs),
        },
        "versions": versions,
    }
    (environment / "training_environment.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--archs", default=os.environ.get("FLASH_ATTN_CUDA_ARCHS", "120"))
    parser.add_argument("--max-jobs", type=int, default=int(os.environ.get("MAX_JOBS", "2")))
    args = parser.parse_args()
    print(json.dumps(install(args.repo, args.policy, args.archs, args.max_jobs),
                     sort_keys=True))
