#!/usr/bin/env python3
"""Check source hashes and optionally the actual inference-image imports on CPU."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def verify(overlay: Path, base_root: Path | None = None, check_imports: bool = False):
    manifest = json.loads((overlay / "manifest.json").read_text())
    if manifest.get("kind") != "cyclo-model-overlay":
        raise ValueError("Not a Cyclo model overlay")
    for relative, expected in manifest["files"].items():
        actual = hashlib.sha256((overlay / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Overlay hash mismatch: {relative}")
    if base_root is not None:
        for relative, expected in manifest["baseline_dependencies"].items():
            if hashlib.sha256((base_root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Inference base differs from pinned dependency: {relative}")
        for relative, expected in manifest["files"].items():
            if hashlib.sha256((base_root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Inference overlay mount differs: {relative}")
    if check_imports:
        if base_root is None:
            raise ValueError("--check-imports requires --base-root pointing at the mounted inference tree")
        sys.path.insert(0, str(base_root.resolve()))
        for relative in manifest["files"]:
            module = importlib.import_module(relative[:-3].replace("/", "."))
            if Path(module.__file__).resolve() != (base_root / relative).resolve():
                raise ValueError(f"Imported a different source tree: {relative}")
    print(f"Verified {len(manifest['files'])} overlay files; baseline {manifest['isaac_groot_baseline']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=Path(__file__).resolve().parents[1] / "model_overrides")
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    verify(args.overlay, args.base_root, args.check_imports)
