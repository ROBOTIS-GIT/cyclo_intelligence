"""Reject changed dependency declarations until the lock is regenerated."""

import hashlib
import json
from pathlib import Path


def check_lock(root):
    root = Path(root)
    receipt = json.loads((root / "lock-inputs.json").read_text())
    expected = {"requirements.in", "requirements.lock", "../requirements-inference.txt"}
    if receipt.get("version") != 1 or set(receipt.get("sha256", {})) != expected:
        raise ValueError("Invalid lock input receipt")
    for name, digest in receipt["sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Dependency lock input changed: {name}; regenerate the lock and receipt")


if __name__ == "__main__":
    check_lock(Path(__file__).resolve().parent)
