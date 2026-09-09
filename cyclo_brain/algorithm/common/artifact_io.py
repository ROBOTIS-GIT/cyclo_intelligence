"""Durable artifact I/O shared by IL and RL training workflows."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_sha256(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash one file without loading the complete artifact into memory."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash exact parameters and buffers independently of their device."""

    import torch
    from torch import nn

    if not isinstance(module, nn.Module):
        raise TypeError("module_state_sha256 requires a torch module")
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def atomic_torch_save(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    sync_directory: bool = False,
) -> Path:
    """Write a Torch mapping completely before atomically replacing ``path``."""

    import torch

    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
        if sync_directory:
            _fsync_directory(resolved.parent)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return resolved


def atomic_json_save(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    compact: bool = False,
    newline: bool = True,
    sync_directory: bool = False,
) -> Path:
    """Write strict JSON completely before atomically replacing ``path``."""

    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                dict(payload),
                stream,
                allow_nan=False,
                ensure_ascii=ensure_ascii,
                indent=None if compact else indent,
                separators=(",", ":") if compact else None,
                sort_keys=True,
            )
            if newline:
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
        if sync_directory:
            _fsync_directory(resolved.parent)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return resolved


__all__ = [
    "atomic_json_save",
    "atomic_torch_save",
    "file_sha256",
    "module_state_sha256",
]
