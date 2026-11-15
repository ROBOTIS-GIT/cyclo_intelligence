"""Shared command-line contracts for ACT-TD3 training entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from cyclo_brain.algorithm.common.cli import json_line as emit_json_line


MAX_SEED = 2**63 - 2
VIDEO_BACKENDS = ("pyav", "torchcodec", "video_reader")


def bounded_integer(
    value: str,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Parse one bounded integer for an ``argparse`` type callback."""

    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
    if not minimum <= result <= maximum:
        raise argparse.ArgumentTypeError(
            f"{name} must be in [{minimum}, {maximum}]"
        )
    return result


def seed_argument(value: str) -> int:
    return bounded_integer(value, name="seed", minimum=0, maximum=MAX_SEED)


def positive_argument(value: str) -> int:
    return bounded_integer(
        value,
        name="value",
        minimum=1,
        maximum=2**31 - 1,
    )


def input_directory(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {resolved}")
    return resolved


def input_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved


def dataset_root_arguments(value: Any) -> tuple[Path, ...]:
    """Normalize argparse and legacy programmatic one-root namespaces."""

    if isinstance(value, Path):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        roots = tuple(value)
        if roots and all(isinstance(root, Path) for root in roots):
            return roots
    raise TypeError("dataset_root must contain one or more paths")


def resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid ACT-TD3 device: {value!r}") from error
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("ACT-TD3 CPU device cannot have an index")
        return device
    if device.type != "cuda" or device.index is None:
        raise ValueError("ACT-TD3 device must be 'cpu' or an explicit CUDA index")
    if not torch.cuda.is_available():
        raise RuntimeError("ACT-TD3 CUDA was requested but is unavailable")
    if not 0 <= device.index < torch.cuda.device_count():
        raise ValueError(f"ACT-TD3 CUDA device index is unavailable: {device.index}")
    torch.cuda.set_device(device)
    return device


def require_local_dataset_layout(dataset_root: Path) -> None:
    required_files = (
        dataset_root / "meta" / "info.json",
        dataset_root / "meta" / "tasks.parquet",
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"LeRobot dataset file is missing: {path}")
    for relative in ("meta/episodes", "data"):
        directory = dataset_root / relative
        if not directory.is_dir() or not any(directory.rglob("*.parquet")):
            raise FileNotFoundError(
                f"LeRobot dataset has no parquet files under: {directory}"
            )


def require_referenced_dataset_files(dataset_root: Path, metadata: Any) -> None:
    episode_indices = tuple(range(int(metadata.total_episodes)))
    referenced = {
        Path(metadata.get_data_file_path(index)) for index in episode_indices
    }
    referenced.update(
        Path(metadata.get_video_file_path(index, video_key))
        for video_key in metadata.video_keys
        for index in episode_indices
    )
    for relative in referenced:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"LeRobot metadata path escapes dataset root: {relative}")
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"LeRobot referenced file is missing: {path}")


__all__ = [
    "MAX_SEED",
    "VIDEO_BACKENDS",
    "bounded_integer",
    "dataset_root_arguments",
    "emit_json_line",
    "input_directory",
    "input_file",
    "positive_argument",
    "require_local_dataset_layout",
    "require_referenced_dataset_files",
    "resolve_device",
    "seed_argument",
]
