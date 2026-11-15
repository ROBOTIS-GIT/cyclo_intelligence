#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small, dependency-light contracts shared by the GR00T RLT CLIs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


def positive_int(value: str) -> int:
    """Parse the positive counters used by both RLT training stages."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def json_line(payload: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    """Emit one canonical compact JSON progress event."""

    print(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=stream,
        flush=True,
    )


def resolved_directory(value: str | Path, name: str) -> Path:
    """Resolve one existing non-symlink directory."""

    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be a real directory: {path}")
    return path


def prepare_output_directory(
    value: str | Path,
    inputs: Sequence[Path],
    *,
    stage_label: str,
    overlap_description: str,
) -> Path:
    """Create an empty output that cannot overlap immutable inputs."""

    output = Path(value).expanduser().absolute()
    if output.is_symlink():
        raise ValueError(f"{stage_label} output must not be a symbolic link")
    for source in inputs:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"{stage_label} output overlaps {overlap_description}")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"{stage_label} output is not empty: {output}")
    else:
        output.mkdir(parents=True)
    return output


__all__ = [
    "json_line",
    "positive_int",
    "prepare_output_directory",
    "resolved_directory",
]
