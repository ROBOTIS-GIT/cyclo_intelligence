#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared path-safety helpers for the RLT supervisor services."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException


def read_json_object(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Read one bounded, regular JSON file whose root value is an object."""

    if path.is_symlink() or not path.is_file():
        raise HTTPException(400, f"Missing or unsafe {label}: {path}")
    try:
        if path.stat().st_size > max_bytes:
            raise HTTPException(400, f"{label} is too large: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, f"Invalid {label}: expected a JSON object")
    return payload


def has_symlink_component(root: Path, candidate: Path) -> bool:
    """Return true when root or any lexical child component is a symlink."""

    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    cursor = root
    if cursor.is_symlink():
        return True
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


def resolve_existing_path(
    raw_path: str,
    *,
    roots: tuple[Path, ...],
    label: str,
    directory: bool = True,
) -> Path:
    """Resolve an absolute existing path constrained beneath allowed roots."""

    value = str(raw_path or "").strip()
    path = Path(value)
    if not value or not path.is_absolute():
        raise HTTPException(400, f"{label} must be an absolute path")
    lexical = Path(os.path.abspath(value))

    for raw_root in roots:
        try:
            lexical.relative_to(raw_root)
        except ValueError:
            continue
        if has_symlink_component(raw_root, lexical):
            raise HTTPException(400, f"{label} must not contain symbolic links")
        try:
            root = raw_root.resolve(strict=True)
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        valid_type = resolved.is_dir() if directory else resolved.is_file()
        if not valid_type:
            expected = "directory" if directory else "file"
            raise HTTPException(400, f"{label} must be a {expected}: {lexical}")
        return resolved

    allowed = ", ".join(str(root) for root in roots)
    raise HTTPException(400, f"{label} must be under: {allowed}")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a mapping with the RLT manifest's canonical JSON encoding."""

    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _version_from_info(info: Mapping[str, Any], unsupported_message: str) -> str:
    value = str(info.get("codebase_version") or "").strip().lower()
    if value.startswith("v"):
        value = value[1:]
    if value.startswith("2.1"):
        return "v2.1"
    if value.startswith("3"):
        return "v3.0"
    raise HTTPException(400, unsupported_message)


def lerobot_dataset_version(
    dataset: Path,
    *,
    max_json_bytes: int,
    unsupported_message: str,
) -> str:
    """Read and normalize the supported LeRobot dataset version."""

    info = read_json_object(
        dataset / "meta" / "info.json",
        label="LeRobot metadata",
        max_bytes=max_json_bytes,
    )
    return _version_from_info(info, unsupported_message)


def validate_lerobot_dataset(
    dataset: Path,
    *,
    expected_version: Literal["v2.1", "v3.0"],
    max_json_bytes: int,
    unsupported_message: str,
    wrong_version_message: str,
    unsafe_metadata_message: str,
    outcome_requirement: Literal["none", "present", "boolean"] = "none",
    missing_outcome_message: str = "",
) -> Path:
    """Validate the metadata layout shared by the two RLT supervisors.

    Callers supply their existing user-facing error strings because Stage 1
    and Stage 2 intentionally describe failures in stage-specific terms.
    Outcome requirements also remain explicit: Stage 1 ignores labels, while
    Stage 2 accepts a legacy v2.1 declaration and requires a boolean v3 field.
    """

    info = read_json_object(
        dataset / "meta" / "info.json",
        label="LeRobot metadata",
        max_bytes=max_json_bytes,
    )
    version = _version_from_info(info, unsupported_message)
    if version != expected_version:
        raise HTTPException(400, wrong_version_message)

    if expected_version == "v2.1":
        required = (
            dataset / "meta" / "episodes.jsonl",
            dataset / "meta" / "tasks.jsonl",
        )
        safe_metadata = not any(
            path.is_symlink() or not path.is_file() for path in required
        )
    else:
        tasks = dataset / "meta" / "tasks.parquet"
        episodes = dataset / "meta" / "episodes"
        shards = (
            sorted(episodes.glob("chunk-*/file-*.parquet"))
            if episodes.is_dir()
            else []
        )
        safe_metadata = not (
            tasks.is_symlink()
            or not tasks.is_file()
            or episodes.is_symlink()
            or not episodes.is_dir()
            or not shards
            or any(path.is_symlink() or not path.is_file() for path in shards)
        )
    if not safe_metadata:
        raise HTTPException(400, unsafe_metadata_message)

    features = info.get("features")
    if outcome_requirement == "present":
        has_outcome = isinstance(features, Mapping) and "episode_success" in features
    elif outcome_requirement == "boolean":
        outcome = (
            features.get("episode_success")
            if isinstance(features, Mapping)
            else None
        )
        has_outcome = isinstance(outcome, Mapping) and outcome.get("dtype") == "bool"
    else:
        has_outcome = True
    if not has_outcome:
        raise HTTPException(400, missing_outcome_message)
    return dataset
