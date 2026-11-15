# Copyright 2026 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0.

"""Verify that converted LeRobot actions came from the live command stream.

RLT inference traces describe candidate chunks.  Only ``action_commanded``
events emitted immediately around ``RobotClient.publish_action`` describe the
commands eligible to become Stage-2 ``executed_actions``.  This module aligns
those events to the converter's absolute frame grid and publishes a small,
tamper-evident qualification manifest.  It never rewrites the source episode.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EPISODE_FORMAT = "cyclo.rlt.episode_execution_verification/v1"
DATASET_FORMAT = "cyclo.rlt.dataset_execution_verification/v1"


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_command_stream(sidecar: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    actor_sha256: set[str] = set()
    summaries = []
    for summary_path in sorted(sidecar.rglob("*_summary.json")):
        summaries.append(_read_json(summary_path))

    for events_path in sorted(sidecar.rglob("events.jsonl")):
        for raw in events_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            metadata = json.loads(raw)
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid RLT event metadata: {events_path}")
            if metadata.get("event") == "inference":
                identity = metadata.get("policy_identity")
                digest = identity.get("actor_sha256") if isinstance(identity, dict) else None
                if isinstance(digest, str) and len(digest) == 64:
                    actor_sha256.add(digest)
            if metadata.get("event") != "action_commanded":
                continue
            filename = metadata.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError(f"Invalid RLT event file name: {events_path}")
            archive_path = events_path.parent / filename
            with np.load(archive_path, allow_pickle=False) as archive:
                if "commanded_action" not in archive:
                    raise ValueError(f"RLT command event has no action: {archive_path}")
                action = np.asarray(archive["commanded_action"], dtype=np.float64)
            if action.ndim == 2 and action.shape[0] == 1:
                action = action[0]
            if action.ndim != 1 or not action.size or not bool(np.isfinite(action).all()):
                raise ValueError(f"RLT commanded action is invalid: {archive_path}")
            started = metadata.get("command_started_ns")
            completed = metadata.get("command_completed_ns")
            if (
                type(started) is not int
                or type(completed) is not int
                or started <= 0
                or completed < started
            ):
                raise ValueError(f"RLT command timestamps are invalid: {archive_path}")
            commands.append(
                {
                    "started_ns": started,
                    "completed_ns": completed,
                    "action": action,
                    "publish_success": metadata.get("publish_success") is True,
                    "action_policy_mode": metadata.get("action_policy_mode"),
                    "recording_id": metadata.get("recording_id"),
                }
            )

    commands.sort(key=lambda row: (row["completed_ns"], row["started_ns"]))
    diagnostics = {
        "summary_count": len(summaries),
        "recorder_dropped": sum(
            int(value.get("recorder_dropped", 0))
            for value in summaries
            if type(value.get("recorder_dropped", 0)) is int
        ),
        "publisher_dropped": sum(
            int(value.get("publisher_dropped", 0))
            for value in summaries
            if type(value.get("publisher_dropped", 0)) is int
        ),
        "publisher_summary_received": bool(summaries) and all(
            value.get("publisher_summary_received") is True
            for value in summaries
        ),
        "summary_errors": [
            str(value.get("error")) for value in summaries if value.get("error")
        ],
        "actor_sha256": sorted(actor_sha256),
    }
    return commands, diagnostics


def verify_episode_execution(
    *,
    sidecar: str | Path,
    episode_index: int,
    grid_log_times_sec: Sequence[float],
    actions: Sequence[np.ndarray],
    episode_success: bool | None,
    fps: float,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Return a deterministic verification record for one converted episode."""

    root = Path(sidecar)
    reasons: list[str] = []
    mismatch_frames: list[int] = []
    max_abs_error = 0.0
    matched = 0
    commands: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "summary_count": 0,
        "recorder_dropped": 0,
        "publisher_dropped": 0,
        "publisher_summary_received": False,
        "summary_errors": [],
        "actor_sha256": [],
    }

    if not root.is_dir():
        reasons.append("RLT sidecar is missing")
    else:
        try:
            commands, diagnostics = _load_command_stream(root)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            reasons.append(f"RLT command stream is invalid: {error}")

    if len(grid_log_times_sec) != len(actions):
        reasons.append("frame timestamps and actions have different lengths")
    if not isinstance(episode_success, bool):
        reasons.append("episode_success is missing")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(float(fps)) or fps <= 0:
        reasons.append("fps is invalid")
        fps = 15.0
    if diagnostics["summary_count"] < 1:
        reasons.append("RLT recording summary is missing")
    if diagnostics["recorder_dropped"]:
        reasons.append("RLT recorder dropped packets")
    if diagnostics["publisher_dropped"]:
        reasons.append("RLT publisher dropped packets")
    if not diagnostics["publisher_summary_received"]:
        reasons.append("RLT publisher completion summary is missing")
    if diagnostics["summary_errors"]:
        reasons.append("RLT recorder reported an error")
    if not commands:
        reasons.append("no commanded robot actions were recorded")
    if any(not row["publish_success"] for row in commands):
        reasons.append("at least one robot command publish failed")
    if any(row["action_policy_mode"] != "rlt" for row in commands):
        reasons.append("episode contains commands outside the RLT route")
    if len(diagnostics["actor_sha256"]) != 1:
        reasons.append("episode does not identify exactly one RLT actor")

    if commands and len(grid_log_times_sec) == len(actions):
        completed = [row["completed_ns"] for row in commands]
        window_ns = int(round(1e9 / float(fps)))
        for frame_index, (timestamp, recorded_action) in enumerate(
            zip(grid_log_times_sec, actions)
        ):
            try:
                target_ns = int(round(float(timestamp) * 1e9))
                expected = np.asarray(recorded_action, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError, OverflowError):
                mismatch_frames.append(frame_index)
                continue
            if not bool(np.isfinite(expected).all()):
                mismatch_frames.append(frame_index)
                continue
            left = bisect_left(completed, target_ns - window_ns)
            right = bisect_right(completed, target_ns + window_ns)
            candidates = commands[left:right]
            errors = [
                float(np.max(np.abs(row["action"] - expected)))
                for row in candidates
                if row["action"].shape == expected.shape
            ]
            if errors:
                best = min(errors)
                max_abs_error = max(max_abs_error, best)
                if best <= atol:
                    matched += 1
                    continue
            mismatch_frames.append(frame_index)

    frame_count = len(actions)
    if mismatch_frames:
        reasons.append("LeRobot actions do not match the commanded action stream")
    unsigned = {
        "format": EPISODE_FORMAT,
        "episode_index": int(episode_index),
        "execution_verified": not reasons and matched == frame_count and frame_count > 0,
        "frame_count": frame_count,
        "matched_frame_count": matched,
        "mismatch_frame_indices": mismatch_frames[:20],
        "max_abs_error": max_abs_error,
        "absolute_tolerance": float(atol),
        "episode_success": episode_success,
        "command_record_count": len(commands),
        "actor_sha256": diagnostics["actor_sha256"],
        "recorder_dropped": diagnostics["recorder_dropped"],
        "publisher_dropped": diagnostics["publisher_dropped"],
        "reasons": reasons,
    }
    return {**unsigned, "verification_fingerprint": _canonical_fingerprint(unsigned)}


def write_dataset_execution_verification(
    output_root: str | Path,
    episode_records: Sequence[Mapping[str, Any]],
) -> Path:
    """Write the dataset-level gate consumed by iterative Stage-2 training."""

    records = [dict(value) for value in episode_records]
    successes = sum(value.get("episode_success") is True for value in records)
    failures = sum(value.get("episode_success") is False for value in records)
    actors = sorted(
        {
            digest
            for value in records
            for digest in value.get("actor_sha256", [])
            if isinstance(digest, str)
        }
    )
    root = Path(output_root)
    action_data_files = []
    for path in sorted((root / "data").rglob("*.parquet")):
        if path.is_symlink() or not path.is_file():
            continue
        action_data_files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "byte_count": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    unsigned = {
        "format": DATASET_FORMAT,
        "execution_verified": bool(records) and all(
            value.get("execution_verified") is True for value in records
        ),
        "episode_count": len(records),
        "success_episode_count": successes,
        "failure_episode_count": failures,
        "actor_sha256": actors,
        "action_data_files": action_data_files,
        "episodes": [
            {
                "episode_index": value.get("episode_index"),
                "execution_verified": value.get("execution_verified"),
                "episode_success": value.get("episode_success"),
                "verification_fingerprint": value.get("verification_fingerprint"),
            }
            for value in records
        ],
    }
    payload = {**unsigned, "verification_fingerprint": _canonical_fingerprint(unsigned)}
    path = root / "rlt" / "execution_verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def validate_dataset_execution_verification(dataset_root: str | Path) -> dict[str, Any]:
    """Fail closed unless a dataset is qualified for iterative RLT training."""

    path = Path(dataset_root) / "rlt" / "execution_verification.json"
    value = _read_json(path)
    fingerprint = value.pop("verification_fingerprint", None)
    if value.get("format") != DATASET_FORMAT:
        raise ValueError("RLT dataset execution verification format is invalid")
    if fingerprint != _canonical_fingerprint(value):
        raise ValueError("RLT dataset execution verification fingerprint disagrees")
    if value.get("execution_verified") is not True:
        raise ValueError("RLT dataset execution has not been verified")
    if not isinstance(value.get("episode_count"), int) or value["episode_count"] < 1:
        raise ValueError("RLT dataset has no verified episodes")
    if value.get("success_episode_count", 0) < 1 or value.get("failure_episode_count", 0) < 1:
        raise ValueError("RLT iterative data requires both success and failure episodes")
    actors = value.get("actor_sha256")
    if not isinstance(actors, list) or len(actors) != 1:
        raise ValueError("RLT iterative data must come from exactly one actor")
    files = value.get("action_data_files")
    if not isinstance(files, list) or not files:
        raise ValueError("RLT dataset execution verification has no action data files")
    root = Path(dataset_root)
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "byte_count",
            "sha256",
        }:
            raise ValueError("RLT action data file record is invalid")
        relative = Path(str(record["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("RLT action data path is unsafe")
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["byte_count"]
            or _file_sha256(path) != record["sha256"]
        ):
            raise ValueError("RLT action data changed after execution verification")
    return {**value, "verification_fingerprint": fingerprint}


__all__ = [
    "DATASET_FORMAT",
    "EPISODE_FORMAT",
    "validate_dataset_execution_verification",
    "verify_episode_execution",
    "write_dataset_execution_verification",
]
