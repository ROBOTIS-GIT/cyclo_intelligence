#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Supervisor routes for GR00T RLT Stage 2 actor-critic training.

Stage 2 always publishes a new, self-contained bundle.  ``new`` initializes
the Action MLP and critics from a frozen GR00T checkpoint plus a completed
Stage-1 RL-token encoder.  ``resume`` restores all trainable state from a
completed Stage-2 bundle, but still writes the next round to a new directory.
The input datasets and bundles are consequently never modified in place.

The subprocess contract is intentionally small and stable.  The one-shot
GR00T runner is ``python -m runtime.rlt_stage2_training_cli`` and reports JSON
lines for progress and its terminal artifact paths.
"""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictInt

try:
    from supervisor_api import rlt_service_common as _rlt_common
except ModuleNotFoundError:  # Direct file loading in focused unit tests.
    from docker.supervisor_api import rlt_service_common as _rlt_common


RLT_STAGE2_DATASET_ROOTS = (Path("/workspace/lerobot"),)
RLT_STAGE2_GROOT_ROOTS = (
    Path("/workspace/model/groot"),
    Path("/workspace/checkpoint"),
)
RLT_STAGE2_ENCODER_ROOTS = (Path("/workspace/checkpoint/rlt/stage1"),)
RLT_STAGE2_BUNDLE_ROOTS = (Path("/workspace/checkpoint/rlt/stage2"),)
RLT_STAGE2_OUTPUT_ROOT = Path("/workspace/checkpoint/rlt/stage2")
RLT_STAGE2_LOG_ROOT = Path("/tmp/cyclo_rlt_stage2")
RLT_STAGE2_CACHE_ROOT = "/tmp/cyclo_rlt_stage2_cache"
RLT_STAGE2_HF_HUB_CACHE = "/huggingface_hub"
def _repository_root() -> Path:
    """Locate the live Cyclo source tree in both host and container layouts."""

    configured = os.environ.get("CYCLO_SUPERVISOR_API_REPO_MOUNT", "").strip()
    candidates = [
        *((Path(configured),) if configured else ()),
        Path(__file__).resolve().parents[2],
        Path("/root/ros2_ws/src/cyclo_intelligence"),
    ]
    for candidate in candidates:
        if (candidate / "cyclo_brain" / "policy" / "groot" / "runtime").is_dir():
            return candidate
    # Retain a deterministic path so readiness reports the missing files rather
    # than failing module import when the repository mount itself is absent.
    return candidates[0]


RLT_STAGE2_RUNTIME_ROOT = (
    _repository_root() / "cyclo_brain" / "policy" / "groot" / "runtime"
)
RLT_STAGE2_REQUIRED_RUNTIME_FILES = (
    RLT_STAGE2_RUNTIME_ROOT / "rlt_stage2_dataset.py",
    RLT_STAGE2_RUNTIME_ROOT / "rlt_stage2_training_cli.py",
)
RLT_STAGE2_TRAIN_UID = 1000
RLT_STAGE2_TRAIN_GID = 1000
RLT_STAGE2_LOG_LINES = 100
RLT_STAGE2_MAX_JSON_BYTES = 2 * 1024 * 1024
RLT_STAGE2_DISCOVERY_LIMIT = 256

_STAGE1_RUN_FORMAT = "cyclo.groot.rlt.stage1_run/v1"
_STAGE2_BUNDLE_FORMAT = "cyclo_brain.rlt.stage2_bundle/v2"
_STAGE2_QUALIFICATION = "training_only_not_deployment_validated"
_STAGE2_OUTPUT_PATTERN = re.compile(r"steps_([0-9]+)_([0-9a-f]{12})")
_STAGE2_SPEC_FIELDS = {
    "reference_contract_fingerprint",
    "rl_token_artifact_fingerprint",
    "rl_token_dim",
    "proprio_dim",
    "reference_horizon",
    "chunk_length",
    "action_dim",
    "action_hz",
    "action_normalization_id",
    "action_codec_id",
    "model_domain",
    "schema_version",
}

_has_symlink_component = _rlt_common.has_symlink_component
_resolve_existing_path = _rlt_common.resolve_existing_path
_file_sha256 = _rlt_common.file_sha256
_canonical_fingerprint = _rlt_common.canonical_json_sha256


class RLTStage2StartRequest(BaseModel):
    initialization_mode: Literal["new", "resume"] = "new"
    dataset_paths: list[str] = Field(min_length=1, max_length=128)
    groot_checkpoint: str = ""
    # Optional compatibility guard for resume callers.  The bundle remains
    # the authoritative source, but it must name this currently selected
    # frozen GR00T checkpoint before a new immutable round may start.
    expected_groot_checkpoint: str = ""
    rl_token_encoder_path: str = ""
    rlt_bundle_path: str = ""
    steps: StrictInt = Field(default=10_000, ge=1, le=10_000_000)
    batch_size: StrictInt = Field(default=64, ge=1, le=4096)
    save_freq: StrictInt = Field(default=1_000, ge=1, le=10_000_000)


class RLTStage2StopRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


class RLTStage2Status(BaseModel):
    ready: bool = True
    status: Literal["idle", "running", "completed", "failed", "stopped"]
    phase: str = "idle"
    percentage: float = 0.0
    job_id: str = ""
    initialization_mode: Literal["new", "resume"] = "new"
    dataset_paths: list[str] = Field(default_factory=list)
    resolved_dataset_paths: list[str] = Field(default_factory=list)
    groot_checkpoint: str = ""
    rl_token_encoder_path: str = ""
    rlt_bundle_path: str = ""
    output_dir: str = ""
    actor_artifact_path: str = ""
    encoder_artifact_path: str = ""
    checkpoint_path: str = ""
    manifest_path: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    batch_size: int = 0
    save_freq: int = 0
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    average_reward: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = ""
    returncode: Optional[int] = None
    log_tail: list[str] = Field(default_factory=list)


@dataclass
class _RLTStage2Job:
    job_id: str
    initialization_mode: str
    dataset_paths: list[str]
    resolved_dataset_paths: list[str]
    groot_checkpoint: str
    rl_token_encoder_path: str
    rlt_bundle_path: str
    output_dir: str
    log_path: str
    total_steps: int
    batch_size: int
    save_freq: int
    status: str = "running"
    phase: str = "starting"
    percentage: float = 0.0
    actor_artifact_path: str = ""
    encoder_artifact_path: str = ""
    checkpoint_path: str = ""
    manifest_path: str = ""
    completed_steps: int = 0
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    average_reward: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = "Starting GR00T RLT Stage 2"
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    stop_confirmed: bool = False
    returncode: Optional[int] = None
    log_tail: list[str] = field(default_factory=list)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    return _rlt_common.read_json_object(
        path,
        label=label,
        max_bytes=RLT_STAGE2_MAX_JSON_BYTES,
    )


def _digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HTTPException(400, f"{label} must be a lowercase SHA-256 digest")
    return value


def _dataset_version(dataset: Path) -> str:
    return _rlt_common.lerobot_dataset_version(
        dataset,
        max_json_bytes=RLT_STAGE2_MAX_JSON_BYTES,
        unsupported_message=f"Unsupported LeRobot codebase_version: {dataset}",
    )


def _validate_v21_dataset(dataset: Path, *, require_outcomes: bool = True) -> Path:
    return _rlt_common.validate_lerobot_dataset(
        dataset,
        expected_version="v2.1",
        max_json_bytes=RLT_STAGE2_MAX_JSON_BYTES,
        unsupported_message=f"Unsupported LeRobot codebase_version: {dataset}",
        wrong_version_message=f"GR00T RLT Stage 2 requires LeRobot v2.1: {dataset}",
        unsafe_metadata_message=f"LeRobot v2.1 metadata is incomplete: {dataset}",
        outcome_requirement="present" if require_outcomes else "none",
        missing_outcome_message=(
            f"RLT Stage 2 dataset is missing episode_success labels: {dataset}"
        ),
    )


def _validate_v30_dataset(dataset: Path, *, require_outcomes: bool = True) -> Path:
    return _rlt_common.validate_lerobot_dataset(
        dataset,
        expected_version="v3.0",
        max_json_bytes=RLT_STAGE2_MAX_JSON_BYTES,
        unsupported_message=f"Unsupported LeRobot codebase_version: {dataset}",
        wrong_version_message=f"GR00T RLT Stage 2 requires LeRobot v3.0: {dataset}",
        unsafe_metadata_message=(
            f"LeRobot v3.0 dataset is missing safe episode/task metadata: {dataset}"
        ),
        outcome_requirement="boolean" if require_outcomes else "none",
        missing_outcome_message=(
            f"RLT Stage 2 dataset is missing episode_success labels: {dataset}"
        ),
    )


def _resolve_datasets(raw_paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        dataset = _resolve_existing_path(
            raw_path,
            roots=RLT_STAGE2_DATASET_ROOTS,
            label="dataset_path",
        )
        dataset = (
            _validate_v21_dataset(dataset)
            if _dataset_version(dataset) == "v2.1"
            else _validate_v30_dataset(dataset)
        )
        if dataset in seen:
            raise HTTPException(400, "dataset_paths resolve to duplicate datasets")
        seen.add(dataset)
        resolved.append(dataset)
    return resolved


def _resolve_groot_checkpoint(raw_path: str) -> Path:
    checkpoint = _resolve_existing_path(
        raw_path,
        roots=RLT_STAGE2_GROOT_ROOTS,
        label="groot_checkpoint",
    )
    config = _read_json(checkpoint / "config.json", label="GR00T config")
    architectures = config.get("architectures")
    model_type = str(config.get("model_type") or "").lower()
    is_groot = "gr00t" in model_type or "groot" in model_type
    if isinstance(architectures, list):
        is_groot = is_groot or any(
            "gr00t" in str(value).lower() or "groot" in str(value).lower()
            for value in architectures
        )
    if not is_groot:
        raise HTTPException(400, f"Checkpoint is not a GR00T model: {checkpoint}")
    weights = list(checkpoint.glob("*.safetensors"))
    if not weights or any(path.is_symlink() or not path.is_file() for path in weights):
        raise HTTPException(400, f"GR00T checkpoint has no safe weights: {checkpoint}")
    for path in (checkpoint / "config.json", *weights):
        mode = path.stat().st_mode
        if not (
            mode & stat.S_IROTH
            or mode & stat.S_IRGRP
            or path.stat().st_uid == RLT_STAGE2_TRAIN_UID
        ):
            raise HTTPException(400, f"GR00T artifact is not readable: {path}")
    return checkpoint


def _resolve_stage1_encoder(raw_path: str, groot_checkpoint: Path) -> Path:
    encoder = _resolve_existing_path(
        raw_path,
        roots=RLT_STAGE2_ENCODER_ROOTS,
        label="rl_token_encoder_path",
        directory=False,
    )
    if encoder.name != "rl_token_encoder.pt" or encoder.parent.name != "artifacts":
        raise HTTPException(400, "RL Token source must be a Stage 1 encoder artifact")
    run_manifest_path = (
        encoder.parent.parent / "training_state" / "rlt_stage1.pt.run.json"
    )
    manifest = _read_json(run_manifest_path, label="RLT Stage 1 run manifest")
    if manifest.get("format") != _STAGE1_RUN_FORMAT or manifest.get("status") != "completed":
        raise HTTPException(400, "RL Token encoder is not from a completed Stage 1 run")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise HTTPException(400, "RLT Stage 1 artifact provenance is missing")
    recorded_path = artifact.get("path")
    if not isinstance(recorded_path, str) or Path(os.path.abspath(recorded_path)) != encoder:
        raise HTTPException(400, "RLT Stage 1 artifact path disagrees with its manifest")
    _digest(artifact.get("artifact_fingerprint"), label="RL Token fingerprint")
    recorded_groot = manifest.get("groot_checkpoint")
    if not isinstance(recorded_groot, str) or Path(os.path.abspath(recorded_groot)) != groot_checkpoint:
        raise HTTPException(
            400,
            "RL Token encoder was trained from a different GR00T checkpoint",
        )
    _digest(manifest.get("policy_weight_fingerprint"), label="GR00T fingerprint")
    checkpoint = encoder.parent.parent / "training_state" / "rlt_stage1.pt"
    if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise HTTPException(400, "RLT Stage 1 training checkpoint is missing")
    return encoder


def _validate_bundle_manifest(bundle: Path, manifest: Mapping[str, Any]) -> None:
    def require_integer(value: object, *, label: str, allow_zero: bool = False) -> int:
        lower = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            qualifier = "non-negative" if allow_zero else "positive"
            raise HTTPException(400, f"RLT Stage 2 {label} must be a {qualifier} integer")
        return value

    def validate_replay_record(
        value: object,
        *,
        expected_relative_path: str,
        label: str,
    ) -> Path:
        if not isinstance(value, Mapping) or set(value) != {
            "relative_path",
            "byte_count",
            "sha256",
        }:
            raise HTTPException(400, f"RLT Stage 2 training round {label} record is invalid")
        if value.get("relative_path") != expected_relative_path:
            raise HTTPException(400, f"RLT Stage 2 training round {label} path is invalid")
        expected_bytes = require_integer(value.get("byte_count"), label=f"{label} byte_count")
        expected_digest = _digest(value.get("sha256"), label=f"{label} SHA-256")
        path = bundle / "replay_cache" / expected_relative_path
        if (
            _has_symlink_component(bundle, path)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_bytes
        ):
            raise HTTPException(400, f"RLT Stage 2 training round {label} artifact is invalid")
        if _file_sha256(path) != expected_digest:
            raise HTTPException(400, f"RLT Stage 2 training round {label} digest disagrees")
        return path

    required = {
        "format",
        "initialization",
        "source",
        "spec",
        "spec_fingerprint",
        "completed_critic_updates",
        "completed_actor_updates",
        "training_round",
        "artifacts",
        "qualification",
        "manifest_fingerprint",
    }
    if set(manifest) != required or manifest.get("format") != _STAGE2_BUNDLE_FORMAT:
        raise HTTPException(400, "RLT Stage 2 bundle manifest fields are invalid")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    fingerprint = _digest(
        manifest.get("manifest_fingerprint"), label="RLT bundle fingerprint"
    )
    if _canonical_fingerprint(unsigned) != fingerprint:
        raise HTTPException(400, "RLT Stage 2 bundle manifest fingerprint disagrees")
    if manifest.get("qualification") != _STAGE2_QUALIFICATION:
        raise HTTPException(400, "RLT Stage 2 bundle qualification is invalid")

    initialization = manifest.get("initialization")
    if not isinstance(initialization, Mapping) or set(initialization) != {
        "mode",
        "parent_bundle_fingerprint",
    }:
        raise HTTPException(400, "RLT Stage 2 initialization record is invalid")
    mode = initialization.get("mode")
    parent = initialization.get("parent_bundle_fingerprint")
    if mode == "new" and parent is not None:
        raise HTTPException(400, "New RLT bundle must not name a parent bundle")
    if mode == "resume":
        _digest(parent, label="parent RLT bundle fingerprint")
    elif mode != "new":
        raise HTTPException(400, "RLT Stage 2 initialization mode is invalid")

    source = manifest.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "groot_checkpoint",
        "groot_checkpoint_fingerprint",
        "representation_contract_fingerprint",
        "rl_token_artifact_fingerprint",
    }:
        raise HTTPException(400, "RLT Stage 2 frozen source is invalid")
    if not isinstance(source.get("groot_checkpoint"), str) or not source.get(
        "groot_checkpoint"
    ):
        raise HTTPException(400, "RLT Stage 2 GR00T source is invalid")
    for name in (
        "groot_checkpoint_fingerprint",
        "representation_contract_fingerprint",
        "rl_token_artifact_fingerprint",
    ):
        _digest(source.get(name), label=name)

    spec = manifest.get("spec")
    if not isinstance(spec, Mapping) or set(spec) != _STAGE2_SPEC_FIELDS:
        raise HTTPException(400, "RLT Stage 2 inference spec is invalid")
    if (
        spec.get("reference_horizon") != 16
        or spec.get("chunk_length") != 10
        or spec.get("action_dim") != 19
        or spec.get("proprio_dim") != 19
        or spec.get("model_domain") != "normalized"
        or spec.get("schema_version") != 1
    ):
        raise HTTPException(400, "RLT Stage 2 bundle does not satisfy the 10x19 contract")
    if spec.get("rl_token_artifact_fingerprint") != source.get(
        "rl_token_artifact_fingerprint"
    ):
        raise HTTPException(400, "RLT Stage 2 encoder fingerprints disagree")
    spec_fingerprint = _digest(
        manifest.get("spec_fingerprint"), label="RLT spec fingerprint"
    )
    if _canonical_fingerprint(spec) != spec_fingerprint:
        raise HTTPException(400, "RLT Stage 2 spec fingerprint disagrees")
    for name in ("completed_critic_updates", "completed_actor_updates"):
        value = manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HTTPException(400, f"RLT Stage 2 {name} is invalid")

    from cyclo_brain.contracts.rlt import validate_training_round

    training_round = manifest.get("training_round")
    try:
        validate_training_round(training_round)
    except (TypeError, ValueError, OverflowError) as error:
        raise HTTPException(400, str(error)) from error
    replay = training_round["replay"]
    replay_manifest_path = validate_replay_record(
        replay.get("manifest"),
        expected_relative_path="manifest.json",
        label="replay manifest",
    )
    validate_replay_record(
        replay.get("artifact"),
        expected_relative_path="replay.pt",
        label="replay artifact",
    )
    replay_manifest_fingerprint = _digest(
        replay.get("manifest_fingerprint"),
        label="RLT replay manifest fingerprint",
    )
    replay_spec_fingerprint = _digest(
        replay.get("spec_fingerprint"),
        label="RLT replay spec fingerprint",
    )
    if replay_spec_fingerprint != spec_fingerprint:
        raise HTTPException(400, "RLT Stage 2 replay spec fingerprint disagrees")
    replay_manifest = _read_json(
        replay_manifest_path,
        label="RLT Stage 2 replay manifest",
    )
    if _canonical_fingerprint(replay_manifest) != replay_manifest_fingerprint:
        raise HTTPException(400, "RLT Stage 2 replay manifest fingerprint disagrees")

    artifacts = manifest.get("artifacts")
    relative_paths = {
        "rl_token_encoder": "artifacts/rl_token_encoder.pt",
        "rlt_actor": "artifacts/rlt_actor.pt",
        "training_state": "training_state/rlt_stage2.pt",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(relative_paths):
        raise HTTPException(400, "RLT Stage 2 artifact records are invalid")
    for name, relative in relative_paths.items():
        record = artifacts.get(name)
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "byte_count",
            "sha256",
        }:
            raise HTTPException(400, f"RLT Stage 2 {name} record is invalid")
        if record.get("relative_path") != relative:
            raise HTTPException(400, f"RLT Stage 2 {name} path is invalid")
        path = bundle / relative
        if (
            _has_symlink_component(bundle, path)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size <= 0
        ):
            raise HTTPException(400, f"RLT resume bundle is incomplete: {path}")
        byte_count = record.get("byte_count")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or (
            byte_count != path.stat().st_size
        ):
            raise HTTPException(400, f"RLT Stage 2 {name} byte count disagrees")
        expected_digest = _digest(record.get("sha256"), label=f"{name} digest")
        if _file_sha256(path) != expected_digest:
            raise HTTPException(400, f"RLT Stage 2 {name} digest disagrees")


def _resolve_resume_bundle(raw_path: str) -> Path:
    bundle = _resolve_existing_path(
        raw_path,
        roots=RLT_STAGE2_BUNDLE_ROOTS,
        label="rlt_bundle_path",
    )
    manifest = _read_json(bundle / "manifest.json", label="RLT Stage 2 manifest")
    _validate_bundle_manifest(bundle, manifest)
    return bundle


def _output_path(job_id: str, steps: int) -> Path:
    root = RLT_STAGE2_OUTPUT_ROOT
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(500, "RLT Stage 2 output root is unavailable") from exc
    if root.is_symlink() or _has_symlink_component(root, root):
        raise HTTPException(500, "RLT Stage 2 output root must not be a symlink")
    output = root / f"steps_{steps:07d}_{job_id[:12]}"
    if output.exists() or output.is_symlink():
        raise HTTPException(409, f"RLT Stage 2 output already exists: {output}")
    try:
        output.resolve(strict=False).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "RLT Stage 2 output escapes its checkpoint root") from exc
    return output


def _prepare_output(output: Path) -> None:
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o775)
        os.chown(output, RLT_STAGE2_TRAIN_UID, RLT_STAGE2_TRAIN_GID)
    except (OSError, PermissionError) as exc:
        raise HTTPException(500, f"Could not create RLT Stage 2 output: {output}") from exc


def _completed_stage2_status(bundle: Path) -> RLTStage2Status:
    """Strictly validate a published bundle and reconstruct terminal status."""

    resolved_bundle = _resolve_resume_bundle(str(bundle))
    root = RLT_STAGE2_OUTPUT_ROOT.resolve(strict=True)
    relative = resolved_bundle.relative_to(root)
    match = _STAGE2_OUTPUT_PATTERN.fullmatch(resolved_bundle.name)
    if len(relative.parts) != 1 or match is None:
        raise HTTPException(400, "RLT Stage 2 output name is invalid")
    manifest_path = resolved_bundle / "manifest.json"
    manifest = _read_json(manifest_path, label="RLT Stage 2 manifest")
    training_round = manifest["training_round"]
    optimization = training_round["optimization"]
    steps = optimization["steps"]
    if steps != int(match.group(1)):
        raise HTTPException(400, "RLT Stage 2 output steps disagree with its manifest")

    # Dataset paths are status metadata only.  Restore them when the immutable
    # replay manifest still names safe, existing datasets; a self-contained
    # bundle remains recoverable even after its source replay was archived.
    dataset_paths: list[str] = []
    replay_manifest = _read_json(
        resolved_bundle / "replay_cache" / "manifest.json",
        label="RLT Stage 2 replay manifest",
    )
    replay_metadata = replay_manifest.get("metadata")
    recorded_datasets = (
        replay_metadata.get("dataset_roots")
        if isinstance(replay_metadata, Mapping)
        else None
    )
    if (
        isinstance(recorded_datasets, list)
        and recorded_datasets
        and all(isinstance(value, str) for value in recorded_datasets)
    ):
        try:
            dataset_paths = [
                str(path) for path in _resolve_datasets(list(recorded_datasets))
            ]
        except HTTPException:
            dataset_paths = []

    source = manifest["source"]
    artifacts = manifest["artifacts"]
    actor_path = resolved_bundle / artifacts["rlt_actor"]["relative_path"]
    encoder_path = resolved_bundle / artifacts["rl_token_encoder"]["relative_path"]
    checkpoint_path = resolved_bundle / artifacts["training_state"]["relative_path"]
    return RLTStage2Status(
        ready=True,
        status="completed",
        phase="complete",
        percentage=100.0,
        initialization_mode=manifest["initialization"]["mode"],
        dataset_paths=list(dataset_paths),
        resolved_dataset_paths=list(dataset_paths),
        groot_checkpoint=source["groot_checkpoint"],
        rl_token_encoder_path=str(encoder_path),
        output_dir=str(resolved_bundle),
        actor_artifact_path=str(actor_path),
        encoder_artifact_path=str(encoder_path),
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(manifest_path),
        completed_steps=steps,
        total_steps=steps,
        batch_size=optimization["batch_size"],
        average_reward=training_round["replay"]["average_reward"],
        eta_seconds=0.0,
        message="Recovered completed GR00T RLT Stage 2 bundle",
        returncode=0,
    )


def _discover_latest_completed_stage2() -> RLTStage2Status | None:
    """Return the newest fully verified direct bundle, ignoring partial runs."""

    root = RLT_STAGE2_OUTPUT_ROOT
    if root.is_symlink() or not root.is_dir():
        return None
    try:
        candidates = []
        for child in root.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if _STAGE2_OUTPUT_PATTERN.fullmatch(child.name) is None:
                continue
            manifest = child / "manifest.json"
            if manifest.is_symlink() or not manifest.is_file():
                continue
            candidates.append((manifest.stat().st_mtime_ns, child.name, child))
    except OSError:
        return None
    candidates.sort(reverse=True)
    for _mtime, _name, candidate in candidates[:RLT_STAGE2_DISCOVERY_LIMIT]:
        try:
            return _completed_stage2_status(candidate)
        except (HTTPException, OSError, KeyError, TypeError, ValueError):
            continue
    return None


def _runtime_readiness() -> tuple[bool, str]:
    """Report whether the one-shot Stage 2 runtime is actually deployable.

    The supervisor/API contract is useful before the trainer lands, but starting
    a Compose job that can only fail with ``No module named ...`` is not.  Keep
    status/configuration available and fail closed until both the raw LeRobot
    materializer and its CLI entry point are present as regular source files.
    """

    missing = [
        path.name
        for path in RLT_STAGE2_REQUIRED_RUNTIME_FILES
        if path.is_symlink() or not path.is_file()
    ]
    if missing:
        return (
            False,
            "GR00T RLT Stage 2 is not ready: missing " + ", ".join(missing),
        )
    return True, "GR00T RLT Stage 2 is ready"


class RLTStage2Supervisor:
    """Own exactly one non-destructive RLT Stage 2 subprocess."""

    def __init__(
        self,
        *,
        compose_command: Callable[[], list[str]],
        compose_environment: Callable[[], dict[str, str]],
        conflict_message: Callable[[], str | None] | None = None,
        interrupt_container: Callable[[str], bool] | None = None,
        readiness_check: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        self._compose_command = compose_command
        self._compose_environment = compose_environment
        self._conflict_message = conflict_message or (lambda: None)
        self._interrupt_container = interrupt_container
        self._readiness_check = readiness_check or _runtime_readiness
        self._lock = threading.Lock()
        self._job: _RLTStage2Job | None = None
        self._recovered_status = _discover_latest_completed_stage2()
        self._starting = False
        self.router = APIRouter(prefix="/rlt-stage2", tags=["rlt-stage2"])
        self.router.add_api_route(
            "/start", self.start, methods=["POST"], response_model=RLTStage2Status
        )
        self.router.add_api_route(
            "/status", self.status, methods=["GET"], response_model=RLTStage2Status
        )
        self.router.add_api_route(
            "/stop", self.stop, methods=["POST"], response_model=RLTStage2Status
        )

    @staticmethod
    def _container_name(job: _RLTStage2Job) -> str:
        return f"cyclo_rlt_stage2_{job.job_id[:12]}"

    def _command(self, job: _RLTStage2Job) -> list[str]:
        command = self._compose_command() + [
            "run", "--rm", "--no-deps", "--pull", "never",
            "--name", self._container_name(job),
            "--user", f"{RLT_STAGE2_TRAIN_UID}:{RLT_STAGE2_TRAIN_GID}",
            "--workdir", "/app",
            "--env", "HOME=/tmp",
            "--env", f"XDG_CACHE_HOME={RLT_STAGE2_CACHE_ROOT}",
            "--env", f"HF_HOME={RLT_STAGE2_CACHE_ROOT}/huggingface",
            "--env", f"HF_HUB_CACHE={RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"HUGGINGFACE_HUB_CACHE={RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"TRANSFORMERS_CACHE={RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"TORCH_HOME={RLT_STAGE2_CACHE_ROOT}/torch",
            "--env", f"TRITON_CACHE_DIR={RLT_STAGE2_CACHE_ROOT}/triton",
            "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1",
            "--env", "HF_DATASETS_OFFLINE=1",
            "--env", "GROOT_HF_LOCAL_FIRST=1",
            "--env", "GROOT_PATCH_MISTRAL=1",
            "--env", "NO_ALBUMENTATIONS_UPDATE=1",
            "--env", "PYTHONPATH=/cyclo_brain_src:/app:/gr00t",
            "--entrypoint", "python", "groot",
            "-m", "runtime.rlt_stage2_training_cli",
            "--initialization-mode", job.initialization_mode,
        ]
        for dataset_path in job.resolved_dataset_paths:
            command.extend(("--dataset-root", dataset_path))
        if job.initialization_mode == "new":
            command.extend((
                "--groot-checkpoint", job.groot_checkpoint,
                "--rl-token-encoder", job.rl_token_encoder_path,
            ))
        else:
            command.extend(("--rlt-bundle", job.rlt_bundle_path))
        command.extend((
            "--output-dir", job.output_dir,
            "--job-id", job.job_id,
            "--steps", str(job.total_steps),
            "--batch-size", str(job.batch_size),
            "--save-freq", str(job.save_freq),
            "--progress-interval", "10",
            "--device", "cuda",
        ))
        return command

    def _status(self, job: _RLTStage2Job | None) -> RLTStage2Status:
        ready, readiness_message = self._readiness_check()
        if job is None:
            if self._recovered_status is not None:
                recovered = self._recovered_status.model_copy(deep=True)
                recovered.ready = ready
                if not ready:
                    recovered.message = (
                        f"{recovered.message}; {readiness_message}"
                    )
                return recovered
            return RLTStage2Status(
                ready=ready,
                status="idle",
                message=readiness_message,
            )
        completed = job.status == "completed"
        return RLTStage2Status(
            ready=ready,
            status=job.status,
            phase=job.phase,
            percentage=job.percentage,
            job_id=job.job_id,
            initialization_mode=job.initialization_mode,
            dataset_paths=list(job.dataset_paths),
            resolved_dataset_paths=list(job.resolved_dataset_paths),
            groot_checkpoint=job.groot_checkpoint,
            rl_token_encoder_path=job.rl_token_encoder_path,
            rlt_bundle_path=job.rlt_bundle_path,
            output_dir=job.output_dir,
            actor_artifact_path=job.actor_artifact_path if completed else "",
            encoder_artifact_path=job.encoder_artifact_path if completed else "",
            checkpoint_path=job.checkpoint_path if completed else "",
            manifest_path=job.manifest_path if completed else "",
            completed_steps=job.completed_steps,
            total_steps=job.total_steps,
            batch_size=job.batch_size,
            save_freq=job.save_freq,
            actor_loss=job.actor_loss,
            critic_loss=job.critic_loss,
            average_reward=job.average_reward,
            eta_seconds=job.eta_seconds,
            message=job.message,
            returncode=job.returncode,
            log_tail=list(job.log_tail),
        )

    @staticmethod
    def _consume_number(job: _RLTStage2Job, payload: Mapping[str, Any], name: str) -> None:
        value = payload.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(job, name, float(value))

    def _consume_event(self, job: _RLTStage2Job, payload: Mapping[str, Any]) -> bool:
        event = str(payload.get("event") or "")
        if event in {"starting", "manifest", "ready", "stage2_training_manifest"}:
            job.phase = str(payload.get("phase") or "preparing_replay")
            job.message = str(payload.get("message") or "Preparing RLT replay features")
            return False
        if event in {"progress", "stage2_training_progress"}:
            job.phase = str(payload.get("phase") or "training_actor_critic")
            step = payload.get("completed_steps", payload.get("completed_critic_updates", payload.get("step")))
            if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
                job.completed_steps = step
            total = payload.get("total_steps", payload.get("total_critic_updates"))
            if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                job.total_steps = total
            for name in ("actor_loss", "critic_loss", "average_reward", "eta_seconds"):
                self._consume_number(job, payload, name)
            percentage = payload.get("percentage")
            if isinstance(percentage, (int, float)) and not isinstance(percentage, bool):
                job.percentage = max(0.0, min(99.0, float(percentage)))
            elif job.total_steps:
                job.percentage = min(99.0, 100.0 * job.completed_steps / job.total_steps)
            job.message = str(payload.get("message") or "Training RLT Action MLP and critics")
            return False
        complete = event in {"completed", "result", "stage2_training_result"} and str(
            payload.get("status") or "completed"
        ) in {"complete", "completed"}
        if complete:
            aliases = {
                "actor_artifact_path": ("actor_artifact_path", "actor_artifact"),
                "encoder_artifact_path": ("encoder_artifact_path", "encoder_artifact"),
                "checkpoint_path": ("checkpoint_path", "checkpoint"),
                "manifest_path": ("manifest_path", "bundle_manifest"),
            }
            for attribute, names in aliases.items():
                value = next((payload.get(name) for name in names if isinstance(payload.get(name), str)), None)
                if isinstance(value, str):
                    setattr(job, attribute, value)
            step = payload.get("completed_steps", payload.get("completed_critic_updates"))
            if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
                job.completed_steps = step
            for name in ("actor_loss", "critic_loss", "average_reward"):
                self._consume_number(job, payload, name)
            job.phase = "verifying"
            job.message = "Verifying the self-contained RLT Stage 2 bundle"
            return all((job.actor_artifact_path, job.encoder_artifact_path, job.checkpoint_path, job.manifest_path))
        if event in {"stopped", "cancelled"} or (
            event == "result" and payload.get("status") == "stopped"
        ):
            job.stop_confirmed = True
            job.phase = "stopping"
            job.message = str(payload.get("message") or "RLT Stage 2 stopped")
        elif event in {"failed", "error"} or (
            event in {"result", "stage2_training_result"}
            and str(payload.get("status") or "").lower() in {"failed", "error"}
        ):
            job.phase = "error"
            job.message = str(payload.get("message") or payload.get("error") or "RLT Stage 2 failed")
        return False

    @staticmethod
    def _verified_artifacts(job: _RLTStage2Job) -> bool:
        output = Path(job.output_dir)
        expected = {
            "actor_artifact_path": output / "artifacts" / "rlt_actor.pt",
            "encoder_artifact_path": output / "artifacts" / "rl_token_encoder.pt",
            "checkpoint_path": output / "training_state" / "rlt_stage2.pt",
            "manifest_path": output / "manifest.json",
        }
        for attribute, path in expected.items():
            if os.path.normpath(getattr(job, attribute)) != str(path):
                return False
            if (
                _has_symlink_component(output, path)
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size <= 0
            ):
                return False
        try:
            manifest = _read_json(expected["manifest_path"], label="RLT Stage 2 manifest")
        except HTTPException:
            return False
        try:
            _validate_bundle_manifest(output, manifest)
        except HTTPException:
            return False
        return True

    @staticmethod
    def _append_log(job: _RLTStage2Job, line: str) -> None:
        if line:
            job.log_tail.append(line)
            del job.log_tail[:-RLT_STAGE2_LOG_LINES]

    def _monitor(self, job: _RLTStage2Job) -> None:
        result_complete = False
        try:
            Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(job.log_path, "a", encoding="utf-8") as log:
                stdout = job.process.stdout if job.process is not None else None
                if stdout is not None:
                    for raw_line in stdout:
                        line = (raw_line.decode(errors="replace") if isinstance(raw_line, bytes) else raw_line).rstrip("\r\n")
                        log.write(line + "\n")
                        log.flush()
                        with self._lock:
                            self._append_log(job, line)
                            try:
                                payload = json.loads(line)
                            except (TypeError, json.JSONDecodeError):
                                continue
                            if isinstance(payload, Mapping):
                                result_complete = self._consume_event(job, payload) or result_complete
                returncode = job.process.wait() if job.process is not None else -1
        except Exception as exc:  # pragma: no cover - worker boundary
            returncode = -1
            with self._lock:
                job.message = f"RLT Stage 2 monitor failed: {exc}"
        with self._lock:
            job.returncode = returncode
            expected_stop_codes = {0, -signal.SIGINT, 128 + signal.SIGINT}
            if (job.stop_requested and returncode in expected_stop_codes) or (
                returncode == 0 and job.stop_confirmed
            ):
                job.status = "stopped"
                job.phase = "stopped"
                job.actor_artifact_path = ""
                job.encoder_artifact_path = ""
                job.checkpoint_path = ""
                job.manifest_path = ""
                job.eta_seconds = None
                job.message = "GR00T RLT Stage 2 stopped"
            elif returncode == 0 and result_complete and self._verified_artifacts(job):
                job.status = "completed"
                job.phase = "complete"
                job.percentage = 100.0
                job.completed_steps = job.total_steps
                job.eta_seconds = 0.0
                job.message = "GR00T RLT Stage 2 completed"
            else:
                job.status = "failed"
                job.phase = "error"
                if not job.message or job.message.startswith("Starting GR00T"):
                    reason = f"exited with code {returncode}" if returncode else "did not publish a verified bundle"
                    job.message = f"GR00T RLT Stage 2 {reason}"

    def is_running(self) -> bool:
        with self._lock:
            return self._starting or (self._job is not None and self._job.status == "running")

    def start(self, request: RLTStage2StartRequest) -> RLTStage2Status:
        with self._lock:
            if self._starting or (self._job is not None and self._job.status == "running"):
                raise HTTPException(409, "An RLT Stage 2 job is already running")
            conflict = self._conflict_message()
            if conflict:
                raise HTTPException(409, conflict)
            ready, readiness_message = self._readiness_check()
            if not ready:
                raise HTTPException(503, readiness_message)
            self._starting = True
        try:
            datasets = _resolve_datasets(request.dataset_paths)
            groot_checkpoint = ""
            encoder_path = ""
            bundle_path = ""
            if request.initialization_mode == "new":
                if request.rlt_bundle_path.strip():
                    raise HTTPException(400, "New RLT initialization must not include rlt_bundle_path")
                checkpoint = _resolve_groot_checkpoint(request.groot_checkpoint)
                encoder = _resolve_stage1_encoder(request.rl_token_encoder_path, checkpoint)
                groot_checkpoint = str(checkpoint)
                encoder_path = str(encoder)
            else:
                if request.groot_checkpoint.strip() or request.rl_token_encoder_path.strip():
                    raise HTTPException(400, "Resume RLT initialization accepts only rlt_bundle_path")
                bundle = _resolve_resume_bundle(request.rlt_bundle_path)
                bundle_path = str(bundle)
                # Preserve the immutable base-policy identity in the job
                # status. The UI uses this value to prevent a completed RLT
                # bundle from being rebound to another GR00T checkpoint.
                manifest = _read_json(
                    bundle / "manifest.json",
                    label="RLT Stage 2 manifest",
                )
                groot_checkpoint = str(manifest["source"]["groot_checkpoint"])
                if request.expected_groot_checkpoint.strip():
                    expected_checkpoint = _resolve_groot_checkpoint(
                        request.expected_groot_checkpoint
                    )
                    recorded_checkpoint = Path(os.path.abspath(groot_checkpoint))
                    if recorded_checkpoint != expected_checkpoint:
                        raise HTTPException(
                            409,
                            "RLT bundle base GR00T checkpoint does not match "
                            "the currently selected GR00T checkpoint",
                        )
            job_id = uuid.uuid4().hex
            output = _output_path(job_id, request.steps)
            if bundle_path and Path(bundle_path) == output:
                raise HTTPException(400, "RLT resume output must differ from its input bundle")
            _prepare_output(output)
            RLT_STAGE2_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        except Exception:
            with self._lock:
                self._starting = False
            raise
        job = _RLTStage2Job(
            job_id=job_id,
            initialization_mode=request.initialization_mode,
            dataset_paths=list(request.dataset_paths),
            resolved_dataset_paths=[str(path) for path in datasets],
            groot_checkpoint=groot_checkpoint,
            rl_token_encoder_path=encoder_path,
            rlt_bundle_path=bundle_path,
            output_dir=str(output),
            log_path=str(RLT_STAGE2_LOG_ROOT / f"{job_id}.log"),
            total_steps=request.steps,
            batch_size=request.batch_size,
            save_freq=request.save_freq,
        )
        try:
            job.process = subprocess.Popen(
                self._command(job),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self._compose_environment(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            job.status = "failed"
            job.phase = "error"
            job.message = f"Could not start GR00T RLT Stage 2: {exc}"
            job.returncode = -1
            with self._lock:
                self._job = job
                self._starting = False
            return self._status(job)
        with self._lock:
            self._job = job
            self._starting = False
        threading.Thread(
            target=self._monitor,
            args=(job,),
            name=f"rlt-stage2-{job_id[:12]}",
            daemon=True,
        ).start()
        return self._status(job)

    def status(self) -> RLTStage2Status:
        with self._lock:
            return self._status(self._job)

    def stop(self, request: RLTStage2StopRequest) -> RLTStage2Status:
        with self._lock:
            job = self._job
            if job is None or request.job_id != job.job_id:
                raise HTTPException(409, "RLT Stage 2 job_id does not match the active job")
            if job.status != "running":
                raise HTTPException(409, f"RLT Stage 2 job is already {job.status}")
            job.stop_requested = True
            job.phase = "stopping"
            job.message = "Stopping GR00T RLT Stage 2"
        signalled = False
        try:
            if self._interrupt_container is not None:
                signalled = self._interrupt_container(self._container_name(job))
            if not signalled and job.process is not None and job.process.poll() is None:
                job.process.send_signal(signal.SIGINT)
                signalled = True
        except (OSError, ProcessLookupError, RuntimeError) as exc:
            with self._lock:
                job.stop_requested = False
                job.phase = "error"
                job.message = f"Could not stop GR00T RLT Stage 2: {exc}"
            raise HTTPException(502, job.message) from exc
        if not signalled:
            with self._lock:
                job.stop_requested = False
            raise HTTPException(409, "RLT Stage 2 exited before it could be stopped")
        with self._lock:
            return self._status(job)


def create_rlt_stage2_router(
    *,
    compose_command: Callable[[], list[str]],
    compose_environment: Callable[[], dict[str, str]],
    conflict_message: Callable[[], str | None] | None = None,
    interrupt_container: Callable[[str], bool] | None = None,
    readiness_check: Callable[[], tuple[bool, str]] | None = None,
) -> tuple[APIRouter, RLTStage2Supervisor]:
    supervisor = RLTStage2Supervisor(
        compose_command=compose_command,
        compose_environment=compose_environment,
        conflict_message=conflict_message,
        interrupt_container=interrupt_container,
        readiness_check=readiness_check,
    )
    return supervisor.router, supervisor
