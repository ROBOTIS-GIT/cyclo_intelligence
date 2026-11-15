#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Supervisor routes for GR00T RLT Stage 2 actor-critic training.

Stage 2 trains in an immutable hidden run and publishes one named bundle per
phase. ``new`` initializes the Action MLP and critics from a frozen GR00T checkpoint plus a completed
Stage-1 RL-token encoder.  ``resume`` restores all trainable state from a
completed Stage-2 bundle, but still writes the next round to a new directory.
The input datasets and bundles are consequently never modified in place.

The subprocess contract is intentionally small and stable.  The one-shot
GR00T runner is ``python -m runtime.rlt_stage2_training_cli`` and reports JSON
lines for progress and its terminal artifact paths.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import signal
import shutil
import stat
import subprocess
import threading
import uuid
import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

try:
    from supervisor_api import rlt_service_common as _rlt_common
except ModuleNotFoundError:  # Direct file loading in focused unit tests.
    from docker.supervisor_api import rlt_service_common as _rlt_common


RLT_STAGE2_DATASET_ROOTS = (Path("/workspace/lerobot"),)
# Match Stage 1 and inference: downloaded models may use owner/model folders.
RLT_STAGE2_GROOT_ROOTS = (
    Path("/workspace/model"),
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
_phase_spec = importlib.util.spec_from_file_location(
    "cyclo_f2_training_contract", RLT_STAGE2_RUNTIME_ROOT / "rlt_training_contract.py"
)
_phase_contract = importlib.util.module_from_spec(_phase_spec)
_phase_spec.loader.exec_module(_phase_contract)

RLT_STAGE2_REQUIRED_RUNTIME_FILES = (
    RLT_STAGE2_RUNTIME_ROOT / "rlt_training_contract.py",
    RLT_STAGE2_RUNTIME_ROOT / "rlt_saved_replay.py",
    RLT_STAGE2_RUNTIME_ROOT / "rlt_stage2_dataset.py",
    RLT_STAGE2_RUNTIME_ROOT / "rlt_stage2_training_cli.py",
)
RLT_STAGE2_TRAIN_UID = 1000
RLT_STAGE2_TRAIN_GID = 1000
RLT_STAGE2_LOG_LINES = 100
RLT_STAGE2_MAX_JSON_BYTES = 2 * 1024 * 1024
RLT_STAGE2_MAX_REPLAY_JSON_BYTES = 64 * 1024 * 1024
RLT_STAGE2_DISCOVERY_LIMIT = 256

_STAGE1_RUN_FORMAT = "cyclo.groot.rlt.stage1_run/v1"
_STAGE2_BUNDLE_FORMAT = "cyclo_brain.rlt.stage2_bundle/v2"
_STAGE2_QUALIFICATION = "training_only_not_deployment_validated"
_STAGE2_OUTPUT_PATTERN = re.compile(
    r"(?:data-[A-Za-z0-9][A-Za-z0-9._+~-]{0,127}__"
    r"rlt_stage2_(?:actor_warmup|critic_warmup|rl_training)_weights__)?"
    r"steps_([0-9]+)_([0-9a-f]{12})(?:_(?:KST_)?([0-9]{14}))?"
)
_STAGE2_PHASE_FOLDERS = {
    "actor_warmup": "actor", "critic_warmup": "critic", "rl": "rl_training",
}
_PUBLICATION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_STAGE2_REFERENCE_ACTION_PROFILES = frozenset({(16, 19), (32, 16)})
_STAGE2_SPEC_FIELDS_V1 = {
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
_STAGE2_SPEC_FIELDS_V2 = _STAGE2_SPEC_FIELDS_V1 | {"reference_offset_steps"}
_EXECUTION_VERIFICATION_FORMAT = "cyclo.rlt.dataset_execution_verification/v1"

_has_symlink_component = _rlt_common.has_symlink_component
_resolve_existing_path = _rlt_common.resolve_existing_path
_file_sha256 = _rlt_common.file_sha256
_canonical_fingerprint = _rlt_common.canonical_json_sha256


class RLTStage2StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    training_phase: Literal["actor_warmup", "critic_warmup", "rl"]
    actor_warmup_data_scope: Literal["all", "success_only"] = "all"
    replay_bundle_path: str = ""
    allow_stale_replay: bool = False
    data_validation_id: str = Field(default="", max_length=32, pattern="^([0-9a-f]{32})?$")

    @model_validator(mode="after")
    def validate_phase(self):
        _phase_contract.validate_training_request(self.training_phase, self.initialization_mode)
        _phase_contract.validate_actor_warmup_data_scope(
            self.training_phase,
            self.actor_warmup_data_scope,
        )
        if self.allow_stale_replay and not (
            self.training_phase == "actor_warmup"
            and self.actor_warmup_data_scope == "success_only"
            and self.replay_bundle_path.strip()
        ):
            raise ValueError(
                "Stale replay is allowed only for success-only Actor warmup "
                "with an explicit replay bundle"
            )
        return self

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


class RLTStage2PublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_path: str = Field(min_length=1)
    name: str = Field(default="latest", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    previous_bundle_path: str = ""


class RLTStage2IterativeStartRequest(BaseModel):
    """Append verified rollout datasets to one immutable parent replay."""

    rlt_bundle_path: str = Field(min_length=1)
    new_dataset_paths: list[str] = Field(min_length=1, max_length=128)
    steps: StrictInt = Field(default=5_000, ge=1, le=10_000_000)
    batch_size: StrictInt = Field(default=16, ge=1, le=4096)
    save_freq: StrictInt = Field(default=1_000, ge=1, le=10_000_000)


class RLTStage2Status(BaseModel):
    ready: bool = True
    phase_contract_version: int = 2
    supported_training_phases: list[str] = Field(default_factory=lambda: list(_phase_contract.TRAINING_PHASES))
    training_phase: str = ""
    actor_warmup_data_scope: str = "all"
    phase_state: dict[str, Any] = Field(default_factory=dict)
    replay_bundle_path: str = ""
    replay_source: str = "saved"
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
    shadow_evaluation_path: str = ""
    shadow_structural_pass: Optional[bool] = None
    shadow_correction_abs_mean: Optional[float] = None
    shadow_correction_abs_p99: Optional[float] = None
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
    training_phase: str = ""
    actor_warmup_data_scope: str = "all"
    replay_bundle_path: str = ""
    allow_stale_replay: bool = False
    phase_state: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    phase: str = "starting"
    percentage: float = 0.0
    actor_artifact_path: str = ""
    encoder_artifact_path: str = ""
    checkpoint_path: str = ""
    manifest_path: str = ""
    shadow_evaluation_path: str = ""
    shadow_structural_pass: Optional[bool] = None
    shadow_correction_abs_mean: Optional[float] = None
    shadow_correction_abs_p99: Optional[float] = None
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


def _read_replay_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read row-level replay manifests without weakening other JSON limits."""

    return _rlt_common.read_json_object(
        path,
        label=label,
        max_bytes=RLT_STAGE2_MAX_REPLAY_JSON_BYTES,
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
    if set(manifest) not in (required, required | {"phase_state"}) or manifest.get("format") != _STAGE2_BUNDLE_FORMAT:
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
    if not isinstance(spec, Mapping) or frozenset(spec) not in {
        frozenset(_STAGE2_SPEC_FIELDS_V1),
        frozenset(_STAGE2_SPEC_FIELDS_V2),
    }:
        raise HTTPException(400, "RLT Stage 2 inference spec is invalid")
    schema_version = spec.get("schema_version")
    schema_fields_valid = (
        schema_version == 1
        and set(spec) == _STAGE2_SPEC_FIELDS_V1
    ) or (
        schema_version == 2
        and set(spec) == _STAGE2_SPEC_FIELDS_V2
        and spec.get("reference_offset_steps") == 6
    )
    if (
        spec.get("chunk_length") != 10
        or (spec.get("reference_horizon"), spec.get("action_dim"))
        not in _STAGE2_REFERENCE_ACTION_PROFILES
        or spec.get("proprio_dim") != spec.get("action_dim")
        or spec.get("model_domain") != "normalized"
        or not schema_fields_valid
    ):
        raise HTTPException(
            400,
            "RLT Stage 2 bundle does not satisfy a supported "
            "10-step 16x19 or 32x16 reference/action contract",
        )
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
    replay_manifest = _read_replay_json(
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


def _tt_dataset_replay(dataset, phase):
    hold = dataset / "rlt/tt_rtc_actor_hold_v1/replay_cache"
    if phase == "actor_warmup" and (hold / "manifest.json").is_file():
        return hold
    name = "tt_rtc_actor_v1" if phase == "actor_warmup" else "tt_rtc_transition_v1"
    return dataset / "rlt" / name / "replay_cache"


def _has_fresh_tt_replay(datasets, phase):
    return bool(datasets) and all((_tt_dataset_replay(p, phase) / "manifest.json").is_file() for p in datasets)


def _saved_dataset_replay(dataset, phase):
    candidates = [_tt_dataset_replay(dataset, phase), dataset / "rlt/tt_rtc_transition_v1/replay_cache",
                  dataset / "rlt/tt_rtc_actor_v1/replay_cache", dataset / "rlt/replay_cache"]
    return next((p for p in candidates if (p / "manifest.json").is_file()), candidates[-1])


def _resolve_resume_bundle(raw_path: str) -> Path:
    requested = Path(raw_path)
    # Old saved UI selections remain resolvable after the one-time migration.
    if (requested.parent == RLT_STAGE2_OUTPUT_ROOT
        and _STAGE2_OUTPUT_PATTERN.fullmatch(requested.name)
        and not requested.exists()):
        raw_path = str(RLT_STAGE2_OUTPUT_ROOT / ".legacy" / requested.name)
    bundle = _resolve_existing_path(
        raw_path,
        roots=RLT_STAGE2_BUNDLE_ROOTS,
        label="rlt_bundle_path",
    )
    manifest = _read_json(bundle / "manifest.json", label="RLT Stage 2 manifest")
    _validate_bundle_manifest(bundle, manifest)
    return bundle


def _validated_execution_manifest(
    dataset: Path,
    *,
    expected_actor_sha256: str,
) -> dict[str, Any]:
    path = dataset / "rlt" / "execution_verification.json"
    if path.is_symlink() or not path.is_file():
        raise HTTPException(
            400,
            f"RLT iterative dataset has no execution verification: {dataset}",
        )
    value = _read_json(path, label="RLT execution verification")
    fingerprint = value.pop("verification_fingerprint", None)
    if (
        value.get("format") != _EXECUTION_VERIFICATION_FORMAT
        or not isinstance(fingerprint, str)
        or _canonical_fingerprint(value) != fingerprint
    ):
        raise HTTPException(400, f"RLT execution verification is invalid: {dataset}")
    if value.get("execution_verified") is not True:
        raise HTTPException(400, f"RLT dataset execution is not verified: {dataset}")
    episode_count = value.get("episode_count")
    successes = value.get("success_episode_count")
    failures = value.get("failure_episode_count")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in (episode_count, successes, failures)
    ):
        raise HTTPException(
            400,
            f"RLT iterative dataset requires verified success and failure episodes: {dataset}",
        )
    actors = value.get("actor_sha256")
    if actors != [expected_actor_sha256]:
        raise HTTPException(
            400,
            "RLT iterative dataset was not collected by the selected parent actor",
        )
    action_files = value.get("action_data_files")
    if not isinstance(action_files, list) or not action_files:
        raise HTTPException(400, "RLT execution verification has no action data files")
    for record in action_files:
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "byte_count",
            "sha256",
        }:
            raise HTTPException(400, "RLT action data verification record is invalid")
        relative = Path(str(record.get("relative_path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise HTTPException(400, "RLT action data verification path is unsafe")
        action_path = dataset / relative
        expected_size = record.get("byte_count")
        expected_digest = record.get("sha256")
        if (
            action_path.is_symlink()
            or not action_path.is_file()
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or action_path.stat().st_size != expected_size
            or not isinstance(expected_digest, str)
            or _file_sha256(action_path) != expected_digest
        ):
            raise HTTPException(
                400,
                "RLT action data changed after execution verification",
            )
    return {**value, "verification_fingerprint": fingerprint}


def _iterative_dataset_paths(
    bundle: Path,
    new_dataset_paths: list[str],
) -> list[str]:
    """Return the exact parent dataset prefix plus verified new rollouts."""

    manifest = _read_json(bundle / "manifest.json", label="RLT Stage 2 manifest")
    actor_sha256 = manifest["artifacts"]["rlt_actor"]["sha256"]
    replay_manifest = _read_replay_json(
        bundle / "replay_cache" / "manifest.json",
        label="RLT Stage 2 replay manifest",
    )
    metadata = replay_manifest.get("metadata")
    parent_paths = metadata.get("dataset_roots") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(parent_paths, list)
        or not parent_paths
        or any(not isinstance(path, str) or not path for path in parent_paths)
    ):
        raise HTTPException(400, "Parent RLT bundle does not record its dataset roots")

    parent_datasets = _resolve_datasets(list(parent_paths))
    new_datasets = _resolve_datasets(new_dataset_paths)
    seen = set(parent_datasets)
    for dataset in new_datasets:
        if dataset in seen:
            raise HTTPException(400, "New iterative datasets must extend the parent replay")
        _validated_execution_manifest(
            dataset,
            expected_actor_sha256=actor_sha256,
        )
        seen.add(dataset)
    return [str(path) for path in (*parent_datasets, *new_datasets)]


def _output_path(
    job_id: str,
    steps: int,
    datasets: list[Path],
    training_phase: str,
) -> Path:
    root = RLT_STAGE2_OUTPUT_ROOT
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(500, "RLT Stage 2 output root is unavailable") from exc
    if root.is_symlink() or _has_symlink_component(root, root):
        raise HTTPException(500, "RLT Stage 2 output root must not be a symlink")
    if training_phase not in _STAGE2_PHASE_FOLDERS:
        raise HTTPException(400, "RLT Stage 2 training phase is invalid")
    # Work in an immutable, hidden run directory. Only verified results become
    # visible in the picker, so failed/restarted jobs cannot replace a good model.
    phase_root = root / _STAGE2_PHASE_FOLDERS[training_phase]
    runs = phase_root / ".runs"
    if _has_symlink_component(root, runs):
        raise HTTPException(400, "RLT output directories must not be symlinks")
    runs.mkdir(parents=True, exist_ok=True)
    output = runs / f"steps_{steps:07d}_{job_id[:12]}"
    if output.exists() or output.is_symlink():
        raise HTTPException(409, f"RLT Stage 2 output already exists: {output}")
    try:
        output.resolve(strict=False).relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "RLT Stage 2 output escapes its checkpoint root") from exc
    return output


def _phase_directory(bundle: Path) -> Path:
    manifest = _read_json(bundle / "manifest.json", label="RLT Stage 2 manifest")
    phase = (manifest.get("phase_state") or {}).get("training_phase")
    folder = _STAGE2_PHASE_FOLDERS.get(phase)
    if folder is None:
        raise HTTPException(400, "Bundle has no supported training phase")
    root = RLT_STAGE2_OUTPUT_ROOT
    destination = root / folder
    if _has_symlink_component(root, destination):
        raise HTTPException(400, "RLT phase directory must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _snapshot_stage2_bundle(bundle: Path) -> Path:
    if bundle.parent.name in {".runs", ".history"}:
        return bundle
    history = _phase_directory(bundle) / ".history"
    if _has_symlink_component(RLT_STAGE2_OUTPUT_ROOT, history):
        raise HTTPException(400, "RLT history directory must not be a symlink")
    history.mkdir(exist_ok=True)
    snapshot = history / uuid.uuid4().hex
    shutil.copytree(bundle, snapshot, symlinks=True)
    _resolve_resume_bundle(str(snapshot))
    return snapshot


def _exchange_directories(left: Path, right: Path) -> None:
    """Linux atomic directory exchange: readers never see a missing latest."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _publish_stage2_bundle(bundle: Path, name: str | None = None) -> Path:
    """Keep one named, verified selection per phase; retain hidden rollback data."""
    bundle = _resolve_resume_bundle(str(bundle))
    phase_root = _phase_directory(bundle)
    index = phase_root / ".publication.json"
    previous_name = ""
    if index.exists():
        previous_name = str(_read_json(index, label="RLT publication").get("name", ""))
        if not _PUBLICATION_NAME.fullmatch(previous_name):
            raise HTTPException(400, "Invalid saved RLT deployment name")
    name = name if name is not None else (previous_name or "latest")
    if not _PUBLICATION_NAME.fullmatch(name):
        raise HTTPException(400, "Use 1–64 letters, numbers, underscores or hyphens for the deployment name")
    destination = phase_root / name
    if _has_symlink_component(RLT_STAGE2_OUTPUT_ROOT, destination):
        raise HTTPException(400, "RLT deployment must not be a symlink")
    if destination.exists():
        existing = _resolve_resume_bundle(str(destination))
        if _phase_directory(existing) != phase_root:
            raise HTTPException(409, "Deployment name belongs to a different training phase")
    if bundle != destination:
        staging = phase_root / f".publish-{uuid.uuid4().hex}"
        try:
            shutil.copytree(bundle, staging, symlinks=True)
            _resolve_resume_bundle(str(staging))
            if destination.exists():
                _exchange_directories(staging, destination)
            else:
                staging.rename(destination)
        finally:
            # After exchange this is the previous published copy; immutable runs
            # and explicit Discard snapshots retain its original weights.
            if staging.exists():
                shutil.rmtree(staging)
    temporary_index = phase_root / f".publication-{uuid.uuid4().hex}.json"
    temporary_index.write_text(json.dumps({"name": name, "source": str(bundle)}), encoding="utf-8")
    os.replace(temporary_index, index)
    if previous_name and previous_name != name:
        previous = phase_root / previous_name
        if previous.exists():
            history = phase_root / ".history"
            if _has_symlink_component(RLT_STAGE2_OUTPUT_ROOT, history):
                raise HTTPException(400, "RLT history directory must not be a symlink")
            history.mkdir(exist_ok=True)
            previous.rename(history / uuid.uuid4().hex)
    return destination


def migrate_legacy_stage2_layout() -> dict[str, str]:
    """One-time, non-destructive migration of pre-layout run directories."""
    root = RLT_STAGE2_OUTPUT_ROOT
    if root.is_symlink() or not root.is_dir():
        raise HTTPException(400, "RLT output root is missing or unsafe")
    legacy = root / ".legacy"
    if _has_symlink_component(root, legacy):
        raise HTTPException(400, "RLT legacy directory must not be a symlink")
    candidates = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                  and _STAGE2_OUTPUT_PATTERN.fullmatch(p.name)]
    newest = {}
    for path in candidates:
        try:
            bundle = _resolve_resume_bundle(str(path))
            phase = _phase_directory(bundle).name
            mtime = (bundle / "manifest.json").stat().st_mtime_ns
            if phase not in newest or mtime > newest[phase][0]:
                newest[phase] = (mtime, bundle)
        except HTTPException:
            continue  # Partial/failed runs are archived, never offered for selection.
    for folder in _STAGE2_PHASE_FOLDERS.values():
        phase_root = root / folder
        if _has_symlink_component(root, phase_root):
            raise HTTPException(400, "RLT phase directory must not be a symlink")
        phase_root.mkdir(exist_ok=True)
        if folder in newest and not (phase_root / ".publication.json").exists():
            _publish_stage2_bundle(newest[folder][1])
    legacy.mkdir(exist_ok=True)
    moved = {}
    for path in candidates:
        archived = legacy / path.name
        if archived.exists():
            raise HTTPException(409, f"Legacy archive already exists: {archived}")
        path.rename(archived)
        moved[str(path)] = str(archived)
    return moved


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
    is_managed = (
        len(relative.parts) in (2, 3)
        and relative.parts[0] in _STAGE2_PHASE_FOLDERS.values()
        and (len(relative.parts) == 2 or relative.parts[1] in {".runs", ".history"})
    )
    if not is_managed and (len(relative.parts) != 1 or match is None):
        raise HTTPException(400, "RLT Stage 2 output name is invalid")
    manifest_path = resolved_bundle / "manifest.json"
    manifest = _read_json(manifest_path, label="RLT Stage 2 manifest")
    training_round = manifest["training_round"]
    optimization = training_round["optimization"]
    steps = optimization["steps"]
    if match is not None and steps != int(match.group(1)):
        raise HTTPException(400, "RLT Stage 2 output steps disagree with its manifest")

    # Dataset paths are status metadata only.  Restore them when the immutable
    # replay manifest still names safe, existing datasets; a self-contained
    # bundle remains recoverable even after its source replay was archived.
    dataset_paths: list[str] = []
    replay_manifest = _read_replay_json(
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
    shadow_path = resolved_bundle / "evaluation" / "shadow_evaluation.json"
    shadow_structural_pass = None
    shadow_correction_abs_mean = None
    shadow_correction_abs_p99 = None
    if shadow_path.is_file() and not shadow_path.is_symlink():
        try:
            shadow = _read_json(shadow_path, label="RLT shadow evaluation")
            if shadow.get("bundle_manifest_fingerprint") == manifest.get(
                "manifest_fingerprint"
            ):
                evaluation = shadow.get("evaluation")
                if isinstance(evaluation, Mapping):
                    passed = evaluation.get("structural_pass")
                    if isinstance(passed, bool):
                        shadow_structural_pass = passed
                    for name in (
                        "correction_abs_mean",
                        "correction_abs_p99",
                    ):
                        value = evaluation.get(name)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if name == "correction_abs_mean":
                                shadow_correction_abs_mean = float(value)
                            else:
                                shadow_correction_abs_p99 = float(value)
        except HTTPException:
            shadow_path = Path()
    return RLTStage2Status(
        ready=True,
        status="completed",
        phase="complete",
        training_phase=(manifest.get("phase_state") or {}).get("training_phase", "legacy_rl"),
        actor_warmup_data_scope=(
            (manifest.get("phase_state") or {}).get("actor_warmup_data") or {}
        ).get("scope", "all"),
        phase_state=manifest.get("phase_state") or {},
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
        shadow_evaluation_path=(
            str(shadow_path) if shadow_path.is_file() else ""
        ),
        shadow_structural_pass=shadow_structural_pass,
        shadow_correction_abs_mean=shadow_correction_abs_mean,
        shadow_correction_abs_p99=shadow_correction_abs_p99,
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
        children = list(root.iterdir())
        for folder in _STAGE2_PHASE_FOLDERS.values():
            phase_root = root / folder
            if phase_root.is_dir() and not phase_root.is_symlink():
                children.extend(p for p in phase_root.iterdir() if not p.name.startswith("."))
        for child in children:
            if child.is_symlink() or not child.is_dir():
                continue
            if child.parent == root and _STAGE2_OUTPUT_PATTERN.fullmatch(child.name) is None:
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
        self._data_validations: dict[str, dict] = {}
        self.router = APIRouter(prefix="/rlt-stage2", tags=["rlt-stage2"])
        self.router.add_api_route("/resolve", self.resolve_bundle, methods=["POST"])
        self.router.add_api_route("/publish", self.publish, methods=["POST"])
        self.router.add_api_route("/validate-data", self.validate_data, methods=["POST"])
        self.router.add_api_route("/validate-data/{validation_id}", self.data_validation_status, methods=["GET"])
        self.router.add_api_route(
            "/start", self.start, methods=["POST"], response_model=RLTStage2Status
        )
        self.router.add_api_route(
            "/start-iterative",
            self.start_iterative,
            methods=["POST"],
            response_model=RLTStage2Status,
        )
        self.router.add_api_route(
            "/status", self.status, methods=["GET"], response_model=RLTStage2Status
        )
        self.router.add_api_route(
            "/stop", self.stop, methods=["POST"], response_model=RLTStage2Status
        )

    def resolve_bundle(self, request: RLTStage2PublishRequest) -> dict:
        source = _resolve_resume_bundle(request.bundle_path)
        phase_root = _phase_directory(source)
        index = phase_root / ".publication.json"
        if index.is_file():
            name = _read_json(index, label="RLT publication").get("name", "")
            if _PUBLICATION_NAME.fullmatch(name):
                published = _resolve_resume_bundle(str(phase_root / name))
                if (_read_json(source / "manifest.json", label="RLT bundle")["manifest_fingerprint"]
                    == _read_json(published / "manifest.json", label="RLT bundle")["manifest_fingerprint"]):
                    return {"bundle_path": str(published)}
        return {"bundle_path": str(source)}

    def publish(self, request: RLTStage2PublishRequest) -> dict:
        with self._lock:
            if self._starting or (self._job is not None and self._job.status == "running"):
                raise HTTPException(409, "Wait for RLT training to finish before deploying")
            conflict = self._conflict_message()
            if conflict:
                raise HTTPException(409, conflict)
            source = _resolve_resume_bundle(request.bundle_path)
            previous = request.previous_bundle_path
            # Preserve Discard even when the same visible name is replaced.
            if previous:
                try:
                    previous_path = _resolve_resume_bundle(previous)
                    previous = str(_snapshot_stage2_bundle(previous_path))
                except HTTPException:
                    pass  # A stale manual selection must not block a valid deployment.
            destination = _publish_stage2_bundle(source, request.name)
            if self._job is None:
                self._recovered_status = _completed_stage2_status(destination)
            return {"bundle_path": str(destination), "previous_bundle_path": previous}

    @staticmethod
    def _container_name(job: _RLTStage2Job) -> str:
        return f"cyclo_rlt_stage2_{job.job_id[:12]}"

    @staticmethod
    def _validation_key(request: RLTStage2StartRequest) -> dict:
        return request.model_dump(exclude={"steps", "batch_size", "save_freq", "data_validation_id"})

    def data_validation_status(self, validation_id: str) -> dict:
        with self._lock:
            record = self._data_validations.get(validation_id)
            if record is None:
                raise HTTPException(404, "Data validation expired; validate the selected data again")
            return dict(record["report"])

    def validate_data(self, request: RLTStage2StartRequest) -> dict:
        datasets = _resolve_datasets(request.dataset_paths)
        expected_spec = None
        if request.initialization_mode == "new":
            checkpoint = _resolve_groot_checkpoint(request.groot_checkpoint)
            encoder = _resolve_stage1_encoder(request.rl_token_encoder_path, checkpoint)
        else:
            bundle = _resolve_resume_bundle(request.rlt_bundle_path)
            manifest = _read_json(bundle / "manifest.json", label="RLT bundle")
            checkpoint = _resolve_groot_checkpoint(manifest["source"]["groot_checkpoint"])
            if request.expected_groot_checkpoint and str(checkpoint) != str(_resolve_groot_checkpoint(request.expected_groot_checkpoint)):
                raise HTTPException(409, "RLT bundle and selected GR00T checkpoint differ")
            encoder = bundle / "artifacts/rl_token_encoder.pt"
            expected_spec = manifest["spec"]
            if expected_spec.get("schema_version") != 1:
                raise HTTPException(400, "Select an F2 10-action schema 1 bundle for data validation")
        explicit = request.replay_bundle_path.strip()
        candidates = []
        if explicit:
            source = _resolve_resume_bundle(explicit)
            if _read_json(source / "manifest.json", label="replay bundle")["spec"].get("schema_version") != 1:
                raise HTTPException(400, "Select an F2 10-action schema 1 replay bundle")
            candidates.append(str(source))
        else:
            if all((_saved_dataset_replay(p, request.training_phase) / "manifest.json").is_file() for p in datasets):
                candidates.append("")
            if request.rlt_bundle_path:
                candidates.append(str(_resolve_resume_bundle(request.rlt_bundle_path)))
            # Discover by identity, never by the latest training timestamp.
            for root in RLT_STAGE2_BUNDLE_ROOTS:
                for candidate in sorted(root.glob("*/replay_cache/manifest.json"))[:RLT_STAGE2_DISCOVERY_LIMIT]:
                    try:
                        metadata = _read_replay_json(candidate, label="saved replay")["metadata"]
                        recorded = metadata.get("dataset_roots", [])
                        if recorded and recorded == [str(p) for p in datasets[:len(recorded)]]:
                            resolved = str(_resolve_resume_bundle(str(candidate.parent.parent)))
                            bundle_spec = _read_json(Path(resolved) / "manifest.json", label="replay bundle")["spec"]
                            if bundle_spec.get("schema_version") != 1:
                                continue
                            if resolved not in candidates:
                                candidates.append(resolved)
                    except (HTTPException, KeyError, TypeError, OSError):
                        continue
        with self._lock:
            if any(r["report"]["status"] == "running" for r in self._data_validations.values()):
                raise HTTPException(409, "Dataset validation is already running")
            validation_id = uuid.uuid4().hex
            output = RLT_STAGE2_OUTPUT_ROOT.parent / "data_validation" / validation_id
            if _has_symlink_component(Path("/"), output):
                raise HTTPException(400, "Unsafe validation output")
            _prepare_output(output)
            report = {"validation_id": validation_id, "status": "running", "message": "저장 입력·모델·실행 구간을 검증 중입니다."}
            self._data_validations[validation_id] = {"report": report, "key": self._validation_key(request), "output": output}
        payload = {"dataset_paths": [str(p) for p in datasets], "checkpoint": str(checkpoint),
                   "encoder": str(encoder), "expected_spec": expected_spec, "candidates": candidates,
                   "training_phase": request.training_phase}
        threading.Thread(target=self._run_data_validation, args=(validation_id, payload), daemon=True).start()
        return dict(report)

    def _run_data_validation(self, validation_id: str, payload: dict) -> None:
        record = self._data_validations[validation_id]
        output = record["output"]
        command = self._compose_command() + [
            "run", "--rm", "--no-deps", "--pull", "never", "--name", f"cyclo_rlt_validate_{validation_id[:12]}",
            "--user", f"{RLT_STAGE2_TRAIN_UID}:{RLT_STAGE2_TRAIN_GID}", "--workdir", "/app",
            "--env", "PYTHONPATH=/cyclo_brain_src:/app:/gr00t", "--env", "CUDA_VISIBLE_DEVICES=",
            "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--env", "OMP_NUM_THREADS=2",
            "--entrypoint", "python", "groot", "-m", "runtime.rlt_data_validation",
            "--request", json.dumps(payload), "--output", str(output),
        ]
        try:
            with (output / "validation.log").open("w") as log:
                process = subprocess.run(command, env=self._compose_environment(), stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
            if process.returncode != 0:
                raise ValueError(f"Dataset validation process failed ({process.returncode}); see {output / 'validation.log'}")
            report = _read_json(output / "report.json", label="data validation")
            if report.get("status") == "ready" and report.get("replay_path") != str(output):
                raise ValueError("Validated replay output path disagrees")
        except (OSError, ValueError, HTTPException, subprocess.SubprocessError) as exc:
            report = {"status": "blocked", "message": str(exc)}
        with self._lock:
            record["report"] = {**report, "validation_id": validation_id}

    def _validated_replay(self, request: RLTStage2StartRequest) -> str:
        with self._lock:
            record = self._data_validations.get(request.data_validation_id)
            if not record or record["report"].get("status") != "ready":
                raise HTTPException(409, "Dataset validation is not ready; validate again")
            if record["key"] != self._validation_key(request):
                raise HTTPException(409, "Dataset/model/phase selection changed; validate again")
            output = record["output"]
            expected = record["report"]["manifest_sha256"]
        if _has_symlink_component(Path("/"), output) or _file_sha256(output / "replay_cache/manifest.json") != expected:
            raise HTTPException(409, "Validated replay changed; validate again")
        return str(output)

    def _command(self, job: _RLTStage2Job) -> list[str]:
        command = self._compose_command() + [
            "run", "--rm", "--no-deps", "--pull", "never",
            "--name", self._container_name(job),
            "--user", f"{RLT_STAGE2_TRAIN_UID}:{RLT_STAGE2_TRAIN_GID}",
            "--workdir", "/app",
            "--env", "HOME=/tmp",
            "--env", "USER=cyclo_rlt",
            "--env", "LOGNAME=cyclo_rlt",
            "--env", f"XDG_CACHE_HOME={RLT_STAGE2_CACHE_ROOT}",
            "--env", f"HF_HOME={RLT_STAGE2_CACHE_ROOT}/huggingface",
            "--env", f"HF_HUB_CACHE={RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"HUGGINGFACE_HUB_CACHE={RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"TRANSFORMERS_CACHE={RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"TORCH_HOME={RLT_STAGE2_CACHE_ROOT}/torch",
            "--env", f"TORCHINDUCTOR_CACHE_DIR={RLT_STAGE2_CACHE_ROOT}/torchinductor",
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
            "--training-phase", job.training_phase,
            "--actor-warmup-data-scope", job.actor_warmup_data_scope,
        ]
        if job.replay_bundle_path:
            command.extend(("--replay-bundle", job.replay_bundle_path))
        if job.allow_stale_replay:
            command.append("--allow-stale-replay")
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
            training_phase=job.training_phase,
            actor_warmup_data_scope=job.actor_warmup_data_scope,
            replay_bundle_path=job.replay_bundle_path,
            phase_state=job.phase_state,
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
            shadow_evaluation_path=(
                job.shadow_evaluation_path if completed else ""
            ),
            shadow_structural_pass=(
                job.shadow_structural_pass if completed else None
            ),
            shadow_correction_abs_mean=(
                job.shadow_correction_abs_mean if completed else None
            ),
            shadow_correction_abs_p99=(
                job.shadow_correction_abs_p99 if completed else None
            ),
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
            if isinstance(payload.get("phase_state"), dict):
                job.phase_state = dict(payload["phase_state"])
            shadow_path = payload.get("shadow_evaluation")
            if isinstance(shadow_path, str):
                job.shadow_evaluation_path = shadow_path
            shadow_pass = payload.get("shadow_structural_pass")
            if isinstance(shadow_pass, bool):
                job.shadow_structural_pass = shadow_pass
            for name in (
                "shadow_correction_abs_mean",
                "shadow_correction_abs_p99",
            ):
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
    def _artifact_verification_error(job: _RLTStage2Job) -> str:
        output = Path(job.output_dir)
        expected = {
            "actor_artifact_path": output / "artifacts" / "rlt_actor.pt",
            "encoder_artifact_path": output / "artifacts" / "rl_token_encoder.pt",
            "checkpoint_path": output / "training_state" / "rlt_stage2.pt",
            "manifest_path": output / "manifest.json",
        }
        for attribute, path in expected.items():
            if os.path.normpath(getattr(job, attribute)) != str(path):
                return f"reported {attribute} does not match {path}"
            if (
                _has_symlink_component(output, path)
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size <= 0
            ):
                return f"required artifact is missing or invalid: {path}"
        try:
            manifest = _read_json(expected["manifest_path"], label="RLT Stage 2 manifest")
        except HTTPException as error:
            return str(error.detail)
        try:
            _validate_bundle_manifest(output, manifest)
        except HTTPException as error:
            return str(error.detail)

        phase_state = manifest.get("phase_state") or {}
        if (phase_state.get("training_phase") != job.training_phase
            or job.training_phase not in phase_state.get("completed_phases", [])):
            return "The completed bundle does not confirm the requested training phase"
        if job.shadow_evaluation_path:
            shadow_path = output / "evaluation" / "shadow_evaluation.json"
            if (
                os.path.normpath(job.shadow_evaluation_path) != str(shadow_path)
                or _has_symlink_component(output, shadow_path)
                or shadow_path.is_symlink()
                or not shadow_path.is_file()
            ):
                return f"shadow evaluation artifact is missing or invalid: {shadow_path}"
            try:
                shadow = _read_json(shadow_path, label="RLT shadow evaluation")
            except HTTPException as error:
                return str(error.detail)
            if shadow.get("bundle_manifest_fingerprint") != manifest.get(
                "manifest_fingerprint"
            ):
                return "shadow evaluation is not bound to the completed bundle"
            qualification_fingerprint = shadow.get("qualification_fingerprint")
            shadow_unsigned = {
                key: value
                for key, value in shadow.items()
                if key != "qualification_fingerprint"
            }
            if (
                not isinstance(qualification_fingerprint, str)
                or _canonical_fingerprint(shadow_unsigned)
                != qualification_fingerprint
            ):
                return "shadow evaluation fingerprint disagrees"
        return ""

    @staticmethod
    def _verified_artifacts(job: _RLTStage2Job) -> bool:
        return not RLTStage2Supervisor._artifact_verification_error(job)

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
        verification_error = (
            self._artifact_verification_error(job)
            if returncode == 0 and result_complete
            else ""
        )
        if returncode == 0 and result_complete and not verification_error and not job.stop_requested:
            if Path(job.output_dir).parent.name == ".runs":
                try:
                    _publish_stage2_bundle(Path(job.output_dir))
                except (HTTPException, OSError) as exc:
                    verification_error = f"could not publish latest result: {exc}"
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
            elif returncode == 0 and result_complete and not verification_error:
                job.status = "completed"
                job.phase = "complete"
                job.percentage = 100.0
                job.completed_steps = job.total_steps
                job.eta_seconds = 0.0
                job.message = "GR00T RLT Stage 2 completed"
            else:
                job.status = "failed"
                job.phase = "error"
                if verification_error:
                    job.message = (
                        "GR00T RLT Stage 2 bundle verification failed: "
                        f"{verification_error}"
                    )
                elif not job.message or job.message.startswith("Starting GR00T"):
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
                try:
                    _phase_contract.validate_phase_manifest(request.training_phase, manifest)
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
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
            replay_bundle = request.replay_bundle_path.strip()
            if not replay_bundle and not _has_fresh_tt_replay(datasets, request.training_phase):
                replay_bundle = bundle_path
            if request.data_validation_id:
                replay_bundle = self._validated_replay(request)
            elif replay_bundle:
                replay_root = _resolve_resume_bundle(replay_bundle)
                replay_manifest = _read_replay_json(
                    replay_root / "replay_cache/manifest.json",
                    label="saved replay",
                )
                recorded = replay_manifest.get("metadata", {}).get("dataset_roots", [])
                if not recorded or [os.path.normpath(str(p)) for p in datasets[:len(recorded)]] != [os.path.normpath(p) for p in recorded]:
                    raise HTTPException(400, "Selected datasets differ from the saved replay bundle; select its exact datasets")
                role = replay_manifest.get("metadata", {}).get("replay_role", "transition")
                replay_subdir = {"transition": "", "tt_rtc_actor/v1": "tt_rtc_actor_v1",
                                 "tt_rtc_transition/v1": "tt_rtc_transition_v1"}.get(role)
                if replay_subdir is None:
                    raise HTTPException(400, "Unsupported saved replay role")
                if role == "tt_rtc_actor/v1" and request.training_phase != "actor_warmup":
                    raise HTTPException(400, "Prepare TT-RTC Critic/RL replay for these datasets first")
                for dataset in datasets[len(recorded):]:
                    selected_cache = (_tt_dataset_replay(dataset, "actor_warmup") if role == "tt_rtc_actor/v1"
                                      else dataset / "rlt" / replay_subdir / "replay_cache")
                    if not (selected_cache / "manifest.json").is_file():
                        raise HTTPException(400, f"New dataset has no saved RLT replay: {dataset}. VLA inference is disabled.")
                replay_spec = _read_json(replay_root / "manifest.json", label="replay source bundle")["spec"]
                if replay_spec.get("schema_version") != 1:
                    raise HTTPException(400, "This saved replay uses an incompatible schema. Select an F2 10-to-10 schema 1 bundle; artifacts are not converted automatically.")
                replay_bundle = str(replay_root)
            else:
                for dataset in datasets:
                    if not (_saved_dataset_replay(dataset, request.training_phase) / "manifest.json").is_file():
                        raise HTTPException(400, "Saved RLT replay is missing. Select a replay source bundle; Stage 2 does not run VLA inference.")
            job_id = uuid.uuid4().hex
            output = _output_path(
                job_id,
                request.steps,
                datasets,
                request.training_phase,
            )
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
            training_phase=request.training_phase,
            actor_warmup_data_scope=request.actor_warmup_data_scope,
            replay_bundle_path=replay_bundle,
            allow_stale_replay=request.allow_stale_replay,
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

    def start_iterative(
        self,
        request: RLTStage2IterativeStartRequest,
    ) -> RLTStage2Status:
        """Resume from one actor using only rollouts verified from that actor."""

        bundle = _resolve_resume_bundle(request.rlt_bundle_path)
        dataset_paths = _iterative_dataset_paths(
            bundle,
            request.new_dataset_paths,
        )
        return self.start(
            RLTStage2StartRequest(
                initialization_mode="resume",
                training_phase="rl",
                dataset_paths=dataset_paths,
                rlt_bundle_path=str(bundle),
                steps=request.steps,
                batch_size=request.batch_size,
                save_freq=request.save_freq,
            )
        )

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
