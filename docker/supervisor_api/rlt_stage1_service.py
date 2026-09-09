#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Supervisor routes for GR00T RLT Stage 1 representation training.

Stage 1 is deliberately separate from imitation-learning policy training:
GR00T stays frozen, while an RL-token encoder/decoder reconstructs frozen
GR00T token features.  Its result is therefore an encoder artifact plus a
resumable training checkpoint, not a deployable robot policy.
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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictInt

try:
    from supervisor_api import rlt_service_common as _rlt_common
except ModuleNotFoundError:  # Direct file loading in focused unit tests.
    from docker.supervisor_api import rlt_service_common as _rlt_common


RLT_STAGE1_DATASET_ROOTS = (Path("/workspace/lerobot"),)
RLT_STAGE1_GROOT_ROOTS = (
    Path("/workspace/model/groot"),
    Path("/workspace/checkpoint"),
)
RLT_STAGE1_OUTPUT_ROOT = Path("/workspace/checkpoint/rlt/stage1")
RLT_STAGE1_LOG_ROOT = Path("/tmp/cyclo_rlt_stage1")
RLT_STAGE1_CACHE_ROOT = "/tmp/cyclo_rlt_stage1_cache"
RLT_STAGE1_HF_HUB_CACHE = "/huggingface_hub"
RLT_STAGE1_TRAIN_UID = 1000
RLT_STAGE1_TRAIN_GID = 1000
RLT_STAGE1_LOG_LINES = 100
RLT_STAGE1_MAX_JSON_BYTES = 2 * 1024 * 1024
RLT_STAGE1_DISCOVERY_LIMIT = 256

_STAGE1_RUN_FORMAT = "cyclo.groot.rlt.stage1_run/v1"
_STAGE1_OUTPUT_PATTERN = re.compile(r"steps_([0-9]+)_([0-9a-f]{12})")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_has_symlink_component = _rlt_common.has_symlink_component
_resolve_existing_path = _rlt_common.resolve_existing_path


class RLTStage1StartRequest(BaseModel):
    dataset_paths: list[str] = Field(min_length=1, max_length=128)
    groot_checkpoint: str
    steps: StrictInt = Field(default=10_000, ge=1, le=1_000_000)
    batch_size: StrictInt = Field(default=1, ge=1, le=64)
    save_freq: StrictInt = Field(default=1_000, ge=1, le=1_000_000)


class RLTStage1StopRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


class RLTStage1Status(BaseModel):
    ready: bool = True
    status: Literal["idle", "running", "completed", "failed", "stopped"]
    phase: str = "idle"
    percentage: float = 0.0
    job_id: str = ""
    dataset_paths: list[str] = Field(default_factory=list)
    resolved_dataset_paths: list[str] = Field(default_factory=list)
    groot_checkpoint: str = ""
    output_dir: str = ""
    encoder_artifact_path: str = ""
    checkpoint_path: str = ""
    completed_steps: int = 0
    total_steps: int = 0
    batch_size: int = 0
    save_freq: int = 0
    reconstruction_loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = ""
    returncode: Optional[int] = None
    log_tail: list[str] = Field(default_factory=list)


@dataclass
class _RLTStage1Job:
    job_id: str
    dataset_paths: list[str]
    resolved_dataset_paths: list[str]
    groot_checkpoint: str
    output_dir: str
    log_path: str
    total_steps: int
    batch_size: int
    save_freq: int
    status: str = "running"
    phase: str = "starting"
    percentage: float = 0.0
    encoder_artifact_path: str = ""
    checkpoint_path: str = ""
    completed_steps: int = 0
    reconstruction_loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = "Starting GR00T RLT Stage 1"
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    stop_confirmed: bool = False
    returncode: Optional[int] = None
    log_tail: list[str] = field(default_factory=list)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    return _rlt_common.read_json_object(
        path,
        label=label,
        max_bytes=RLT_STAGE1_MAX_JSON_BYTES,
    )


def _dataset_version(dataset: Path) -> str:
    return _rlt_common.lerobot_dataset_version(
        dataset,
        max_json_bytes=RLT_STAGE1_MAX_JSON_BYTES,
        unsupported_message=(
            f"Unsupported LeRobot codebase_version in {dataset / 'meta' / 'info.json'}"
        ),
    )


def _validate_v21_dataset(dataset: Path) -> Path:
    return _rlt_common.validate_lerobot_dataset(
        dataset,
        expected_version="v2.1",
        max_json_bytes=RLT_STAGE1_MAX_JSON_BYTES,
        unsupported_message=(
            f"Unsupported LeRobot codebase_version in {dataset / 'meta' / 'info.json'}"
        ),
        wrong_version_message=f"GR00T RLT Stage 1 requires LeRobot v2.1: {dataset}",
        unsafe_metadata_message=(
            f"LeRobot v2.1 dataset is missing episode/task metadata: {dataset}"
        ),
    )


def _validate_v30_dataset(dataset: Path) -> Path:
    return _rlt_common.validate_lerobot_dataset(
        dataset,
        expected_version="v3.0",
        max_json_bytes=RLT_STAGE1_MAX_JSON_BYTES,
        unsupported_message=(
            f"Unsupported LeRobot codebase_version in {dataset / 'meta' / 'info.json'}"
        ),
        wrong_version_message=f"GR00T RLT Stage 1 requires LeRobot v3.0: {dataset}",
        unsafe_metadata_message=(
            f"LeRobot v3.0 dataset is missing safe episode/task metadata: {dataset}"
        ),
    )


def _resolve_datasets(raw_paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        dataset = _resolve_existing_path(
            raw_path,
            roots=RLT_STAGE1_DATASET_ROOTS,
            label="dataset_path",
        )
        version = _dataset_version(dataset)
        dataset = (
            _validate_v21_dataset(dataset)
            if version == "v2.1"
            else _validate_v30_dataset(dataset)
        )
        if dataset in seen:
            raise HTTPException(
                400,
                "dataset_paths resolve to duplicate LeRobot datasets",
            )
        seen.add(dataset)
        resolved.append(dataset)
    return resolved


def _resolve_groot_checkpoint(raw_path: str) -> Path:
    checkpoint = _resolve_existing_path(
        raw_path,
        roots=RLT_STAGE1_GROOT_ROOTS,
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
        raise HTTPException(400, f"GR00T checkpoint has no safe safetensors weights: {checkpoint}")
    for path in (checkpoint / "config.json", *weights):
        mode = path.stat().st_mode
        if not (
            mode & stat.S_IROTH
            or mode & stat.S_IRGRP
            or path.stat().st_uid == RLT_STAGE1_TRAIN_UID
        ):
            raise HTTPException(
                400,
                f"GR00T artifact is not readable by the training user: {path}",
            )
    return checkpoint


def _output_path(job_id: str, steps: int) -> Path:
    root = RLT_STAGE1_OUTPUT_ROOT
    raw_checkpoint_root = next(
        (candidate for candidate in RLT_STAGE1_GROOT_ROOTS if root.is_relative_to(candidate)),
        None,
    )
    checkpoint_root = None
    if raw_checkpoint_root is not None:
        try:
            checkpoint_root = raw_checkpoint_root.resolve(strict=True)
        except OSError as exc:
            raise HTTPException(500, "RLT Stage 1 checkpoint root is unavailable") from exc
        if _has_symlink_component(raw_checkpoint_root, root):
            raise HTTPException(
                500,
                "RLT Stage 1 output root must not contain symbolic links",
            )
    if checkpoint_root is None:
        # Tests may replace the output root with an isolated safe directory.
        parent = root.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            checkpoint_root = parent.resolve(strict=True)
        except OSError as exc:
            raise HTTPException(500, "RLT Stage 1 output parent is unavailable") from exc
    output = root / f"steps_{steps:06d}_{job_id[:12]}"
    if output.exists() or output.is_symlink():
        raise HTTPException(409, f"RLT Stage 1 output already exists: {output}")
    try:
        output.resolve(strict=False).relative_to(checkpoint_root)
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "RLT Stage 1 output escapes its checkpoint root") from exc
    return output


def _prepare_output(output: Path) -> None:
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o775)
        os.chown(output, RLT_STAGE1_TRAIN_UID, RLT_STAGE1_TRAIN_GID)
    except (OSError, PermissionError) as exc:
        raise HTTPException(
            500,
            f"Could not create a writable RLT Stage 1 output: {output}",
        ) from exc


def _completed_stage1_status(output: Path) -> RLTStage1Status:
    """Validate one published Stage-1 run and reconstruct terminal status.

    Discovery is intentionally stricter than a directory-exists check.  Only
    the immutable terminal manifest and its exact, non-symlink artifact paths
    can make a run visible again after the supervisor restarts.
    """

    resolved_output = _resolve_existing_path(
        str(output),
        roots=(RLT_STAGE1_OUTPUT_ROOT,),
        label="RLT Stage 1 output",
    )
    root = RLT_STAGE1_OUTPUT_ROOT.resolve(strict=True)
    relative = resolved_output.relative_to(root)
    match = _STAGE1_OUTPUT_PATTERN.fullmatch(resolved_output.name)
    if len(relative.parts) != 1 or match is None:
        raise HTTPException(400, "RLT Stage 1 output name is invalid")

    manifest_path = (
        resolved_output / "training_state" / "rlt_stage1.pt.run.json"
    )
    manifest = _read_json(manifest_path, label="RLT Stage 1 run manifest")
    required = {
        "format",
        "status",
        "job_id",
        "dataset_roots",
        "groot_checkpoint",
        "policy_weight_fingerprint",
        "policy_checkpoint_fingerprint",
        "policy_processor_fingerprint",
        "action_normalization_id",
        "action_codec_id",
        "completed_steps",
        "batch_size",
        "artifact",
        "checkpoint",
    }
    if (
        set(manifest) != required
        or manifest.get("format") != _STAGE1_RUN_FORMAT
        or manifest.get("status") != "completed"
    ):
        raise HTTPException(400, "RLT Stage 1 run manifest fields are invalid")

    job_id = manifest.get("job_id")
    if (
        not isinstance(job_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", job_id) is None
        or job_id[:12] != match.group(2)
    ):
        raise HTTPException(400, "RLT Stage 1 run job identity is invalid")
    completed_steps = manifest.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or not 1 <= completed_steps <= 1_000_000
        or completed_steps != int(match.group(1))
    ):
        raise HTTPException(400, "RLT Stage 1 completed steps are invalid")
    batch_size = manifest.get("batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 64
    ):
        raise HTTPException(400, "RLT Stage 1 batch size is invalid")

    datasets = manifest.get("dataset_roots")
    if (
        not isinstance(datasets, list)
        or not 1 <= len(datasets) <= 128
        or any(not isinstance(value, str) or not Path(value).is_absolute() for value in datasets)
    ):
        raise HTTPException(400, "RLT Stage 1 dataset provenance is invalid")
    for value in datasets:
        lexical = Path(os.path.abspath(value))
        if not any(
            lexical.is_relative_to(Path(os.path.abspath(root_path)))
            for root_path in RLT_STAGE1_DATASET_ROOTS
        ):
            raise HTTPException(400, "RLT Stage 1 dataset provenance escapes its root")

    checkpoint = _resolve_groot_checkpoint(str(manifest.get("groot_checkpoint") or ""))
    for name in (
        "policy_weight_fingerprint",
        "policy_checkpoint_fingerprint",
        "policy_processor_fingerprint",
    ):
        value = manifest.get(name)
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise HTTPException(400, f"RLT Stage 1 {name} is invalid")
    for name in ("action_normalization_id", "action_codec_id"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise HTTPException(400, f"RLT Stage 1 {name} is invalid")

    expected_encoder = resolved_output / "artifacts" / "rl_token_encoder.pt"
    expected_checkpoint = resolved_output / "training_state" / "rlt_stage1.pt"
    artifact = manifest.get("artifact")
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(artifact, dict) or set(artifact) != {
        "path",
        "artifact_fingerprint",
    }:
        raise HTTPException(400, "RLT Stage 1 artifact record is invalid")
    if not isinstance(checkpoint_record, dict) or set(checkpoint_record) != {"path"}:
        raise HTTPException(400, "RLT Stage 1 checkpoint record is invalid")
    if (
        Path(os.path.abspath(str(artifact.get("path") or ""))) != expected_encoder
        or Path(os.path.abspath(str(checkpoint_record.get("path") or "")))
        != expected_checkpoint
        or not isinstance(artifact.get("artifact_fingerprint"), str)
        or _SHA256_PATTERN.fullmatch(artifact["artifact_fingerprint"]) is None
    ):
        raise HTTPException(400, "RLT Stage 1 artifact provenance disagrees")
    for path in (expected_encoder, expected_checkpoint):
        if (
            _has_symlink_component(resolved_output, path)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size <= 0
        ):
            raise HTTPException(400, f"RLT Stage 1 artifact is incomplete: {path}")

    return RLTStage1Status(
        status="completed",
        phase="complete",
        percentage=100.0,
        job_id=job_id,
        dataset_paths=list(datasets),
        resolved_dataset_paths=list(datasets),
        groot_checkpoint=str(checkpoint),
        output_dir=str(resolved_output),
        encoder_artifact_path=str(expected_encoder),
        checkpoint_path=str(expected_checkpoint),
        completed_steps=completed_steps,
        total_steps=completed_steps,
        batch_size=batch_size,
        eta_seconds=0.0,
        message="Recovered completed GR00T RLT Stage 1 artifact",
        returncode=0,
    )


def _discover_latest_completed_stage1() -> RLTStage1Status | None:
    """Return the newest verified direct child, ignoring partial publications."""

    root = RLT_STAGE1_OUTPUT_ROOT
    if root.is_symlink() or not root.is_dir():
        return None
    try:
        candidates = []
        for child in root.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if _STAGE1_OUTPUT_PATTERN.fullmatch(child.name) is None:
                continue
            manifest = child / "training_state" / "rlt_stage1.pt.run.json"
            if manifest.is_symlink() or not manifest.is_file():
                continue
            candidates.append((manifest.stat().st_mtime_ns, child.name, child))
    except OSError:
        return None
    candidates.sort(reverse=True)
    for _mtime, _name, candidate in candidates[:RLT_STAGE1_DISCOVERY_LIMIT]:
        try:
            return _completed_stage1_status(candidate)
        except (HTTPException, OSError, ValueError):
            continue
    return None


class RLTStage1Supervisor:
    """Own exactly one frozen-GR00T Stage 1 subprocess."""

    def __init__(
        self,
        *,
        compose_command: Callable[[], list[str]],
        compose_environment: Callable[[], dict[str, str]],
        conflict_message: Callable[[], str | None] | None = None,
        interrupt_container: Callable[[str], bool] | None = None,
    ) -> None:
        self._compose_command = compose_command
        self._compose_environment = compose_environment
        self._conflict_message = conflict_message or (lambda: None)
        self._interrupt_container = interrupt_container
        self._lock = threading.Lock()
        self._job: _RLTStage1Job | None = None
        self._recovered_status = _discover_latest_completed_stage1()
        self._starting = False
        self.router = APIRouter(prefix="/rlt-stage1", tags=["rlt-stage1"])
        self.router.add_api_route(
            "/start", self.start, methods=["POST"], response_model=RLTStage1Status
        )
        self.router.add_api_route(
            "/status", self.status, methods=["GET"], response_model=RLTStage1Status
        )
        self.router.add_api_route(
            "/stop", self.stop, methods=["POST"], response_model=RLTStage1Status
        )

    @staticmethod
    def _container_name(job: _RLTStage1Job) -> str:
        return f"cyclo_rlt_stage1_{job.job_id[:12]}"

    def _command(self, job: _RLTStage1Job) -> list[str]:
        command = self._compose_command() + [
            "run",
            "--rm",
            "--no-deps",
            "--pull",
            "never",
            "--name",
            self._container_name(job),
            "--user",
            f"{RLT_STAGE1_TRAIN_UID}:{RLT_STAGE1_TRAIN_GID}",
            "--workdir",
            "/app",
            "--env",
            "HOME=/tmp",
            "--env",
            f"XDG_CACHE_HOME={RLT_STAGE1_CACHE_ROOT}",
            "--env",
            f"HF_HOME={RLT_STAGE1_CACHE_ROOT}/huggingface",
            "--env",
            f"HF_HUB_CACHE={RLT_STAGE1_HF_HUB_CACHE}",
            "--env",
            f"HUGGINGFACE_HUB_CACHE={RLT_STAGE1_HF_HUB_CACHE}",
            "--env",
            f"TRANSFORMERS_CACHE={RLT_STAGE1_HF_HUB_CACHE}",
            "--env",
            f"TORCH_HOME={RLT_STAGE1_CACHE_ROOT}/torch",
            "--env",
            f"TRITON_CACHE_DIR={RLT_STAGE1_CACHE_ROOT}/triton",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "HF_DATASETS_OFFLINE=1",
            "--env",
            "GROOT_HF_LOCAL_FIRST=1",
            "--env",
            "GROOT_PATCH_MISTRAL=1",
            "--env",
            "NO_ALBUMENTATIONS_UPDATE=1",
            "--env",
            "PYTHONPATH=/cyclo_brain_src:/app:/gr00t",
            "--entrypoint",
            "python",
            "groot",
            "-m",
            "runtime.rlt_stage1_training_cli",
            "--groot-checkpoint",
            job.groot_checkpoint,
        ]
        for dataset_path in job.resolved_dataset_paths:
            command.extend(("--dataset-root", dataset_path))
        command.extend(
            (
                "--output-dir",
                job.output_dir,
                "--job-id",
                job.job_id,
                "--steps",
                str(job.total_steps),
                "--batch-size",
                str(job.batch_size),
                "--save-freq",
                str(job.save_freq),
                "--progress-interval",
                "10",
                "--device",
                "cuda",
            )
        )
        return command

    @staticmethod
    def _status(job: _RLTStage1Job | None) -> RLTStage1Status:
        if job is None:
            return RLTStage1Status(
                status="idle",
                message="GR00T RLT Stage 1 is ready",
            )
        completed = job.status == "completed"
        return RLTStage1Status(
            status=job.status,
            phase=job.phase,
            percentage=job.percentage,
            job_id=job.job_id,
            dataset_paths=list(job.dataset_paths),
            resolved_dataset_paths=list(job.resolved_dataset_paths),
            groot_checkpoint=job.groot_checkpoint,
            output_dir=job.output_dir,
            encoder_artifact_path=job.encoder_artifact_path if completed else "",
            checkpoint_path=job.checkpoint_path if completed else "",
            completed_steps=job.completed_steps,
            total_steps=job.total_steps,
            batch_size=job.batch_size,
            save_freq=job.save_freq,
            reconstruction_loss=job.reconstruction_loss,
            eta_seconds=job.eta_seconds,
            message=job.message,
            returncode=job.returncode,
            log_tail=list(job.log_tail),
        )

    @staticmethod
    def _append_log(job: _RLTStage1Job, line: str) -> None:
        if line:
            job.log_tail.append(line)
            del job.log_tail[:-RLT_STAGE1_LOG_LINES]

    @staticmethod
    def _consume_number(job: _RLTStage1Job, payload: dict[str, Any], name: str) -> None:
        value = payload.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(job, name, float(value))

    def _consume_event(self, job: _RLTStage1Job, payload: dict[str, Any]) -> bool:
        event = str(payload.get("event") or "")
        if event in {"starting", "manifest", "ready"}:
            job.phase = str(payload.get("phase") or "extracting_features")
            job.message = str(
                payload.get("message")
                or "Extracting frozen GR00T token features"
            )
            return False
        if event == "progress":
            job.phase = str(payload.get("phase") or "training_rl_token")
            step = payload.get("completed_steps", payload.get("step"))
            if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
                job.completed_steps = step
            total = payload.get("total_steps")
            if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                job.total_steps = total
            self._consume_number(job, payload, "reconstruction_loss")
            self._consume_number(job, payload, "eta_seconds")
            percentage = payload.get("percentage")
            if isinstance(percentage, (int, float)) and not isinstance(percentage, bool):
                phase_percentage = max(0.0, min(100.0, float(percentage)))
                job.percentage = (
                    0.5 * phase_percentage
                    if job.phase == "extracting"
                    else min(99.0, 50.0 + 0.5 * phase_percentage)
                )
            elif job.total_steps:
                job.percentage = max(
                    50.0,
                    min(
                        99.0,
                        50.0 + 50.0 * job.completed_steps / job.total_steps,
                    ),
                )
            default_message = (
                "Extracting frozen GR00T token features"
                if job.phase == "extracting"
                else "Training RL Token network"
            )
            job.message = str(payload.get("message") or default_message)
            return False

        complete = event in {"completed", "result"} and payload.get("status") in {
            "complete",
            "completed",
        }
        if complete:
            encoder = payload.get("encoder_artifact_path")
            checkpoint = payload.get("checkpoint_path")
            if isinstance(encoder, str):
                job.encoder_artifact_path = encoder
            if isinstance(checkpoint, str):
                job.checkpoint_path = checkpoint
            step = payload.get("completed_steps", payload.get("step"))
            if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
                job.completed_steps = step
            self._consume_number(job, payload, "reconstruction_loss")
            job.phase = "verifying"
            job.message = "Verifying the RL Token encoder and Stage 1 checkpoint"
            return bool(job.encoder_artifact_path and job.checkpoint_path)
        if event in {"stopped", "cancelled"} or (
            event == "result" and payload.get("status") == "stopped"
        ):
            job.stop_confirmed = True
            job.phase = "stopping"
            job.message = str(payload.get("message") or "RLT Stage 1 stopped")
        elif event in {"failed", "error"} or (
            event == "result"
            and str(payload.get("status") or "").lower() in {"failed", "error"}
        ):
            job.phase = "error"
            job.message = str(
                payload.get("message")
                or payload.get("error")
                or "RLT Stage 1 failed"
            )
        return False

    @staticmethod
    def _verified_artifacts(job: _RLTStage1Job) -> bool:
        output = Path(job.output_dir)
        expected_encoder = output / "artifacts" / "rl_token_encoder.pt"
        expected_checkpoint = output / "training_state" / "rlt_stage1.pt"
        if os.path.normpath(job.encoder_artifact_path) != str(expected_encoder):
            return False
        if os.path.normpath(job.checkpoint_path) != str(expected_checkpoint):
            return False
        for path in (expected_encoder, expected_checkpoint):
            if (
                _has_symlink_component(output, path)
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size <= 0
            ):
                return False
        return True

    def _monitor(self, job: _RLTStage1Job) -> None:
        result_complete = False
        try:
            Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(job.log_path, "a", encoding="utf-8") as log:
                stdout = job.process.stdout if job.process is not None else None
                if stdout is not None:
                    for raw_line in stdout:
                        line = (
                            raw_line.decode(errors="replace")
                            if isinstance(raw_line, bytes)
                            else raw_line
                        ).rstrip("\r\n")
                        log.write(line + "\n")
                        log.flush()
                        with self._lock:
                            self._append_log(job, line)
                            try:
                                payload = json.loads(line)
                            except (TypeError, json.JSONDecodeError):
                                continue
                            if isinstance(payload, dict):
                                result_complete = (
                                    self._consume_event(job, payload)
                                    or result_complete
                                )
                returncode = job.process.wait() if job.process is not None else -1
        except Exception as exc:  # pragma: no cover - worker boundary
            returncode = -1
            with self._lock:
                job.message = f"RLT Stage 1 monitor failed: {exc}"

        with self._lock:
            job.returncode = returncode
            expected_stop_codes = {0, -signal.SIGINT, 128 + signal.SIGINT}
            if (job.stop_requested and returncode in expected_stop_codes) or (
                returncode == 0 and job.stop_confirmed
            ):
                job.status = "stopped"
                job.phase = "stopped"
                job.encoder_artifact_path = ""
                job.checkpoint_path = ""
                job.eta_seconds = None
                job.message = "GR00T RLT Stage 1 stopped"
            elif returncode == 0 and result_complete and self._verified_artifacts(job):
                job.status = "completed"
                job.phase = "complete"
                job.percentage = 100.0
                job.completed_steps = job.total_steps
                job.eta_seconds = 0.0
                job.message = "GR00T RLT Stage 1 completed"
            else:
                job.status = "failed"
                job.phase = "error"
                if not job.message or job.message.startswith("Starting GR00T"):
                    reason = (
                        f"exited with code {returncode}"
                        if returncode != 0
                        else "did not report both verified Stage 1 artifacts"
                    )
                    job.message = f"GR00T RLT Stage 1 {reason}"

    def is_running(self) -> bool:
        with self._lock:
            return self._starting or (
                self._job is not None and self._job.status == "running"
            )

    def start(self, request: RLTStage1StartRequest) -> RLTStage1Status:
        with self._lock:
            if self._starting or (
                self._job is not None and self._job.status == "running"
            ):
                raise HTTPException(409, "An RLT Stage 1 job is already running")
            conflict = self._conflict_message()
            if conflict:
                raise HTTPException(409, conflict)
            self._starting = True

        try:
            datasets = _resolve_datasets(request.dataset_paths)
            checkpoint = _resolve_groot_checkpoint(request.groot_checkpoint)
            job_id = uuid.uuid4().hex
            output = _output_path(job_id, request.steps)
            _prepare_output(output)
            RLT_STAGE1_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        except Exception:
            with self._lock:
                self._starting = False
            raise
        job = _RLTStage1Job(
            job_id=job_id,
            dataset_paths=list(request.dataset_paths),
            resolved_dataset_paths=[str(path) for path in datasets],
            groot_checkpoint=str(checkpoint),
            output_dir=str(output),
            log_path=str(RLT_STAGE1_LOG_ROOT / f"{job_id}.log"),
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
            job.message = f"Could not start GR00T RLT Stage 1: {exc}"
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
            name=f"rlt-stage1-{job_id[:12]}",
            daemon=True,
        ).start()
        return self._status(job)

    def status(self) -> RLTStage1Status:
        with self._lock:
            if self._job is None and self._recovered_status is not None:
                return self._recovered_status.model_copy(deep=True)
            return self._status(self._job)

    def stop(self, request: RLTStage1StopRequest) -> RLTStage1Status:
        with self._lock:
            job = self._job
            if job is None or request.job_id != job.job_id:
                raise HTTPException(409, "RLT Stage 1 job_id does not match the active job")
            if job.status != "running":
                raise HTTPException(409, f"RLT Stage 1 job is already {job.status}")
            job.stop_requested = True
            job.phase = "stopping"
            job.message = "Stopping GR00T RLT Stage 1"

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
                job.message = f"Could not stop GR00T RLT Stage 1: {exc}"
            raise HTTPException(502, job.message) from exc
        if not signalled:
            with self._lock:
                job.stop_requested = False
            raise HTTPException(409, "RLT Stage 1 exited before it could be stopped")
        with self._lock:
            return self._status(job)


def create_rlt_stage1_router(
    *,
    compose_command: Callable[[], list[str]],
    compose_environment: Callable[[], dict[str, str]],
    conflict_message: Callable[[], str | None] | None = None,
    interrupt_container: Callable[[str], bool] | None = None,
) -> tuple[APIRouter, RLTStage1Supervisor]:
    supervisor = RLTStage1Supervisor(
        compose_command=compose_command,
        compose_environment=compose_environment,
        conflict_message=conflict_message,
        interrupt_container=interrupt_container,
    )
    return supervisor.router, supervisor
