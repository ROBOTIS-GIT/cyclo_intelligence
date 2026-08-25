#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dedicated supervisor routes for live MultiTaskDiT Flow-SDE PPO.

The existing ``/offline-rl`` API deliberately remains ACT/TD3-specific.
Flow-SDE PPO is on-policy, owns a live action-step session, and therefore has
different lifecycle and episode-outcome semantics.  Keeping it on a separate
router prevents a UI selection from accidentally falling through to TD3.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictInt


FLOW_SDE_POLICY_ROOTS = (
    Path("/workspace/checkpoint/multi_task_dit"),
    Path("/workspace/model/lerobot"),
)
FLOW_SDE_DATASET_ROOTS = (Path("/workspace/lerobot"),)
FLOW_SDE_OUTPUT_ROOT = Path(
    "/workspace/checkpoint/multi_task_dit/flow_sde_ppo"
)
FLOW_SDE_VALUE_WARMUP_ROOT = FLOW_SDE_OUTPUT_ROOT / "value_warmup"
FLOW_SDE_LOG_ROOT = Path("/tmp/cyclo_flow_sde_ppo")
FLOW_SDE_CACHE_ROOT = "/tmp/cyclo_flow_sde_ppo_cache"
FLOW_SDE_LOG_LINES = 100
FLOW_SDE_TRAIN_UID = 1000
FLOW_SDE_TRAIN_GID = 1000
FLOW_SDE_POLICY_ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
FLOW_SDE_ONLINE_BUNDLE_FORMAT = "cyclo.flow_sde_ppo.online.bundle.v1"
FLOW_SDE_ONLINE_STARTUP_FORMAT = "cyclo.flow_sde_ppo.online_startup.v1"
FLOW_SDE_ONLINE_EXPORT_FORMAT = "cyclo.flow_sde_ppo.actor.v1"


class FlowSDEPPOStartRequest(BaseModel):
    policy_checkpoint: str
    policy_type: Literal["multi_task_dit"] = "multi_task_dit"
    algorithm: Literal["flow_sde_ppo"] = "flow_sde_ppo"
    robot_type: str
    task_instruction: str = "pick up the jelly bag"
    episodes: StrictInt = Field(default=1, ge=1, le=100)
    ppo_epochs: StrictInt = Field(default=4, ge=1, le=64)
    minibatch_size: StrictInt = Field(default=4, ge=1, le=256)
    max_chunk_decisions: StrictInt = Field(default=20, ge=1, le=1000)
    actor_learning_rate: float = Field(default=3.0e-5, gt=0.0, le=1.0)
    value_learning_rate: float = Field(default=1.0e-4, gt=0.0, le=1.0)
    ack_timeout_seconds: float = Field(default=5.0, gt=0.0, le=120.0)
    sensor_timeout_seconds: float = Field(default=15.0, gt=0.0, le=300.0)
    value_warmup_bundle: Optional[str] = None
    resume_checkpoint: Optional[str] = None


class FlowSDEPPOStopRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


class FlowSDEPPOOutcomeRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)
    outcome: Literal["success", "fail", "cancel"]


class FlowSDEValueWarmupStartRequest(BaseModel):
    policy_checkpoint: str
    dataset_paths: list[str] = Field(min_length=1, max_length=128)
    policy_type: Literal["multi_task_dit"] = "multi_task_dit"
    task_instruction: str = "pick up the jelly bag"
    steps: StrictInt = Field(default=2000, ge=1, le=1_000_000)
    batch_size: StrictInt = Field(default=8, ge=1, le=256)
    value_learning_rate: float = Field(default=1.0e-4, gt=0.0, le=1.0)
    discount: float = Field(default=0.99, ge=0.0, le=1.0)


class FlowSDEPPOStatus(BaseModel):
    ready: bool = True
    status: Literal["idle", "running", "completed", "failed", "stopped"]
    phase: str = "idle"
    percentage: float = 0.0
    job_id: str = ""
    policy_checkpoint: str = ""
    lineage_policy_checkpoint: str = ""
    value_warmup_bundle: str = ""
    resume_checkpoint: str = ""
    resume_source_job_id: str = ""
    task_instruction: str = ""
    output_dir: str = ""
    checkpoint_path: str = ""
    model_path: str = ""
    episode: int = 0
    episodes: int = 1
    chunk_decisions: int = 0
    update_step: int = 0
    episode_return: Optional[float] = None
    actor_loss: Optional[float] = None
    value_loss: Optional[float] = None
    approx_kl: Optional[float] = None
    clip_fraction: Optional[float] = None
    eta_seconds: Optional[float] = None
    awaiting_outcome: bool = False
    message: str = ""
    returncode: Optional[int] = None
    log_tail: list[str] = Field(default_factory=list)


class FlowSDEValueWarmupStatus(BaseModel):
    ready: bool = True
    status: Literal["idle", "running", "completed", "failed", "stopped"]
    phase: str = "idle"
    percentage: float = 0.0
    job_id: str = ""
    policy_checkpoint: str = ""
    dataset_paths: list[str] = Field(default_factory=list)
    task_instruction: str = ""
    output_dir: str = ""
    bundle_path: str = ""
    checkpoint_path: str = ""
    model_path: str = ""
    step: int = 0
    total_steps: int = 0
    batch_size: int = 0
    value_learning_rate: float = 0.0
    discount: float = 0.0
    value_loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = ""
    returncode: Optional[int] = None
    log_tail: list[str] = Field(default_factory=list)


@dataclass
class _FlowSDEPPOJob:
    job_id: str
    policy_checkpoint: str
    robot_type: str
    task_instruction: str
    output_dir: str
    control_file: str
    log_path: str
    episodes: int
    ppo_epochs: int
    minibatch_size: int
    max_chunk_decisions: int
    actor_learning_rate: float
    value_learning_rate: float
    ack_timeout_seconds: float
    sensor_timeout_seconds: float
    value_warmup_bundle: str = ""
    resume_checkpoint: str = ""
    resume_source_job_id: str = ""
    lineage_policy_checkpoint: str = ""
    status: str = "running"
    phase: str = "starting"
    percentage: float = 0.0
    checkpoint_path: str = ""
    model_path: str = ""
    episode: int = 0
    chunk_decisions: int = 0
    update_step: int = 0
    episode_return: Optional[float] = None
    actor_loss: Optional[float] = None
    value_loss: Optional[float] = None
    approx_kl: Optional[float] = None
    clip_fraction: Optional[float] = None
    eta_seconds: Optional[float] = None
    awaiting_outcome: bool = False
    message: str = "Starting Flow-SDE PPO"
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    stop_confirmed: bool = False
    returncode: Optional[int] = None
    log_tail: list[str] = field(default_factory=list)


@dataclass
class _FlowSDEValueWarmupJob:
    job_id: str
    policy_checkpoint: str
    dataset_paths: list[str]
    task_instruction: str
    output_dir: str
    log_path: str
    total_steps: int
    batch_size: int
    value_learning_rate: float
    discount: float
    status: str = "running"
    phase: str = "starting"
    percentage: float = 0.0
    bundle_path: str = ""
    checkpoint_path: str = ""
    model_path: str = ""
    step: int = 0
    value_loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = "Starting Flow-SDE PPO value warm-up"
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    stop_confirmed: bool = False
    returncode: Optional[int] = None
    log_tail: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ResolvedFlowSDEResume:
    checkpoint: Path
    source_job_id: str
    source_output_dir: Path
    source_policy_checkpoint: Path
    source_model_path: Path
    lineage_policy_checkpoint: Path
    task_instruction: str
    robot_type: str
    ppo_config: dict[str, Any]
    update_step: int


def _resolve_policy_checkpoint(raw_path: str) -> Path:
    value = str(raw_path or "").strip()
    if not value or not Path(value).is_absolute():
        raise HTTPException(400, "policy_checkpoint must be an absolute path")
    lexical = Path(os.path.abspath(value))
    allowed_root: Path | None = None
    for root in FLOW_SDE_POLICY_ROOTS:
        try:
            lexical.relative_to(root)
        except ValueError:
            continue
        allowed_root = root
        break
    if allowed_root is None:
        roots = ", ".join(str(root) for root in FLOW_SDE_POLICY_ROOTS)
        raise HTTPException(400, f"policy_checkpoint must be under one of: {roots}")
    try:
        root_resolved = allowed_root.resolve(strict=True)
        candidate = lexical.resolve(strict=True)
        candidate.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, "policy_checkpoint does not exist or escapes its root") from exc
    if not candidate.is_dir():
        raise HTTPException(400, "policy_checkpoint must be a real directory")

    nested_candidates = (
        candidate,
        candidate / "pretrained_model",
        candidate / "checkpoints" / "last" / "pretrained_model",
    )
    for path in nested_candidates:
        if all((path / name).is_file() for name in FLOW_SDE_POLICY_ARTIFACTS):
            try:
                resolved_path = path.resolve(strict=True)
                resolved_path.relative_to(root_resolved)
            except (OSError, ValueError) as exc:
                raise HTTPException(400, "resolved policy checkpoint escapes its root") from exc
            if any(
                (resolved_path / name).is_symlink()
                or not (resolved_path / name).is_file()
                for name in FLOW_SDE_POLICY_ARTIFACTS
            ):
                raise HTTPException(
                    400,
                    "policy checkpoint artifacts must be regular files",
                )
            inaccessible = [
                name
                for name in FLOW_SDE_POLICY_ARTIFACTS
                if not _training_user_can_read(resolved_path / name)
            ]
            directories = [resolved_path]
            current = resolved_path
            while current != root_resolved:
                current = current.parent
                directories.append(current)
            if inaccessible or any(
                not _training_user_can_execute(directory)
                for directory in directories
            ):
                raise HTTPException(
                    400,
                    "policy checkpoint is not readable by the LeRobot training user "
                    f"(uid={FLOW_SDE_TRAIN_UID}, gid={FLOW_SDE_TRAIN_GID}); "
                    f"unreadable artifacts={inaccessible!r}",
                )
            return resolved_path
    raise HTTPException(
        400,
        "policy_checkpoint is missing a complete LeRobot policy/processor artifact set",
    )


def _training_user_mode_bit(path: Path, *, owner: int, group: int, other: int) -> bool:
    metadata = path.stat()
    if metadata.st_uid == FLOW_SDE_TRAIN_UID:
        bit = owner
    elif metadata.st_gid == FLOW_SDE_TRAIN_GID:
        bit = group
    else:
        bit = other
    return bool(metadata.st_mode & bit)


def _training_user_can_read(path: Path) -> bool:
    return _training_user_mode_bit(
        path,
        owner=stat.S_IRUSR,
        group=stat.S_IRGRP,
        other=stat.S_IROTH,
    )


def _training_user_can_execute(path: Path) -> bool:
    return _training_user_mode_bit(
        path,
        owner=stat.S_IXUSR,
        group=stat.S_IXGRP,
        other=stat.S_IXOTH,
    )


def _has_symlink_component(root: Path, path: Path) -> bool:
    """Return whether an existing path component below ``root`` is a symlink."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _resolve_value_warmup_dataset(raw_path: str) -> Path:
    """Resolve one immutable, labeled LeRobot v3 root for value warm-up."""

    value = str(raw_path or "").strip()
    if not value or not Path(value).is_absolute():
        raise HTTPException(400, "dataset_paths entries must be absolute paths")
    lexical = Path(os.path.abspath(value))
    allowed_root: Path | None = None
    for configured_root in FLOW_SDE_DATASET_ROOTS:
        root = Path(os.path.abspath(str(configured_root)))
        try:
            lexical.relative_to(root)
        except ValueError:
            continue
        allowed_root = root
        break
    if allowed_root is None:
        roots = ", ".join(str(root) for root in FLOW_SDE_DATASET_ROOTS)
        raise HTTPException(400, f"dataset_paths entries must be under one of: {roots}")
    if _has_symlink_component(allowed_root, lexical):
        raise HTTPException(400, "dataset_paths entries must not contain symbolic links")
    try:
        root_resolved = allowed_root.resolve(strict=True)
        candidate = lexical.resolve(strict=True)
        candidate.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, "dataset path does not exist or escapes its root") from exc
    if not candidate.is_dir():
        raise HTTPException(400, "dataset_paths entries must be real directories")

    directories = [candidate]
    current = candidate
    while current != root_resolved:
        current = current.parent
        directories.append(current)
    if any(not _training_user_can_execute(directory) for directory in directories):
        raise HTTPException(
            400,
            "dataset path is not traversable by the LeRobot training user "
            f"(uid={FLOW_SDE_TRAIN_UID}, gid={FLOW_SDE_TRAIN_GID})",
        )

    info_path = candidate / "meta" / "info.json"
    if (
        _has_symlink_component(candidate, info_path)
        or info_path.is_symlink()
        or not info_path.is_file()
        or not _training_user_can_read(info_path)
    ):
        raise HTTPException(400, "dataset is missing a safe, readable meta/info.json")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "dataset meta/info.json is not valid JSON") from exc
    if not isinstance(info, dict) or info.get("codebase_version") != "v3.0":
        raise HTTPException(400, "Flow-SDE PPO value warm-up requires LeRobot v3.0")
    features = info.get("features")
    success_feature = (
        features.get("episode_success") if isinstance(features, dict) else None
    )
    if not isinstance(success_feature, dict) or success_feature.get("dtype") != "bool":
        raise HTTPException(
            400,
            "Flow-SDE PPO value warm-up requires a boolean episode_success feature",
        )
    return candidate


def _resolve_value_warmup_datasets(raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        raise HTTPException(400, "dataset_paths must contain at least one dataset")
    datasets: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        dataset = _resolve_value_warmup_dataset(raw_path)
        if dataset in seen:
            raise HTTPException(400, "dataset_paths must not contain duplicates")
        seen.add(dataset)
        datasets.append(dataset)
    return datasets


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""

    first = first.resolve(strict=True)
    second = second.resolve(strict=False)
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"{name} is not valid readable JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, f"{name} must contain a JSON object")
    return payload


def _is_job_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _expected_online_ppo_config(
    *,
    ppo_epochs: int,
    minibatch_size: int,
    actor_learning_rate: float,
    value_learning_rate: float,
) -> dict[str, Any]:
    """Mirror the immutable ``FlowSDEPPOConfig`` used by ``live_cli``.

    A training checkpoint contains AdamW moments for both networks.  Resuming
    it under a different optimizer/training contract would be ambiguous, so
    the supervisor validates the complete public v1 contract without loading
    the pickle-backed Torch artifact itself.
    """

    return {
        "num_denoising_steps": 4,
        "noise_level": 0.5,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "discount": 0.99,
        "gae_lambda": 0.95,
        "value_clip": 0.2,
        "value_loss_coefficient": 0.5,
        "actor_learning_rate": float(actor_learning_rate),
        "value_learning_rate": float(value_learning_rate),
        "ppo_epochs": int(ppo_epochs),
        "minibatch_size": int(minibatch_size),
        "actor_max_grad_norm": 1.0,
        "value_max_grad_norm": 1.0,
        "normalize_advantages": True,
    }


def _safe_completed_online_paths(output_dir: Path) -> dict[str, Path]:
    paths = {
        "checkpoint": output_dir / "training_state" / "trainer_state.pt",
        "model": output_dir / "pretrained_model",
        "startup": output_dir / "startup_manifest.json",
        "summary": output_dir / "summary.json",
        "manifest": output_dir / "run_manifest.json",
        "progress": output_dir / "progress.jsonl",
        "export": output_dir / "pretrained_model" / "flow_sde_ppo_export.json",
    }
    files = tuple(path for name, path in paths.items() if name != "model")
    policy_files = tuple(
        paths["model"] / name for name in FLOW_SDE_POLICY_ARTIFACTS
    )
    if any(
        _has_symlink_component(output_dir, path)
        for path in (*files, *policy_files, paths["model"])
    ):
        raise HTTPException(400, "resume source artifacts must not contain symbolic links")
    if not paths["model"].is_dir() or any(
        path.is_symlink() or not path.is_file() for path in (*files, *policy_files)
    ):
        raise HTTPException(400, "resume source is missing completed online PPO artifacts")
    unreadable = [
        str(path.relative_to(output_dir))
        for path in (*files, *policy_files)
        if not _training_user_can_read(path)
    ]
    directories = [output_dir, output_dir / "training_state", paths["model"]]
    if unreadable or any(
        not _training_user_can_execute(directory) for directory in directories
    ):
        raise HTTPException(
            400,
            "resume source is not readable by the LeRobot training user; "
            f"unreadable artifacts={unreadable!r}",
        )
    return paths


def _online_output_from_checkpoint(raw_path: str) -> tuple[Path, Path, str]:
    value = str(raw_path or "").strip()
    if not value or not Path(value).is_absolute():
        raise HTTPException(400, "resume_checkpoint must be an absolute path")
    root = Path(os.path.abspath(str(FLOW_SDE_OUTPUT_ROOT)))
    lexical = Path(os.path.abspath(value))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            400,
            f"resume_checkpoint must be under {root}",
        ) from exc
    if (
        len(relative.parts) != 3
        or not _is_job_id(relative.parts[0])
        or relative.parts[1:] != ("training_state", "trainer_state.pt")
    ):
        raise HTTPException(
            400,
            "resume_checkpoint must be exactly "
            f"{root}/<32hex>/training_state/trainer_state.pt",
        )
    source_job_id = relative.parts[0]
    source_output = root / source_job_id
    if _has_symlink_component(root, lexical):
        raise HTTPException(400, "resume_checkpoint must not contain symbolic links")
    try:
        root_resolved = root.resolve(strict=True)
        checkpoint = lexical.resolve(strict=True)
        checkpoint.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            400,
            "resume_checkpoint does not exist or escapes its output root",
        ) from exc
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise HTTPException(400, "resume_checkpoint must be a regular file")
    return checkpoint, source_output.resolve(strict=True), source_job_id


def _manifest_policy_artifacts(
    manifest: dict[str, Any],
) -> tuple[dict[str, str], str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HTTPException(400, "online PPO run manifest artifacts are invalid")
    model = artifacts.get("pretrained_model")
    trainer = artifacts.get("trainer_checkpoint")
    if not isinstance(model, dict) or not isinstance(trainer, dict):
        raise HTTPException(400, "online PPO run manifest artifacts are incomplete")
    if model.get("path") != "pretrained_model":
        raise HTTPException(400, "online PPO pretrained-model artifact path is invalid")
    file_hashes = model.get("files")
    if not isinstance(file_hashes, dict) or set(file_hashes) != set(
        FLOW_SDE_POLICY_ARTIFACTS
    ):
        raise HTTPException(400, "online PPO policy artifact hash contract is invalid")
    if any(not _is_sha256(value) for value in file_hashes.values()):
        raise HTTPException(400, "online PPO policy artifact hashes are invalid")
    if trainer.get("path") != "training_state/trainer_state.pt" or not _is_sha256(
        trainer.get("sha256")
    ):
        raise HTTPException(400, "online PPO trainer-checkpoint contract is invalid")
    for key, expected in (
        ("startup_manifest_path", "startup_manifest.json"),
        ("progress_path", "progress.jsonl"),
        ("summary_path", "summary.json"),
    ):
        if artifacts.get(key) != expected:
            raise HTTPException(400, f"online PPO {key} contract is invalid")
    return {str(key): str(value) for key, value in file_hashes.items()}, str(
        trainer["sha256"]
    )


def _verify_policy_hashes(model_path: Path, hashes: dict[str, str]) -> None:
    for name in FLOW_SDE_POLICY_ARTIFACTS:
        if _sha256_file(model_path / name) != hashes[name]:
            raise HTTPException(
                400,
                f"resume source policy artifact hash mismatch: {name}",
            )


def _policy_hash_contract(payload: Any, *, name: str) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != set(FLOW_SDE_POLICY_ARTIFACTS):
        raise HTTPException(400, f"online PPO {name} hash contract is invalid")
    if any(not _is_sha256(value) for value in payload.values()):
        raise HTTPException(400, f"online PPO {name} hashes are invalid")
    return {str(key): str(value) for key, value in payload.items()}


def _validate_online_source_lineage(
    manifest: dict[str, Any],
    *,
    lineage_policy: Path,
) -> None:
    source_lineage = manifest.get("source_lineage")
    if not isinstance(source_lineage, dict):
        raise HTTPException(400, "resume source lineage provenance is invalid")
    resume = source_lineage.get("resume")
    value_initialization = source_lineage.get("value_initialization")
    if value_initialization is not None and not isinstance(value_initialization, dict):
        raise HTTPException(400, "resume source value initialization is invalid")
    if resume is None:
        return
    if not isinstance(resume, dict):
        raise HTTPException(400, "resume source checkpoint provenance is invalid")
    if resume.get("format") != "cyclo.flow_sde_ppo.resume.v1" or resume.get(
        "mode"
    ) not in {"explicit_checkpoint", "legacy_output"}:
        raise HTTPException(400, "resume source checkpoint provenance format is invalid")
    if not _is_sha256(resume.get("checkpoint_sha256")) or not _is_sha256(
        resume.get("source_manifest_sha256")
    ):
        raise HTTPException(400, "resume source provenance hashes are invalid")
    source_step = resume.get("source_update_step")
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step < 1:
        raise HTTPException(400, "resume source provenance update step is invalid")
    prior_checkpoint, prior_output, _ = _online_output_from_checkpoint(
        str(resume.get("checkpoint_path") or "")
    )
    prior_manifest = prior_output / "run_manifest.json"
    if os.path.normpath(str(resume.get("source_manifest_path") or "")) != str(
        prior_manifest
    ):
        raise HTTPException(400, "resume source provenance manifest path is invalid")
    if (
        prior_manifest.is_symlink()
        or not prior_manifest.is_file()
        or not _training_user_can_read(prior_manifest)
        or _sha256_file(prior_manifest) != resume["source_manifest_sha256"]
    ):
        raise HTTPException(400, "resume source provenance manifest hash is stale")
    prior_payload = _read_json_object(
        prior_manifest,
        name="prior online PPO run manifest",
    )
    prior_result = prior_payload.get("result")
    if (
        prior_payload.get("format") != FLOW_SDE_ONLINE_BUNDLE_FORMAT
        or prior_payload.get("status") != "complete"
        or not isinstance(prior_result, dict)
        or prior_result.get("update_step") != source_step
        or prior_payload.get("lineage_policy_checkpoint") != str(lineage_policy)
    ):
        raise HTTPException(400, "resume source provenance chain is inconsistent")
    # The current source run already loaded and validated this prior checkpoint.
    # Confirm it remains a safe regular artifact and preserve its recorded hash;
    # the current trainer checkpoint is independently re-hashed below.
    if prior_checkpoint.is_symlink() or not prior_checkpoint.is_file():
        raise HTTPException(400, "resume source provenance checkpoint is missing")


def _resolve_online_resume_checkpoint(
    raw_path: str,
    *,
    policy_checkpoint: Path,
    robot_type: str,
    task_instruction: str,
    ppo_epochs: int,
    minibatch_size: int,
    actor_learning_rate: float,
    value_learning_rate: float,
) -> _ResolvedFlowSDEResume:
    """Validate one completed online job without deserializing Torch state."""

    checkpoint, source_output, source_job_id = _online_output_from_checkpoint(raw_path)
    paths = _safe_completed_online_paths(source_output)
    manifest = _read_json_object(paths["manifest"], name="online PPO run manifest")
    startup = _read_json_object(paths["startup"], name="online PPO startup manifest")
    summary = _read_json_object(paths["summary"], name="online PPO summary")
    export = _read_json_object(paths["export"], name="online PPO export metadata")

    if manifest.get("format") != FLOW_SDE_ONLINE_BUNDLE_FORMAT:
        raise HTTPException(400, "resume source has an unsupported online bundle format")
    if manifest.get("status") != "complete" or summary.get("status") != "completed":
        raise HTTPException(400, "resume source online PPO job is not complete")
    if startup.get("format") != FLOW_SDE_ONLINE_STARTUP_FORMAT:
        raise HTTPException(400, "resume source startup manifest format is invalid")
    if startup.get("status") != "ready":
        raise HTTPException(400, "resume source startup manifest is not ready")
    if export.get("format") != FLOW_SDE_ONLINE_EXPORT_FORMAT:
        raise HTTPException(400, "resume source export metadata format is invalid")
    if any(
        payload.get("job_id") != source_job_id
        for payload in (manifest, startup, summary)
    ):
        raise HTTPException(400, "resume source job identity is inconsistent")

    immediate_base_raw = manifest.get("base_checkpoint")
    lineage_raw = manifest.get("lineage_policy_checkpoint")
    if not isinstance(immediate_base_raw, str) or not isinstance(lineage_raw, str):
        raise HTTPException(400, "resume source policy lineage is incomplete")
    source_policy = _resolve_policy_checkpoint(immediate_base_raw)
    lineage_policy = _resolve_policy_checkpoint(lineage_raw)
    _validate_online_source_lineage(
        manifest,
        lineage_policy=lineage_policy,
    )
    for payload in (startup, summary):
        if payload.get("base_checkpoint") != str(source_policy):
            raise HTTPException(400, "resume source immediate base checkpoint is inconsistent")
        if payload.get("lineage_policy_checkpoint") != str(lineage_policy):
            raise HTTPException(400, "resume source root policy lineage is inconsistent")
        for name in (
            "base_policy_artifacts",
            "lineage_policy_artifacts",
            "source_lineage",
        ):
            if payload.get(name) != manifest.get(name):
                raise HTTPException(400, f"resume source {name} is inconsistent")
    for payload in (manifest, startup, summary):
        if payload.get("task_instruction") != task_instruction:
            raise HTTPException(400, "resume source task instruction does not match online PPO")
        if payload.get("robot_type") != robot_type:
            raise HTTPException(400, "resume source robot type does not match online PPO")

    expected_config = _expected_online_ppo_config(
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        actor_learning_rate=actor_learning_rate,
        value_learning_rate=value_learning_rate,
    )
    manifest_config = manifest.get("ppo_config")
    if (
        manifest_config != expected_config
        or startup.get("ppo_config") != expected_config
        or export.get("ppo_config") != expected_config
    ):
        raise HTTPException(400, "resume source PPO configuration does not match this run")

    policy_hashes, checkpoint_hash = _manifest_policy_artifacts(manifest)
    base_policy_hashes = _policy_hash_contract(
        manifest.get("base_policy_artifacts"),
        name="immediate base policy",
    )
    lineage_policy_hashes = _policy_hash_contract(
        manifest.get("lineage_policy_artifacts"),
        name="lineage policy",
    )
    if _sha256_file(checkpoint) != checkpoint_hash:
        raise HTTPException(400, "resume checkpoint hash does not match its run manifest")
    _verify_policy_hashes(paths["model"], policy_hashes)

    expected_checkpoint = source_output / "training_state" / "trainer_state.pt"
    expected_model = source_output / "pretrained_model"
    if (
        os.path.normpath(str(checkpoint)) != str(expected_checkpoint)
        or summary.get("trainer_checkpoint") != str(expected_checkpoint)
        or summary.get("pretrained_model") != str(expected_model)
        or summary.get("base_checkpoint") != str(source_policy)
        or summary.get("run_manifest") != str(paths["manifest"])
        or summary.get("run_manifest_sha256") != _sha256_file(paths["manifest"])
    ):
        raise HTTPException(400, "resume source artifact paths are inconsistent")
    selected = policy_checkpoint.resolve(strict=True)
    allowed_selected = {
        source_policy.resolve(strict=True),
        lineage_policy.resolve(strict=True),
        expected_model.resolve(strict=True),
    }
    if selected not in allowed_selected:
        raise HTTPException(
            400,
            "selected policy is outside the resume source policy lineage",
        )
    if selected == expected_model.resolve(strict=True):
        _verify_policy_hashes(selected, policy_hashes)
    elif selected == source_policy.resolve(strict=True):
        _verify_policy_hashes(selected, base_policy_hashes)
    elif selected == lineage_policy.resolve(strict=True):
        _verify_policy_hashes(selected, lineage_policy_hashes)

    result = manifest.get("result")
    update_step = result.get("update_step") if isinstance(result, dict) else None
    if (
        isinstance(update_step, bool)
        or not isinstance(update_step, int)
        or update_step < 1
        or summary.get("updates") != update_step
        or export.get("source_update_step") != update_step
    ):
        raise HTTPException(400, "resume source update-step contract is inconsistent")
    if not isinstance(result, dict) or not all(
        _is_sha256(result.get(name))
        for name in ("actor_sha256", "critic_sha256", "frozen_policy_sha256")
    ):
        raise HTTPException(400, "resume source network hashes are invalid")

    return _ResolvedFlowSDEResume(
        checkpoint=checkpoint,
        source_job_id=source_job_id,
        source_output_dir=source_output,
        source_policy_checkpoint=source_policy,
        source_model_path=expected_model,
        lineage_policy_checkpoint=lineage_policy,
        task_instruction=task_instruction,
        robot_type=robot_type,
        ppo_config=dict(expected_config),
        update_step=update_step,
    )


def _discover_latest_online_job() -> _FlowSDEPPOJob | None:
    """Recover the newest verified resumable job after an API restart."""

    root = Path(os.path.abspath(str(FLOW_SDE_OUTPUT_ROOT)))
    if not root.is_dir() or root.is_symlink():
        return None
    try:
        candidates = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and _is_job_id(path.name)
                and (path / "run_manifest.json").is_file()
            ),
            key=lambda path: (path / "run_manifest.json").stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return None

    for output in candidates:
        try:
            manifest = _read_json_object(
                output / "run_manifest.json",
                name="online PPO run manifest",
            )
            config = manifest.get("ppo_config")
            if not isinstance(config, dict):
                continue
            task_instruction = str(manifest.get("task_instruction") or "")
            robot_type = str(manifest.get("robot_type") or "")
            resolved = _resolve_online_resume_checkpoint(
                str(output / "training_state" / "trainer_state.pt"),
                policy_checkpoint=_resolve_policy_checkpoint(
                    str(output / "pretrained_model")
                ),
                robot_type=robot_type,
                task_instruction=task_instruction,
                ppo_epochs=int(config["ppo_epochs"]),
                minibatch_size=int(config["minibatch_size"]),
                actor_learning_rate=float(config["actor_learning_rate"]),
                value_learning_rate=float(config["value_learning_rate"]),
            )
            summary = _read_json_object(output / "summary.json", name="online PPO summary")
            episodes = summary.get("episodes")
            if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
                continue
            source_lineage = manifest.get("source_lineage")
            source_lineage = source_lineage if isinstance(source_lineage, dict) else {}
            prior_resume = source_lineage.get("resume")
            prior_resume = prior_resume if isinstance(prior_resume, dict) else {}
            previous_checkpoint = prior_resume.get("checkpoint_path")
            previous_checkpoint = (
                previous_checkpoint if isinstance(previous_checkpoint, str) else ""
            )
            previous_job_id = ""
            if previous_checkpoint:
                try:
                    _, _, previous_job_id = _online_output_from_checkpoint(
                        previous_checkpoint
                    )
                except HTTPException:
                    previous_checkpoint = ""
            value_initialization = source_lineage.get("value_initialization")
            value_initialization = (
                value_initialization if isinstance(value_initialization, dict) else {}
            )
            warmup_bundle = value_initialization.get("bundle_path")
            warmup_bundle = warmup_bundle if isinstance(warmup_bundle, str) else ""
            return _FlowSDEPPOJob(
                job_id=resolved.source_job_id,
                policy_checkpoint=str(resolved.source_policy_checkpoint),
                robot_type=resolved.robot_type,
                task_instruction=resolved.task_instruction,
                output_dir=str(resolved.source_output_dir),
                control_file=str(resolved.source_output_dir / "control" / "outcome.json"),
                log_path=str(FLOW_SDE_LOG_ROOT / f"{resolved.source_job_id}.log"),
                episodes=episodes,
                ppo_epochs=int(config["ppo_epochs"]),
                minibatch_size=int(config["minibatch_size"]),
                max_chunk_decisions=20,
                actor_learning_rate=float(config["actor_learning_rate"]),
                value_learning_rate=float(config["value_learning_rate"]),
                ack_timeout_seconds=5.0,
                sensor_timeout_seconds=15.0,
                value_warmup_bundle=warmup_bundle,
                resume_checkpoint=previous_checkpoint,
                resume_source_job_id=previous_job_id,
                lineage_policy_checkpoint=str(resolved.lineage_policy_checkpoint),
                status="completed",
                phase="complete",
                percentage=100.0,
                checkpoint_path=str(resolved.checkpoint),
                model_path=str(resolved.source_model_path),
                episode=episodes,
                update_step=resolved.update_step,
                message="Recovered completed Flow-SDE PPO job",
                returncode=0,
            )
        except (HTTPException, KeyError, TypeError, ValueError, OSError):
            continue
    return None


def _resolve_value_warmup_bundle(
    raw_path: str,
    *,
    policy_checkpoint: Path,
    task_instruction: str,
    forbidden_output: Path | None = None,
) -> Path:
    """Validate one immutable completed warm-up bundle for online PPO.

    The bundle is treated as executable training input because its checkpoint
    is loaded with ``torch.load``.  It must therefore be a regular, readable
    artifact produced below the dedicated warm-up root, never a symlink or an
    arbitrary workspace path.
    """

    value = str(raw_path or "").strip()
    if not value or not Path(value).is_absolute():
        raise HTTPException(400, "value_warmup_bundle must be an absolute path")
    warmup_root = Path(os.path.abspath(str(FLOW_SDE_VALUE_WARMUP_ROOT)))
    lexical = Path(os.path.abspath(value))
    try:
        lexical.relative_to(warmup_root)
    except ValueError as exc:
        raise HTTPException(
            400,
            f"value_warmup_bundle must be under {warmup_root}",
        ) from exc
    if _has_symlink_component(warmup_root, lexical):
        raise HTTPException(
            400,
            "value_warmup_bundle must not contain symbolic links",
        )
    try:
        root_resolved = warmup_root.resolve(strict=True)
        candidate = lexical.resolve(strict=True)
        candidate.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            400,
            "value_warmup_bundle does not exist or escapes its root",
        ) from exc
    if not candidate.is_dir():
        raise HTTPException(400, "value_warmup_bundle must be a real directory")
    if forbidden_output is not None and _paths_overlap(candidate, forbidden_output):
        raise HTTPException(
            400,
            "value_warmup_bundle cannot be the current online PPO output",
        )

    expected_model = candidate / "pretrained_model"
    expected_checkpoint = candidate / "training_state" / "value_warmup.pt"
    expected_manifest = candidate / "run_manifest.json"
    expected_progress = candidate / "progress.jsonl"
    policy_artifacts = tuple(
        expected_model / name for name in FLOW_SDE_POLICY_ARTIFACTS
    )
    required_files = policy_artifacts + (
        expected_checkpoint,
        expected_manifest,
        expected_progress,
    )
    if any(_has_symlink_component(candidate, path) for path in required_files):
        raise HTTPException(
            400,
            "value_warmup_bundle artifacts must not contain symbolic links",
        )
    if any(path.is_symlink() or not path.is_file() for path in required_files):
        raise HTTPException(
            400,
            "value_warmup_bundle is incomplete or has non-regular artifacts",
        )
    if any(not _training_user_can_read(path) for path in required_files):
        raise HTTPException(
            400,
            "value_warmup_bundle is not readable by the LeRobot training user",
        )
    required_directories = (candidate, expected_model, expected_checkpoint.parent)
    if any(
        not path.is_dir() or not _training_user_can_execute(path)
        for path in required_directories
    ):
        raise HTTPException(
            400,
            "value_warmup_bundle is not traversable by the LeRobot training user",
        )
    if expected_checkpoint.stat().st_size <= 0:
        raise HTTPException(400, "value_warmup_bundle critic checkpoint is empty")

    try:
        manifest = json.loads(expected_manifest.read_text(encoding="utf-8"))
        model_config = json.loads(
            (expected_model / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            400,
            "value_warmup_bundle manifest or policy config is invalid",
        ) from exc
    if not isinstance(manifest, dict):
        raise HTTPException(400, "value_warmup_bundle manifest must be an object")
    if manifest.get("format") != "cyclo.flow_sde_ppo.value_warmup.bundle.v1":
        raise HTTPException(400, "value_warmup_bundle has an unsupported format")
    if manifest.get("status") != "complete":
        raise HTTPException(400, "value_warmup_bundle is not complete")
    if not isinstance(model_config, dict) or model_config.get("type") != "multi_task_dit":
        raise HTTPException(400, "value_warmup_bundle policy is not MultiTaskDiT")
    if manifest.get("artifacts") != {
        "model_path": "pretrained_model",
        "checkpoint_path": "training_state/value_warmup.pt",
        "progress_path": "progress.jsonl",
    }:
        raise HTTPException(400, "value_warmup_bundle artifact contract is invalid")

    base = manifest.get("base")
    if not isinstance(base, dict) or base.get("path") != str(policy_checkpoint):
        raise HTTPException(
            400,
            "value_warmup_bundle was trained from a different policy checkpoint",
        )
    config = manifest.get("config")
    if not isinstance(config, dict) or config.get("task_instruction") != task_instruction:
        raise HTTPException(
            400,
            "value_warmup_bundle task instruction does not match online PPO",
        )
    result = manifest.get("result")
    base_policy_hash = base.get("policy_sha256")
    if (
        not isinstance(result, dict)
        or not isinstance(base_policy_hash, str)
        or result.get("policy_sha256_before") != base_policy_hash
        or result.get("policy_sha256_after") != base_policy_hash
    ):
        raise HTTPException(
            400,
            "value_warmup_bundle policy integrity contract is invalid",
        )

    base_artifacts = base.get("artifacts")
    if not isinstance(base_artifacts, dict) or set(base_artifacts) != set(
        FLOW_SDE_POLICY_ARTIFACTS
    ):
        raise HTTPException(
            400,
            "value_warmup_bundle base artifact hashes are incomplete",
        )
    for name in FLOW_SDE_POLICY_ARTIFACTS:
        expected_hash = base_artifacts.get(name)
        try:
            source_hash = _sha256_file(policy_checkpoint / name)
            bundled_hash = _sha256_file(expected_model / name)
        except OSError as exc:
            raise HTTPException(
                400,
                "value_warmup_bundle policy artifacts cannot be hashed",
            ) from exc
        if expected_hash != source_hash:
            raise HTTPException(
                400,
                "value_warmup_bundle base policy is stale or was modified",
            )
        if expected_hash != bundled_hash:
            raise HTTPException(
                400,
                "value_warmup_bundle copied policy artifacts failed integrity check",
            )
    return candidate


def _discover_latest_value_warmup_job() -> _FlowSDEValueWarmupJob | None:
    """Recover the newest compatible completed warm-up after API restart."""

    policy_root = Path(os.path.abspath(str(FLOW_SDE_POLICY_ROOTS[0])))
    warmup_root = Path(os.path.abspath(str(FLOW_SDE_VALUE_WARMUP_ROOT)))
    try:
        warmup_root.relative_to(policy_root)
        policy_root.resolve(strict=True)
        root_resolved = warmup_root.resolve(strict=True)
    except (OSError, ValueError):
        return None
    if (
        _has_symlink_component(policy_root, warmup_root)
        or not root_resolved.is_dir()
    ):
        return None
    try:
        candidates = sorted(
            (
                path
                for path in root_resolved.iterdir()
                if path.is_dir()
                and not path.is_symlink()
                and len(path.name) == 32
                and all(character in "0123456789abcdef" for character in path.name)
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return None

    for candidate in candidates:
        manifest_path = candidate / "run_manifest.json"
        if (
            _has_symlink_component(candidate, manifest_path)
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            base = manifest.get("base") if isinstance(manifest, dict) else None
            config = manifest.get("config") if isinstance(manifest, dict) else None
            if not isinstance(base, dict) or not isinstance(config, dict):
                continue
            task_instruction = _safe_instruction(config.get("task_instruction"))
            policy_checkpoint = _resolve_policy_checkpoint(base.get("path"))
            bundle = _resolve_value_warmup_bundle(
                str(candidate),
                policy_checkpoint=policy_checkpoint,
                task_instruction=task_instruction,
            )
            result = manifest.get("result")
            datasets = manifest.get("datasets")
            if not isinstance(result, dict):
                continue
            dataset_paths = [
                str(entry["path"])
                for entry in datasets
                if isinstance(entry, dict) and isinstance(entry.get("path"), str)
            ] if isinstance(datasets, list) else []
            total_steps = int(config.get("steps") or 0)
            completed_steps = int(result.get("completed_steps") or total_steps)
            batch_size = int(config.get("batch_size") or 0)
            value_learning_rate = float(config.get("value_lr") or 0.0)
            discount = float(config.get("gamma") or 0.0)
            final_value_loss = result.get("final_value_loss")
            value_loss = (
                float(final_value_loss)
                if isinstance(final_value_loss, (int, float))
                and not isinstance(final_value_loss, bool)
                else None
            )
        except (HTTPException, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        return _FlowSDEValueWarmupJob(
            job_id=bundle.name,
            policy_checkpoint=str(policy_checkpoint),
            dataset_paths=dataset_paths,
            task_instruction=task_instruction,
            output_dir=str(bundle),
            log_path=str(FLOW_SDE_LOG_ROOT / f"value_warmup_{bundle.name}.log"),
            total_steps=total_steps,
            batch_size=batch_size,
            value_learning_rate=value_learning_rate,
            discount=discount,
            status="completed",
            phase="complete",
            percentage=100.0,
            bundle_path=str(bundle),
            checkpoint_path=str(bundle / "training_state" / "value_warmup.pt"),
            model_path=str(bundle / "pretrained_model"),
            step=completed_steps,
            value_loss=value_loss,
            message="Recovered completed Flow-SDE PPO value warm-up bundle",
            returncode=0,
        )
    return None


def _flow_sde_value_warmup_output_path(job_id: str) -> Path:
    """Resolve a fresh value-warm-up bundle directory without symlinks."""

    policy_root = Path(os.path.abspath(str(FLOW_SDE_POLICY_ROOTS[0])))
    warmup_root = Path(os.path.abspath(str(FLOW_SDE_VALUE_WARMUP_ROOT)))
    try:
        warmup_root.relative_to(policy_root)
        policy_root_resolved = policy_root.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "Flow-SDE PPO value warm-up root is invalid") from exc
    if _has_symlink_component(policy_root, warmup_root):
        raise HTTPException(
            500,
            "Flow-SDE PPO value warm-up root must not contain symbolic links",
        )
    if warmup_root.exists():
        try:
            warmup_root.resolve(strict=True).relative_to(policy_root_resolved)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                500,
                "Flow-SDE PPO value warm-up root escapes its policy root",
            ) from exc
        if not warmup_root.is_dir():
            raise HTTPException(
                500,
                "Flow-SDE PPO value warm-up root must be a directory",
            )
    output = warmup_root / job_id
    if output.exists():
        raise HTTPException(409, f"Flow-SDE PPO value warm-up output exists: {output}")
    return output


def _prepare_value_warmup_directory(output_dir: Path) -> None:
    """Prepare only the parent; the CLI atomically creates ``output_dir``."""

    warmup_root = output_dir.parent
    created = not warmup_root.exists()
    try:
        warmup_root.mkdir(parents=True, exist_ok=True)
        os.chown(warmup_root, FLOW_SDE_TRAIN_UID, FLOW_SDE_TRAIN_GID)
    except OSError as exc:
        if created:
            try:
                warmup_root.rmdir()
            except OSError:
                pass
        raise HTTPException(
            503,
            f"Could not create writable Flow-SDE PPO value warm-up output: {exc}",
        ) from exc


def _flow_sde_output_path(job_id: str) -> Path:
    """Resolve a fresh job directory without following workspace symlinks."""

    policy_root = FLOW_SDE_POLICY_ROOTS[0]
    output_root = Path(os.path.abspath(str(FLOW_SDE_OUTPUT_ROOT)))
    try:
        output_root.relative_to(policy_root)
        policy_root_resolved = policy_root.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "Flow-SDE PPO output root is invalid") from exc
    if _has_symlink_component(policy_root, output_root):
        raise HTTPException(500, "Flow-SDE PPO output root must not contain symbolic links")
    if output_root.exists():
        try:
            output_root.resolve(strict=True).relative_to(policy_root_resolved)
        except (OSError, ValueError) as exc:
            raise HTTPException(500, "Flow-SDE PPO output root escapes its policy root") from exc
        if not output_root.is_dir():
            raise HTTPException(500, "Flow-SDE PPO output root must be a directory")
    output = output_root / job_id
    if output.exists():
        raise HTTPException(409, f"Flow-SDE PPO output already exists: {output}")
    return output


def _remove_empty_job_directories(output_dir: Path, control_dir: Path) -> None:
    """Best-effort cleanup restricted to the two freshly-created directories."""

    for path in (control_dir, output_dir):
        try:
            path.rmdir()
        except OSError:
            pass


def _prepare_job_directories(output_dir: Path) -> Path:
    """Create the job tree and transfer it to the uid used by LeRobot."""

    control_dir = output_dir / "control"
    try:
        control_dir.mkdir(parents=True, exist_ok=False)
        os.chown(output_dir, FLOW_SDE_TRAIN_UID, FLOW_SDE_TRAIN_GID)
        os.chown(control_dir, FLOW_SDE_TRAIN_UID, FLOW_SDE_TRAIN_GID)
    except OSError as exc:
        _remove_empty_job_directories(output_dir, control_dir)
        raise HTTPException(
            503,
            f"Could not create writable Flow-SDE PPO output: {exc}",
        ) from exc
    return control_dir


def _safe_robot_type(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or any(
        not (character.isalnum() or character in "_.-")
        for character in normalized
    ):
        raise HTTPException(400, "robot_type contains unsupported characters")
    return normalized


def _safe_instruction(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise HTTPException(400, "task_instruction is required")
    if len(normalized) > 1000:
        raise HTTPException(400, "task_instruction is too long")
    return normalized


def _flow_sde_ros_domain_id() -> str:
    raw_value = str(os.environ.get("ROS_DOMAIN_ID") or "30").strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise HTTPException(500, "ROS_DOMAIN_ID must be an integer") from exc
    if value < 0 or value > 232:
        raise HTTPException(500, "ROS_DOMAIN_ID must be between 0 and 232")
    return str(value)


class FlowSDEPPOSupervisor:
    """Own exactly one live PPO subprocess and its sparse outcome channel."""

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
        self._job: _FlowSDEPPOJob | None = None
        self._value_warmup_job: _FlowSDEValueWarmupJob | None = None
        self.router = APIRouter(prefix="/flow-sde-ppo", tags=["flow-sde-ppo"])
        self.router.add_api_route(
            "/start",
            self.start,
            methods=["POST"],
            response_model=FlowSDEPPOStatus,
        )
        self.router.add_api_route(
            "/status",
            self.status,
            methods=["GET"],
            response_model=FlowSDEPPOStatus,
        )
        self.router.add_api_route(
            "/stop",
            self.stop,
            methods=["POST"],
            response_model=FlowSDEPPOStatus,
        )
        self.router.add_api_route(
            "/outcome",
            self.outcome,
            methods=["POST"],
            response_model=FlowSDEPPOStatus,
        )
        self.router.add_api_route(
            "/value-warmup/start",
            self.start_value_warmup,
            methods=["POST"],
            response_model=FlowSDEValueWarmupStatus,
        )
        self.router.add_api_route(
            "/value-warmup/status",
            self.value_warmup_status,
            methods=["GET"],
            response_model=FlowSDEValueWarmupStatus,
        )
        self.router.add_api_route(
            "/value-warmup/stop",
            self.stop_value_warmup,
            methods=["POST"],
            response_model=FlowSDEValueWarmupStatus,
        )

    @staticmethod
    def _container_name(job: _FlowSDEPPOJob) -> str:
        return f"cyclo_flow_sde_ppo_{job.job_id[:12]}"

    @staticmethod
    def _value_warmup_container_name(job: _FlowSDEValueWarmupJob) -> str:
        return f"cyclo_flow_sde_value_warmup_{job.job_id[:12]}"

    def _command(self, job: _FlowSDEPPOJob) -> list[str]:
        command = self._compose_command() + [
            "run",
            "--rm",
            "--no-deps",
            "--pull",
            "never",
            "--name",
            self._container_name(job),
            "--user",
            "1000:1000",
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/tmp",
            "--env",
            f"XDG_CACHE_HOME={FLOW_SDE_CACHE_ROOT}",
            "--env",
            f"HF_HOME={FLOW_SDE_CACHE_ROOT}/huggingface",
            "--env",
            f"HF_LEROBOT_HOME={FLOW_SDE_CACHE_ROOT}/huggingface/lerobot",
            "--env",
            "HF_HUB_CACHE=/huggingface_hub",
            "--env",
            "HUGGINGFACE_HUB_CACHE=/huggingface_hub",
            "--env",
            "TRANSFORMERS_CACHE=/huggingface_hub",
            "--env",
            f"TORCH_HOME={FLOW_SDE_CACHE_ROOT}/torch",
            "--env",
            f"TRITON_CACHE_DIR={FLOW_SDE_CACHE_ROOT}/triton",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "HF_DATASETS_OFFLINE=1",
            "--env",
            "ZENOH_SDK_PATH=/zenoh_sdk",
            "--env",
            "ZENOH_ROS2_SDK_CACHE=/zenoh_cache",
            "--env",
            f"ROS_DOMAIN_ID={_flow_sde_ros_domain_id()}",
            "--env",
            "ROBOT_CLIENT_SDK_PATH=/robot_client_sdk",
            "--env",
            "PYTHONPATH=/cyclo_brain_src:/robot_client_sdk:/app:/policy_runtime",
            "--entrypoint",
            "/lerobot/.venv/bin/python",
            "lerobot",
            "-m",
            "cyclo_brain.algorithm.rl.flow_sde_ppo.live_cli",
            "--base-checkpoint",
            job.policy_checkpoint,
            "--output-dir",
            job.output_dir,
            "--job-id",
            job.job_id,
            "--control-file",
            job.control_file,
            "--robot-type",
            job.robot_type,
            "--task-instruction",
            job.task_instruction,
            "--episodes",
            str(job.episodes),
            "--ppo-epochs",
            str(job.ppo_epochs),
            "--minibatch-size",
            str(job.minibatch_size),
            "--max-chunk-decisions",
            str(job.max_chunk_decisions),
            "--actor-lr",
            str(job.actor_learning_rate),
            "--value-lr",
            str(job.value_learning_rate),
            "--ack-timeout",
            str(job.ack_timeout_seconds),
            "--sensor-timeout",
            str(job.sensor_timeout_seconds),
        ]
        if job.value_warmup_bundle:
            command.extend(("--value-warmup-bundle", job.value_warmup_bundle))
        if job.resume_checkpoint:
            command.extend(("--resume-checkpoint", job.resume_checkpoint))
        return command

    def _value_warmup_command(self, job: _FlowSDEValueWarmupJob) -> list[str]:
        command = self._compose_command() + [
            "run",
            "--rm",
            "--no-deps",
            "--pull",
            "never",
            "--name",
            self._value_warmup_container_name(job),
            "--user",
            "1000:1000",
            "--workdir",
            "/workspace",
            "--env",
            "HOME=/tmp",
            "--env",
            f"XDG_CACHE_HOME={FLOW_SDE_CACHE_ROOT}",
            "--env",
            f"HF_HOME={FLOW_SDE_CACHE_ROOT}/huggingface",
            "--env",
            f"HF_LEROBOT_HOME={FLOW_SDE_CACHE_ROOT}/huggingface/lerobot",
            "--env",
            "HF_HUB_CACHE=/huggingface_hub",
            "--env",
            "HUGGINGFACE_HUB_CACHE=/huggingface_hub",
            "--env",
            "TRANSFORMERS_CACHE=/huggingface_hub",
            "--env",
            f"TORCH_HOME={FLOW_SDE_CACHE_ROOT}/torch",
            "--env",
            f"TRITON_CACHE_DIR={FLOW_SDE_CACHE_ROOT}/triton",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "HF_DATASETS_OFFLINE=1",
            "--env",
            "PYTHONPATH=/cyclo_brain_src:/app:/policy_runtime",
            "--entrypoint",
            "/lerobot/.venv/bin/python",
            "lerobot",
            "-m",
            "cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup_cli",
            "--base-checkpoint",
            job.policy_checkpoint,
        ]
        for dataset_path in job.dataset_paths:
            command.extend(("--dataset-root", dataset_path))
        command.extend(
            (
                "--output-dir",
                job.output_dir,
                "--steps",
                str(job.total_steps),
                "--batch-size",
                str(job.batch_size),
                "--value-lr",
                str(job.value_learning_rate),
                "--gamma",
                str(job.discount),
                "--task-instruction",
                job.task_instruction,
                "--seed",
                "17",
                "--device",
                "cuda",
            )
        )
        return command

    @staticmethod
    def _status(job: _FlowSDEPPOJob | None) -> FlowSDEPPOStatus:
        if job is None:
            return FlowSDEPPOStatus(
                status="idle",
                message=(
                    "Flow-SDE PPO is ready; start cyclo_lab with atomic "
                    "action-step transport before training"
                ),
            )
        return FlowSDEPPOStatus(
            status=job.status,
            phase=job.phase,
            percentage=job.percentage,
            job_id=job.job_id,
            policy_checkpoint=job.policy_checkpoint,
            lineage_policy_checkpoint=job.lineage_policy_checkpoint,
            value_warmup_bundle=job.value_warmup_bundle,
            resume_checkpoint=job.resume_checkpoint,
            resume_source_job_id=job.resume_source_job_id,
            task_instruction=job.task_instruction,
            output_dir=job.output_dir,
            checkpoint_path=job.checkpoint_path,
            model_path=job.model_path if job.status == "completed" else "",
            episode=job.episode,
            episodes=job.episodes,
            chunk_decisions=job.chunk_decisions,
            update_step=job.update_step,
            episode_return=job.episode_return,
            actor_loss=job.actor_loss,
            value_loss=job.value_loss,
            approx_kl=job.approx_kl,
            clip_fraction=job.clip_fraction,
            eta_seconds=job.eta_seconds,
            awaiting_outcome=job.awaiting_outcome,
            message=job.message,
            returncode=job.returncode,
            log_tail=list(job.log_tail),
        )

    @staticmethod
    def _value_warmup_status(
        job: _FlowSDEValueWarmupJob | None,
    ) -> FlowSDEValueWarmupStatus:
        if job is None:
            return FlowSDEValueWarmupStatus(
                status="idle",
                message="Flow-SDE PPO value warm-up is ready",
            )
        completed = job.status == "completed"
        return FlowSDEValueWarmupStatus(
            status=job.status,
            phase=job.phase,
            percentage=job.percentage,
            job_id=job.job_id,
            policy_checkpoint=job.policy_checkpoint,
            dataset_paths=list(job.dataset_paths),
            task_instruction=job.task_instruction,
            output_dir=job.output_dir,
            bundle_path=job.bundle_path if completed else "",
            checkpoint_path=job.checkpoint_path if completed else "",
            model_path=job.model_path if completed else "",
            step=job.step,
            total_steps=job.total_steps,
            batch_size=job.batch_size,
            value_learning_rate=job.value_learning_rate,
            discount=job.discount,
            value_loss=job.value_loss,
            eta_seconds=job.eta_seconds,
            message=job.message,
            returncode=job.returncode,
            log_tail=list(job.log_tail),
        )

    @staticmethod
    def _append_value_warmup_log(job: _FlowSDEValueWarmupJob, line: str) -> None:
        if not line:
            return
        job.log_tail.append(line)
        del job.log_tail[:-FLOW_SDE_LOG_LINES]

    @staticmethod
    def _value_warmup_number(
        job: _FlowSDEValueWarmupJob,
        payload: dict[str, Any],
        name: str,
    ) -> None:
        value = payload.get(name)
        current = getattr(job, name)
        if isinstance(current, int) and not isinstance(current, bool):
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                setattr(job, name, value)
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(job, name, float(value))

    def _consume_value_warmup_event(
        self,
        job: _FlowSDEValueWarmupJob,
        payload: dict[str, Any],
    ) -> bool:
        event = str(payload.get("event") or "")
        if event in {"starting", "manifest"}:
            phase = payload.get("phase", payload.get("stage"))
            if isinstance(phase, str) and phase:
                job.phase = phase
            else:
                job.phase = "loading_data"
            message = payload.get("message")
            job.message = (
                message
                if isinstance(message, str) and message
                else "Loading the policy and labeled LeRobot datasets"
            )
            total_steps = payload.get("total_steps", payload.get("steps"))
            if (
                isinstance(total_steps, int)
                and not isinstance(total_steps, bool)
                and total_steps > 0
            ):
                job.total_steps = total_steps
            return False
        if event == "progress":
            phase = payload.get("phase")
            if isinstance(phase, str) and phase:
                job.phase = phase
            else:
                job.phase = "training_value"
            message = payload.get("message")
            if isinstance(message, str) and message:
                job.message = message
            else:
                job.message = "Warming up the Flow-SDE PPO value network"
            step = payload.get("step", payload.get("completed_steps"))
            if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
                job.step = step
            total_steps = payload.get("total_steps")
            if (
                isinstance(total_steps, int)
                and not isinstance(total_steps, bool)
                and total_steps > 0
            ):
                job.total_steps = total_steps
            self._value_warmup_number(job, payload, "value_loss")
            self._value_warmup_number(job, payload, "eta_seconds")
            percentage = payload.get("percentage")
            if isinstance(percentage, (int, float)) and not isinstance(
                percentage,
                bool,
            ):
                job.percentage = max(0.0, min(99.0, float(percentage)))
            elif job.total_steps > 0:
                job.percentage = max(
                    0.0,
                    min(99.0, 100.0 * float(job.step) / job.total_steps),
                )
            return False

        terminal_complete = (
            event == "result" and payload.get("status") == "complete"
        ) or (
            event == "completed" and payload.get("status") == "completed"
        )
        if terminal_complete:
            bundle_path = payload.get("bundle_path")
            model_path = payload.get("model_path", payload.get("pretrained_model"))
            checkpoint_path = payload.get(
                "checkpoint_path",
                payload.get("trainer_checkpoint"),
            )
            if isinstance(bundle_path, str):
                job.bundle_path = bundle_path
            if isinstance(model_path, str):
                job.model_path = model_path
            if isinstance(checkpoint_path, str):
                job.checkpoint_path = checkpoint_path
            completed_steps = payload.get("completed_steps", payload.get("step"))
            if (
                isinstance(completed_steps, int)
                and not isinstance(completed_steps, bool)
                and completed_steps >= 0
            ):
                job.step = completed_steps
            job.phase = "verifying"
            job.message = "Verifying the Flow-SDE PPO value warm-up bundle"
            return bool(job.bundle_path and job.model_path and job.checkpoint_path)

        stopped = event in {"stopped", "cancelled"} or (
            event == "result" and payload.get("status") == "stopped"
        )
        if stopped:
            job.stop_confirmed = True
            job.phase = "stopping"
            job.message = str(
                payload.get("message") or "Flow-SDE PPO value warm-up stopped"
            )
            return False
        if event in {"failed", "error"}:
            job.phase = "error"
            error_type = payload.get("error_type")
            message = str(
                payload.get("message") or "Flow-SDE PPO value warm-up failed"
            )
            job.message = f"{error_type}: {message}" if error_type else message
        return False

    @staticmethod
    def _verified_value_warmup_bundle(job: _FlowSDEValueWarmupJob) -> bool:
        output = Path(job.output_dir)
        expected_model = output / "pretrained_model"
        expected_checkpoint = output / "training_state" / "value_warmup.pt"
        expected_manifest = output / "run_manifest.json"
        expected_progress = output / "progress.jsonl"
        expected_paths = (
            (job.bundle_path, output),
            (job.model_path, expected_model),
            (job.checkpoint_path, expected_checkpoint),
        )
        if any(os.path.normpath(actual) != str(expected) for actual, expected in expected_paths):
            return False
        if any(
            _has_symlink_component(output, path)
            for path in (
                expected_model,
                expected_checkpoint,
                expected_manifest,
                expected_progress,
            )
        ):
            return False
        required_policy = tuple(expected_model / name for name in FLOW_SDE_POLICY_ARTIFACTS)
        required_files = required_policy + (
            expected_checkpoint,
            expected_manifest,
            expected_progress,
        )
        if any(path.is_symlink() or not path.is_file() for path in required_files):
            return False
        try:
            config = json.loads(
                (expected_model / "config.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(expected_manifest.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            return False
        if not isinstance(config, dict) or config.get("type") != "multi_task_dit":
            return False
        if not isinstance(manifest, dict):
            return False
        if manifest.get("format") != "cyclo.flow_sde_ppo.value_warmup.bundle.v1":
            return False
        if manifest.get("status") != "complete":
            return False
        artifacts = manifest.get("artifacts")
        return isinstance(artifacts, dict) and artifacts == {
            "model_path": "pretrained_model",
            "checkpoint_path": "training_state/value_warmup.pt",
            "progress_path": "progress.jsonl",
        }

    def _monitor_value_warmup(self, job: _FlowSDEValueWarmupJob) -> None:
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
                        )
                        line = line.rstrip("\r\n")
                        log.write(line + "\n")
                        log.flush()
                        with self._lock:
                            self._append_value_warmup_log(job, line)
                            try:
                                payload = json.loads(line)
                            except (TypeError, json.JSONDecodeError):
                                continue
                            if isinstance(payload, dict):
                                result_complete = (
                                    self._consume_value_warmup_event(job, payload)
                                    or result_complete
                                )
                returncode = job.process.wait() if job.process is not None else -1
        except Exception as exc:  # pragma: no cover - worker boundary
            returncode = -1
            with self._lock:
                job.message = f"Flow-SDE PPO value warm-up monitor failed: {exc}"

        with self._lock:
            job.returncode = returncode
            expected_stop_returncodes = {0, -signal.SIGINT, 128 + signal.SIGINT}
            if (
                job.stop_requested and returncode in expected_stop_returncodes
            ) or (returncode == 0 and job.stop_confirmed):
                job.status = "stopped"
                job.phase = "stopped"
                job.bundle_path = ""
                job.checkpoint_path = ""
                job.model_path = ""
                job.message = "Flow-SDE PPO value warm-up stopped"
            elif (
                returncode == 0
                and result_complete
                and self._verified_value_warmup_bundle(job)
            ):
                job.status = "completed"
                job.phase = "complete"
                job.percentage = 100.0
                job.step = job.total_steps
                job.message = "Flow-SDE PPO value warm-up completed"
            else:
                job.status = "failed"
                job.phase = "error"
                if not job.message or job.message == "Starting Flow-SDE PPO value warm-up":
                    detail = job.log_tail[-1] if job.log_tail else ""
                    reason = (
                        f"exited with code {returncode}"
                        if returncode != 0
                        else "result or verified bundle is missing"
                    )
                    job.message = f"Flow-SDE PPO value warm-up {reason}"
                    if detail:
                        job.message += f": {detail}"

    @staticmethod
    def _append_log(job: _FlowSDEPPOJob, line: str) -> None:
        if not line:
            return
        job.log_tail.append(line)
        del job.log_tail[:-FLOW_SDE_LOG_LINES]

    @staticmethod
    def _number(job: _FlowSDEPPOJob, payload: dict[str, Any], name: str) -> None:
        value = payload.get(name)
        current = getattr(job, name)
        if isinstance(current, int) and not isinstance(current, bool):
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                setattr(job, name, value)
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(job, name, float(value))

    def _consume_event(self, job: _FlowSDEPPOJob, payload: dict[str, Any]) -> bool:
        event = str(payload.get("event") or "")
        if event == "starting":
            job.phase = str(payload.get("stage") or "starting")
            job.message = "Loading the MultiTaskDiT checkpoint"
            return False
        if event == "ready":
            job.phase = str(payload.get("stage") or "rollout")
            job.message = "Flow-SDE PPO is ready to collect an episode"
            return False
        if event == "episode_started":
            self._number(job, payload, "episode")
            job.chunk_decisions = 0
            job.awaiting_outcome = True
            job.phase = "collecting"
            job.message = "Episode is running; submit Success, Fail, or Cancel"
            if job.episodes > 0:
                job.percentage = max(
                    0.0,
                    min(99.0, 100.0 * float(job.episode - 1) / job.episodes),
                )
            return False
        if event == "episode_updated":
            self._number(job, payload, "episode")
            self._number(job, payload, "episode_return")
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                metric_names = (
                    "update_step",
                    "actor_loss",
                    "value_loss",
                    "approx_kl",
                    "clip_fraction",
                )
                for name in metric_names:
                    self._number(job, metrics, name)
                transitions = metrics.get("transitions")
                if (
                    isinstance(transitions, int)
                    and not isinstance(transitions, bool)
                    and transitions >= 0
                ):
                    job.chunk_decisions = transitions
            checkpoint = payload.get("checkpoint")
            if isinstance(checkpoint, str):
                job.checkpoint_path = checkpoint
            job.awaiting_outcome = False
            job.phase = "updating"
            job.message = "Episode collected and PPO update completed"
            if job.episodes > 0:
                job.percentage = max(
                    0.0,
                    min(99.0, 100.0 * float(job.episode) / job.episodes),
                )
            return False
        if event == "episode_cancelled":
            job.awaiting_outcome = False
            job.phase = "resetting"
            job.message = "Episode discarded; resetting for a fresh rollout"
            return False
        if event == "completed":
            job.awaiting_outcome = False
            job.phase = "exporting"
            job.message = "Verifying the exported MultiTaskDiT policy"
            model_path = payload.get("pretrained_model")
            if isinstance(model_path, str):
                job.model_path = model_path
            checkpoint = payload.get("trainer_checkpoint")
            if isinstance(checkpoint, str):
                job.checkpoint_path = checkpoint
            updates = payload.get("updates")
            if isinstance(updates, int) and not isinstance(updates, bool) and updates >= 0:
                job.update_step = updates
            return payload.get("status") == "completed" and bool(job.model_path)
        if event in {"cancelled", "stopped"}:
            job.awaiting_outcome = False
            job.stop_confirmed = True
            job.phase = "stopping"
            fallback = (
                "Flow-SDE PPO training stopped"
                if event == "stopped"
                else "Flow-SDE PPO was cancelled"
            )
            job.message = str(payload.get("message") or fallback)
            return False
        if event == "failed":
            job.awaiting_outcome = False
            job.phase = "error"
            error_type = str(payload.get("error_type") or "FlowSDEPPOError")
            message = str(payload.get("message") or "Flow-SDE PPO failed")
            job.message = f"{error_type}: {message}"
            return False

        # Retain the original generic event contract for compatibility with
        # older standalone runners and diagnostic fixtures.
        if event in {"manifest", "progress"}:
            phase = payload.get("phase")
            if isinstance(phase, str) and phase:
                job.phase = phase
            message = payload.get("message")
            if isinstance(message, str) and message:
                job.message = message
            for name in (
                "episode",
                "chunk_decisions",
                "update_step",
                "episode_return",
                "actor_loss",
                "value_loss",
                "approx_kl",
                "clip_fraction",
                "eta_seconds",
            ):
                self._number(job, payload, name)
            percentage = payload.get("percentage")
            if isinstance(percentage, (int, float)) and not isinstance(percentage, bool):
                job.percentage = max(0.0, min(100.0, float(percentage)))
            awaiting = payload.get("awaiting_outcome")
            if isinstance(awaiting, bool):
                job.awaiting_outcome = awaiting
            checkpoint_path = payload.get("checkpoint_path")
            if isinstance(checkpoint_path, str):
                job.checkpoint_path = checkpoint_path
            return False
        if event == "result":
            self._consume_event(job, {**payload, "event": "progress"})
            model_path = payload.get("model_path")
            if isinstance(model_path, str):
                job.model_path = model_path
            if payload.get("status") == "stopped":
                job.stop_confirmed = True
            return payload.get("status") == "complete" and bool(job.model_path)
        if event == "error":
            job.phase = "error"
            job.message = str(payload.get("message") or "Flow-SDE PPO failed")
        return False

    @staticmethod
    def _verified_model(job: _FlowSDEPPOJob) -> bool:
        expected = Path(job.output_dir) / "pretrained_model"
        if os.path.normpath(job.model_path) != str(expected):
            return False
        output = Path(job.output_dir)
        if _has_symlink_component(output, expected) or not expected.is_dir():
            return False
        required = (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        for name in required:
            path = expected / name
            if path.is_symlink() or not path.is_file():
                return False
        try:
            config = json.loads((expected / "config.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            return False
        return isinstance(config, dict) and config.get("type") == "multi_task_dit"

    @staticmethod
    def _verified_completed_bundle(job: _FlowSDEPPOJob) -> bool:
        """Verify the deployable actor and resumable actor+critic state together."""

        if not FlowSDEPPOSupervisor._verified_model(job):
            return False
        output = Path(job.output_dir)
        expected_model = output / "pretrained_model"
        expected_checkpoint = output / "training_state" / "trainer_state.pt"
        if (
            os.path.normpath(job.model_path) != str(expected_model)
            or os.path.normpath(job.checkpoint_path) != str(expected_checkpoint)
        ):
            return False
        try:
            resume = _resolve_online_resume_checkpoint(
                str(expected_checkpoint),
                policy_checkpoint=expected_model,
                robot_type=job.robot_type,
                task_instruction=job.task_instruction,
                ppo_epochs=job.ppo_epochs,
                minibatch_size=job.minibatch_size,
                actor_learning_rate=job.actor_learning_rate,
                value_learning_rate=job.value_learning_rate,
            )
        except (HTTPException, OSError, TypeError, ValueError):
            return False
        expected_lineage = Path(
            job.lineage_policy_checkpoint or job.policy_checkpoint
        ).resolve(strict=True)
        return (
            resume.source_job_id == job.job_id
            and resume.source_model_path == expected_model.resolve(strict=True)
            and resume.checkpoint == expected_checkpoint.resolve(strict=True)
            and resume.update_step == job.update_step
            and resume.lineage_policy_checkpoint == expected_lineage
        )

    def _monitor(self, job: _FlowSDEPPOJob) -> None:
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
                        )
                        line = line.rstrip("\r\n")
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
                job.message = f"Flow-SDE PPO monitor failed: {exc}"

        bundle_verified = (
            returncode == 0
            and result_complete
            and self._verified_completed_bundle(job)
        )
        with self._lock:
            job.returncode = returncode
            job.awaiting_outcome = False
            expected_stop_returncodes = {0, -signal.SIGINT, 128 + signal.SIGINT}
            if (
                job.stop_requested
                and returncode in expected_stop_returncodes
            ) or (returncode == 0 and job.stop_confirmed):
                job.status = "stopped"
                job.phase = "stopped"
                job.model_path = ""
                job.message = "Flow-SDE PPO training stopped"
            elif bundle_verified:
                job.status = "completed"
                job.phase = "complete"
                job.percentage = 100.0
                job.message = "Flow-SDE PPO training completed"
            else:
                job.status = "failed"
                job.phase = "error"
                if not job.message or job.message == "Starting Flow-SDE PPO":
                    job.message = (
                        f"Flow-SDE PPO exited with code {returncode}"
                        if returncode != 0
                        else (
                            "Flow-SDE PPO result or verified actor+critic bundle "
                            "is missing"
                        )
                    )

    def start(self, request: FlowSDEPPOStartRequest) -> FlowSDEPPOStatus:
        conflict = self._conflict_message()
        if conflict:
            raise HTTPException(409, conflict)
        checkpoint = _resolve_policy_checkpoint(request.policy_checkpoint)
        robot_type = _safe_robot_type(request.robot_type)
        task_instruction = _safe_instruction(request.task_instruction)
        if (
            request.value_warmup_bundle is not None
            and request.resume_checkpoint is not None
        ):
            raise HTTPException(
                400,
                "resume_checkpoint and value_warmup_bundle are mutually exclusive",
            )
        value_warmup_bundle: Path | None = None
        if request.value_warmup_bundle is not None:
            value_warmup_bundle = _resolve_value_warmup_bundle(
                request.value_warmup_bundle,
                policy_checkpoint=checkpoint,
                task_instruction=task_instruction,
            )
        resume: _ResolvedFlowSDEResume | None = None
        if request.resume_checkpoint is not None:
            resume = _resolve_online_resume_checkpoint(
                request.resume_checkpoint,
                policy_checkpoint=checkpoint,
                robot_type=robot_type,
                task_instruction=task_instruction,
                ppo_epochs=request.ppo_epochs,
                minibatch_size=request.minibatch_size,
                actor_learning_rate=request.actor_learning_rate,
                value_learning_rate=request.value_learning_rate,
            )

        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise HTTPException(409, "A Flow-SDE PPO job is already running")
            if (
                self._value_warmup_job is not None
                and self._value_warmup_job.status == "running"
            ):
                raise HTTPException(
                    409,
                    "Stop Flow-SDE PPO value warm-up before starting online PPO",
                )
            job_id = uuid.uuid4().hex
            output_dir = _flow_sde_output_path(job_id)
            if value_warmup_bundle is not None and _paths_overlap(
                value_warmup_bundle,
                output_dir,
            ):
                raise HTTPException(
                    400,
                    "value_warmup_bundle cannot be the current online PPO output",
                )
            control_dir = _prepare_job_directories(output_dir)
            job = _FlowSDEPPOJob(
                job_id=job_id,
                policy_checkpoint=str(checkpoint),
                robot_type=robot_type,
                task_instruction=task_instruction,
                output_dir=str(output_dir),
                control_file=str(control_dir / "outcome.json"),
                log_path=str(FLOW_SDE_LOG_ROOT / f"{job_id}.log"),
                episodes=request.episodes,
                ppo_epochs=request.ppo_epochs,
                minibatch_size=request.minibatch_size,
                max_chunk_decisions=request.max_chunk_decisions,
                actor_learning_rate=request.actor_learning_rate,
                value_learning_rate=request.value_learning_rate,
                ack_timeout_seconds=request.ack_timeout_seconds,
                sensor_timeout_seconds=request.sensor_timeout_seconds,
                value_warmup_bundle=(
                    str(value_warmup_bundle)
                    if value_warmup_bundle is not None
                    else ""
                ),
                resume_checkpoint=(str(resume.checkpoint) if resume is not None else ""),
                resume_source_job_id=(
                    resume.source_job_id if resume is not None else ""
                ),
                lineage_policy_checkpoint=str(
                    resume.lineage_policy_checkpoint
                    if resume is not None
                    else checkpoint
                ),
            )
            try:
                environment = self._compose_environment()
            except Exception as exc:  # noqa: BLE001 - Docker mount boundary
                _remove_empty_job_directories(output_dir, control_dir)
                raise HTTPException(
                    503,
                    f"Could not resolve Docker workspace: {exc}",
                ) from exc
            try:
                process = subprocess.Popen(
                    self._command(job),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=environment,
                )
            except OSError as exc:
                _remove_empty_job_directories(output_dir, control_dir)
                raise HTTPException(503, f"Could not launch Flow-SDE PPO: {exc}") from exc
            job.process = process
            self._job = job

        threading.Thread(
            target=self._monitor,
            args=(job,),
            daemon=True,
            name=f"flow-sde-ppo-{job_id[:12]}",
        ).start()
        with self._lock:
            return self._status(job)

    def start_value_warmup(
        self,
        request: FlowSDEValueWarmupStartRequest,
    ) -> FlowSDEValueWarmupStatus:
        conflict = self._conflict_message()
        if conflict:
            raise HTTPException(409, conflict)
        checkpoint = _resolve_policy_checkpoint(request.policy_checkpoint)
        datasets = _resolve_value_warmup_datasets(request.dataset_paths)
        task_instruction = _safe_instruction(request.task_instruction)

        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise HTTPException(
                    409,
                    "Stop online Flow-SDE PPO before starting value warm-up",
                )
            if (
                self._value_warmup_job is not None
                and self._value_warmup_job.status == "running"
            ):
                raise HTTPException(
                    409,
                    "A Flow-SDE PPO value warm-up job is already running",
                )
            job_id = uuid.uuid4().hex
            output_dir = _flow_sde_value_warmup_output_path(job_id)
            _prepare_value_warmup_directory(output_dir)
            job = _FlowSDEValueWarmupJob(
                job_id=job_id,
                policy_checkpoint=str(checkpoint),
                dataset_paths=[str(dataset) for dataset in datasets],
                task_instruction=task_instruction,
                output_dir=str(output_dir),
                log_path=str(FLOW_SDE_LOG_ROOT / f"value_warmup_{job_id}.log"),
                total_steps=request.steps,
                batch_size=request.batch_size,
                value_learning_rate=request.value_learning_rate,
                discount=request.discount,
            )
            try:
                environment = self._compose_environment()
            except Exception as exc:  # noqa: BLE001 - Docker mount boundary
                raise HTTPException(
                    503,
                    f"Could not resolve Docker workspace: {exc}",
                ) from exc
            try:
                process = subprocess.Popen(
                    self._value_warmup_command(job),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=environment,
                )
            except OSError as exc:
                raise HTTPException(
                    503,
                    f"Could not launch Flow-SDE PPO value warm-up: {exc}",
                ) from exc
            job.process = process
            self._value_warmup_job = job

        threading.Thread(
            target=self._monitor_value_warmup,
            args=(job,),
            daemon=True,
            name=f"flow-sde-value-warmup-{job_id[:12]}",
        ).start()
        with self._lock:
            return self._value_warmup_status(job)

    def status(self) -> FlowSDEPPOStatus:
        with self._lock:
            if self._job is None:
                self._job = _discover_latest_online_job()
            return self._status(self._job)

    def value_warmup_status(self) -> FlowSDEValueWarmupStatus:
        with self._lock:
            if self._value_warmup_job is None:
                self._value_warmup_job = _discover_latest_value_warmup_job()
            return self._value_warmup_status(self._value_warmup_job)

    def is_running(self) -> bool:
        with self._lock:
            online_running = self._job is not None and self._job.status == "running"
            warmup_running = (
                self._value_warmup_job is not None
                and self._value_warmup_job.status == "running"
            )
            return online_running or warmup_running

    def _write_outcome(self, job: _FlowSDEPPOJob, outcome: str) -> None:
        path = Path(job.control_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": job.job_id,
            "outcome": outcome,
            "sequence": time.time_ns(),
            "timestamp": time.time(),
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(payload) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def outcome(self, request: FlowSDEPPOOutcomeRequest) -> FlowSDEPPOStatus:
        with self._lock:
            job = self._job
            if job is None or request.job_id.strip() != job.job_id:
                raise HTTPException(409, "Flow-SDE PPO job_id is stale or no longer current")
            if job.status != "running":
                raise HTTPException(409, f"Flow-SDE PPO job is already {job.status}")
            if not job.awaiting_outcome:
                raise HTTPException(409, "Flow-SDE PPO is not awaiting an episode outcome")
            try:
                self._write_outcome(job, request.outcome)
            except OSError as exc:
                raise HTTPException(503, f"Could not submit episode outcome: {exc}") from exc
            job.message = (
                "Cancelling Flow-SDE PPO episode"
                if request.outcome == "cancel"
                else f"Episode marked {request.outcome}"
            )
            job.awaiting_outcome = False
            return self._status(job)

    def _interrupt(self, job: _FlowSDEPPOJob) -> bool:
        container_error: Exception | None = None
        if self._interrupt_container is not None:
            try:
                if self._interrupt_container(self._container_name(job)):
                    return True
            except Exception as exc:  # noqa: BLE001 - Docker control boundary
                container_error = exc
        process = job.process
        if process is None or process.poll() is not None:
            if container_error is not None:
                raise container_error
            return False
        try:
            process.send_signal(signal.SIGINT)
        except (OSError, ProcessLookupError) as exc:
            if container_error is not None:
                raise RuntimeError(f"{container_error}; {exc}") from exc
            return False
        return True

    def _interrupt_value_warmup(self, job: _FlowSDEValueWarmupJob) -> bool:
        container_error: Exception | None = None
        if self._interrupt_container is not None:
            try:
                if self._interrupt_container(self._value_warmup_container_name(job)):
                    return True
            except Exception as exc:  # noqa: BLE001 - Docker control boundary
                container_error = exc
        process = job.process
        if process is None or process.poll() is not None:
            if container_error is not None:
                raise container_error
            return False
        try:
            process.send_signal(signal.SIGINT)
        except (OSError, ProcessLookupError) as exc:
            if container_error is not None:
                raise RuntimeError(f"{container_error}; {exc}") from exc
            return False
        return True

    def stop(self, request: FlowSDEPPOStopRequest) -> FlowSDEPPOStatus:
        with self._lock:
            job = self._job
            if job is None or request.job_id.strip() != job.job_id:
                raise HTTPException(409, "Flow-SDE PPO job_id is stale or no longer current")
            if job.status == "stopped":
                return self._status(job)
            if job.status != "running":
                raise HTTPException(409, f"Flow-SDE PPO job is already {job.status}")
            if job.stop_requested:
                return self._status(job)
            job.stop_requested = True
            job.message = "Stopping Flow-SDE PPO training"
            try:
                interrupted = self._interrupt(job)
            except Exception as exc:  # noqa: BLE001 - Docker/subprocess boundary
                job.stop_requested = False
                raise HTTPException(
                    503,
                    f"Could not stop Flow-SDE PPO job: {exc}",
                ) from exc
            if not interrupted:
                job.stop_requested = False
                raise HTTPException(409, "Flow-SDE PPO job exited before it could be stopped")
            return self._status(job)

    def stop_value_warmup(
        self,
        request: FlowSDEPPOStopRequest,
    ) -> FlowSDEValueWarmupStatus:
        with self._lock:
            job = self._value_warmup_job
            if job is None or request.job_id.strip() != job.job_id:
                raise HTTPException(
                    409,
                    "Flow-SDE PPO value warm-up job_id is stale or no longer current",
                )
            if job.status == "stopped":
                return self._value_warmup_status(job)
            if job.status != "running":
                raise HTTPException(
                    409,
                    f"Flow-SDE PPO value warm-up job is already {job.status}",
                )
            if job.stop_requested:
                return self._value_warmup_status(job)
            job.stop_requested = True
            job.message = "Stopping Flow-SDE PPO value warm-up"
            try:
                interrupted = self._interrupt_value_warmup(job)
            except Exception as exc:  # noqa: BLE001 - Docker/subprocess boundary
                job.stop_requested = False
                raise HTTPException(
                    503,
                    f"Could not stop Flow-SDE PPO value warm-up: {exc}",
                ) from exc
            if not interrupted:
                job.stop_requested = False
                raise HTTPException(
                    409,
                    "Flow-SDE PPO value warm-up exited before it could be stopped",
                )
            return self._value_warmup_status(job)


def create_flow_sde_ppo_router(
    *,
    compose_command: Callable[[], list[str]],
    compose_environment: Callable[[], dict[str, str]],
    conflict_message: Callable[[], str | None] | None = None,
    interrupt_container: Callable[[str], bool] | None = None,
) -> tuple[APIRouter, FlowSDEPPOSupervisor]:
    supervisor = FlowSDEPPOSupervisor(
        compose_command=compose_command,
        compose_environment=compose_environment,
        conflict_message=conflict_message,
        interrupt_container=interrupt_container,
    )
    return supervisor.router, supervisor
