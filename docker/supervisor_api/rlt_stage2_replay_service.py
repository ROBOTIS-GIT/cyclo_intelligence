#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Supervisor API for preparing fresh, dataset-owned RLT Stage-2 replay."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from supervisor_api import rlt_stage2_service as _stage2


_REPLAY_MANIFEST_FORMAT = "cyclo.groot.rlt.stage2_feature_replay_manifest/v1"


class RLTStage2ReplayPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_paths: list[str] = Field(min_length=1, max_length=128)
    groot_checkpoint: str = ""
    rl_token_encoder_path: str = ""
    rlt_bundle_path: str = ""
    feature_batch_size: StrictInt = Field(default=2, ge=1, le=64)
    replay_mode: Literal["legacy", "tt_rtc_actor", "tt_rtc_rl"] = "legacy"

    @model_validator(mode="after")
    def validate_sources(self):
        if self.rlt_bundle_path:
            if self.groot_checkpoint or self.rl_token_encoder_path:
                raise ValueError("Select a Resume bundle or GR00T/encoder pair, not both")
        elif not self.groot_checkpoint or not self.rl_token_encoder_path:
            raise ValueError("GR00T checkpoint and RL Token encoder are required")
        return self


class RLTStage2ReplayStopRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


class RLTStage2ReplayStatus(BaseModel):
    ready: bool = True
    status: Literal["idle", "running", "completed", "failed", "stopped"]
    phase: str = "idle"
    percentage: float = 0.0
    job_id: str = ""
    dataset_paths: list[str] = Field(default_factory=list)
    resolved_dataset_paths: list[str] = Field(default_factory=list)
    groot_checkpoint: str = ""
    rl_token_encoder_path: str = ""
    replay_paths: list[str] = Field(default_factory=list)
    feature_batch_size: int = 0
    replay_mode: str = "legacy"
    rlt_bundle_path: str = ""
    completed_datasets: int = 0
    total_datasets: int = 0
    completed_features: int = 0
    total_features: int = 0
    transition_count: int = 0
    average_reward: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = ""
    returncode: Optional[int] = None
    log_tail: list[str] = Field(default_factory=list)


@dataclass
class _ReplayJob:
    job_id: str
    dataset_paths: list[str]
    resolved_dataset_paths: list[str]
    groot_checkpoint: str
    rl_token_encoder_path: str
    feature_batch_size: int
    log_path: str
    replay_mode: str = "legacy"
    rlt_bundle_path: str = ""
    status: str = "running"
    phase: str = "starting"
    percentage: float = 0.0
    replay_paths: list[str] = field(default_factory=list)
    completed_datasets: int = 0
    completed_features: int = 0
    total_features: int = 0
    transition_count: int = 0
    average_reward: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = "Starting Stage 2 replay preparation"
    process: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    stop_confirmed: bool = False
    returncode: Optional[int] = None
    log_tail: list[str] = field(default_factory=list)


def _replay_relative(mode):
    return {"legacy": "rlt/replay_cache", "tt_rtc_actor": "rlt/tt_rtc_actor_hold_v1/replay_cache",
            "tt_rtc_rl": "rlt/tt_rtc_transition_v1/replay_cache"}[mode]


def _resolve_frozen_sources(request):
    if request.rlt_bundle_path:
        bundle = _stage2._resolve_resume_bundle(request.rlt_bundle_path)
        manifest = _stage2._read_json(bundle / "manifest.json", label="RLT bundle")
        checkpoint = _stage2._resolve_groot_checkpoint(manifest["source"]["groot_checkpoint"])
        return checkpoint, bundle / "artifacts/rl_token_encoder.pt", str(bundle)
    checkpoint = _stage2._resolve_groot_checkpoint(request.groot_checkpoint)
    return checkpoint, _stage2._resolve_stage1_encoder(request.rl_token_encoder_path, checkpoint), ""


def _runtime_readiness() -> tuple[bool, str]:
    required = (
        _stage2.RLT_STAGE2_RUNTIME_ROOT / "rlt_stage2_replay_cli.py",
        _stage2.RLT_STAGE2_RUNTIME_ROOT / "rlt_stage2_dataset.py",
        _stage2.RLT_STAGE2_RUNTIME_ROOT / "rlt_tt_rtc_transition_replay.py",
    )
    missing = [path.name for path in required if path.is_symlink() or not path.is_file()]
    if missing:
        return False, "Stage 2 replay preparation is not ready: missing " + ", ".join(missing)
    return True, "Stage 2 replay preparation is ready"


def _prepare_replay_output_parent(dataset: Path, replay_mode: str) -> None:
    """Create the dataset-owned replay parent for the unprivileged worker."""

    target_parent = dataset / Path(_replay_relative(replay_mode)).parent
    dataset_root = dataset.resolve(strict=True)
    try:
        target_parent.resolve(strict=False).relative_to(dataset_root)
    except (OSError, ValueError) as exc:
        raise HTTPException(409, f"Stage 2 replay path escapes its dataset: {target_parent}") from exc
    current = dataset
    for part in target_parent.relative_to(dataset).parts:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise HTTPException(409, f"Stage 2 replay path is unsafe: {current}")
        try:
            current.mkdir(mode=0o775, exist_ok=True)
            current_stat = current.stat()
            if (
                current_stat.st_uid != _stage2.RLT_STAGE2_TRAIN_UID
                or current_stat.st_gid != _stage2.RLT_STAGE2_TRAIN_GID
            ):
                os.chown(
                    current,
                    _stage2.RLT_STAGE2_TRAIN_UID,
                    _stage2.RLT_STAGE2_TRAIN_GID,
                )
            if current.stat().st_mode & 0o770 != 0o770:
                os.chmod(current, current.stat().st_mode | 0o770)
        except OSError as exc:
            raise HTTPException(
                500,
                f"Could not prepare writable Stage 2 replay path: {current}",
            ) from exc


class RLTStage2ReplaySupervisor:
    """Own one GR00T feature extraction process and publish no model weights."""

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
        self._job: _ReplayJob | None = None
        self._starting = False
        self.router = APIRouter(prefix="/rlt-stage2/replay", tags=["rlt-stage2-replay"])
        self.router.add_api_route(
            "/prepare",
            self.prepare,
            methods=["POST"],
            response_model=RLTStage2ReplayStatus,
        )
        self.router.add_api_route(
            "/status",
            self.status,
            methods=["GET"],
            response_model=RLTStage2ReplayStatus,
        )
        self.router.add_api_route(
            "/stop",
            self.stop,
            methods=["POST"],
            response_model=RLTStage2ReplayStatus,
        )

    @staticmethod
    def _container_name(job: _ReplayJob) -> str:
        return f"cyclo_rlt_replay_{job.job_id[:12]}"

    def _command(self, job: _ReplayJob) -> list[str]:
        command = self._compose_command() + [
            "run", "--rm", "--no-deps", "--pull", "never",
            "--name", self._container_name(job),
            "--user", f"{_stage2.RLT_STAGE2_TRAIN_UID}:{_stage2.RLT_STAGE2_TRAIN_GID}",
            "--workdir", "/app",
            "--env", "HOME=/tmp",
            "--env", "USER=cyclo_rlt",
            "--env", "LOGNAME=cyclo_rlt",
            "--env", f"XDG_CACHE_HOME={_stage2.RLT_STAGE2_CACHE_ROOT}",
            "--env", f"HF_HOME={_stage2.RLT_STAGE2_CACHE_ROOT}/huggingface",
            "--env", f"HF_HUB_CACHE={_stage2.RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"HUGGINGFACE_HUB_CACHE={_stage2.RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"TRANSFORMERS_CACHE={_stage2.RLT_STAGE2_HF_HUB_CACHE}",
            "--env", f"TORCH_HOME={_stage2.RLT_STAGE2_CACHE_ROOT}/torch",
            "--env", f"TORCHINDUCTOR_CACHE_DIR={_stage2.RLT_STAGE2_CACHE_ROOT}/torchinductor",
            "--env", f"TRITON_CACHE_DIR={_stage2.RLT_STAGE2_CACHE_ROOT}/triton",
            "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1",
            "--env", "HF_DATASETS_OFFLINE=1",
            "--env", "GROOT_HF_LOCAL_FIRST=1",
            "--env", "GROOT_PATCH_MISTRAL=1",
            "--env", "NO_ALBUMENTATIONS_UPDATE=1",
            "--env", "PYTHONPATH=/cyclo_brain_src:/app:/gr00t",
            "--entrypoint", "python", "groot",
            "-m", "runtime.rlt_stage2_replay_cli",
        ]
        for dataset in job.resolved_dataset_paths:
            command.extend(("--dataset-root", dataset))
        command.extend((
            "--groot-checkpoint", job.groot_checkpoint,
            "--rl-token-encoder", job.rl_token_encoder_path,
            "--job-id", job.job_id,
            "--feature-batch-size", str(job.feature_batch_size),
            "--device", "cuda",
        ))
        command.extend(("--replay-mode", job.replay_mode))
        return command

    def _status(self, job: _ReplayJob | None) -> RLTStage2ReplayStatus:
        ready, readiness_message = self._readiness_check()
        if job is None:
            return RLTStage2ReplayStatus(
                ready=ready,
                status="idle",
                message=readiness_message,
            )
        return RLTStage2ReplayStatus(
            ready=ready,
            status=job.status,
            phase=job.phase,
            percentage=job.percentage,
            job_id=job.job_id,
            dataset_paths=list(job.dataset_paths),
            resolved_dataset_paths=list(job.resolved_dataset_paths),
            groot_checkpoint=job.groot_checkpoint,
            rl_token_encoder_path=job.rl_token_encoder_path,
            replay_paths=list(job.replay_paths),
            feature_batch_size=job.feature_batch_size,
            replay_mode=job.replay_mode,
            rlt_bundle_path=job.rlt_bundle_path,
            completed_datasets=job.completed_datasets,
            total_datasets=len(job.resolved_dataset_paths),
            completed_features=job.completed_features,
            total_features=job.total_features,
            transition_count=job.transition_count,
            average_reward=job.average_reward,
            eta_seconds=job.eta_seconds,
            message=job.message,
            returncode=job.returncode,
            log_tail=list(job.log_tail),
        )

    @staticmethod
    def _consume_number(job: _ReplayJob, payload: Mapping[str, Any], name: str) -> None:
        value = payload.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(job, name, float(value))

    def _consume_event(self, job: _ReplayJob, payload: Mapping[str, Any]) -> bool:
        event = str(payload.get("event") or "")
        if event == "stage2_replay_prepare_manifest":
            job.phase = str(payload.get("phase") or "loading_frozen_sources")
            job.message = str(payload.get("message") or "Loading frozen replay sources")
            return False
        if event == "stage2_replay_prepare_progress":
            job.phase = str(payload.get("phase") or "extracting_features")
            for name in (
                "completed_datasets", "completed_features", "total_features"
            ):
                value = payload.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    setattr(job, name, value)
            percentage = payload.get("percentage")
            if isinstance(percentage, (int, float)) and not isinstance(percentage, bool):
                job.percentage = max(0.0, min(99.0, float(percentage)))
            self._consume_number(job, payload, "eta_seconds")
            job.message = str(payload.get("message") or "Preparing Stage 2 replay")
            return False
        if event == "stage2_replay_prepare_result":
            status = str(payload.get("status") or "")
            if status == "completed":
                paths = payload.get("replay_paths")
                if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
                    job.replay_paths = list(paths)
                transitions = payload.get("transition_count")
                if isinstance(transitions, int) and not isinstance(transitions, bool):
                    job.transition_count = transitions
                self._consume_number(job, payload, "average_reward")
                job.phase = "verifying"
                job.message = "Verifying prepared Stage 2 replay"
                return len(job.replay_paths) == len(job.resolved_dataset_paths)
            if status == "stopped":
                job.stop_confirmed = True
                job.phase = "stopping"
                job.message = str(payload.get("message") or "Replay preparation stopped")
            return False
        if event == "stage2_replay_prepare_error":
            job.phase = "error"
            job.message = str(payload.get("message") or payload.get("error") or "Replay preparation failed")
        return False

    @staticmethod
    def _verification_error(job: _ReplayJob) -> str:
        relative = _replay_relative(job.replay_mode)
        expected = [Path(path) / relative for path in job.resolved_dataset_paths]
        if [os.path.normpath(path) for path in job.replay_paths] != [
            os.path.normpath(str(path)) for path in expected
        ]:
            return "reported replay paths do not match the selected datasets"
        for dataset, replay_root in zip(job.resolved_dataset_paths, expected, strict=True):
            manifest_path = replay_root / "manifest.json"
            replay_path = replay_root / "replay.pt"
            if (
                _stage2._has_symlink_component(Path(dataset), manifest_path)
                or manifest_path.is_symlink()
                or replay_path.is_symlink()
                or not manifest_path.is_file()
                or not replay_path.is_file()
            ):
                return f"prepared replay artifacts are missing or unsafe: {replay_root}"
            try:
                manifest = _stage2._read_replay_json(
                    manifest_path,
                    label="prepared replay",
                )
            except HTTPException as error:
                return str(error.detail)
            metadata = manifest.get("metadata")
            if (
                manifest.get("format") != _REPLAY_MANIFEST_FORMAT
                or manifest.get("file") != "replay.pt"
                or not isinstance(metadata, Mapping)
                or metadata.get("dataset_roots") != [dataset]
                or manifest.get("byte_count") != replay_path.stat().st_size
                or manifest.get("sha256") != _stage2._file_sha256(replay_path)
                or not isinstance(manifest.get("transition_count"), int)
                or manifest.get("transition_count", 0) < 1
            ):
                return f"prepared replay manifest is invalid: {manifest_path}"
        return ""

    @staticmethod
    def _append_log(job: _ReplayJob, line: str) -> None:
        if line:
            job.log_tail.append(line)
            del job.log_tail[:-_stage2.RLT_STAGE2_LOG_LINES]

    def _monitor(self, job: _ReplayJob) -> None:
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
                            if isinstance(payload, Mapping):
                                result_complete = self._consume_event(job, payload) or result_complete
                returncode = job.process.wait() if job.process is not None else -1
        except Exception as exc:  # pragma: no cover - worker boundary
            returncode = -1
            with self._lock:
                job.message = f"Stage 2 replay monitor failed: {exc}"
        verification_error = (
            self._verification_error(job)
            if returncode == 0 and result_complete
            else ""
        )
        with self._lock:
            job.returncode = returncode
            expected_stop_codes = {0, -signal.SIGINT, 128 + signal.SIGINT}
            if (job.stop_requested and returncode in expected_stop_codes) or (
                returncode == 0 and job.stop_confirmed
            ):
                job.status = "stopped"
                job.phase = "stopped"
                job.replay_paths = []
                job.eta_seconds = None
                job.message = "Stage 2 replay preparation stopped"
            elif returncode == 0 and result_complete and not verification_error:
                job.status = "completed"
                job.phase = "complete"
                job.percentage = 100.0
                job.completed_datasets = len(job.resolved_dataset_paths)
                job.eta_seconds = 0.0
                job.message = ("TT-RTC Critic/RL replay preparation completed" if job.replay_mode == "tt_rtc_rl"
                               else "TT-RTC Actor replay preparation completed" if job.replay_mode == "tt_rtc_actor"
                               else "Stage 2 replay preparation completed")
            else:
                job.status = "failed"
                job.phase = "error"
                if verification_error:
                    job.message = f"Stage 2 replay verification failed: {verification_error}"
                elif not job.message or job.message.startswith("Starting Stage 2"):
                    reason = f"exited with code {returncode}" if returncode else "did not publish verified replay"
                    job.message = f"Stage 2 replay preparation {reason}"

    def is_running(self) -> bool:
        with self._lock:
            return self._starting or (self._job is not None and self._job.status == "running")

    def prepare(self, request: RLTStage2ReplayPrepareRequest) -> RLTStage2ReplayStatus:
        with self._lock:
            if self._starting or (self._job is not None and self._job.status == "running"):
                raise HTTPException(409, "Stage 2 replay preparation is already running")
            conflict = self._conflict_message()
            if conflict:
                raise HTTPException(409, conflict)
            ready, readiness_message = self._readiness_check()
            if not ready:
                raise HTTPException(503, readiness_message)
            self._starting = True
        try:
            datasets = _stage2._resolve_datasets(request.dataset_paths)
            checkpoint, encoder, bundle_path = _resolve_frozen_sources(request)
            for dataset in datasets:
                relative = _replay_relative(request.replay_mode)
                replay = dataset / relative
                if replay.is_symlink() or (replay.exists() and not replay.is_dir()):
                    raise HTTPException(
                        409,
                        f"Stage 2 replay path is unsafe: {replay}",
                    )
                _prepare_replay_output_parent(dataset, request.replay_mode)
            job_id = uuid.uuid4().hex
            _stage2.RLT_STAGE2_LOG_ROOT.mkdir(parents=True, exist_ok=True)
        except Exception:
            with self._lock:
                self._starting = False
            raise
        job = _ReplayJob(
            job_id=job_id,
            dataset_paths=list(request.dataset_paths),
            resolved_dataset_paths=[str(path) for path in datasets],
            groot_checkpoint=str(checkpoint),
            rl_token_encoder_path=str(encoder),
            feature_batch_size=1 if request.replay_mode != "legacy" else request.feature_batch_size,
            replay_mode=request.replay_mode,
            rlt_bundle_path=bundle_path,
            log_path=str(_stage2.RLT_STAGE2_LOG_ROOT / f"replay_{job_id}.log"),
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
            job.message = f"Could not start Stage 2 replay preparation: {exc}"
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
            name=f"rlt-replay-{job_id[:12]}",
            daemon=True,
        ).start()
        return self._status(job)

    def status(self) -> RLTStage2ReplayStatus:
        with self._lock:
            return self._status(self._job)

    def stop(self, request: RLTStage2ReplayStopRequest) -> RLTStage2ReplayStatus:
        with self._lock:
            job = self._job
            if job is None or request.job_id != job.job_id:
                raise HTTPException(409, "Replay preparation job_id does not match the active job")
            if job.status != "running":
                raise HTTPException(409, f"Replay preparation is already {job.status}")
            job.stop_requested = True
            job.phase = "stopping"
            job.message = "Stopping Stage 2 replay preparation"
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
                job.message = f"Could not stop replay preparation: {exc}"
            raise HTTPException(502, job.message) from exc
        if not signalled:
            with self._lock:
                job.stop_requested = False
            raise HTTPException(409, "Replay preparation exited before it could be stopped")
        with self._lock:
            return self._status(job)


def create_rlt_stage2_replay_router(
    *,
    compose_command: Callable[[], list[str]],
    compose_environment: Callable[[], dict[str, str]],
    conflict_message: Callable[[], str | None] | None = None,
    interrupt_container: Callable[[str], bool] | None = None,
    readiness_check: Callable[[], tuple[bool, str]] | None = None,
) -> tuple[APIRouter, RLTStage2ReplaySupervisor]:
    supervisor = RLTStage2ReplaySupervisor(
        compose_command=compose_command,
        compose_environment=compose_environment,
        conflict_message=conflict_message,
        interrupt_container=interrupt_container,
        readiness_check=readiness_check,
    )
    return supervisor.router, supervisor
