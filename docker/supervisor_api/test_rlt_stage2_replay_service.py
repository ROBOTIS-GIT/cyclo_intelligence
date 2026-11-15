#!/usr/bin/env python3

"""Focused tests for fresh RLT Stage-2 replay preparation."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError


DOCKER_ROOT = Path(__file__).resolve().parents[1]
if str(DOCKER_ROOT) not in sys.path:
    sys.path.insert(0, str(DOCKER_ROOT))

from supervisor_api import rlt_stage2_replay_service as service  # noqa: E402


def _supervisor(**overrides):
    values = {
        "compose_command": lambda: ["docker", "compose", "-f", "/tmp/compose.yml"],
        "compose_environment": lambda: {"COMPOSE_PROJECT_NAME": "cyclo"},
        "readiness_check": lambda: (True, "ready"),
    }
    values.update(overrides)
    return service.RLTStage2ReplaySupervisor(**values)


def _job(tmp_path: Path) -> service._ReplayJob:
    return service._ReplayJob(
        job_id="a" * 32,
        dataset_paths=["/workspace/lerobot/task"],
        resolved_dataset_paths=["/workspace/lerobot/task"],
        groot_checkpoint="/workspace/model/groot/task",
        rl_token_encoder_path=(
            "/workspace/checkpoint/rlt/stage1/run/artifacts/rl_token_encoder.pt"
        ),
        feature_batch_size=2,
        log_path=str(tmp_path / "replay.log"),
    )


def _write_replay(dataset: Path) -> Path:
    replay_root = dataset / "rlt" / "replay_cache"
    replay_root.mkdir(parents=True)
    replay = replay_root / "replay.pt"
    replay.write_bytes(b"immutable replay")
    manifest = {
        "format": service._REPLAY_MANIFEST_FORMAT,
        "file": "replay.pt",
        "byte_count": replay.stat().st_size,
        "sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
        "transition_count": 12,
        "metadata": {"dataset_roots": [str(dataset)]},
    }
    (replay_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return replay_root


class _FakeProcess:
    def __init__(self, stdout, *, on_wait=None, returncode=0):
        self.stdout = list(stdout)
        self._on_wait = on_wait
        self._returncode = returncode
        self._poll = None
        self.signals = []

    def wait(self):
        if self._on_wait is not None:
            self._on_wait()
        self._poll = self._returncode
        return self._returncode

    def poll(self):
        return self._poll

    def send_signal(self, requested_signal):
        self.signals.append(requested_signal)


def test_prepare_request_requires_all_frozen_sources():
    with pytest.raises(ValidationError):
        service.RLTStage2ReplayPrepareRequest(dataset_paths=[])
    with pytest.raises(ValidationError):
        service.RLTStage2ReplayPrepareRequest(
            dataset_paths=["/workspace/lerobot/task"],
            groot_checkpoint="/workspace/model/groot/task",
            rl_token_encoder_path="",
        )


def test_command_extracts_features_without_stage2_weights(tmp_path):
    command = _supervisor()._command(_job(tmp_path))

    assert command[command.index("-m") + 1] == "runtime.rlt_stage2_replay_cli"
    assert command[command.index("--dataset-root") + 1] == "/workspace/lerobot/task"
    assert command[command.index("--rl-token-encoder") + 1].endswith(
        "/artifacts/rl_token_encoder.pt"
    )
    assert "--rlt-bundle" not in command
    assert "--replay-bundle" not in command
    assert "--initialization-mode" not in command


def test_prepare_replay_output_parent_is_worker_writable(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    ownership = []
    monkeypatch.setattr(service._stage2, "RLT_STAGE2_TRAIN_UID", 12345)
    monkeypatch.setattr(service._stage2, "RLT_STAGE2_TRAIN_GID", 12346)
    monkeypatch.setattr(
        service.os,
        "chown",
        lambda path, uid, gid: ownership.append((Path(path), uid, gid)),
    )

    service._prepare_replay_output_parent(dataset, "tt_rtc_actor")

    parent = dataset / "rlt" / "tt_rtc_actor_hold_v1"
    assert parent.is_dir()
    assert parent.stat().st_mode & 0o770 == 0o770
    assert ownership[-1] == (
        parent,
        12345,
        12346,
    )


def test_prepare_replay_output_parent_rejects_symlink(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (dataset / "rlt").symlink_to(outside, target_is_directory=True)

    with pytest.raises(service.HTTPException, match="escapes|unsafe"):
        service._prepare_replay_output_parent(dataset, "tt_rtc_actor")


def test_verification_accepts_dataset_owned_replay(tmp_path):
    dataset = tmp_path / "dataset"
    replay_root = _write_replay(dataset)
    job = _job(tmp_path)
    job.resolved_dataset_paths = [str(dataset)]
    job.replay_paths = [str(replay_root)]

    assert service.RLTStage2ReplaySupervisor._verification_error(job) == ""

    (replay_root / "replay.pt").write_bytes(b"tampered")
    assert "manifest is invalid" in (
        service.RLTStage2ReplaySupervisor._verification_error(job)
    )


def test_verification_accepts_row_level_manifest_over_general_json_limit(tmp_path):
    dataset = tmp_path / "dataset"
    replay_root = _write_replay(dataset)
    manifest_path = replay_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["row_provenance"] = [
        {"transition_index": index, "action_sources": ["human"] * 10}
        for index in range(20_000)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert manifest_path.stat().st_size > service._stage2.RLT_STAGE2_MAX_JSON_BYTES

    job = _job(tmp_path)
    job.resolved_dataset_paths = [str(dataset)]
    job.replay_paths = [str(replay_root)]

    assert service.RLTStage2ReplaySupervisor._verification_error(job) == ""


def test_prepare_publishes_only_after_verified_result(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    checkpoint = tmp_path / "groot"
    checkpoint.mkdir()
    encoder = tmp_path / "rl_token_encoder.pt"
    encoder.write_bytes(b"encoder")
    log_root = tmp_path / "logs"

    monkeypatch.setattr(service._stage2, "RLT_STAGE2_LOG_ROOT", log_root)
    monkeypatch.setattr(
        service._stage2, "_resolve_datasets", lambda _paths: [dataset]
    )
    monkeypatch.setattr(
        service._stage2, "_resolve_groot_checkpoint", lambda _path: checkpoint
    )
    monkeypatch.setattr(
        service._stage2,
        "_resolve_stage1_encoder",
        lambda _path, _checkpoint: encoder,
    )

    result = {
        "event": "stage2_replay_prepare_result",
        "status": "completed",
        "replay_paths": [str(dataset / "rlt" / "replay_cache")],
        "transition_count": 12,
        "average_reward": 0.5,
    }
    process = _FakeProcess(
        [json.dumps(result) + "\n"],
        on_wait=lambda: _write_replay(dataset),
    )
    monkeypatch.setattr(service.subprocess, "Popen", lambda *_args, **_kwargs: process)

    supervisor = _supervisor()
    status = supervisor.prepare(
        service.RLTStage2ReplayPrepareRequest(
            dataset_paths=[str(dataset)],
            groot_checkpoint=str(checkpoint),
            rl_token_encoder_path=str(encoder),
            feature_batch_size=2,
        )
    )
    assert status.status in {"running", "completed"}

    deadline = time.monotonic() + 2.0
    while supervisor.status().status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    completed = supervisor.status()
    assert completed.status == "completed"
    assert completed.percentage == 100.0
    assert completed.transition_count == 12
    assert completed.replay_paths == [str(dataset / "rlt" / "replay_cache")]


def test_prepare_revalidates_existing_replay_without_overwriting(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_replay(dataset)
    checkpoint = tmp_path / "groot"
    checkpoint.mkdir()
    encoder = tmp_path / "rl_token_encoder.pt"
    encoder.write_bytes(b"encoder")
    monkeypatch.setattr(
        service._stage2, "_resolve_datasets", lambda _paths: [dataset]
    )
    monkeypatch.setattr(
        service._stage2, "_resolve_groot_checkpoint", lambda _path: checkpoint
    )
    monkeypatch.setattr(
        service._stage2,
        "_resolve_stage1_encoder",
        lambda _path, _checkpoint: encoder,
    )
    original = (dataset / "rlt" / "replay_cache" / "replay.pt").read_bytes()
    result = {
        "event": "stage2_replay_prepare_result",
        "status": "completed",
        "replay_paths": [str(dataset / "rlt" / "replay_cache")],
        "transition_count": 12,
        "average_reward": 0.5,
    }
    process = _FakeProcess([json.dumps(result) + "\n"])
    monkeypatch.setattr(service.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(service._stage2, "RLT_STAGE2_LOG_ROOT", tmp_path / "logs")

    supervisor = _supervisor()
    supervisor.prepare(
        service.RLTStage2ReplayPrepareRequest(
            dataset_paths=[str(dataset)],
            groot_checkpoint=str(checkpoint),
            rl_token_encoder_path=str(encoder),
        )
    )
    deadline = time.monotonic() + 2.0
    while supervisor.status().status == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert supervisor.status().status == "completed"
    assert (dataset / "rlt" / "replay_cache" / "replay.pt").read_bytes() == original


def test_tt_rtc_actor_request_command_and_output_path(tmp_path):
    request = service.RLTStage2ReplayPrepareRequest(
        dataset_paths=["/workspace/lerobot/task"], groot_checkpoint="/workspace/model/task",
        rl_token_encoder_path="/workspace/encoder.pt", replay_mode="tt_rtc_actor")
    assert request.replay_mode == "tt_rtc_actor"
    job = _job(tmp_path)
    job.replay_mode = request.replay_mode
    job.feature_batch_size = 1
    command = _supervisor()._command(job)
    assert command[command.index("--replay-mode") + 1] == "tt_rtc_actor"
    assert command[command.index("--feature-batch-size") + 1] == "1"
    dataset = tmp_path / "dataset"
    legacy = _write_replay(dataset)
    job.resolved_dataset_paths = [str(dataset)]
    job.replay_paths = [str(legacy)]
    assert service.RLTStage2ReplaySupervisor._verification_error(job)
    target = dataset / "rlt/tt_rtc_actor_hold_v1/replay_cache"
    target.parent.mkdir(parents=True)
    legacy.rename(target)
    job.replay_paths = [str(target)]
    assert service.RLTStage2ReplaySupervisor._verification_error(job) == ""
    assert _supervisor()._status(job).replay_mode == "tt_rtc_actor"


def test_prepare_can_use_verified_resume_bundle_encoder(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    checkpoint = tmp_path / "groot"
    checkpoint.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"source": {"groot_checkpoint": str(checkpoint)}}))
    verified = []
    monkeypatch.setattr(service._stage2, "_resolve_resume_bundle", lambda p: verified.append(p) or bundle)
    monkeypatch.setattr(service._stage2, "_resolve_groot_checkpoint", lambda p: Path(p))
    request = service.RLTStage2ReplayPrepareRequest(dataset_paths=["/workspace/lerobot/task"],
        rlt_bundle_path=str(bundle), replay_mode="tt_rtc_rl")
    groot, encoder, source = service._resolve_frozen_sources(request)
    assert verified == [str(bundle)]
    assert groot == checkpoint
    assert encoder == bundle / "artifacts/rl_token_encoder.pt"
    assert source == str(bundle)
    job = _job(tmp_path)
    job.replay_mode = request.replay_mode
    job.rlt_bundle_path = str(bundle)
    assert _supervisor()._status(job).rlt_bundle_path == str(bundle)
    command = _supervisor()._command(job)
    assert command[command.index("--replay-mode") + 1] == "tt_rtc_rl"
    assert service._replay_relative("tt_rtc_rl") == "rlt/tt_rtc_transition_v1/replay_cache"
    with pytest.raises(ValidationError):
        service.RLTStage2ReplayPrepareRequest(dataset_paths=["/workspace/lerobot/task"],
            rlt_bundle_path=str(bundle), groot_checkpoint=str(checkpoint), replay_mode="tt_rtc_rl")
