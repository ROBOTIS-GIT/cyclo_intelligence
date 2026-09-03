#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for the GR00T RLT Stage 2 supervisor."""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException


SERVICE_PATH = Path(__file__).resolve().with_name("rlt_stage2_service.py")
SPEC = importlib.util.spec_from_file_location("rlt_stage2_service_under_test", SERVICE_PATH)
assert SPEC is not None and SPEC.loader is not None
service = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = service
SPEC.loader.exec_module(service)


class FakeProcess:
    def __init__(self, *, stdout=(), returncode=0):
        self.stdout = list(stdout)
        self._returncode = returncode
        self._poll = None
        self.signals = []

    def wait(self):
        self._poll = self._returncode
        return self._returncode

    def poll(self):
        return self._poll

    def send_signal(self, requested_signal):
        self.signals.append(requested_signal)


def _write_v21(path: Path, *, outcomes: bool = True) -> Path:
    meta = path / "meta"
    meta.mkdir(parents=True)
    features = {"episode_success": {"dtype": "bool"}} if outcomes else {}
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "total_episodes": 2,
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        '{"episode_index": 0}\n{"episode_index": 1}\n', encoding="utf-8"
    )
    (meta / "tasks.jsonl").write_text('{"task_index": 0}\n', encoding="utf-8")
    return path


def _write_groot(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "Gr00tN1d7", "architectures": ["Gr00tN1d7"]}),
        encoding="utf-8",
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return path


def _write_stage1_encoder(root: Path, groot: Path) -> Path:
    encoder = root / "artifacts" / "rl_token_encoder.pt"
    checkpoint = root / "training_state" / "rlt_stage1.pt"
    encoder.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    encoder.write_bytes(b"encoder")
    checkpoint.write_bytes(b"training-state")
    (checkpoint.parent / "rlt_stage1.pt.run.json").write_text(
        json.dumps(
            {
                "format": service._STAGE1_RUN_FORMAT,
                "status": "completed",
                "groot_checkpoint": str(groot),
                "policy_weight_fingerprint": "a" * 64,
                "artifact": {
                    "path": str(encoder),
                    "artifact_fingerprint": "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    return encoder


def _write_stage2_bundle(root: Path) -> Path:
    encoder = root / "artifacts" / "rl_token_encoder.pt"
    actor = root / "artifacts" / "rlt_actor.pt"
    checkpoint = root / "training_state" / "rlt_stage2.pt"
    encoder.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    encoder.write_bytes(b"encoder")
    actor.write_bytes(b"actor")
    checkpoint.write_bytes(b"checkpoint")
    spec = {
        "reference_contract_fingerprint": "c" * 64,
        "rl_token_artifact_fingerprint": "b" * 64,
        "rl_token_dim": 64,
        "proprio_dim": 19,
        "reference_horizon": 16,
        "chunk_length": 10,
        "action_dim": 19,
        "action_hz": 15.0,
        "action_normalization_id": "showroom-normalized-19d",
        "action_codec_id": "normalized-chunk-10x19",
        "model_domain": "normalized",
        "schema_version": 1,
    }
    artifacts = {}
    for name, relative in {
        "rl_token_encoder": "artifacts/rl_token_encoder.pt",
        "rlt_actor": "artifacts/rlt_actor.pt",
        "training_state": "training_state/rlt_stage2.pt",
    }.items():
        path = root / relative
        artifacts[name] = {
            "relative_path": relative,
            "byte_count": path.stat().st_size,
            "sha256": service._file_sha256(path),
        }
    unsigned = {
        "format": service._STAGE2_BUNDLE_FORMAT,
        "initialization": {
            "mode": "new",
            "parent_bundle_fingerprint": None,
        },
        "source": {
            "groot_checkpoint": "/workspace/model/groot/showroom_groot",
            "groot_checkpoint_fingerprint": "a" * 64,
            "representation_contract_fingerprint": "c" * 64,
            "rl_token_artifact_fingerprint": "b" * 64,
        },
        "spec": spec,
        "spec_fingerprint": service._canonical_fingerprint(spec),
        "completed_critic_updates": 100,
        "completed_actor_updates": 50,
        "artifacts": artifacts,
        "qualification": service._STAGE2_QUALIFICATION,
    }
    manifest = {
        **unsigned,
        "manifest_fingerprint": service._canonical_fingerprint(unsigned),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def roots(tmp_path, monkeypatch):
    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    model_root = tmp_path / "model" / "groot"
    model_root.mkdir(parents=True)
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    stage1_root = checkpoint_root / "rlt" / "stage1"
    stage1_root.mkdir(parents=True)
    stage2_root = checkpoint_root / "rlt" / "stage2"
    stage2_root.mkdir()
    log_root = tmp_path / "logs"
    monkeypatch.setattr(service, "RLT_STAGE2_DATASET_ROOTS", (dataset_root,))
    monkeypatch.setattr(
        service, "RLT_STAGE2_GROOT_ROOTS", (model_root, checkpoint_root)
    )
    monkeypatch.setattr(service, "RLT_STAGE2_ENCODER_ROOTS", (stage1_root,))
    monkeypatch.setattr(service, "RLT_STAGE2_BUNDLE_ROOTS", (stage2_root,))
    monkeypatch.setattr(service, "RLT_STAGE2_OUTPUT_ROOT", stage2_root)
    monkeypatch.setattr(service, "RLT_STAGE2_LOG_ROOT", log_root)
    monkeypatch.setattr(service.os, "chown", lambda *args: None)
    return dataset_root, model_root, stage1_root, stage2_root, log_root


def _supervisor(
    *,
    conflict=lambda: None,
    interrupt_container=None,
    readiness=lambda: (True, "GR00T RLT Stage 2 is ready"),
):
    return service.RLTStage2Supervisor(
        compose_command=lambda: ["docker", "compose", "-f", "/tmp/compose.yml"],
        compose_environment=lambda: {"COMPOSE_PROJECT_NAME": "cyclo"},
        conflict_message=conflict,
        interrupt_container=interrupt_container,
        readiness_check=readiness,
    )


def _job(tmp_path: Path, **overrides):
    values = {
        "job_id": "a" * 32,
        "initialization_mode": "new",
        "dataset_paths": ["/workspace/lerobot/selected"],
        "resolved_dataset_paths": ["/workspace/lerobot/selected-v21"],
        "groot_checkpoint": "/workspace/model/groot/showroom_groot",
        "rl_token_encoder_path": "/workspace/checkpoint/rlt/stage1/run/artifacts/rl_token_encoder.pt",
        "rlt_bundle_path": "",
        "output_dir": str(tmp_path / "stage2-output"),
        "log_path": str(tmp_path / "logs" / "stage2.log"),
        "total_steps": 100,
        "batch_size": 64,
        "save_freq": 25,
    }
    values.update(overrides)
    return service._RLTStage2Job(**values)


def test_routes_are_dedicated_to_stage2():
    supervisor = _supervisor()
    assert {route.path for route in supervisor.router.routes} == {
        "/rlt-stage2/start",
        "/rlt-stage2/status",
        "/rlt-stage2/stop",
    }
    assert supervisor.status().status == "idle"
    assert supervisor.status().ready is True


def test_missing_runtime_keeps_status_available_but_blocks_start():
    supervisor = _supervisor(
        readiness=lambda: (
            False,
            "GR00T RLT Stage 2 is not ready: missing rlt_stage2_training_cli.py",
        )
    )

    status = supervisor.status()
    assert status.status == "idle"
    assert status.ready is False
    assert "not ready" in status.message
    with pytest.raises(HTTPException, match="not ready") as error:
        supervisor.start(
            service.RLTStage2StartRequest(
                dataset_paths=["/not-inspected"],
                groot_checkpoint="/not-inspected",
                rl_token_encoder_path="/not-inspected",
            )
        )
    assert error.value.status_code == 503


def test_dataset_requires_outcome_labels(roots):
    dataset_root, *_ = roots
    dataset = _write_v21(dataset_root / "unlabelled", outcomes=False)
    with pytest.raises(HTTPException, match="episode_success"):
        service._resolve_datasets([str(dataset)])


def test_new_encoder_requires_completed_matching_stage1(roots):
    _, model_root, stage1_root, *_ = roots
    groot = _write_groot(model_root / "showroom_groot")
    encoder = _write_stage1_encoder(stage1_root / "run", groot)
    assert service._resolve_stage1_encoder(str(encoder), groot.resolve()) == encoder.resolve()

    other = _write_groot(model_root / "other_groot")
    with pytest.raises(HTTPException, match="different GR00T"):
        service._resolve_stage1_encoder(str(encoder), other.resolve())


def test_resume_requires_self_contained_bundle(roots):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(stage2_root / "round-1")
    assert service._resolve_resume_bundle(str(bundle)) == bundle.resolve()
    (bundle / "artifacts" / "rlt_actor.pt").unlink()
    with pytest.raises(HTTPException, match="incomplete"):
        service._resolve_resume_bundle(str(bundle))


def test_new_and_resume_commands_have_disjoint_sources(tmp_path):
    supervisor = _supervisor()
    new_job = _job(tmp_path)
    new_command = supervisor._command(new_job)
    assert "runtime.rlt_stage2_training_cli" in new_command
    assert new_command[new_command.index("--initialization-mode") + 1] == "new"
    assert new_command[new_command.index("--groot-checkpoint") + 1] == new_job.groot_checkpoint
    assert new_command[new_command.index("--rl-token-encoder") + 1] == new_job.rl_token_encoder_path
    assert "--rlt-bundle" not in new_command

    resume_job = _job(
        tmp_path,
        initialization_mode="resume",
        groot_checkpoint="",
        rl_token_encoder_path="",
        rlt_bundle_path="/workspace/checkpoint/rlt/stage2/round-1",
    )
    resume_command = supervisor._command(resume_job)
    assert resume_command[resume_command.index("--initialization-mode") + 1] == "resume"
    assert resume_command[resume_command.index("--rlt-bundle") + 1] == resume_job.rlt_bundle_path
    assert "--groot-checkpoint" not in resume_command
    assert "--rl-token-encoder" not in resume_command


def test_monitor_requires_exact_self_contained_bundle(tmp_path):
    output = _write_stage2_bundle(tmp_path / "stage2-output")
    actor = output / "artifacts" / "rlt_actor.pt"
    encoder = output / "artifacts" / "rl_token_encoder.pt"
    checkpoint = output / "training_state" / "rlt_stage2.pt"
    manifest = output / "manifest.json"
    lines = [
        json.dumps(
            {
                "event": "stage2_training_progress",
                "completed_critic_updates": 50,
                "total_critic_updates": 100,
                "actor_loss": -0.25,
                "critic_loss": 0.5,
                "average_reward": 0.75,
            }
        )
        + "\n",
        json.dumps(
            {
                "event": "stage2_training_result",
                "status": "completed",
                "completed_critic_updates": 100,
                "actor_artifact": str(actor),
                "encoder_artifact": str(encoder),
                "checkpoint": str(checkpoint),
                "bundle_manifest": str(manifest),
            }
        )
        + "\n",
    ]
    job = _job(tmp_path, output_dir=str(output), process=FakeProcess(stdout=lines))
    supervisor = _supervisor()
    supervisor._job = job
    supervisor._monitor(job)
    status = supervisor.status()
    assert status.status == "completed"
    assert status.percentage == 100.0
    assert status.actor_loss == pytest.approx(-0.25)
    assert status.critic_loss == pytest.approx(0.5)
    assert status.average_reward == pytest.approx(0.75)
    assert status.actor_artifact_path == str(actor)


def test_start_new_and_resume_validate_then_launch(roots, monkeypatch):
    dataset_root, model_root, stage1_root, stage2_root, _ = roots
    dataset = _write_v21(dataset_root / "selected")
    groot = _write_groot(model_root / "showroom_groot")
    encoder = _write_stage1_encoder(stage1_root / "run", groot)
    bundle = _write_stage2_bundle(stage2_root / "round-1")
    captured = []

    def fake_popen(command, **kwargs):
        captured.append((command, kwargs))
        return FakeProcess()

    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(service.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(service.threading, "Thread", NoopThread)
    supervisor = _supervisor()
    new_status = supervisor.start(
        service.RLTStage2StartRequest(
            initialization_mode="new",
            dataset_paths=[str(dataset)],
            groot_checkpoint=str(groot),
            rl_token_encoder_path=str(encoder),
            steps=20,
        )
    )
    assert new_status.status == "running"
    assert new_status.initialization_mode == "new"
    supervisor._job.status = "completed"
    resume_status = supervisor.start(
        service.RLTStage2StartRequest(
            initialization_mode="resume",
            dataset_paths=[str(dataset)],
            rlt_bundle_path=str(bundle),
            steps=20,
        )
    )
    assert resume_status.status == "running"
    assert resume_status.initialization_mode == "resume"
    assert resume_status.rlt_bundle_path == str(bundle.resolve())
    assert resume_status.groot_checkpoint == "/workspace/model/groot/showroom_groot"
    assert resume_status.output_dir != str(bundle.resolve())
    assert captured[-1][1]["env"] == {"COMPOSE_PROJECT_NAME": "cyclo"}


def test_modes_reject_ambiguous_sources(roots):
    dataset_root, *_ = roots
    dataset = _write_v21(dataset_root / "selected")
    supervisor = _supervisor()
    with pytest.raises(HTTPException, match="must not include"):
        supervisor.start(
            service.RLTStage2StartRequest(
                initialization_mode="new",
                dataset_paths=[str(dataset)],
                groot_checkpoint="/missing",
                rl_token_encoder_path="/missing",
                rlt_bundle_path="/also-not-allowed",
            )
        )
    with pytest.raises(HTTPException, match="accepts only"):
        supervisor.start(
            service.RLTStage2StartRequest(
                initialization_mode="resume",
                dataset_paths=[str(dataset)],
                groot_checkpoint="/ambiguous",
                rlt_bundle_path="/missing",
            )
        )


def test_conflict_is_checked_before_paths():
    supervisor = _supervisor(conflict=lambda: "GPU training is busy")
    with pytest.raises(HTTPException, match="GPU training is busy"):
        supervisor.start(
            service.RLTStage2StartRequest(
                dataset_paths=["/missing"],
                groot_checkpoint="/missing",
                rl_token_encoder_path="/missing",
            )
        )


def test_stop_signals_only_owned_container(tmp_path):
    interrupted = []
    process = FakeProcess()
    job = _job(tmp_path, process=process)
    supervisor = _supervisor(
        interrupt_container=lambda name: interrupted.append(name) or True
    )
    supervisor._job = job
    with pytest.raises(HTTPException, match="does not match"):
        supervisor.stop(service.RLTStage2StopRequest(job_id="stale"))
    status = supervisor.stop(service.RLTStage2StopRequest(job_id=job.job_id))
    assert status.phase == "stopping"
    assert interrupted == ["cyclo_rlt_stage2_aaaaaaaaaaaa"]
    assert process.signals == []


def test_stop_falls_back_to_compose_process_signal(tmp_path):
    process = FakeProcess()
    job = _job(tmp_path, process=process)
    supervisor = _supervisor(interrupt_container=lambda _name: False)
    supervisor._job = job
    supervisor.stop(service.RLTStage2StopRequest(job_id=job.job_id))
    assert process.signals == [signal.SIGINT]
