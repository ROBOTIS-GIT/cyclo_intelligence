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
import os
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


def _write_v30(path: Path, *, outcomes: bool = True) -> Path:
    meta = path / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    features = {"episode_success": {"dtype": "bool"}} if outcomes else {}
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": 2,
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    (meta / "tasks.parquet").write_bytes(b"tasks")
    (meta / "episodes/chunk-000/file-000.parquet").write_bytes(b"episodes")
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


def _write_stage2_bundle(
    root: Path,
    *,
    groot_checkpoint: str = "/workspace/model/groot/showroom_groot",
) -> Path:
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
    spec_fingerprint = service._canonical_fingerprint(spec)
    replay_root = root / "replay_cache"
    replay_root.mkdir()
    replay_artifact = replay_root / "replay.pt"
    replay_artifact.write_bytes(b"replay")
    dataset_content_fingerprint = "d" * 64
    dataset_collection_fingerprint = service._canonical_fingerprint(
        {
            "format": "cyclo.groot.rlt.dataset_collection/v1",
            "ordered_content_fingerprints": [dataset_content_fingerprint],
        }
    )
    reference_extraction = {"seed": 0, "feature_batch_size": 2}
    reward_contract = "terminal_success_plus_one_discounted_within_chunk/v1"
    replay_manifest = {
        "format": "cyclo.groot.rlt.stage2_feature_replay_manifest/v1",
        "file": "replay.pt",
        "byte_count": replay_artifact.stat().st_size,
        "sha256": service._file_sha256(replay_artifact),
        "spec_fingerprint": spec_fingerprint,
        "transition_count": 20,
        "average_reward": 0.5,
        "metadata": {
            "transition_count": 20,
            "reward_contract": reward_contract,
            "dataset_snapshot_fingerprint": dataset_collection_fingerprint,
            "reference_extraction": reference_extraction,
        },
    }
    replay_manifest_path = replay_root / "manifest.json"
    replay_manifest_path.write_text(json.dumps(replay_manifest), encoding="utf-8")
    training_round_unsigned = {
        "format": "cyclo_brain.rlt.stage2_training_round/v1",
        "replay": {
            "manifest": {
                "relative_path": "manifest.json",
                "byte_count": replay_manifest_path.stat().st_size,
                "sha256": service._file_sha256(replay_manifest_path),
            },
            "artifact": {
                "relative_path": "replay.pt",
                "byte_count": replay_artifact.stat().st_size,
                "sha256": service._file_sha256(replay_artifact),
            },
            "manifest_fingerprint": service._canonical_fingerprint(replay_manifest),
            "spec_fingerprint": spec_fingerprint,
            "transition_count": 20,
            "average_reward": 0.5,
            "reward_contract": reward_contract,
        },
        "datasets": {
            "snapshot_fingerprint": dataset_collection_fingerprint,
            "snapshots": [
                {
                    "ordinal": 0,
                    "file_count": 3,
                    "total_byte_count": 1024,
                    "content_fingerprint": dataset_content_fingerprint,
                }
            ],
        },
        "reference_extraction": reference_extraction,
        "optimization": {
            "sampling_seed": 0,
            "batch_size": 4,
            "steps": 100,
            "starting_critic_updates": 0,
        },
    }
    training_round = {
        **training_round_unsigned,
        "round_fingerprint": service._canonical_fingerprint(training_round_unsigned),
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
            "groot_checkpoint": groot_checkpoint,
            "groot_checkpoint_fingerprint": "a" * 64,
            "representation_contract_fingerprint": "c" * 64,
            "rl_token_artifact_fingerprint": "b" * 64,
        },
        "spec": spec,
        "spec_fingerprint": spec_fingerprint,
        "completed_critic_updates": 100,
        "completed_actor_updates": 50,
        "training_round": training_round,
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


def _rewrite_bundle_manifest(
    bundle: Path,
    mutate,
    *,
    refresh_round_fingerprint: bool = True,
) -> dict:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    training_round = manifest.get("training_round")
    if refresh_round_fingerprint and isinstance(training_round, dict):
        round_unsigned = {
            key: value
            for key, value in training_round.items()
            if key != "round_fingerprint"
        }
        training_round["round_fingerprint"] = service._canonical_fingerprint(
            round_unsigned
        )
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    }
    manifest["manifest_fingerprint"] = service._canonical_fingerprint(unsigned)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


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


def test_restart_recovers_newest_verified_completed_stage2(roots):
    *_, stage2_root, _ = roots
    older = _write_stage2_bundle(
        stage2_root / f"steps_{100:07d}_{'a' * 12}"
    )
    newer = _write_stage2_bundle(
        stage2_root / f"steps_{100:07d}_{'b' * 12}"
    )
    os.utime(older / "manifest.json", ns=(1, 1))
    os.utime(newer / "manifest.json", ns=(2, 2))

    status = _supervisor().status()

    assert status.status == "completed"
    assert status.output_dir == str(newer.resolve())
    assert status.actor_artifact_path == str(
        newer.resolve() / "artifacts/rlt_actor.pt"
    )
    assert status.encoder_artifact_path == str(
        newer.resolve() / "artifacts/rl_token_encoder.pt"
    )
    assert status.checkpoint_path == str(
        newer.resolve() / "training_state/rlt_stage2.pt"
    )
    assert status.manifest_path == str(newer.resolve() / "manifest.json")
    assert status.total_steps == 100
    assert status.batch_size == 4
    assert status.save_freq == 0
    assert status.average_reward == pytest.approx(0.5)
    assert "Recovered" in status.message


def test_restart_skips_newer_tampered_stage2_bundle(roots):
    *_, stage2_root, _ = roots
    valid = _write_stage2_bundle(
        stage2_root / f"steps_{100:07d}_{'a' * 12}"
    )
    tampered = _write_stage2_bundle(
        stage2_root / f"steps_{100:07d}_{'b' * 12}"
    )
    (tampered / "replay_cache/replay.pt").write_bytes(b"tampered replay")
    os.utime(valid / "manifest.json", ns=(1, 1))
    os.utime(tampered / "manifest.json", ns=(2, 2))

    status = _supervisor().status()

    assert status.status == "completed"
    assert status.output_dir == str(valid.resolve())


def test_restart_ignores_stage2_bundle_with_symlink_artifact(roots, tmp_path):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(
        stage2_root / f"steps_{100:07d}_{'a' * 12}"
    )
    actor = bundle / "artifacts/rlt_actor.pt"
    actor.unlink()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"actor")
    actor.symlink_to(outside)

    assert _supervisor().status().status == "idle"


def test_recovered_stage2_status_preserves_runtime_readiness(roots):
    *_, stage2_root, _ = roots
    _write_stage2_bundle(stage2_root / f"steps_{100:07d}_{'a' * 12}")
    supervisor = _supervisor(
        readiness=lambda: (False, "RLT runtime file is missing")
    )

    status = supervisor.status()

    assert status.status == "completed"
    assert status.ready is False
    assert "RLT runtime file is missing" in status.message


def test_repository_root_uses_live_container_mount(tmp_path, monkeypatch):
    runtime = tmp_path / "cyclo_brain" / "policy" / "groot" / "runtime"
    runtime.mkdir(parents=True)
    monkeypatch.setenv("CYCLO_SUPERVISOR_API_REPO_MOUNT", str(tmp_path))

    assert service._repository_root() == tmp_path


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


def test_native_v30_dataset_is_accepted_and_requires_outcomes(roots):
    dataset_root, *_ = roots
    dataset = _write_v30(dataset_root / "selected-v30")
    assert service._resolve_datasets([str(dataset)]) == [dataset.resolve()]

    unlabeled = _write_v30(dataset_root / "unlabelled-v30", outcomes=False)
    with pytest.raises(HTTPException, match="episode_success"):
        service._resolve_datasets([str(unlabeled)])


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


@pytest.mark.parametrize("ordinal", [False, 0.0, "0", -1])
def test_training_and_service_share_round_validation(roots, ordinal):
    from cyclo_brain.contracts.rlt import validate_training_round

    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(stage2_root / "round-1")

    def tamper(manifest):
        manifest["training_round"]["datasets"]["snapshots"][0]["ordinal"] = ordinal

    _rewrite_bundle_manifest(bundle, tamper)
    manifest = json.loads((bundle / "manifest.json").read_text())
    with pytest.raises(ValueError, match="ordinal"):
        validate_training_round(manifest["training_round"])
    with pytest.raises(HTTPException, match="ordinal") as error:
        service._resolve_resume_bundle(str(bundle))
    assert error.value.status_code == 400


def test_resume_rejects_v2_bundle_without_training_round(roots):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(stage2_root / "round-1")
    _rewrite_bundle_manifest(bundle, lambda manifest: manifest.pop("training_round"))

    with pytest.raises(HTTPException, match="manifest fields"):
        service._resolve_resume_bundle(str(bundle))


def test_resume_rejects_tampered_training_round_fingerprint(roots):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(stage2_root / "round-1")

    def tamper(manifest):
        manifest["training_round"]["optimization"]["steps"] += 1

    _rewrite_bundle_manifest(
        bundle,
        tamper,
        refresh_round_fingerprint=False,
    )

    with pytest.raises(HTTPException, match="training round fingerprint disagrees"):
        service._resolve_resume_bundle(str(bundle))


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("replay", "reward_contract", "training round replay fields"),
        ("datasets", "snapshots", "training round datasets"),
        ("reference_extraction", "feature_batch_size", "reference extraction fields"),
        ("optimization", "batch_size", "optimization provenance fields"),
    ],
)
def test_resume_rejects_incomplete_training_round_sections(
    roots,
    section,
    field,
    message,
):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(stage2_root / "round-1")

    def remove_field(manifest):
        del manifest["training_round"][section][field]

    _rewrite_bundle_manifest(bundle, remove_field)

    with pytest.raises(HTTPException, match=message):
        service._resolve_resume_bundle(str(bundle))


def test_resume_rejects_training_round_dataset_collection_mismatch(roots):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(stage2_root / "round-1")

    def tamper(manifest):
        manifest["training_round"]["datasets"]["snapshot_fingerprint"] = "e" * 64

    _rewrite_bundle_manifest(bundle, tamper)

    with pytest.raises(HTTPException, match="dataset summary collection disagrees"):
        service._resolve_resume_bundle(str(bundle))


def test_resume_rejects_tampered_training_round_replay_artifact(roots):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(stage2_root / "round-1")
    (bundle / "replay_cache" / "replay.pt").write_bytes(b"forged")

    with pytest.raises(HTTPException, match="replay artifact"):
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
    runtime_environment = [
        new_command[index + 1]
        for index, value in enumerate(new_command[:-1])
        if value == "--env"
    ]
    assert f"HF_HOME={service.RLT_STAGE2_CACHE_ROOT}/huggingface" in runtime_environment
    assert f"HF_HUB_CACHE={service.RLT_STAGE2_HF_HUB_CACHE}" in runtime_environment
    assert f"HUGGINGFACE_HUB_CACHE={service.RLT_STAGE2_HF_HUB_CACHE}" in runtime_environment
    assert f"TRANSFORMERS_CACHE={service.RLT_STAGE2_HF_HUB_CACHE}" in runtime_environment
    assert "HF_HUB_OFFLINE=1" in runtime_environment
    assert "TRANSFORMERS_OFFLINE=1" in runtime_environment
    assert "GROOT_HF_LOCAL_FIRST=1" in runtime_environment
    assert "GROOT_PATCH_MISTRAL=1" in runtime_environment
    assert "NO_ALBUMENTATIONS_UPDATE=1" in runtime_environment
    assert all("/root/.cache" not in value for value in runtime_environment)

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


def test_monitor_preserves_cli_failure_detail(tmp_path):
    failure = "PermissionError: [Errno 13] Permission denied: '/root/.cache'"
    lines = [
        json.dumps(
            {
                "event": "result",
                "status": "failed",
                "error": failure,
            }
        )
        + "\n"
    ]
    job = _job(tmp_path, process=FakeProcess(stdout=lines, returncode=1))
    supervisor = _supervisor()
    supervisor._job = job

    supervisor._monitor(job)

    status = supervisor.status()
    assert status.status == "failed"
    assert status.phase == "error"
    assert status.message == failure


def test_start_new_and_resume_validate_then_launch(roots, monkeypatch):
    dataset_root, model_root, stage1_root, stage2_root, _ = roots
    dataset = _write_v21(dataset_root / "selected")
    groot = _write_groot(model_root / "showroom_groot")
    encoder = _write_stage1_encoder(stage1_root / "run", groot)
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        groot_checkpoint=str(groot.resolve()),
    )
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
            expected_groot_checkpoint=str(groot),
            rlt_bundle_path=str(bundle),
            steps=20,
        )
    )
    assert resume_status.status == "running"
    assert resume_status.initialization_mode == "resume"
    assert resume_status.rlt_bundle_path == str(bundle.resolve())
    assert resume_status.groot_checkpoint == str(groot.resolve())
    assert resume_status.output_dir != str(bundle.resolve())
    assert captured[-1][1]["env"] == {"COMPOSE_PROJECT_NAME": "cyclo"}


def test_resume_rejects_bundle_from_different_selected_groot(roots, monkeypatch):
    dataset_root, model_root, _, stage2_root, _ = roots
    dataset = _write_v21(dataset_root / "selected")
    bundle_groot = _write_groot(model_root / "bundle_groot")
    selected_groot = _write_groot(model_root / "selected_groot")
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        groot_checkpoint=str(bundle_groot.resolve()),
    )
    popen_called = False

    def fail_popen(*_args, **_kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("mismatched resume must not launch")

    monkeypatch.setattr(service.subprocess, "Popen", fail_popen)
    supervisor = _supervisor()

    with pytest.raises(HTTPException, match="does not match") as error:
        supervisor.start(
            service.RLTStage2StartRequest(
                initialization_mode="resume",
                dataset_paths=[str(dataset)],
                expected_groot_checkpoint=str(selected_groot),
                rlt_bundle_path=str(bundle),
            )
        )

    assert error.value.status_code == 409
    assert popen_called is False
    assert supervisor.status().status == "idle"


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
