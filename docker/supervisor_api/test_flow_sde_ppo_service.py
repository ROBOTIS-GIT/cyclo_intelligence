#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Focused lifecycle tests for the Flow-SDE PPO supervisor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException


SERVICE_PATH = Path(__file__).resolve().with_name("flow_sde_ppo_service.py")
SPEC = importlib.util.spec_from_file_location(
    "flow_sde_ppo_service_under_test",
    SERVICE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
service = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = service
SPEC.loader.exec_module(service)


class FakeProcess:
    def __init__(self, *, stdout=(), returncode=0, signal_error=False):
        self.stdout = list(stdout)
        self._wait_returncode = returncode
        self._poll_returncode = None
        self.signal_error = signal_error
        self.signals = []

    def wait(self):
        self._poll_returncode = self._wait_returncode
        return self._wait_returncode

    def poll(self):
        return self._poll_returncode

    def send_signal(self, requested_signal):
        if self.signal_error:
            raise OSError("signal failed")
        self.signals.append(requested_signal)


def _write_policy(path: Path, *, policy_type: str = "multi_task_dit") -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"type": policy_type}),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (path / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    return path


def _write_dataset(
    path: Path,
    *,
    version: str = "v3.0",
    success_dtype: str | None = "bool",
) -> Path:
    (path / "meta").mkdir(parents=True)
    features = {}
    if success_dtype is not None:
        features["episode_success"] = {
            "dtype": success_dtype,
            "shape": [1],
        }
    (path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "total_episodes": 1,
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    policy_root = tmp_path / "checkpoint" / "multi_task_dit"
    policy_root.mkdir(parents=True)
    output_root = policy_root / "flow_sde_ppo"
    warmup_root = output_root / "value_warmup"
    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    log_root = tmp_path / "logs"
    monkeypatch.setattr(service, "FLOW_SDE_POLICY_ROOTS", (policy_root,))
    monkeypatch.setattr(service, "FLOW_SDE_DATASET_ROOTS", (dataset_root,))
    monkeypatch.setattr(service, "FLOW_SDE_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(service, "FLOW_SDE_VALUE_WARMUP_ROOT", warmup_root)
    monkeypatch.setattr(service, "FLOW_SDE_LOG_ROOT", log_root)
    return policy_root, output_root, log_root


def _supervisor(*, conflict=lambda: None, interrupt_container=None):
    return service.FlowSDEPPOSupervisor(
        compose_command=lambda: ["docker", "compose", "-f", "/tmp/compose.yml"],
        compose_environment=lambda: {"COMPOSE_PROJECT_NAME": "cyclo"},
        conflict_message=conflict,
        interrupt_container=interrupt_container,
    )


def _job(tmp_path: Path, **overrides):
    output_dir = tmp_path / "output"
    values = {
        "job_id": "a" * 32,
        "policy_checkpoint": "/workspace/checkpoint/multi_task_dit/base",
        "robot_type": "ffw_sg2_rev1",
        "task_instruction": "pick up the jelly bag",
        "output_dir": str(output_dir),
        "control_file": str(output_dir / "control" / "outcome.json"),
        "log_path": str(tmp_path / "logs" / "job.log"),
        "episodes": 2,
        "ppo_epochs": 4,
        "minibatch_size": 4,
        "max_chunk_decisions": 20,
        "actor_learning_rate": 3.0e-5,
        "value_learning_rate": 1.0e-4,
        "ack_timeout_seconds": 5.0,
        "sensor_timeout_seconds": 15.0,
    }
    values.update(overrides)
    return service._FlowSDEPPOJob(**values)


def _warmup_job(tmp_path: Path, **overrides):
    output_dir = tmp_path / "warmup-output"
    values = {
        "job_id": "c" * 32,
        "policy_checkpoint": "/workspace/checkpoint/multi_task_dit/base",
        "dataset_paths": [
            "/workspace/lerobot/data_epoch_0000",
            "/workspace/lerobot/data_epoch_0001",
        ],
        "task_instruction": "pick up the jelly bag",
        "output_dir": str(output_dir),
        "log_path": str(tmp_path / "logs" / "warmup.log"),
        "total_steps": 2000,
        "batch_size": 8,
        "value_learning_rate": 1.0e-4,
        "discount": 0.99,
    }
    values.update(overrides)
    return service._FlowSDEValueWarmupJob(**values)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _policy_hashes(path: Path) -> dict[str, str]:
    return {name: _sha256(path / name) for name in service.FLOW_SDE_POLICY_ARTIFACTS}


def _ppo_config(**overrides):
    values = service._expected_online_ppo_config(
        ppo_epochs=4,
        minibatch_size=4,
        actor_learning_rate=3.0e-5,
        value_learning_rate=1.0e-4,
    )
    values.update(overrides)
    return values


def _write_completed_online_bundle(
    output_root: Path,
    base_policy: Path,
    *,
    job_id: str = "d" * 32,
    task_instruction: str = "pick up the jelly bag",
    robot_type: str = "ffw_sg2_rev1",
    ppo_config: dict | None = None,
    update_step: int = 3,
) -> tuple[Path, Path, Path]:
    output = output_root / job_id
    model = _write_policy(output / "pretrained_model")
    checkpoint = output / "training_state" / "trainer_state.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"actor+critic+optimizer-state")
    (output / "progress.jsonl").write_text("{}\n", encoding="utf-8")
    config = dict(ppo_config or _ppo_config())
    base_policy = base_policy.resolve()
    base_hashes = _policy_hashes(base_policy)
    source_lineage = {"resume": None, "value_initialization": None}
    startup = {
        "format": service.FLOW_SDE_ONLINE_STARTUP_FORMAT,
        "status": "ready",
        "job_id": job_id,
        "base_checkpoint": str(base_policy),
        "base_policy_artifacts": base_hashes,
        "lineage_policy_checkpoint": str(base_policy),
        "lineage_policy_artifacts": base_hashes,
        "task_instruction": task_instruction,
        "robot_type": robot_type,
        "ppo_config": config,
        "resume_checkpoint": "",
        "source_lineage": source_lineage,
    }
    (output / "startup_manifest.json").write_text(
        json.dumps(startup),
        encoding="utf-8",
    )
    export = {
        "format": service.FLOW_SDE_ONLINE_EXPORT_FORMAT,
        "source_update_step": update_step,
        "ppo_config": config,
    }
    (model / "flow_sde_ppo_export.json").write_text(
        json.dumps(export),
        encoding="utf-8",
    )
    summary = {
        "status": "completed",
        "job_id": job_id,
        "episodes": 2,
        "updates": update_step,
        "trainer_checkpoint": str(checkpoint),
        "pretrained_model": str(model),
        "base_checkpoint": str(base_policy),
        "base_policy_artifacts": base_hashes,
        "lineage_policy_checkpoint": str(base_policy),
        "lineage_policy_artifacts": base_hashes,
        "task_instruction": task_instruction,
        "robot_type": robot_type,
        "source_lineage": source_lineage,
        "run_manifest": str(output / "run_manifest.json"),
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    digest = "sha256:" + "1" * 64
    manifest = {
        "format": service.FLOW_SDE_ONLINE_BUNDLE_FORMAT,
        "status": "complete",
        "job_id": job_id,
        "base_checkpoint": str(base_policy),
        "base_policy_artifacts": base_hashes,
        "lineage_policy_checkpoint": str(base_policy),
        "lineage_policy_artifacts": base_hashes,
        "task_instruction": task_instruction,
        "robot_type": robot_type,
        "ppo_config": config,
        "result": {
            "episodes": 2,
            "update_step": update_step,
            "actor_sha256": digest,
            "critic_sha256": digest,
            "frozen_policy_sha256": digest,
        },
        "source_lineage": source_lineage,
        "artifacts": {
            "pretrained_model": {
                "path": "pretrained_model",
                "files": _policy_hashes(model),
            },
            "trainer_checkpoint": {
                "path": "training_state/trainer_state.pt",
                "sha256": _sha256(checkpoint),
            },
            "startup_manifest_path": "startup_manifest.json",
            "progress_path": "progress.jsonl",
            "summary_path": "summary.json",
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary["run_manifest_sha256"] = _sha256(manifest_path)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return output, model, checkpoint


def test_idle_status_and_routes_are_explicit(roots):
    supervisor = _supervisor()

    status = supervisor.status()
    paths = {route.path for route in supervisor.router.routes}

    assert status.ready is True
    assert status.status == "idle"
    assert status.phase == "idle"
    assert paths == {
        "/flow-sde-ppo/start",
        "/flow-sde-ppo/status",
        "/flow-sde-ppo/stop",
        "/flow-sde-ppo/outcome",
        "/flow-sde-ppo/value-warmup/start",
        "/flow-sde-ppo/value-warmup/status",
        "/flow-sde-ppo/value-warmup/stop",
    }
    warmup = supervisor.value_warmup_status()
    assert warmup.status == "idle"
    assert warmup.ready is True


def test_policy_checkpoint_accepts_nested_export_and_rejects_escapes(
    roots,
    tmp_path,
):
    policy_root, _, _ = roots
    run = policy_root / "run"
    expected = _write_policy(run / "checkpoints" / "last" / "pretrained_model")

    assert service._resolve_policy_checkpoint(str(run)) == expected.resolve()

    with pytest.raises(HTTPException, match="absolute path"):
        service._resolve_policy_checkpoint("relative/checkpoint")
    outside = _write_policy(tmp_path / "outside")
    with pytest.raises(HTTPException, match="must be under"):
        service._resolve_policy_checkpoint(str(outside))


def test_policy_checkpoint_allows_internal_last_symlink(roots):
    policy_root, _, _ = roots
    run = policy_root / "run"
    target = _write_policy(run / "checkpoints" / "000010" / "pretrained_model")
    last = run / "checkpoints" / "last"
    last.symlink_to("000010", target_is_directory=True)

    assert service._resolve_policy_checkpoint(str(run)) == target.resolve()
    assert service._resolve_policy_checkpoint(
        str(last / "pretrained_model")
    ) == target.resolve()


def test_policy_checkpoint_rejects_symlink_escape(roots, tmp_path):
    policy_root, _, _ = roots
    outside = _write_policy(tmp_path / "outside-target")
    alias = policy_root / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(HTTPException, match="escapes its root"):
        service._resolve_policy_checkpoint(str(alias))


def test_policy_checkpoint_rejects_symlinked_artifact(roots):
    policy_root, _, _ = roots
    checkpoint = policy_root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "real-config.json").write_text(
        json.dumps({"type": "multi_task_dit"}),
        encoding="utf-8",
    )
    (checkpoint / "config.json").symlink_to("real-config.json")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (checkpoint / "policy_postprocessor.json").write_text("{}", encoding="utf-8")

    with pytest.raises(HTTPException, match="regular files"):
        service._resolve_policy_checkpoint(str(checkpoint))


def test_policy_checkpoint_rejects_artifact_unreadable_by_training_uid(roots):
    policy_root, _, _ = roots
    checkpoint = _write_policy(policy_root / "checkpoint")
    weights = checkpoint / "model.safetensors"
    weights.chmod(0o600)

    with pytest.raises(HTTPException, match="not readable by the LeRobot training user"):
        service._resolve_policy_checkpoint(str(checkpoint))


def test_value_warmup_datasets_require_safe_labeled_lerobot_v3_roots(
    roots,
    tmp_path,
):
    dataset_root = service.FLOW_SDE_DATASET_ROOTS[0]
    first = _write_dataset(dataset_root / "data_epoch_0000")
    second = _write_dataset(dataset_root / "data_epoch_0001")

    assert service._resolve_value_warmup_datasets(
        [str(first), str(second)]
    ) == [first.resolve(), second.resolve()]

    with pytest.raises(HTTPException, match="duplicates"):
        service._resolve_value_warmup_datasets([str(first), str(first)])
    with pytest.raises(HTTPException, match="absolute paths"):
        service._resolve_value_warmup_datasets(["relative/dataset"])
    outside = _write_dataset(tmp_path / "outside-dataset")
    with pytest.raises(HTTPException, match="must be under"):
        service._resolve_value_warmup_datasets([str(outside)])

    old = _write_dataset(dataset_root / "old", version="v2.1")
    with pytest.raises(HTTPException, match="LeRobot v3.0"):
        service._resolve_value_warmup_datasets([str(old)])
    unlabeled = _write_dataset(dataset_root / "unlabeled", success_dtype=None)
    with pytest.raises(HTTPException, match="boolean episode_success"):
        service._resolve_value_warmup_datasets([str(unlabeled)])
    wrong_dtype = _write_dataset(dataset_root / "wrong-dtype", success_dtype="int64")
    with pytest.raises(HTTPException, match="boolean episode_success"):
        service._resolve_value_warmup_datasets([str(wrong_dtype)])


def test_value_warmup_dataset_rejects_symlink_components(roots):
    dataset_root = service.FLOW_SDE_DATASET_ROOTS[0]
    target = _write_dataset(dataset_root / "target")
    alias = dataset_root / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(HTTPException, match="symbolic links"):
        service._resolve_value_warmup_datasets([str(alias)])


def test_output_path_rejects_symlink_root(roots, tmp_path, monkeypatch):
    policy_root, _, _ = roots
    outside = tmp_path / "outside-output"
    outside.mkdir()
    linked_output = policy_root / "linked-output"
    linked_output.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(service, "FLOW_SDE_OUTPUT_ROOT", linked_output)

    with pytest.raises(HTTPException, match="symbolic links"):
        service._flow_sde_output_path("job")


def test_prepare_job_directories_transfers_uid_1000_ownership(
    roots,
    monkeypatch,
):
    _, output_root, _ = roots
    output_dir = output_root / "job"
    ownership = []
    monkeypatch.setattr(
        service.os,
        "chown",
        lambda path, uid, gid: ownership.append((Path(path), uid, gid)),
    )

    control_dir = service._prepare_job_directories(output_dir)

    assert control_dir == output_dir / "control"
    assert ownership == [
        (output_dir, 1000, 1000),
        (control_dir, 1000, 1000),
    ]


def test_prepare_job_directories_cleans_empty_tree_on_chown_failure(
    roots,
    monkeypatch,
):
    _, output_root, _ = roots
    output_dir = output_root / "job"
    monkeypatch.setattr(
        service.os,
        "chown",
        lambda *args: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(HTTPException, match="writable Flow-SDE PPO output"):
        service._prepare_job_directories(output_dir)

    assert not output_dir.exists()


def test_value_warmup_output_is_dedicated_and_owned_by_training_uid(
    roots,
    monkeypatch,
):
    output = service._flow_sde_value_warmup_output_path("job")
    ownership = []
    monkeypatch.setattr(
        service.os,
        "chown",
        lambda path, uid, gid: ownership.append((Path(path), uid, gid)),
    )

    service._prepare_value_warmup_directory(output)

    assert output == service.FLOW_SDE_VALUE_WARMUP_ROOT / "job"
    assert not output.exists()
    assert ownership == [(service.FLOW_SDE_VALUE_WARMUP_ROOT, 1000, 1000)]


def test_value_warmup_output_rejects_symlink_root(roots, tmp_path, monkeypatch):
    policy_root, _, _ = roots
    outside = tmp_path / "outside-warmup"
    outside.mkdir()
    linked = policy_root / "linked-warmup"
    linked.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(service, "FLOW_SDE_VALUE_WARMUP_ROOT", linked)

    with pytest.raises(HTTPException, match="symbolic links"):
        service._flow_sde_value_warmup_output_path("job")


def test_command_uses_dedicated_cli_container_and_writable_caches(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)
    supervisor = _supervisor()
    job = _job(tmp_path)

    command = supervisor._command(job)

    assert command[:5] == [
        "docker",
        "compose",
        "-f",
        "/tmp/compose.yml",
        "run",
    ]
    assert command[command.index("--name") + 1] == "cyclo_flow_sde_ppo_aaaaaaaaaaaa"
    assert "HOME=/tmp" in command
    assert "HF_LEROBOT_HOME=/tmp/cyclo_flow_sde_ppo_cache/huggingface/lerobot" in command
    assert "HF_HUB_CACHE=/huggingface_hub" in command
    assert "HUGGINGFACE_HUB_CACHE=/huggingface_hub" in command
    assert "TRANSFORMERS_CACHE=/huggingface_hub" in command
    assert "ZENOH_ROS2_SDK_CACHE=/zenoh_cache" in command
    assert "ROS_DOMAIN_ID=30" in command
    assert command[command.index("--entrypoint") + 1] == "/lerobot/.venv/bin/python"
    assert "cyclo_brain.algorithm.rl.flow_sde_ppo.live_cli" in command
    assert command[command.index("--base-checkpoint") + 1] == job.policy_checkpoint
    assert command[command.index("--max-chunk-decisions") + 1] == "20"
    assert command[command.index("--ack-timeout") + 1] == "5.0"
    assert command[command.index("--sensor-timeout") + 1] == "15.0"
    assert "--value-warmup-bundle" not in command


def test_online_command_and_status_include_optional_value_warmup_bundle(tmp_path):
    bundle = "/workspace/checkpoint/multi_task_dit/flow_sde_ppo/value_warmup/bundle"
    supervisor = _supervisor()
    job = _job(tmp_path, value_warmup_bundle=bundle)

    command = supervisor._command(job)
    status = supervisor._status(job)

    assert command[command.index("--value-warmup-bundle") + 1] == bundle
    assert status.value_warmup_bundle == bundle


def test_online_request_without_warmup_preserves_existing_contract():
    request = service.FlowSDEPPOStartRequest(
        policy_checkpoint="/workspace/checkpoint/multi_task_dit/base",
        robot_type="ffw_sg2_rev1",
    )

    assert request.value_warmup_bundle is None
    assert request.resume_checkpoint is None


def test_online_command_and_status_include_explicit_resume(tmp_path):
    checkpoint = (
        "/workspace/checkpoint/multi_task_dit/flow_sde_ppo/"
        + "d" * 32
        + "/training_state/trainer_state.pt"
    )
    supervisor = _supervisor()
    job = _job(
        tmp_path,
        resume_checkpoint=checkpoint,
        resume_source_job_id="d" * 32,
        lineage_policy_checkpoint="/workspace/checkpoint/multi_task_dit/base",
    )

    command = supervisor._command(job)
    status = supervisor._status(job)

    assert command[command.index("--resume-checkpoint") + 1] == checkpoint
    assert status.resume_checkpoint == checkpoint
    assert status.resume_source_job_id == "d" * 32
    assert status.lineage_policy_checkpoint.endswith("/base")
    assert status.task_instruction == "pick up the jelly bag"


def test_online_resume_bundle_validates_actor_critic_config_and_lineage(roots):
    policy_root, output_root, _ = roots
    base = _write_policy(policy_root / "base")
    output, model, checkpoint = _write_completed_online_bundle(output_root, base)

    resolved = service._resolve_online_resume_checkpoint(
        str(checkpoint),
        policy_checkpoint=model,
        robot_type="ffw_sg2_rev1",
        task_instruction="pick up the jelly bag",
        ppo_epochs=4,
        minibatch_size=4,
        actor_learning_rate=3.0e-5,
        value_learning_rate=1.0e-4,
    )

    assert resolved.checkpoint == checkpoint.resolve()
    assert resolved.source_output_dir == output.resolve()
    assert resolved.source_job_id == "d" * 32
    assert resolved.source_model_path == model.resolve()
    assert resolved.lineage_policy_checkpoint == base.resolve()
    assert resolved.update_step == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("task", "task instruction"),
        ("config", "configuration"),
        ("checkpoint", "checkpoint hash"),
        ("model", "policy artifact hash"),
    ),
)
def test_online_resume_rejects_stale_or_incompatible_bundle(
    roots,
    mutation,
    message,
):
    policy_root, output_root, _ = roots
    base = _write_policy(policy_root / "base")
    _, model, checkpoint = _write_completed_online_bundle(output_root, base)
    kwargs = {
        "policy_checkpoint": model,
        "robot_type": "ffw_sg2_rev1",
        "task_instruction": "pick up the jelly bag",
        "ppo_epochs": 4,
        "minibatch_size": 4,
        "actor_learning_rate": 3.0e-5,
        "value_learning_rate": 1.0e-4,
    }
    if mutation == "task":
        kwargs["task_instruction"] = "different task"
    elif mutation == "config":
        kwargs["ppo_epochs"] = 5
    elif mutation == "checkpoint":
        checkpoint.write_bytes(b"tampered critic")
    elif mutation == "model":
        (model / "model.safetensors").write_bytes(b"tampered actor")

    with pytest.raises(HTTPException, match=message):
        service._resolve_online_resume_checkpoint(str(checkpoint), **kwargs)


def test_online_resume_path_is_exact_and_warmup_is_mutually_exclusive(
    roots,
):
    policy_root, output_root, _ = roots
    base = _write_policy(policy_root / "base")
    _, _, checkpoint = _write_completed_online_bundle(output_root, base)

    with pytest.raises(HTTPException, match="exactly"):
        service._resolve_online_resume_checkpoint(
            str(checkpoint.parent.parent / "trainer_state.pt"),
            policy_checkpoint=base,
            robot_type="ffw_sg2_rev1",
            task_instruction="pick up the jelly bag",
            ppo_epochs=4,
            minibatch_size=4,
            actor_learning_rate=3.0e-5,
            value_learning_rate=1.0e-4,
        )
    with pytest.raises(HTTPException, match="mutually exclusive"):
        _supervisor().start(
            service.FlowSDEPPOStartRequest(
                policy_checkpoint=str(base),
                robot_type="ffw_sg2_rev1",
                resume_checkpoint=str(checkpoint),
                value_warmup_bundle="/not/used",
            )
        )


def test_online_start_carries_validated_resume_and_recovery_survives_restart(
    roots,
    monkeypatch,
):
    policy_root, output_root, _ = roots
    base = _write_policy(policy_root / "base")
    _, model, checkpoint = _write_completed_online_bundle(output_root, base)
    commands = []
    monkeypatch.setattr(service.os, "chown", lambda *args: None)
    monkeypatch.setattr(
        service.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or FakeProcess(),
    )
    monkeypatch.setattr(
        service.threading,
        "Thread",
        lambda **kwargs: type("Thread", (), {"start": lambda self: None})(),
    )

    status = _supervisor().start(
        service.FlowSDEPPOStartRequest(
            policy_checkpoint=str(model),
            robot_type="ffw_sg2_rev1",
            resume_checkpoint=str(checkpoint),
        )
    )

    assert status.resume_checkpoint == str(checkpoint.resolve())
    assert status.resume_source_job_id == "d" * 32
    assert status.lineage_policy_checkpoint == str(base.resolve())
    assert commands[0][commands[0].index("--resume-checkpoint") + 1] == str(
        checkpoint.resolve()
    )

    recovered = _supervisor().status()
    assert recovered.status == "completed"
    assert recovered.job_id == "d" * 32
    assert recovered.checkpoint_path == str(checkpoint.resolve())
    assert recovered.model_path == str(model.resolve())
    assert recovered.lineage_policy_checkpoint == str(base.resolve())


def test_online_start_carries_validated_warmup_into_job_and_process_command(
    roots,
    monkeypatch,
):
    policy_root, _, _ = roots
    policy = _write_policy(policy_root / "base")
    bundle = _write_online_compatible_warmup_bundle(
        service.FLOW_SDE_VALUE_WARMUP_ROOT / ("c" * 32),
        policy_checkpoint=policy,
    )
    commands = []
    monkeypatch.setattr(service.os, "chown", lambda *args: None)
    monkeypatch.setattr(
        service.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or FakeProcess(),
    )
    monkeypatch.setattr(
        service.threading,
        "Thread",
        lambda **kwargs: type("Thread", (), {"start": lambda self: None})(),
    )

    status = _supervisor().start(
        service.FlowSDEPPOStartRequest(
            policy_checkpoint=str(policy),
            robot_type="ffw_sg2_rev1",
            value_warmup_bundle=str(bundle),
        )
    )

    assert status.value_warmup_bundle == str(bundle.resolve())
    assert commands[0][commands[0].index("--value-warmup-bundle") + 1] == str(
        bundle.resolve()
    )


def test_command_allows_ros_domain_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "42")

    command = _supervisor()._command(_job(tmp_path))

    assert "ROS_DOMAIN_ID=42" in command


def test_value_warmup_command_invokes_offline_cli_with_ordered_datasets(tmp_path):
    supervisor = _supervisor()
    job = _warmup_job(tmp_path)

    command = supervisor._value_warmup_command(job)

    assert command[command.index("--name") + 1] == (
        "cyclo_flow_sde_value_warmup_cccccccccccc"
    )
    assert "cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup_cli" in command
    assert "cyclo_brain.algorithm.rl.flow_sde_ppo.live_cli" not in command
    dataset_values = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--dataset-root"
    ]
    assert dataset_values == job.dataset_paths
    assert command[command.index("--base-checkpoint") + 1] == job.policy_checkpoint
    assert command[command.index("--output-dir") + 1] == job.output_dir
    assert command[command.index("--steps") + 1] == "2000"
    assert command[command.index("--batch-size") + 1] == "8"
    assert command[command.index("--value-lr") + 1] == "0.0001"
    assert command[command.index("--gamma") + 1] == "0.99"
    assert command[command.index("--task-instruction") + 1] == job.task_instruction
    assert command[command.index("--device") + 1] == "cuda"
    assert "ZENOH_SDK_PATH=/zenoh_sdk" not in command


def test_start_conflict_fails_before_path_or_process_validation(monkeypatch):
    supervisor = _supervisor(conflict=lambda: "another GPU trainer is running")
    monkeypatch.setattr(
        service.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Popen must not be called"),
    )
    request = service.FlowSDEPPOStartRequest(
        policy_checkpoint="/missing",
        robot_type="ffw_sg2_rev1",
    )

    with pytest.raises(HTTPException, match="another GPU trainer") as exc_info:
        supervisor.start(request)

    assert exc_info.value.status_code == 409


def test_start_launch_failure_removes_only_fresh_empty_job_tree(
    roots,
    monkeypatch,
):
    policy_root, output_root, _ = roots
    checkpoint = _write_policy(policy_root / "base")
    job_id = "b" * 32
    monkeypatch.setattr(service.uuid, "uuid4", lambda: type("U", (), {"hex": job_id})())
    monkeypatch.setattr(service.os, "chown", lambda *args: None)
    monkeypatch.setattr(
        service.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("launch failed")),
    )
    supervisor = _supervisor()

    with pytest.raises(HTTPException, match="Could not launch") as exc_info:
        supervisor.start(
            service.FlowSDEPPOStartRequest(
                policy_checkpoint=str(checkpoint),
                robot_type="ffw_sg2_rev1",
            )
        )

    assert exc_info.value.status_code == 503
    assert not (output_root / job_id).exists()
    assert supervisor.status().status == "idle"


def test_value_warmup_start_conflict_precedes_path_validation(monkeypatch):
    supervisor = _supervisor(conflict=lambda: "another GPU trainer is running")
    monkeypatch.setattr(
        service.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Popen must not be called"),
    )

    with pytest.raises(HTTPException, match="another GPU trainer") as exc_info:
        supervisor.start_value_warmup(
            service.FlowSDEValueWarmupStartRequest(
                policy_checkpoint="/missing",
                dataset_paths=["/missing"],
            )
        )

    assert exc_info.value.status_code == 409


def test_value_warmup_launch_failure_removes_fresh_output(
    roots,
    monkeypatch,
):
    policy_root, _, _ = roots
    checkpoint = _write_policy(policy_root / "base")
    dataset = _write_dataset(service.FLOW_SDE_DATASET_ROOTS[0] / "dataset")
    job_id = "d" * 32
    monkeypatch.setattr(service.uuid, "uuid4", lambda: type("U", (), {"hex": job_id})())
    monkeypatch.setattr(service.os, "chown", lambda *args: None)
    monkeypatch.setattr(
        service.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("launch failed")),
    )
    supervisor = _supervisor()

    with pytest.raises(HTTPException, match="Could not launch") as exc_info:
        supervisor.start_value_warmup(
            service.FlowSDEValueWarmupStartRequest(
                policy_checkpoint=str(checkpoint),
                dataset_paths=[str(dataset)],
            )
        )

    assert exc_info.value.status_code == 503
    assert not (service.FLOW_SDE_VALUE_WARMUP_ROOT / job_id).exists()
    assert supervisor.value_warmup_status().status == "idle"


def test_online_and_value_warmup_jobs_are_mutually_exclusive(
    roots,
    monkeypatch,
    tmp_path,
):
    policy_root, _, _ = roots
    checkpoint = _write_policy(policy_root / "base")
    dataset = _write_dataset(service.FLOW_SDE_DATASET_ROOTS[0] / "dataset")
    supervisor = _supervisor()
    supervisor._job = _job(tmp_path)

    with pytest.raises(HTTPException, match="Stop online Flow-SDE PPO"):
        supervisor.start_value_warmup(
            service.FlowSDEValueWarmupStartRequest(
                policy_checkpoint=str(checkpoint),
                dataset_paths=[str(dataset)],
            )
        )

    supervisor._job = None
    supervisor._value_warmup_job = _warmup_job(tmp_path)
    monkeypatch.setattr(
        service.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Popen must not be called"),
    )
    with pytest.raises(HTTPException, match="Stop Flow-SDE PPO value warm-up"):
        supervisor.start(
            service.FlowSDEPPOStartRequest(
                policy_checkpoint=str(checkpoint),
                robot_type="ffw_sg2_rev1",
            )
        )


def test_progress_events_update_typed_status_and_ignore_invalid_counters(tmp_path):
    supervisor = _supervisor()
    job = _job(tmp_path)

    completed = supervisor._consume_event(
        job,
        {
            "event": "progress",
            "phase": "collecting",
            "episode": 1,
            "chunk_decisions": 3,
            "update_step": 2,
            "actor_loss": -0.25,
            "value_loss": 0.5,
            "percentage": 145,
            "awaiting_outcome": True,
            "message": "Awaiting Success/Fail",
        },
    )

    assert completed is False
    assert job.phase == "collecting"
    assert job.episode == 1
    assert job.chunk_decisions == 3
    assert job.update_step == 2
    assert job.actor_loss == pytest.approx(-0.25)
    assert job.percentage == 100.0
    assert job.awaiting_outcome is True

    supervisor._consume_event(
        job,
        {
            "event": "progress",
            "episode": 1.5,
            "chunk_decisions": -7,
            "update_step": True,
        },
    )
    assert (job.episode, job.chunk_decisions, job.update_step) == (1, 3, 2)


def test_live_cli_episode_events_drive_outcome_and_nested_metrics(tmp_path):
    supervisor = _supervisor()
    job = _job(tmp_path)

    supervisor._consume_event(
        job,
        {"event": "starting", "stage": "load_checkpoint"},
    )
    assert job.phase == "load_checkpoint"
    supervisor._consume_event(job, {"event": "ready", "stage": "rollout"})
    assert job.phase == "rollout"
    supervisor._consume_event(
        job,
        {"event": "episode_started", "episode": 1, "episodes": 2},
    )
    assert job.awaiting_outcome is True
    assert job.phase == "collecting"

    supervisor._consume_event(
        job,
        {
            "event": "episode_cancelled",
            "episode": 1,
            "episodes": 2,
        },
    )
    assert job.awaiting_outcome is False
    assert job.phase == "resetting"

    supervisor._consume_event(
        job,
        {"event": "episode_started", "episode": 1, "episodes": 2, "attempt": 2},
    )
    assert job.awaiting_outcome is True

    supervisor._consume_event(
        job,
        {
            "event": "episode_updated",
            "episode": 1,
            "episodes": 2,
            "episode_return": 1.0,
            "checkpoint": "/tmp/trainer_state.pt",
            "metrics": {
                "update_step": 1,
                "transitions": 4,
                "actor_loss": -0.1,
                "value_loss": 0.2,
                "approx_kl": 0.03,
                "clip_fraction": 0.25,
            },
        },
    )
    assert job.awaiting_outcome is False
    assert job.episode_return == pytest.approx(1.0)
    assert job.update_step == 1
    assert job.chunk_decisions == 4
    assert job.actor_loss == pytest.approx(-0.1)
    assert job.value_loss == pytest.approx(0.2)
    assert job.approx_kl == pytest.approx(0.03)
    assert job.clip_fraction == pytest.approx(0.25)
    assert job.checkpoint_path == "/tmp/trainer_state.pt"
    assert job.percentage == pytest.approx(50.0)


def test_outcome_is_atomic_current_and_only_allowed_while_awaiting(tmp_path):
    supervisor = _supervisor()
    job = _job(tmp_path, awaiting_outcome=True)
    supervisor._job = job

    status = supervisor.outcome(
        service.FlowSDEPPOOutcomeRequest(job_id=job.job_id, outcome="success")
    )

    path = Path(job.control_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["job_id"] == job.job_id
    assert payload["outcome"] == "success"
    assert isinstance(payload["sequence"], int)
    assert status.awaiting_outcome is False
    assert list(path.parent.glob("*.tmp")) == []
    assert list(path.parent.glob(".*.tmp")) == []

    with pytest.raises(HTTPException, match="not awaiting"):
        supervisor.outcome(
            service.FlowSDEPPOOutcomeRequest(job_id=job.job_id, outcome="fail")
        )
    with pytest.raises(HTTPException, match="stale"):
        supervisor.outcome(
            service.FlowSDEPPOOutcomeRequest(job_id="stale", outcome="fail")
        )


def _write_deployable_model(output_dir: Path, *, policy_type="multi_task_dit") -> Path:
    model = output_dir / "pretrained_model"
    _write_policy(model, policy_type=policy_type)
    (model / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    (model / "policy_postprocessor.json").write_text("{}", encoding="utf-8")
    return model


def _write_value_warmup_bundle(
    output_dir: Path,
    *,
    policy_type: str = "multi_task_dit",
    manifest_format: str = "cyclo.flow_sde_ppo.value_warmup.bundle.v1",
) -> tuple[Path, Path]:
    model = _write_deployable_model(output_dir, policy_type=policy_type)
    checkpoint = output_dir / "training_state" / "value_warmup.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"warmup-state")
    (output_dir / "progress.jsonl").write_text(
        json.dumps({"event": "progress", "step": 2000}) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "format": manifest_format,
                "status": "complete",
                "base": {"path": "/base", "policy_sha256": "abc"},
                "datasets": [{"path": "/dataset", "identity_sha256": "def"}],
                "config": {"steps": 2000},
                "dataset_contract": {"version": "v3.0"},
                "result": {"completed_steps": 2000},
                "artifacts": {
                    "model_path": "pretrained_model",
                    "checkpoint_path": "training_state/value_warmup.pt",
                    "progress_path": "progress.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    return model, checkpoint


def _write_online_compatible_warmup_bundle(
    output_dir: Path,
    *,
    policy_checkpoint: Path,
    task_instruction: str = "pick up the jelly bag",
    status: str = "complete",
) -> Path:
    _write_value_warmup_bundle(output_dir)
    hashes = {}
    for name in service.FLOW_SDE_POLICY_ARTIFACTS:
        digest = hashlib.sha256((policy_checkpoint / name).read_bytes()).hexdigest()
        hashes[name] = f"sha256:{digest}"
    base_hash = "sha256:policy-contract"
    manifest = {
        "format": "cyclo.flow_sde_ppo.value_warmup.bundle.v1",
        "status": status,
        "base": {
            "path": str(policy_checkpoint.resolve()),
            "policy_sha256": base_hash,
            "artifacts": hashes,
        },
        "config": {
            "steps": 1000,
            "task_instruction": task_instruction,
        },
        "result": {
            "completed_steps": 1000,
            "policy_sha256_before": base_hash,
            "policy_sha256_after": base_hash,
        },
        "artifacts": {
            "model_path": "pretrained_model",
            "checkpoint_path": "training_state/value_warmup.pt",
            "progress_path": "progress.jsonl",
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return output_dir


def test_online_value_warmup_bundle_is_validated_against_policy_and_task(roots):
    policy_root, _, _ = roots
    policy = _write_policy(policy_root / "base")
    bundle = _write_online_compatible_warmup_bundle(
        service.FLOW_SDE_VALUE_WARMUP_ROOT / "bundle",
        policy_checkpoint=policy,
    )

    assert service._resolve_value_warmup_bundle(
        str(bundle),
        policy_checkpoint=policy.resolve(),
        task_instruction="pick up the jelly bag",
    ) == bundle.resolve()

    stale_policy = _write_policy(policy_root / "stale")
    with pytest.raises(HTTPException, match="different policy checkpoint"):
        service._resolve_value_warmup_bundle(
            str(bundle),
            policy_checkpoint=stale_policy.resolve(),
            task_instruction="pick up the jelly bag",
        )
    with pytest.raises(HTTPException, match="task instruction"):
        service._resolve_value_warmup_bundle(
            str(bundle),
            policy_checkpoint=policy.resolve(),
            task_instruction="pick up the peanuts",
        )


def test_online_value_warmup_bundle_requires_absolute_dedicated_root(
    roots,
    tmp_path,
):
    policy_root, _, _ = roots
    policy = _write_policy(policy_root / "base")
    outside = _write_online_compatible_warmup_bundle(
        tmp_path / "outside-bundle",
        policy_checkpoint=policy,
    )

    with pytest.raises(HTTPException, match="absolute path"):
        service._resolve_value_warmup_bundle(
            "relative-bundle",
            policy_checkpoint=policy.resolve(),
            task_instruction="pick up the jelly bag",
        )
    with pytest.raises(HTTPException, match="must be under"):
        service._resolve_value_warmup_bundle(
            str(outside),
            policy_checkpoint=policy.resolve(),
            task_instruction="pick up the jelly bag",
        )


def test_online_value_warmup_bundle_rejects_stale_policy_contents(roots):
    policy_root, _, _ = roots
    policy = _write_policy(policy_root / "base")
    bundle = _write_online_compatible_warmup_bundle(
        service.FLOW_SDE_VALUE_WARMUP_ROOT / "bundle",
        policy_checkpoint=policy,
    )
    (policy / "model.safetensors").write_bytes(b"changed-after-warmup")

    with pytest.raises(HTTPException, match="stale or was modified"):
        service._resolve_value_warmup_bundle(
            str(bundle),
            policy_checkpoint=policy.resolve(),
            task_instruction="pick up the jelly bag",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_checkpoint", "incomplete"),
        ("incomplete_manifest", "not complete"),
        ("missing_manifest", "incomplete"),
    ],
)
def test_online_value_warmup_bundle_rejects_incomplete_artifacts(
    roots,
    mutation,
    message,
):
    policy_root, _, _ = roots
    policy = _write_policy(policy_root / "base")
    bundle = _write_online_compatible_warmup_bundle(
        service.FLOW_SDE_VALUE_WARMUP_ROOT / mutation,
        policy_checkpoint=policy,
    )
    if mutation == "missing_checkpoint":
        (bundle / "training_state" / "value_warmup.pt").unlink()
    elif mutation == "missing_manifest":
        (bundle / "run_manifest.json").unlink()
    else:
        manifest_path = bundle / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HTTPException, match=message):
        service._resolve_value_warmup_bundle(
            str(bundle),
            policy_checkpoint=policy.resolve(),
            task_instruction="pick up the jelly bag",
        )


def test_online_value_warmup_bundle_rejects_symlinks_and_output_overlap(
    roots,
    tmp_path,
):
    policy_root, _, _ = roots
    policy = _write_policy(policy_root / "base")
    bundle = _write_online_compatible_warmup_bundle(
        service.FLOW_SDE_VALUE_WARMUP_ROOT / "bundle",
        policy_checkpoint=policy,
    )
    alias = service.FLOW_SDE_VALUE_WARMUP_ROOT / "alias"
    alias.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(HTTPException, match="symbolic links"):
        service._resolve_value_warmup_bundle(
            str(alias),
            policy_checkpoint=policy.resolve(),
            task_instruction="pick up the jelly bag",
        )

    with pytest.raises(HTTPException, match="current online PPO output"):
        service._resolve_value_warmup_bundle(
            str(bundle),
            policy_checkpoint=policy.resolve(),
            task_instruction="pick up the jelly bag",
            forbidden_output=bundle / "online-output",
        )


def test_value_warmup_status_recovers_latest_compatible_completed_bundle(roots):
    policy_root, _, _ = roots
    policy = _write_policy(policy_root / "base")
    valid = _write_online_compatible_warmup_bundle(
        service.FLOW_SDE_VALUE_WARMUP_ROOT / ("a" * 32),
        policy_checkpoint=policy,
    )
    incompatible = _write_online_compatible_warmup_bundle(
        service.FLOW_SDE_VALUE_WARMUP_ROOT / ("b" * 32),
        policy_checkpoint=policy,
        status="running",
    )
    os.utime(valid, ns=(1_000_000_000, 1_000_000_000))
    os.utime(incompatible, ns=(2_000_000_000, 2_000_000_000))

    supervisor = _supervisor()
    status = supervisor.value_warmup_status()

    assert status.status == "completed"
    assert status.job_id == "a" * 32
    assert status.bundle_path == str(valid.resolve())
    assert status.policy_checkpoint == str(policy.resolve())
    assert status.task_instruction == "pick up the jelly bag"
    assert status.checkpoint_path.endswith("training_state/value_warmup.pt")


def test_value_warmup_progress_and_terminal_events_update_typed_status(tmp_path):
    supervisor = _supervisor()
    job = _warmup_job(tmp_path)

    assert supervisor._consume_value_warmup_event(
        job,
        {
            "event": "progress",
            "completed_steps": 500,
            "total_steps": 2000,
            "value_loss": 0.25,
            "eta_seconds": 12.5,
        },
    ) is False
    assert job.step == 500
    assert job.percentage == pytest.approx(25.0)
    assert job.value_loss == pytest.approx(0.25)
    assert job.eta_seconds == pytest.approx(12.5)

    output = Path(job.output_dir)
    assert supervisor._consume_value_warmup_event(
        job,
        {
            "event": "completed",
            "status": "completed",
            "bundle_path": str(output),
            "pretrained_model": str(output / "pretrained_model"),
            "trainer_checkpoint": str(
                output / "training_state" / "value_warmup.pt"
            ),
        },
    ) is True
    assert job.phase == "verifying"


def test_value_warmup_monitor_requires_terminal_result_and_verified_bundle(
    roots,
    tmp_path,
):
    supervisor = _supervisor()
    job = _warmup_job(tmp_path)
    output = Path(job.output_dir)
    model, checkpoint = _write_value_warmup_bundle(output)
    job.process = FakeProcess(
        stdout=[
            json.dumps(
                {
                    "event": "progress",
                    "step": 1000,
                    "total_steps": 2000,
                    "value_loss": 0.3,
                }
            )
            + "\n",
            json.dumps(
                {
                    "event": "result",
                    "status": "complete",
                    "model_path": str(model),
                    "checkpoint_path": str(checkpoint),
                    "bundle_path": str(output),
                }
            )
            + "\n",
        ],
        returncode=0,
    )

    supervisor._monitor_value_warmup(job)

    assert job.status == "completed"
    assert job.phase == "complete"
    assert job.step == 2000
    assert job.percentage == 100.0
    status = supervisor._value_warmup_status(job)
    assert status.bundle_path == str(output)
    assert status.model_path == str(model)
    assert status.checkpoint_path == str(checkpoint)


def test_value_warmup_monitor_rejects_missing_artifact_after_complete_event(
    roots,
    tmp_path,
):
    supervisor = _supervisor()
    job = _warmup_job(tmp_path)
    output = Path(job.output_dir)
    model, checkpoint = _write_value_warmup_bundle(output)
    (output / "progress.jsonl").unlink()
    job.process = FakeProcess(
        stdout=[
            json.dumps(
                {
                    "event": "result",
                    "status": "complete",
                    "model_path": str(model),
                    "checkpoint_path": str(checkpoint),
                    "bundle_path": str(output),
                }
            )
            + "\n"
        ],
        returncode=0,
    )

    supervisor._monitor_value_warmup(job)

    assert job.status == "failed"
    assert supervisor._value_warmup_status(job).bundle_path == ""


def test_monitor_requires_completed_event_and_verified_deployable_model(roots, tmp_path):
    policy_root, output_root, log_root = roots
    supervisor = _supervisor()
    base = _write_policy(policy_root / "base")
    output, model, checkpoint = _write_completed_online_bundle(
        output_root,
        base,
        update_step=2,
    )
    job = _job(
        tmp_path,
        job_id="d" * 32,
        policy_checkpoint=str(base.resolve()),
        lineage_policy_checkpoint=str(base.resolve()),
        output_dir=str(output),
        control_file=str(output / "control" / "outcome.json"),
    )
    job.process = FakeProcess(
        stdout=[
            json.dumps({
                "event": "episode_updated",
                "episode": 2,
                "episodes": 2,
                "checkpoint": str(checkpoint),
                "metrics": {"update_step": 2, "transitions": 3},
            }) + "\n",
            json.dumps({
                "event": "completed",
                "status": "completed",
                "pretrained_model": str(model),
                "trainer_checkpoint": str(checkpoint),
                "updates": 2,
            }) + "\n",
        ],
        returncode=0,
    )

    supervisor._monitor(job)

    assert Path(job.log_path).is_file()
    assert job.status == "completed"
    assert job.phase == "complete"
    assert job.percentage == 100.0
    assert supervisor._status(job).model_path == str(model)
    assert log_root.exists()


def test_monitor_rejects_wrong_policy_type_even_after_complete_result(roots, tmp_path):
    supervisor = _supervisor()
    job = _job(tmp_path)
    model = _write_deployable_model(Path(job.output_dir), policy_type="act")
    job.process = FakeProcess(
        stdout=[json.dumps({
            "event": "completed",
            "status": "completed",
            "pretrained_model": str(model),
        }) + "\n"],
        returncode=0,
    )

    supervisor._monitor(job)

    assert job.status == "failed"
    assert supervisor._status(job).model_path == ""


def test_stop_signals_process_without_reusing_episode_cancel(tmp_path):
    supervisor = _supervisor()
    process = FakeProcess(
        stdout=[json.dumps({"event": "cancelled", "status": "cancelled"}) + "\n"],
        returncode=0,
    )
    job = _job(tmp_path, process=process, awaiting_outcome=True)
    supervisor._job = job

    status = supervisor.stop(service.FlowSDEPPOStopRequest(job_id=job.job_id))

    assert status.status == "running"
    assert job.stop_requested is True
    assert process.signals == [signal.SIGINT]
    assert not Path(job.control_file).exists()

    supervisor._monitor(job)
    assert job.status == "stopped"
    assert job.phase == "stopped"


def test_stop_does_not_report_success_when_signal_delivery_fails(tmp_path):
    supervisor = _supervisor()
    process = FakeProcess(signal_error=True)
    job = _job(tmp_path, process=process)
    supervisor._job = job

    with pytest.raises(HTTPException, match="exited before") as exc_info:
        supervisor.stop(service.FlowSDEPPOStopRequest(job_id=job.job_id))

    assert exc_info.value.status_code == 409
    assert job.stop_requested is False


def test_stop_prefers_exact_named_container_over_compose_wrapper(tmp_path):
    interrupted = []
    supervisor = _supervisor(
        interrupt_container=lambda name: interrupted.append(name) or True,
    )
    process = FakeProcess()
    job = _job(tmp_path, process=process)
    supervisor._job = job

    supervisor.stop(service.FlowSDEPPOStopRequest(job_id=job.job_id))

    assert interrupted == ["cyclo_flow_sde_ppo_aaaaaaaaaaaa"]
    assert process.signals == []


def test_value_warmup_stop_prefers_exact_named_container(tmp_path):
    interrupted = []
    supervisor = _supervisor(
        interrupt_container=lambda name: interrupted.append(name) or True,
    )
    process = FakeProcess()
    job = _warmup_job(tmp_path, process=process)
    supervisor._value_warmup_job = job

    status = supervisor.stop_value_warmup(
        service.FlowSDEPPOStopRequest(job_id=job.job_id)
    )

    assert status.status == "running"
    assert job.stop_requested is True
    assert interrupted == ["cyclo_flow_sde_value_warmup_cccccccccccc"]
    assert process.signals == []


def test_value_warmup_stop_event_finishes_as_stopped(tmp_path):
    supervisor = _supervisor()
    job = _warmup_job(
        tmp_path,
        process=FakeProcess(
            stdout=[
                json.dumps({"event": "result", "status": "stopped"}) + "\n"
            ],
            returncode=0,
        ),
        stop_requested=True,
    )

    supervisor._monitor_value_warmup(job)

    assert job.status == "stopped"
    assert job.phase == "stopped"
    assert job.bundle_path == ""


@pytest.mark.parametrize("returncode", [-signal.SIGINT, 128 + signal.SIGINT])
def test_expected_sigint_exit_is_stopped_without_deployable_model(
    tmp_path,
    returncode,
):
    supervisor = _supervisor()
    job = _job(
        tmp_path,
        process=FakeProcess(returncode=returncode),
        stop_requested=True,
        model_path=str(tmp_path / "must-not-deploy"),
    )

    supervisor._monitor(job)

    assert job.status == "stopped"
    assert job.model_path == ""


def test_is_running_is_true_only_for_active_job(tmp_path):
    supervisor = _supervisor()
    assert supervisor.is_running() is False
    job = _job(tmp_path)
    supervisor._job = job
    assert supervisor.is_running() is True
    job.status = "completed"
    assert supervisor.is_running() is False

    warmup = _warmup_job(tmp_path)
    supervisor._value_warmup_job = warmup
    assert supervisor.is_running() is True
    warmup.status = "failed"
    assert supervisor.is_running() is False
