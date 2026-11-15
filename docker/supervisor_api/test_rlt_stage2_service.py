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


@pytest.fixture
def workspace_checkpoint_roots(tmp_path, monkeypatch):
    # Preserve the production allowlist layout under an isolated workspace.
    roots = tuple(tmp_path / path.relative_to("/workspace")
                  for path in service.RLT_STAGE2_GROOT_ROOTS)
    for path in roots:
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(service, "RLT_STAGE2_GROOT_ROOTS", roots)
    return tmp_path


@pytest.mark.parametrize("relative", [
    "model/Dongkkka/Grootn17_Drill_Into_Dynamixel_VT_5K",
    "model/groot/task",
    "checkpoint/groot/task",
])
def test_resolves_supported_groot_model_locations(workspace_checkpoint_roots, relative):
    checkpoint = _write_groot(workspace_checkpoint_roots / relative)
    assert service._resolve_groot_checkpoint(str(checkpoint)) == checkpoint


@pytest.mark.parametrize("relative", ["outside/task", "model-other/task"])
def test_rejects_checkpoint_outside_allowed_roots(workspace_checkpoint_roots, relative):
    checkpoint = _write_groot(workspace_checkpoint_roots / relative)
    with pytest.raises(HTTPException, match="must be under"):
        service._resolve_groot_checkpoint(str(checkpoint))


def test_namespaced_model_still_rejects_symlinks(workspace_checkpoint_roots):
    root = workspace_checkpoint_roots
    checkpoint = _write_groot(root / "outside/task")
    (root / "model/Dongkkka").symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(HTTPException, match="symbolic links"):
        service._resolve_groot_checkpoint(str(root / "model/Dongkkka"))


def test_namespaced_model_still_requires_groot_config(workspace_checkpoint_roots):
    checkpoint = _write_groot(workspace_checkpoint_roots / "model/owner/not-groot")
    (checkpoint / "config.json").write_text('{"model_type": "act"}')
    with pytest.raises(HTTPException, match="not a GR00T"):
        service._resolve_groot_checkpoint(str(checkpoint))


def test_success_only_scope_is_rejected_outside_actor_warmup():
    with pytest.raises(ValueError, match="only for Actor warmup"):
        service.RLTStage2StartRequest(
            training_phase="critic_warmup",
            actor_warmup_data_scope="success_only",
            initialization_mode="resume",
            dataset_paths=["/workspace/lerobot/data"],
            rlt_bundle_path="/workspace/checkpoint/rlt/stage2/parent",
        )


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


def _write_execution_verification(path: Path, actor_sha256: str) -> Path:
    action_file = path / "data" / "chunk-000" / "episode_000000.parquet"
    action_file.parent.mkdir(parents=True, exist_ok=True)
    action_file.write_bytes(b"verified-action-data")
    unsigned = {
        "format": service._EXECUTION_VERIFICATION_FORMAT,
        "execution_verified": True,
        "episode_count": 2,
        "success_episode_count": 1,
        "failure_episode_count": 1,
        "actor_sha256": [actor_sha256],
        "action_data_files": [
            {
                "relative_path": action_file.relative_to(path).as_posix(),
                "byte_count": action_file.stat().st_size,
                "sha256": service._file_sha256(action_file),
            }
        ],
        "episodes": [
            {
                "episode_index": 0,
                "execution_verified": True,
                "episode_success": True,
                "verification_fingerprint": "1" * 64,
            },
            {
                "episode_index": 1,
                "execution_verified": True,
                "episode_success": False,
                "verification_fingerprint": "2" * 64,
            },
        ],
    }
    payload = {
        **unsigned,
        "verification_fingerprint": service._canonical_fingerprint(unsigned),
    }
    output = path / "rlt" / "execution_verification.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    return output


def _write_execution_verification(path: Path, actor_sha256: str) -> Path:
    action_file = path / "data" / "chunk-000" / "episode_000000.parquet"
    action_file.parent.mkdir(parents=True, exist_ok=True)
    action_file.write_bytes(b"verified-action-data")
    unsigned = {
        "format": service._EXECUTION_VERIFICATION_FORMAT,
        "execution_verified": True,
        "episode_count": 2,
        "success_episode_count": 1,
        "failure_episode_count": 1,
        "actor_sha256": [actor_sha256],
        "action_data_files": [
            {
                "relative_path": action_file.relative_to(path).as_posix(),
                "byte_count": action_file.stat().st_size,
                "sha256": service._file_sha256(action_file),
            }
        ],
        "episodes": [
            {
                "episode_index": 0,
                "execution_verified": True,
                "episode_success": True,
                "verification_fingerprint": "1" * 64,
            },
            {
                "episode_index": 1,
                "execution_verified": True,
                "episode_success": False,
                "verification_fingerprint": "2" * 64,
            },
        ],
    }
    payload = {
        **unsigned,
        "verification_fingerprint": service._canonical_fingerprint(unsigned),
    }
    output = path / "rlt" / "execution_verification.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(payload), encoding="utf-8")
    return output


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
    reference_horizon: int = 16,
    action_dim: int = 19,
    schema_version: int = 2,
    dataset_roots: list[str] | None = None,
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
        "proprio_dim": action_dim,
        "reference_horizon": reference_horizon,
        "chunk_length": 10,
        "action_dim": action_dim,
        "action_hz": 15.0,
        "action_normalization_id": "showroom-normalized-19d",
        "action_codec_id": "normalized-chunk-10x19",
        "model_domain": "normalized",
        "schema_version": schema_version,
    }
    if schema_version == 2:
        spec["reference_offset_steps"] = 6
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
    if dataset_roots is not None:
        replay_manifest["metadata"]["dataset_roots"] = list(dataset_roots)
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
        "phase_state": {
            "training_phase": "rl",
            "completed_phases": ["actor_warmup", "critic_warmup", "rl"],
        },
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
        "training_phase": "rl",
    }
    values.update(overrides)
    return service._RLTStage2Job(**values)


def test_routes_are_dedicated_to_stage2(roots):
    supervisor = _supervisor()
    assert {route.path for route in supervisor.router.routes} == {
        "/rlt-stage2/start",
        "/rlt-stage2/publish",
        "/rlt-stage2/resolve",
        "/rlt-stage2/start-iterative",
        "/rlt-stage2/status",
        "/rlt-stage2/stop",
        "/rlt-stage2/validate-data",
        "/rlt-stage2/validate-data/{validation_id}",
    }
    assert supervisor.status().status == "idle"
    assert supervisor.status().ready is True


def test_restart_recovers_newest_verified_completed_stage2(roots):
    *_, stage2_root, _ = roots
    older = _write_stage2_bundle(
        stage2_root / f"steps_{100:07d}_{'a' * 12}"
    )
    newer = _write_stage2_bundle(
        stage2_root / f"steps_{100:07d}_{'b' * 12}_20260923070102"
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


@pytest.mark.parametrize(
    ("training_phase", "weight_label"),
    [
        ("actor_warmup", "actor_warmup"),
        ("critic_warmup", "critic_warmup"),
        ("rl", "rl_training"),
    ],
)
def test_output_path_uses_hidden_phase_run(
    roots,
    monkeypatch,
    training_phase,
    weight_label,
):
    dataset_root, *_, stage2_root, _ = roots
    dataset = dataset_root / "data_epoch_0001" / "task_lerobot_v30"

    output = service._output_path(
        "a" * 32,
        100,
        [dataset],
        training_phase,
    )

    folder = service._STAGE2_PHASE_FOLDERS[training_phase]
    assert output == stage2_root / folder / ".runs" / "steps_0000100_aaaaaaaaaaaa"



def test_output_path_rejects_unknown_weight_type(roots):
    dataset_root, *_ = roots
    with pytest.raises(HTTPException, match="training phase is invalid"):
        service._output_path(
            "a" * 32,
            100,
            [dataset_root / "selected"],
            "unknown",
        )


def test_restart_recovers_dual_arm_32x16_stage2_bundle(roots):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(
        stage2_root / f"steps_{100:07d}_{'c' * 12}",
        reference_horizon=32,
        action_dim=16,
    )

    status = _supervisor().status()

    assert status.status == "completed"
    assert status.output_dir == str(bundle.resolve())


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


def test_missing_runtime_keeps_status_available_but_blocks_start(roots):
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
                training_phase="actor_warmup",
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


def test_resume_accepts_legacy_8ecc_schema_v1_bundle(roots):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(
        stage2_root / "round-v1",
        reference_horizon=32,
        action_dim=16,
        schema_version=1,
    )

    assert service._resolve_resume_bundle(str(bundle)) == bundle.resolve()


def test_iterative_replay_appends_only_parent_actor_verified_data(roots):
    dataset_root, _model, _stage1, stage2_root, _logs = roots
    parent_dataset = _write_v21(dataset_root / "parent")
    rollout_dataset = _write_v21(dataset_root / "rollout")
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        reference_horizon=32,
        action_dim=16,
        schema_version=1,
        dataset_roots=[str(parent_dataset)],
    )
    actor_sha256 = service._file_sha256(bundle / "artifacts" / "rlt_actor.pt")
    _write_execution_verification(rollout_dataset, actor_sha256)

    paths = service._iterative_dataset_paths(bundle, [str(rollout_dataset)])

    assert paths == [str(parent_dataset.resolve()), str(rollout_dataset.resolve())]


def test_iterative_replay_rejects_rollout_from_another_actor(roots):
    dataset_root, _model, _stage1, stage2_root, _logs = roots
    parent_dataset = _write_v21(dataset_root / "parent")
    rollout_dataset = _write_v21(dataset_root / "rollout")
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        dataset_roots=[str(parent_dataset)],
    )
    _write_execution_verification(rollout_dataset, "f" * 64)

    with pytest.raises(HTTPException, match="selected parent actor"):
        service._iterative_dataset_paths(bundle, [str(rollout_dataset)])


def test_iterative_replay_rejects_actions_changed_after_verification(roots):
    dataset_root, _model, _stage1, stage2_root, _logs = roots
    parent_dataset = _write_v21(dataset_root / "parent")
    rollout_dataset = _write_v21(dataset_root / "rollout")
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        dataset_roots=[str(parent_dataset)],
    )
    actor_sha256 = service._file_sha256(bundle / "artifacts" / "rlt_actor.pt")
    _write_execution_verification(rollout_dataset, actor_sha256)
    (rollout_dataset / "data/chunk-000/episode_000000.parquet").write_bytes(
        b"mutated"
    )

    with pytest.raises(HTTPException, match="changed after execution verification"):
        service._iterative_dataset_paths(bundle, [str(rollout_dataset)])


def test_resume_accepts_legacy_8ecc_schema_v1_bundle(roots):
    *_, stage2_root, _ = roots
    bundle = _write_stage2_bundle(
        stage2_root / "round-v1",
        reference_horizon=32,
        action_dim=16,
        schema_version=1,
    )

    assert service._resolve_resume_bundle(str(bundle)) == bundle.resolve()


def test_iterative_replay_appends_only_parent_actor_verified_data(roots):
    dataset_root, _model, _stage1, stage2_root, _logs = roots
    parent_dataset = _write_v21(dataset_root / "parent")
    rollout_dataset = _write_v21(dataset_root / "rollout")
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        reference_horizon=32,
        action_dim=16,
        schema_version=1,
        dataset_roots=[str(parent_dataset)],
    )
    actor_sha256 = service._file_sha256(bundle / "artifacts" / "rlt_actor.pt")
    _write_execution_verification(rollout_dataset, actor_sha256)

    paths = service._iterative_dataset_paths(bundle, [str(rollout_dataset)])

    assert paths == [str(parent_dataset.resolve()), str(rollout_dataset.resolve())]


def test_iterative_replay_rejects_rollout_from_another_actor(roots):
    dataset_root, _model, _stage1, stage2_root, _logs = roots
    parent_dataset = _write_v21(dataset_root / "parent")
    rollout_dataset = _write_v21(dataset_root / "rollout")
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        dataset_roots=[str(parent_dataset)],
    )
    _write_execution_verification(rollout_dataset, "f" * 64)

    with pytest.raises(HTTPException, match="selected parent actor"):
        service._iterative_dataset_paths(bundle, [str(rollout_dataset)])


def test_iterative_replay_rejects_actions_changed_after_verification(roots):
    dataset_root, _model, _stage1, stage2_root, _logs = roots
    parent_dataset = _write_v21(dataset_root / "parent")
    rollout_dataset = _write_v21(dataset_root / "rollout")
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        dataset_roots=[str(parent_dataset)],
    )
    actor_sha256 = service._file_sha256(bundle / "artifacts" / "rlt_actor.pt")
    _write_execution_verification(rollout_dataset, actor_sha256)
    (rollout_dataset / "data/chunk-000/episode_000000.parquet").write_bytes(
        b"mutated"
    )

    with pytest.raises(HTTPException, match="changed after execution verification"):
        service._iterative_dataset_paths(bundle, [str(rollout_dataset)])


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
    assert "USER=cyclo_rlt" in runtime_environment
    assert "LOGNAME=cyclo_rlt" in runtime_environment
    assert (
        f"TORCHINDUCTOR_CACHE_DIR={service.RLT_STAGE2_CACHE_ROOT}/torchinductor"
        in runtime_environment
    )
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


def test_monitor_publishes_bundle_bound_shadow_metrics(tmp_path):
    output = _write_stage2_bundle(tmp_path / "stage2-output")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    shadow_unsigned = {
        "format": "cyclo.groot.rlt.shadow_qualification/v1",
        "bundle_manifest_fingerprint": manifest["manifest_fingerprint"],
        "evaluation": {
            "format": "cyclo.groot.rlt.shadow_evaluation/v1",
            "structural_pass": True,
            "correction_abs_mean": 0.02,
            "correction_abs_p99": 0.08,
        },
    }
    shadow = {
        **shadow_unsigned,
        "qualification_fingerprint": service._canonical_fingerprint(
            shadow_unsigned
        ),
    }
    shadow_path = output / "evaluation" / "shadow_evaluation.json"
    shadow_path.parent.mkdir()
    shadow_path.write_text(json.dumps(shadow), encoding="utf-8")
    lines = [
        json.dumps(
            {
                "event": "stage2_training_result",
                "status": "completed",
                "actor_artifact": str(output / "artifacts/rlt_actor.pt"),
                "encoder_artifact": str(output / "artifacts/rl_token_encoder.pt"),
                "checkpoint": str(output / "training_state/rlt_stage2.pt"),
                "bundle_manifest": str(output / "manifest.json"),
                "shadow_evaluation": str(shadow_path),
                "shadow_structural_pass": True,
                "shadow_correction_abs_mean": 0.02,
                "shadow_correction_abs_p99": 0.08,
            }
        ) + "\n"
    ]
    job = _job(tmp_path, output_dir=str(output), process=FakeProcess(stdout=lines))
    supervisor = _supervisor()
    supervisor._job = job

    supervisor._monitor(job)
    status = supervisor.status()

    assert status.status == "completed"
    assert status.shadow_evaluation_path == str(shadow_path)
    assert status.shadow_structural_pass is True
    assert status.shadow_correction_abs_mean == pytest.approx(0.02)
    assert status.shadow_correction_abs_p99 == pytest.approx(0.08)


def test_monitor_publishes_bundle_bound_shadow_metrics(tmp_path):
    output = _write_stage2_bundle(tmp_path / "stage2-output")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    shadow_unsigned = {
        "format": "cyclo.groot.rlt.shadow_qualification/v1",
        "bundle_manifest_fingerprint": manifest["manifest_fingerprint"],
        "evaluation": {
            "format": "cyclo.groot.rlt.shadow_evaluation/v1",
            "structural_pass": True,
            "correction_abs_mean": 0.02,
            "correction_abs_p99": 0.08,
        },
    }
    shadow = {
        **shadow_unsigned,
        "qualification_fingerprint": service._canonical_fingerprint(
            shadow_unsigned
        ),
    }
    shadow_path = output / "evaluation" / "shadow_evaluation.json"
    shadow_path.parent.mkdir()
    shadow_path.write_text(json.dumps(shadow), encoding="utf-8")
    lines = [
        json.dumps(
            {
                "event": "stage2_training_result",
                "status": "completed",
                "actor_artifact": str(output / "artifacts/rlt_actor.pt"),
                "encoder_artifact": str(output / "artifacts/rl_token_encoder.pt"),
                "checkpoint": str(output / "training_state/rlt_stage2.pt"),
                "bundle_manifest": str(output / "manifest.json"),
                "shadow_evaluation": str(shadow_path),
                "shadow_structural_pass": True,
                "shadow_correction_abs_mean": 0.02,
                "shadow_correction_abs_p99": 0.08,
            }
        ) + "\n"
    ]
    job = _job(tmp_path, output_dir=str(output), process=FakeProcess(stdout=lines))
    supervisor = _supervisor()
    supervisor._job = job

    supervisor._monitor(job)
    status = supervisor.status()

    assert status.status == "completed"
    assert status.shadow_evaluation_path == str(shadow_path)
    assert status.shadow_structural_pass is True
    assert status.shadow_correction_abs_mean == pytest.approx(0.02)
    assert status.shadow_correction_abs_p99 == pytest.approx(0.08)


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
    actor_replay = dataset / "rlt/tt_rtc_actor_hold_v1/replay_cache"
    actor_replay.mkdir(parents=True)
    (actor_replay / "manifest.json").write_text("{}", encoding="utf-8")
    bundle = _write_stage2_bundle(
        stage2_root / "round-1",
        groot_checkpoint=str(groot.resolve()),
        schema_version=1,
        dataset_roots=[str(dataset.resolve())],
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
            training_phase="actor_warmup",
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
            training_phase="rl",
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
                training_phase="rl",
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
                training_phase="actor_warmup",
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
                training_phase="rl",
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
                training_phase="actor_warmup",
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


@pytest.mark.parametrize("phase,folder", [("actor_warmup", "actor"), ("critic_warmup", "critic"), ("rl", "rl_training")])
def test_named_publication_is_single_selectable_bundle_and_recovers(roots, phase, folder):
    *_, root, _ = roots
    first = _write_stage2_bundle(root / folder / ".runs" / "steps_0000100_aaaaaaaaaaaa")
    _rewrite_bundle_manifest(first, lambda m: m["phase_state"].update(training_phase=phase))
    destination = service._publish_stage2_bundle(first)
    assert destination == root / folder / "latest"
    assert service._resolve_resume_bundle(str(destination)) == destination
    second = _write_stage2_bundle(root / folder / ".runs" / "steps_0000100_bbbbbbbbbbbb")
    _rewrite_bundle_manifest(second, lambda m: m["phase_state"].update(training_phase=phase, revision=2))
    assert service._publish_stage2_bundle(second) == destination
    assert json.loads((destination / "manifest.json").read_text())["phase_state"]["revision"] == 2
    renamed = service._publish_stage2_bundle(second, "screw_v2")
    assert renamed.name == "screw_v2"
    assert not destination.exists()
    assert [p.name for p in (root / folder).iterdir() if not p.name.startswith(".")] == ["screw_v2"]
    assert service._publish_stage2_bundle(first) == renamed  # keep the user name on the next round
    assert _supervisor().status().output_dir == str(renamed)
    assert first.is_dir() and second.is_dir()


def test_failed_publication_keeps_previous_weights(roots, monkeypatch):
    *_, root, _ = roots
    first = _write_stage2_bundle(root / "rl_training/.runs/first")
    published = service._publish_stage2_bundle(first)
    original = (published / "manifest.json").read_bytes()
    broken = _write_stage2_bundle(root / "rl_training/.runs/broken")
    (broken / "artifacts/rlt_actor.pt").write_bytes(b"tampered")
    with pytest.raises(HTTPException):
        service._publish_stage2_bundle(broken)
    assert (published / "manifest.json").read_bytes() == original
    good = _write_stage2_bundle(root / "rl_training/.runs/good")
    def fail_exchange(*args):
        raise OSError("exchange failed")
    monkeypatch.setattr(service, "_exchange_directories", fail_exchange)
    with pytest.raises(OSError, match="exchange failed"):
        service._publish_stage2_bundle(good)
    assert service._resolve_resume_bundle(str(published)) == published
    assert not list(published.parent.glob(".publish-*"))


def test_publish_preserves_discard_snapshot_and_rejects_running_job(roots):
    *_, root, _ = roots
    first = _write_stage2_bundle(root / "rl_training/.runs/first")
    selected = service._publish_stage2_bundle(first)
    second = _write_stage2_bundle(root / "rl_training/.runs/second")
    _rewrite_bundle_manifest(second, lambda m: m["phase_state"].update(revision=2))
    supervisor = _supervisor()
    request = service.RLTStage2PublishRequest(bundle_path=str(second), name="screw", previous_bundle_path=str(selected))
    result = supervisor.publish(request)
    assert supervisor.status().output_dir == result["bundle_path"]
    previous = Path(result["previous_bundle_path"])
    assert previous.parent.name == ".history"
    assert "revision" not in json.loads((previous / "manifest.json").read_text())["phase_state"]
    assert service._resolve_resume_bundle(result["bundle_path"]).name == "screw"
    supervisor._job = _job(root)
    with pytest.raises(HTTPException, match="Wait for RLT"):
        supervisor.publish(request)


@pytest.mark.parametrize("name", ["../escape", ".hidden", "a/b", "", "a" * 65, "한글"])
def test_invalid_deployment_names_are_rejected(name):
    with pytest.raises(ValueError):
        service.RLTStage2PublishRequest(bundle_path="/workspace/checkpoint/rlt/stage2/run", name=name)


def test_phase_directory_symlinks_are_rejected(roots, tmp_path):
    *_, root, _ = roots
    source = _write_stage2_bundle(root / "legacy")
    (root / "rl_training").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(HTTPException, match="symlink"):
        service._publish_stage2_bundle(source)


def test_legacy_saved_selection_resolves_after_migration(roots):
    *_, root, _ = roots
    old = root / "steps_0000100_aaaaaaaaaaaa"
    archived = _write_stage2_bundle(root / ".legacy" / old.name)
    published = service._publish_stage2_bundle(archived)
    assert service._resolve_resume_bundle(str(old)) == archived
    result = _supervisor().resolve_bundle(service.RLTStage2PublishRequest(bundle_path=str(old)))
    assert result["bundle_path"] == str(published)


def test_nested_picker_marks_only_complete_bundle_selectable(roots):
    spec = importlib.util.spec_from_file_location("rlt_test_browser", SERVICE_PATH.parents[2] / "orchestrator/orchestrator/internal/file_browser/file_browse_utils.py")
    browser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(browser)
    *_, root, _ = roots
    run = _write_stage2_bundle(root / "actor/.runs/first")
    _rewrite_bundle_manifest(run, lambda m: m["phase_state"].update(training_phase="actor_warmup"))
    service._publish_stage2_bundle(run)
    util = browser.FileBrowseUtils()
    required = dict(target_files={"manifest.json"}, target_folders={"artifacts", "training_state", "replay_cache"})
    groups = util._get_directories_with_target_check(str(root), **required)
    assert [(i["name"], i["has_target_file"]) for i in groups] == [("actor", False)]
    choices = util._get_directories_with_target_check(str(root / "actor"), **required)
    assert [(i["name"], i["has_target_file"]) for i in choices] == [("latest", True)]


def test_migration_archives_failed_runs_and_keeps_only_latest_complete(roots):
    *_, root, _ = roots
    complete = _write_stage2_bundle(root / "steps_0000100_aaaaaaaaaaaa")
    _rewrite_bundle_manifest(complete, lambda m: m["phase_state"].update(training_phase="actor_warmup"))
    failed = root / "steps_0000100_bbbbbbbbbbbb"
    failed.mkdir()
    partial = root / "steps_0000100_cccccccccccc"
    (partial / "training_state").mkdir(parents=True)
    result = service.migrate_legacy_stage2_layout()
    assert len(result) == 3
    assert sorted(p.name for p in root.iterdir() if not p.name.startswith(".")) == ["actor", "critic", "rl_training"]
    assert service._resolve_resume_bundle(str(complete)).parent.name == ".legacy"
    assert _supervisor().status().output_dir == str(root / "actor/latest")
    assert service.migrate_legacy_stage2_layout() == {}


def test_monitor_publishes_completed_hidden_run(roots):
    *_, root, _ = roots
    output = _write_stage2_bundle(root / "rl_training/.runs/steps_0000100_aaaaaaaaaaaa")
    payload = {"event": "stage2_training_result", "status": "completed",
               "actor_artifact": str(output / "artifacts/rlt_actor.pt"),
               "encoder_artifact": str(output / "artifacts/rl_token_encoder.pt"),
               "checkpoint": str(output / "training_state/rlt_stage2.pt"),
               "bundle_manifest": str(output / "manifest.json")}
    job = _job(root, output_dir=str(output), process=FakeProcess(stdout=[json.dumps(payload)]))
    _supervisor()._monitor(job)
    assert job.status == "completed"
    assert service._resolve_resume_bundle(str(root / "rl_training/latest")).is_dir()
    assert job.output_dir == str(output)  # candidate identity changes on every new run
