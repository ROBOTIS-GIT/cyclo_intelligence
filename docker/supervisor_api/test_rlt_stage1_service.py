#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused tests for the GR00T RLT Stage 1 supervisor."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException


SERVICE_PATH = Path(__file__).resolve().with_name("rlt_stage1_service.py")
SPEC = importlib.util.spec_from_file_location("rlt_stage1_service_under_test", SERVICE_PATH)
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


def _write_v21(path: Path) -> Path:
    meta = path / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "total_episodes": 1}),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text('{"episode_index": 0}\n', encoding="utf-8")
    (meta / "tasks.jsonl").write_text('{"task_index": 0}\n', encoding="utf-8")
    return path


def _write_v30(path: Path) -> Path:
    meta = path / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_episodes": 1}),
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


@pytest.mark.parametrize("relative", ["Dongkkka/Grootn17_Drill_Into_Dynamixel_VT_5K", "groot/Dongkkka/legacy"])
def test_checkpoint_supports_current_and_legacy_model_layouts(tmp_path, monkeypatch, relative):
    model_root = tmp_path / "model"
    checkpoint = _write_groot(model_root / relative)
    assert Path("/workspace/model") in service.RLT_STAGE1_GROOT_ROOTS
    monkeypatch.setattr(service, "RLT_STAGE1_GROOT_ROOTS", (model_root,))
    assert service._resolve_groot_checkpoint(str(checkpoint)) == checkpoint


def _write_completed_stage1_run(
    output_root: Path,
    *,
    dataset: Path,
    groot: Path,
    steps: int,
    job_id: str,
    timestamp: str = "",
) -> Path:
    suffix = f"_{timestamp}" if timestamp else ""
    output = output_root / f"steps_{steps:06d}_{job_id[:12]}{suffix}"
    encoder = output / "artifacts" / "rl_token_encoder.pt"
    checkpoint = output / "training_state" / "rlt_stage1.pt"
    encoder.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    encoder.write_bytes(b"encoder")
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parent / "rlt_stage1.pt.run.json").write_text(
        json.dumps(
            {
                "format": service._STAGE1_RUN_FORMAT,
                "status": "completed",
                "job_id": job_id,
                "dataset_roots": [str(dataset)],
                "groot_checkpoint": str(groot),
                "policy_weight_fingerprint": "a" * 64,
                "policy_checkpoint_fingerprint": "b" * 64,
                "policy_processor_fingerprint": "c" * 64,
                "action_normalization_id": "showroom-normalized-19d",
                "action_codec_id": "normalized-chunk-10x19",
                "completed_steps": steps,
                "batch_size": 2,
                "artifact": {
                    "path": str(encoder),
                    "artifact_fingerprint": "d" * 64,
                },
                "checkpoint": {"path": str(checkpoint)},
            }
        ),
        encoding="utf-8",
    )
    return output


@pytest.fixture
def roots(tmp_path, monkeypatch):
    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    model_root = tmp_path / "model" / "groot"
    model_root.mkdir(parents=True)
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    output_root = checkpoint_root / "rlt" / "stage1"
    log_root = tmp_path / "logs"
    monkeypatch.setattr(service, "RLT_STAGE1_DATASET_ROOTS", (dataset_root,))
    monkeypatch.setattr(
        service,
        "RLT_STAGE1_GROOT_ROOTS",
        (model_root, checkpoint_root),
    )
    monkeypatch.setattr(service, "RLT_STAGE1_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(service, "RLT_STAGE1_LOG_ROOT", log_root)
    monkeypatch.setattr(service.os, "chown", lambda *args: None)
    return dataset_root, model_root, checkpoint_root, output_root, log_root


def _supervisor(*, conflict=lambda: None, interrupt_container=None):
    return service.RLTStage1Supervisor(
        compose_command=lambda: ["docker", "compose", "-f", "/tmp/compose.yml"],
        compose_environment=lambda: {"COMPOSE_PROJECT_NAME": "cyclo"},
        conflict_message=conflict,
        interrupt_container=interrupt_container,
    )


def _job(tmp_path: Path, **overrides):
    output = tmp_path / "stage1-output"
    values = {
        "job_id": "a" * 32,
        "dataset_paths": ["/workspace/lerobot/selected-v30"],
        "resolved_dataset_paths": ["/workspace/lerobot/paired-v21"],
        "groot_checkpoint": "/workspace/model/groot/showroom_groot",
        "output_dir": str(output),
        "log_path": str(tmp_path / "logs" / "stage1.log"),
        "total_steps": 100,
        "batch_size": 1,
        "save_freq": 25,
    }
    values.update(overrides)
    return service._RLTStage1Job(**values)


def test_routes_are_dedicated_to_stage1(roots):
    supervisor = _supervisor()
    paths = {route.path for route in supervisor.router.routes}
    assert paths == {
        "/rlt-stage1/start",
        "/rlt-stage1/status",
        "/rlt-stage1/stop",
    }
    assert supervisor.status().status == "idle"


def test_restart_recovers_newest_verified_completed_stage1(roots):
    dataset_root, model_root, _, output_root, _ = roots
    dataset = _write_v30(dataset_root / "selected")
    groot = _write_groot(model_root / "showroom_groot")
    older = _write_completed_stage1_run(
        output_root,
        dataset=dataset,
        groot=groot,
        steps=100,
        job_id="a" * 32,
    )
    newer = _write_completed_stage1_run(
        output_root,
        dataset=dataset,
        groot=groot,
        steps=200,
        job_id="b" * 32,
        timestamp="20260923070102",
    )
    os.utime(older / "training_state/rlt_stage1.pt.run.json", ns=(1, 1))
    os.utime(newer / "training_state/rlt_stage1.pt.run.json", ns=(2, 2))

    status = _supervisor().status()

    assert status.status == "completed"
    assert status.job_id == "b" * 32
    assert status.output_dir == str(newer.resolve())
    assert status.encoder_artifact_path == str(
        newer.resolve() / "artifacts/rl_token_encoder.pt"
    )
    assert status.checkpoint_path == str(
        newer.resolve() / "training_state/rlt_stage1.pt"
    )
    assert status.total_steps == 200
    assert status.batch_size == 2
    assert status.save_freq == 0
    assert "Recovered" in status.message


def test_restart_skips_newer_partial_stage1_publication(roots):
    dataset_root, model_root, _, output_root, _ = roots
    dataset = _write_v30(dataset_root / "selected")
    groot = _write_groot(model_root / "showroom_groot")
    valid = _write_completed_stage1_run(
        output_root,
        dataset=dataset,
        groot=groot,
        steps=100,
        job_id="a" * 32,
    )
    partial = output_root / f"steps_{200:06d}_{'b' * 12}"
    (partial / "training_state").mkdir(parents=True)
    partial_manifest = partial / "training_state/rlt_stage1.pt.run.json"
    partial_manifest.write_text("{}", encoding="utf-8")
    os.utime(valid / "training_state/rlt_stage1.pt.run.json", ns=(1, 1))
    os.utime(partial_manifest, ns=(2, 2))

    status = _supervisor().status()

    assert status.status == "completed"
    assert status.output_dir == str(valid.resolve())


def test_restart_ignores_stage1_run_with_symlink_artifact(roots, tmp_path):
    dataset_root, model_root, _, output_root, _ = roots
    dataset = _write_v30(dataset_root / "selected")
    groot = _write_groot(model_root / "showroom_groot")
    output = _write_completed_stage1_run(
        output_root,
        dataset=dataset,
        groot=groot,
        steps=100,
        job_id="a" * 32,
    )
    encoder = output / "artifacts/rl_token_encoder.pt"
    encoder.unlink()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"encoder")
    encoder.symlink_to(outside)

    assert _supervisor().status().status == "idle"


def test_direct_v21_dataset_is_accepted(roots):
    dataset_root, *_ = roots
    dataset = _write_v21(dataset_root / "direct-v21")
    assert service._resolve_datasets([str(dataset)]) == [dataset.resolve()]


def test_v30_is_accepted_directly_even_when_a_v21_pair_exists(roots):
    dataset_root, *_ = roots
    epoch = dataset_root / "data_epoch_0007"
    v30 = _write_v30(epoch / "task_lerobot_v30")
    v21 = _write_v21(epoch / "task_lerobot_v21")
    (epoch / "cyclo_data_epoch.json").write_text(
        json.dumps(
            {
                "formats": ["v2.1", "v3.0"],
                "expected_outputs": {"v30": str(v30), "v21": str(v21)},
            }
        ),
        encoding="utf-8",
    )

    assert service._resolve_datasets([str(v30)]) == [v30.resolve()]


def test_v30_without_explicit_v21_pair_is_accepted_directly(roots):
    dataset_root, *_ = roots
    epoch = dataset_root / "data_epoch_0008"
    v30 = _write_v30(epoch / "task_lerobot_v30")
    (epoch / "cyclo_data_epoch.json").write_text(
        json.dumps({"expected_outputs": {"v30": str(v30)}}),
        encoding="utf-8",
    )

    assert service._resolve_datasets([str(v30)]) == [v30.resolve()]


def test_v30_does_not_require_or_infer_an_unrecorded_sibling(roots):
    dataset_root, *_ = roots
    epoch = dataset_root / "data_epoch_0009"
    v30 = _write_v30(epoch / "task_lerobot_v30")
    _write_v21(epoch / "task_lerobot_v21")

    assert service._resolve_datasets([str(v30)]) == [v30.resolve()]


def test_v30_and_v21_are_both_supported_sources(roots):
    dataset_root, *_ = roots
    epoch = dataset_root / "data_epoch_0010"
    v30 = _write_v30(epoch / "task_lerobot_v30")
    v21 = _write_v21(epoch / "task_lerobot_v21")
    (epoch / "cyclo_data_epoch.json").write_text(
        json.dumps({"expected_outputs": {"v30": str(v30), "v21": str(v21)}}),
        encoding="utf-8",
    )

    assert service._resolve_datasets([str(v30), str(v21)]) == [
        v30.resolve(),
        v21.resolve(),
    ]


def test_groot_checkpoint_requires_groot_config_and_weights(roots):
    _, model_root, *_ = roots
    checkpoint = _write_groot(model_root / "showroom_groot")
    assert service._resolve_groot_checkpoint(str(checkpoint)) == checkpoint.resolve()

    wrong = model_root / "act"
    wrong.mkdir()
    (wrong / "config.json").write_text('{"model_type": "act"}', encoding="utf-8")
    (wrong / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(HTTPException, match="not a GR00T model"):
        service._resolve_groot_checkpoint(str(wrong))


def test_paths_reject_symlink_escape(roots, tmp_path):
    dataset_root, *_ = roots
    outside = _write_v21(tmp_path / "outside")
    alias = dataset_root / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(HTTPException, match="symbolic links"):
        service._resolve_datasets([str(alias)])


def test_output_rejects_symlink_component(roots, tmp_path, monkeypatch):
    _, _, checkpoint_root, _, _ = roots
    outside = tmp_path / "outside-output"
    outside.mkdir()
    linked = checkpoint_root / "rlt"
    linked.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(service, "RLT_STAGE1_OUTPUT_ROOT", linked / "stage1")

    with pytest.raises(HTTPException, match="symbolic links"):
        service._output_path("a" * 32, 100, [tmp_path / "dataset"])


def test_output_path_identifies_dataset_weight_type_and_kst(roots, monkeypatch):
    dataset_root, *_, output_root, _ = roots
    monkeypatch.setattr(
        service,
        "_kst_run_timestamp",
        lambda: "20260923070102",
    )
    dataset = dataset_root / "data_epoch_0001" / "task_lerobot_v30"

    output = service._output_path("a" * 32, 100, [dataset])

    assert output == output_root / (
        "data-data_epoch_0001--task_lerobot_v30__"
        "rlt_stage1_rl_token_encoder_weights__"
        "steps_000100_aaaaaaaaaaaa_KST_20260923070102"
    )
    assert service._STAGE1_OUTPUT_PATTERN.fullmatch(output.name)


def test_run_timestamp_timezone_is_kst():
    assert service._KST.utcoffset(None) == service.timedelta(hours=9)
    assert service._KST.tzname(None) == "KST"


def test_dataset_path_label_records_multiple_paths_and_bounds_long_names(roots):
    dataset_root, *_ = roots
    datasets = [
        dataset_root / "epoch_0001" / "success_lerobot_v30",
        dataset_root / "epoch_0002" / "failure_lerobot_v30",
    ]

    assert service._dataset_path_label(datasets) == (
        "epoch_0001--success_lerobot_v30+epoch_0002--failure_lerobot_v30"
    )

    long_paths = [dataset_root / ("dataset_" + "x" * 120)]
    label = service._dataset_path_label(long_paths)
    assert len(label) <= service._DATASET_LABEL_MAX_LENGTH
    assert "--set-1-" in label


def test_command_uses_one_shot_groot_stage1_cli(tmp_path):
    supervisor = _supervisor()
    job = _job(tmp_path)
    command = supervisor._command(job)

    assert command[:5] == ["docker", "compose", "-f", "/tmp/compose.yml", "run"]
    assert command[command.index("--name") + 1] == "cyclo_rlt_stage1_aaaaaaaaaaaa"
    assert command[command.index("--entrypoint") + 1] == "python"
    assert "groot" in command
    assert "runtime.rlt_stage1_training_cli" in command
    assert command[command.index("--groot-checkpoint") + 1] == job.groot_checkpoint
    assert command[command.index("--dataset-root") + 1] == job.resolved_dataset_paths[0]
    assert command[command.index("--steps") + 1] == "100"
    assert command[command.index("--batch-size") + 1] == "1"
    assert command[command.index("--save-freq") + 1] == "25"
    runtime_environment = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--env"
    ]
    assert f"HF_HOME={service.RLT_STAGE1_CACHE_ROOT}/huggingface" in runtime_environment
    assert f"HF_HUB_CACHE={service.RLT_STAGE1_HF_HUB_CACHE}" in runtime_environment
    assert f"HUGGINGFACE_HUB_CACHE={service.RLT_STAGE1_HF_HUB_CACHE}" in runtime_environment
    assert f"TRANSFORMERS_CACHE={service.RLT_STAGE1_HF_HUB_CACHE}" in runtime_environment
    assert "USER=cyclo_rlt" in runtime_environment
    assert "LOGNAME=cyclo_rlt" in runtime_environment
    assert (
        f"TORCHINDUCTOR_CACHE_DIR={service.RLT_STAGE1_CACHE_ROOT}/torchinductor"
        in runtime_environment
    )
    assert "HF_HUB_OFFLINE=1" in runtime_environment
    assert "TRANSFORMERS_OFFLINE=1" in runtime_environment
    assert "GROOT_HF_LOCAL_FIRST=1" in runtime_environment
    assert "GROOT_PATCH_MISTRAL=1" in runtime_environment
    assert "NO_ALBUMENTATIONS_UPDATE=1" in runtime_environment
    assert all("/root/.cache" not in value for value in runtime_environment)


def test_monitor_requires_reported_existing_encoder_and_checkpoint(tmp_path):
    output = tmp_path / "stage1-output"
    encoder = output / "artifacts" / "rl_token_encoder.pt"
    checkpoint = output / "training_state" / "rlt_stage1.pt"
    encoder.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    encoder.write_bytes(b"encoder")
    checkpoint.write_bytes(b"checkpoint")
    lines = [
        json.dumps(
            {
                "event": "progress",
                "phase": "training_rl_token",
                "completed_steps": 50,
                "total_steps": 100,
                "reconstruction_loss": 0.125,
                "eta_seconds": 2.0,
            }
        )
        + "\n",
        json.dumps(
            {
                "event": "completed",
                "status": "completed",
                "completed_steps": 100,
                "encoder_artifact_path": str(encoder),
                "checkpoint_path": str(checkpoint),
            }
        )
        + "\n",
    ]
    job = _job(tmp_path, process=FakeProcess(stdout=lines), output_dir=str(output))
    supervisor = _supervisor()
    supervisor._job = job

    supervisor._monitor(job)

    status = supervisor.status()
    assert status.status == "completed"
    assert status.phase == "complete"
    assert status.percentage == 100.0
    assert status.reconstruction_loss == pytest.approx(0.125)
    assert status.encoder_artifact_path == str(encoder)
    assert status.checkpoint_path == str(checkpoint)


def test_monitor_rejects_success_without_terminal_artifact_report(tmp_path):
    job = _job(
        tmp_path,
        process=FakeProcess(
            stdout=[json.dumps({"event": "progress", "completed_steps": 100}) + "\n"]
        ),
    )
    supervisor = _supervisor()
    supervisor._job = job
    supervisor._monitor(job)
    assert supervisor.status().status == "failed"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "event": "error",
            "status": "failed",
            "message": "RLT Stage 1 failed: PermissionError: denied",
            "error": "PermissionError: denied",
        },
        {
            "event": "result",
            "status": "failed",
            "error": "PermissionError: denied",
        },
    ],
    ids=("standard-error", "legacy-result"),
)
def test_monitor_preserves_cli_failure_detail(tmp_path, payload):
    job = _job(
        tmp_path,
        process=FakeProcess(stdout=[json.dumps(payload) + "\n"], returncode=1),
    )
    supervisor = _supervisor()
    supervisor._job = job

    supervisor._monitor(job)

    status = supervisor.status()
    assert status.status == "failed"
    assert status.phase == "error"
    assert "PermissionError: denied" in status.message


def test_start_validates_then_launches_exact_job(roots, monkeypatch):
    dataset_root, model_root, *_ = roots
    dataset = _write_v21(dataset_root / "direct-v21")
    checkpoint = _write_groot(model_root / "showroom_groot")
    process = FakeProcess()
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(service.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(service.threading, "Thread", NoopThread)
    supervisor = _supervisor()
    status = supervisor.start(
        service.RLTStage1StartRequest(
            dataset_paths=[str(dataset)],
            groot_checkpoint=str(checkpoint),
            steps=123,
            batch_size=2,
            save_freq=50,
        )
    )

    assert status.status == "running"
    assert status.resolved_dataset_paths == [str(dataset.resolve())]
    assert status.total_steps == 123
    assert captured["command"][captured["command"].index("--batch-size") + 1] == "2"
    assert captured["kwargs"]["env"] == {"COMPOSE_PROJECT_NAME": "cyclo"}


def test_conflict_is_checked_before_paths():
    supervisor = _supervisor(conflict=lambda: "GPU training is busy")
    with pytest.raises(HTTPException, match="GPU training is busy"):
        supervisor.start(
            service.RLTStage1StartRequest(
                dataset_paths=["/missing"],
                groot_checkpoint="/missing",
            )
        )


def test_stop_requires_current_job_id_and_signals_owned_container(tmp_path):
    interrupted = []
    process = FakeProcess()
    job = _job(tmp_path, process=process)
    supervisor = _supervisor(
        interrupt_container=lambda name: interrupted.append(name) or True
    )
    supervisor._job = job

    with pytest.raises(HTTPException, match="does not match"):
        supervisor.stop(service.RLTStage1StopRequest(job_id="stale"))
    status = supervisor.stop(service.RLTStage1StopRequest(job_id=job.job_id))
    assert status.phase == "stopping"
    assert interrupted == ["cyclo_rlt_stage1_aaaaaaaaaaaa"]
    assert process.signals == []


def test_stop_falls_back_to_compose_process_signal(tmp_path):
    process = FakeProcess()
    job = _job(tmp_path, process=process)
    supervisor = _supervisor(interrupt_container=lambda _name: False)
    supervisor._job = job
    supervisor.stop(service.RLTStage1StopRequest(job_id=job.job_id))
    assert process.signals == [signal.SIGINT]


def test_explicit_imported_seed_preserves_remote_identity_and_overrides_discovery(roots, monkeypatch):
    try:
        from supervisor_api import rlt_stage2_service as stage2
    except ModuleNotFoundError:
        from docker.supervisor_api import rlt_stage2_service as stage2
    _, model_root, _, output_root, _ = roots
    groot = _write_groot(model_root / 'base')
    output = _write_completed_stage1_run(
        output_root, dataset=Path('/upstream/dataset'), groot=groot,
        steps=10000, job_id='remote-training-job',
    )
    monkeypatch.setattr(stage2, 'RLT_STAGE2_ENCODER_ROOTS', (output_root,))
    encoder = output / 'artifacts' / 'rl_token_encoder.pt'
    (output_root / 'selected_seed.json').write_text(json.dumps({
        'encoder_artifact_path': str(encoder), 'groot_checkpoint': str(groot),
    }))
    supervisor = _supervisor()
    status = supervisor.status()
    assert status.status == 'completed'
    assert status.encoder_artifact_path == str(encoder)
    assert status.job_id == 'remote-training-job'
    assert status.dataset_paths == []  # Do not populate local dataset selectors with remote paths.
    assert 'imported' in status.message
    supervisor._job = _job(output_root, status='running')
    assert supervisor.status().status == 'running'


def test_invalid_explicit_seed_does_not_fall_back_to_previous_seed(roots):
    _, model_root, _, output_root, _ = roots
    groot = _write_groot(model_root / 'base')
    _write_completed_stage1_run(
        output_root, dataset=roots[0] / 'dataset', groot=groot,
        steps=10000, job_id='a' * 32,
    )
    (output_root / 'selected_seed.json').write_text('{}')
    status = _supervisor().status()
    assert status.status == 'failed'
    assert not status.encoder_artifact_path
    assert 'Invalid selected' in status.message
