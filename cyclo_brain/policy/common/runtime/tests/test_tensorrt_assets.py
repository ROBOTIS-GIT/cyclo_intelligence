"""Checkpoint/GPU binding must reject stale or tampered TensorRT plans."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("trt_assets_under_test", Path(__file__).parents[1] / "tensorrt_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def make_plan(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"type":"act"}')
    (tmp_path / "model.safetensors").write_bytes(b"original weights")
    engine = tmp_path / "act_model_fp32.trt"
    engine.write_bytes(b"verified engine")
    monkeypatch.setattr(assets, "runtime_identity", lambda: {"gpu": "Orin", "tensorrt": "10.7"})
    manifest = {"status": "ready", "adapter": "lerobot_act", "runtime_identity": assets.runtime_identity(),
                "checkpoint_identity": assets.checkpoint_identity(tmp_path), "engine_sha256": assets.sha256_file(engine)}
    assets.write_manifest(engine, manifest)
    return engine, manifest


def test_verified_plan_survives_added_registration_metadata(tmp_path, monkeypatch):
    engine, manifest = make_plan(tmp_path, monkeypatch)
    (tmp_path / "cyclo_inference.json").write_text('{"action_keys":["arm_right"]}')
    (tmp_path / "other.trt.json").write_text('{}')
    assert assets.validate_manifest(tmp_path, engine, "lerobot_act") == manifest


@pytest.mark.parametrize("change", ["weights", "config", "engine", "gpu", "adapter", "status"])
def test_rejects_changed_plan_inputs(tmp_path, monkeypatch, change):
    engine, manifest = make_plan(tmp_path, monkeypatch)
    if change == "weights":
        (tmp_path / "model.safetensors").write_bytes(b"different weights")
    elif change == "config":
        (tmp_path / "config.json").write_text('{"type":"act","chunk_size":5}')
    elif change == "engine":
        engine.write_bytes(b"modified engine")
    elif change == "gpu":
        monkeypatch.setattr(assets, "runtime_identity", lambda: {"gpu": "different", "tensorrt": "10.7"})
    else:
        manifest[change] = "failed" if change == "status" else "groot_dit"
        assets.write_manifest(engine, manifest)
    with pytest.raises(RuntimeError):
        assets.validate_manifest(tmp_path, engine, "lerobot_act")
