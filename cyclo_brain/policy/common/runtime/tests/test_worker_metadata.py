"""Workers and Cyclo must accept the same deployment catalog."""

from pathlib import Path
import sys

import pytest
import yaml

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT.parent))

from catalog import load_runtime_catalog, resolve_runtime
from engine_process.worker import _worker_metadata


@pytest.mark.parametrize("backend", ["lerobot", "groot", "rldx"])
def test_worker_metadata_matches_validated_catalog(monkeypatch, backend):
    path = RUNTIME_ROOT.parents[1] / backend / "manifest.yaml"
    monkeypatch.setenv("POLICY_MANIFEST_PATH", str(path))
    runtime = resolve_runtime(load_runtime_catalog(path, backend), backend)
    ids, capabilities = _worker_metadata(backend)
    assert ids == [model["policy_id"] for model in runtime["models"]]
    assert capabilities == runtime["capabilities"]


def test_missing_manifest_prevents_worker_start(monkeypatch, tmp_path):
    monkeypatch.setenv("POLICY_MANIFEST_PATH", str(tmp_path / "missing.yaml"))
    with pytest.raises(RuntimeError, match="invalid worker manifest"):
        _worker_metadata("lerobot")


def test_invalid_manifest_fails_before_engine_import_or_service_creation(monkeypatch, tmp_path):
    from engine_process import worker

    monkeypatch.setenv("POLICY_BACKEND", "lerobot")
    monkeypatch.setenv("POLICY_MANIFEST_PATH", str(tmp_path / "missing.yaml"))

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid deployment must fail before initializing model or transport")

    monkeypatch.setattr(worker, "resolve_engine", unexpected)
    monkeypatch.setattr(worker, "EngineWorker", unexpected)
    with pytest.raises(RuntimeError, match="invalid worker manifest"):
        worker.main()


@pytest.mark.parametrize("mutation", [
    "empty_models", "duplicate_model", "malformed_model", "unknown_field",
    "wrong_prefix", "wrong_runtime", "invalid_capabilities",
])
def test_invalid_manifest_is_not_silently_advertised(monkeypatch, tmp_path, mutation):
    source = RUNTIME_ROOT.parents[1] / "lerobot" / "manifest.yaml"
    manifest = yaml.safe_load(source.read_text())
    if mutation == "empty_models":
        manifest["models"] = []
    elif mutation == "duplicate_model":
        manifest["models"].append(dict(manifest["models"][0]))
    elif mutation == "malformed_model":
        manifest["models"].append("misspelled model entry")
    elif mutation == "unknown_field":
        manifest["models"][0]["require_instruction"] = True
    elif mutation == "wrong_prefix":
        manifest["runtime"]["service_prefix"] = "other"
    elif mutation == "wrong_runtime":
        manifest["runtime"]["id"] = "other"
    elif mutation == "invalid_capabilities":
        manifest["runtime"]["capabilities"]["action_request_modes"] = ["typo"]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))
    monkeypatch.setenv("POLICY_MANIFEST_PATH", str(path))
    with pytest.raises(RuntimeError, match="invalid worker manifest"):
        _worker_metadata("lerobot")
