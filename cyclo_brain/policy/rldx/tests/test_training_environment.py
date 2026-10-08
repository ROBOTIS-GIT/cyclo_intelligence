"""Pin and lock validation without downloading CUDA packages."""

from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import install_training_environment as module


def test_pinned_upstream_manifest_and_lock():
    repo = SCRIPTS.parent / "RLDX-1"
    if not (repo / "pixi.lock").is_file():
        pytest.skip("RLDX upstream submodule is not initialized")
    policy = module.validate(repo, module.POLICY, "120", 2)
    assert policy["pyarrow_version"] == "24.0.0"
    assert policy["flash_attention_commit"] == "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"


def test_lock_mismatch_fails_before_install(tmp_path):
    repo = tmp_path / "rldx"
    repo.mkdir()
    (repo / "pixi.toml").write_text("wrong\n")
    (repo / "pixi.lock").write_text("wrong\n")
    with pytest.raises(ValueError, match="pixi.toml"):
        module.validate(repo, module.POLICY, "120", 2)

    source = SCRIPTS.parent / "RLDX-1" / "pixi.toml"
    if source.is_file():
        (repo / "pixi.toml").write_bytes(source.read_bytes())
        with pytest.raises(ValueError, match="pixi.lock"):
            module.validate(repo, module.POLICY, "120", 2)


def test_wrong_pixi_version_fails_before_install(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "validate", lambda *args: {"pixi_version": "0.65.0"})
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "pixi 0.70.1")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("install started"))
    with pytest.raises(ValueError, match="Expected pixi 0.65.0"):
        module.install(tmp_path, module.POLICY, "120", 2)


@pytest.mark.parametrize("archs", ["", "75", "120;75", "12.0"])
def test_unsupported_flash_arch_is_rejected(tmp_path, archs):
    repo = SCRIPTS.parent / "RLDX-1"
    if not (repo / "pixi.lock").is_file():
        pytest.skip("RLDX upstream submodule is not initialized")
    with pytest.raises(ValueError, match="architecture"):
        module.validate(repo, module.POLICY, archs, 2)
