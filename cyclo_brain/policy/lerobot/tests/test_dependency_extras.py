"""Validate the Cyclo-owned LeRobot installation profiles."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import tomllib
import yaml

POLICY_DIR = Path(__file__).resolve().parents[1]
PROFILE = POLICY_DIR / "dependency_profile/dependencies.toml"
PROJECT = POLICY_DIR / "lerobot"
spec = importlib.util.spec_from_file_location("lerobot_install", POLICY_DIR / "dependency_profile/install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_inference_profile_matches_manifest():
    catalog = yaml.safe_load((POLICY_DIR / "manifest.yaml").read_text())
    selected = installer.resolve(PROFILE, PROJECT, "inference")
    assert set(selected["models"]) == {model["id"] for model in catalog["models"]}
    assert selected["extras"] == [
        "training", "hilserl", "async", "peft", "diffusion", "smolvla",
        "xvla", "pi", "molmoact2", "vla-jepa", "fastwam", "wallx", "groot",
    ]
    assert all(not extra.startswith("cyclo-") for extra in selected["extras"])
    assert len(selected["profile_sha256"]) == len(selected["lock_sha256"]) == 64


def test_training_image_is_a_distinct_installation_profile():
    selected = installer.resolve(PROFILE, PROJECT, "training-image")
    assert selected["extras"][:2] == ["training", "peft"]
    assert "hilserl" not in selected["extras"]
    assert "async" not in selected["extras"]
    assert "multi-task-dit" not in selected["extras"]
    assert "training_supported" not in tomllib.loads(PROFILE.read_text())


def test_profile_does_not_require_cyclo_extras_in_fork(tmp_path):
    extras = installer.resolve(PROFILE, PROJECT, "inference")["extras"]
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "lerobot"\n[project.optional-dependencies]\n'
        + "".join(f'"{name}" = []\n' for name in extras)
    )
    (tmp_path / "uv.lock").write_text("locked checkout")
    assert installer.resolve(PROFILE, tmp_path, "inference")["extras"] == extras


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("schema_version = 1", "schema_version = 2", "schema"),
        ('extra = "diffusion"', 'extra = "missing"', "not defined"),
        ('extra = "diffusion"', 'extra = "diffusion;rm"', "Invalid LeRobot extra"),
        ('profiles = ["inference", "training-image"]', 'profiles = ["unknown"]', "Unknown profile"),
        ('features = ["training", "hilserl", "async", "peft"]', 'features = ["training", "training"]', "Invalid features"),
        ('features = ["training", "hilserl", "async", "peft"]', 'features = ["training", 1]', "Invalid features"),
        ('profiles = ["inference", "training-image"]', 'profiles = ["inference", 1]', "Invalid profiles"),
    ],
)
def test_invalid_profile_fails_before_install(tmp_path, old, new, error):
    modified = PROFILE.read_text().replace(old, new, 1)
    config = tmp_path / "dependencies.toml"
    config.write_text(modified)
    with pytest.raises(ValueError, match=error):
        installer.resolve(config, PROJECT, "inference")


def test_missing_lockfile_is_rejected(tmp_path):
    (tmp_path / "pyproject.toml").write_bytes((PROJECT / "pyproject.toml").read_bytes())
    with pytest.raises(ValueError, match="uv.lock"):
        installer.resolve(PROFILE, tmp_path, "inference")


@pytest.mark.parametrize("profile", ["inference", "training-image"])
def test_unknown_profile_is_rejected(profile):
    with pytest.raises(ValueError, match="Unknown installation profile"):
        installer.resolve(PROFILE, PROJECT, profile + "-unknown")


@pytest.mark.parametrize("manager", ["uv", "pip"])
def test_install_command_uses_only_resolved_extras(monkeypatch, capsys, manager):
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["install.py", "--profile", "inference", "--project", str(PROJECT), "--manager", manager],
    )
    installer.main()
    selected = json.loads(capsys.readouterr().out)
    command, kwargs = calls[0]
    assert kwargs == {"cwd": PROJECT, "check": True}
    if manager == "uv":
        assert command[:5] == ["uv", "sync", "--locked", "--no-dev", "--no-cache"]
        assert command[5:] == [item for extra in selected["extras"] for item in ("--extra", extra)]
    else:
        assert command == ["pip", "install", "--no-cache-dir", f".[{','.join(selected['extras'])}]"]


def test_pinned_hash_mismatch_fails_before_install(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "install.py", "--profile", "inference", "--project", str(PROJECT),
        "--expected-profile-sha256", "0" * 64,
    ])
    with pytest.raises(ValueError, match="profile hash"):
        installer.main()


def test_pinned_lock_hash_mismatch_fails_before_install(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "install.py", "--profile", "inference", "--project", str(PROJECT),
        "--expected-lock-sha256", "0" * 64,
    ])
    with pytest.raises(ValueError, match="lockfile hash"):
        installer.main()


def test_dockerfiles_use_cyclo_profile():
    for architecture, manager in (("amd64", "uv"), ("arm64", "pip")):
        dockerfile = (POLICY_DIR / f"Dockerfile.{architecture}").read_text()
        assert "COPY cyclo_brain/policy/lerobot/dependency_profile/" in dockerfile
        assert f"--profile inference --project /lerobot --manager {manager}" in dockerfile
        assert "lerobot/uv.lock" in dockerfile
        assert "--extra smolvla" not in dockerfile
        assert '--expected-profile-sha256 "$CYCLO_PROFILE_SHA256"' in dockerfile
        assert '--expected-lock-sha256 "$LEROBOT_LOCK_SHA256"' in dockerfile
        assert 'org.robotis.cyclo.profile-sha256="$CYCLO_PROFILE_SHA256"' in dockerfile
        assert 'org.robotis.lerobot.lock-sha256="$LEROBOT_LOCK_SHA256"' in dockerfile


def test_compose_passes_provenance_to_lerobot_only():
    root = POLICY_DIR.parents[2]
    compose = yaml.safe_load((root / "docker/docker-compose.yml").read_text())
    args = compose["services"]["lerobot"]["build"]["args"]
    provenance = {
        "CYCLO_SOURCE_COMMIT", "LEROBOT_SOURCE_COMMIT",
        "CYCLO_SOURCE_STATE", "LEROBOT_SOURCE_STATE",
        "CYCLO_PROFILE_SHA256", "LEROBOT_LOCK_SHA256",
    }
    assert provenance <= args.keys()
    assert all(not provenance.intersection(service.get("build", {}).get("args", {}))
               for name, service in compose["services"].items() if name != "lerobot")
    script = (root / "docker/container.sh").read_text()
    assert "prepare_lerobot_build_metadata" in script
    assert 'recorded_lerobot_commit="$(git -C "$repo" rev-parse HEAD:cyclo_brain/policy/lerobot/lerobot)"' in script
