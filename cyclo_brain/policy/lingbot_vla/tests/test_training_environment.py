import importlib.util
import io
from pathlib import Path
import tarfile
import tomllib

import pytest


POLICY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lingbot_training_environment", POLICY / "training/environment.py")
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


def test_recorded_build_inputs_match_checkout():
    if not (POLICY / "lingbot-vla-v2/requirements.txt").is_file():
        pytest.skip("Upstream submodule is not initialized")
    contract = environment.verify_inputs(POLICY / "training", POLICY / "lingbot-vla-v2")
    assert contract["upstream_commit"] == "ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8"


def test_docker_and_source_contract_match():
    contract = tomllib.loads((POLICY / "training/environment.toml").read_text())
    docker = (POLICY / "Dockerfile.training").read_text()
    assert f'FROM {contract["base_image"]}' in docker
    assert f'COPY --from={contract["uv_image"]}' in docker
    flash = contract["flash_attention"]
    assert flash["url"] + "#sha256=" + flash["sha256"] in docker
    assert contract["python_path"] in docker
    assert "environment.py verify" in docker
    assert "--require-hashes" in docker
    assert "requirements-build.lock" in docker
    assert "start-lingbot" not in docker and 'ENTRYPOINT ["/init"]' not in docker


def test_no_teacher_training_tools_in_lock():
    lock = (POLICY / "training/requirements.lock").read_text()
    for package in ("mlflow", "plyfile", "gradio", "xformers"):
        assert f"\n{package}==" not in lock
    for package in ("pyarrow==21.0.0", "numpy==1.26.4", "torch==2.8.0", "utils3d @"):
        assert f"\n{package}" in lock
    assert "--hash=sha256:" in lock


@pytest.fixture
def recorded_inputs(tmp_path):
    policy = tmp_path / "policy"
    upstream = tmp_path / "upstream"
    policy.mkdir()
    upstream.mkdir()
    (upstream / "model.py").write_text("original source\n")
    (policy / "requirements.lock").write_text("example==1.0\n")
    (policy / "environment.toml").write_text(
        f'upstream_tree_sha256 = "{environment.source_tree_sha256(upstream)}"\n'
        f'[policy_hashes]\n"requirements.lock" = "{environment.sha256(policy / "requirements.lock")}"\n'
    )
    return policy, upstream


@pytest.mark.parametrize("change", ["lock", "upstream", "missing"])
def test_source_or_lock_drift_fails(recorded_inputs, change):
    policy, upstream = recorded_inputs
    if change == "lock":
        (policy / "requirements.lock").write_text("example==2.0\n")
    elif change == "upstream":
        (upstream / "model.py").write_text("changed source\n")
    else:
        (upstream / "model.py").unlink()
    with pytest.raises(ValueError):
        environment.verify_inputs(policy, upstream)


def test_generated_bytecode_does_not_change_source_identity(recorded_inputs):
    policy, upstream = recorded_inputs
    (upstream / "__pycache__").mkdir()
    (upstream / "__pycache__/model.pyc").write_bytes(b"bytecode")
    environment.verify_inputs(policy, upstream)


def test_source_symlinks_rejected(recorded_inputs):
    _, upstream = recorded_inputs
    (upstream / "alias.py").symlink_to(upstream / "model.py")
    with pytest.raises(ValueError, match="symlink"):
        environment.source_tree_sha256(upstream)


@pytest.mark.parametrize("case", ["success", "hash_mismatch", "traversal", "link", "existing"])
def test_source_fetch_is_verified_and_confined(tmp_path, monkeypatch, case):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../escape" if case == "traversal" else "source/model.py")
        if case == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            tar.addfile(member)
        else:
            member.size = 4
            tar.addfile(member, io.BytesIO(b"code"))
    monkeypatch.setattr(environment.urllib.request, "urlopen", lambda *a, **kw: archive.open("rb"))
    destination = tmp_path / "installed"
    asset = {
        "url": "https://example.invalid/pinned.tar.gz",
        "sha256": "0" * 64 if case == "hash_mismatch" else environment.sha256(archive),
        "archive_root": "source",
    }
    if case == "existing":
        destination.mkdir()
        (destination / "preserve").write_text("existing")
    if case == "success":
        environment.fetch_source(asset, destination)
        assert (destination / "model.py").read_bytes() == b"code"
    else:
        with pytest.raises(ValueError):
            environment.fetch_source(asset, destination)
        if case == "existing":
            assert (destination / "preserve").read_text() == "existing"
        else:
            assert not destination.exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("link_in_src", [False, True])
def test_source_subtree_ignores_documentation_links_but_rejects_code_links(
    tmp_path, monkeypatch, link_in_src
):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        code = tarfile.TarInfo("source/src/package/__init__.py")
        code.size = 4
        tar.addfile(code, io.BytesIO(b"code"))
        license_file = tarfile.TarInfo("source/LICENSE")
        license_file.size = 7
        tar.addfile(license_file, io.BytesIO(b"license"))
        link = tarfile.TarInfo("source/src/link.py" if link_in_src else "source/src/README.md")
        link.type = tarfile.SYMTYPE
        link.linkname = "../README.md"
        tar.addfile(link)
    monkeypatch.setattr(environment.urllib.request, "urlopen", lambda *a, **kw: archive.open("rb"))
    destination = tmp_path / "installed"
    asset = {
        "url": "https://example.invalid/source.tar.gz",
        "sha256": environment.sha256(archive),
        "archive_root": "source",
        "source_subdir": "src",
        "exclude_suffixes": [".md", ".mdx"],
        "root_files": ["LICENSE"],
    }
    if link_in_src:
        with pytest.raises(ValueError, match="Unsupported"):
            environment.fetch_source(asset, destination)
        assert not destination.exists()
    else:
        environment.fetch_source(asset, destination)
        assert (destination / "src/package/__init__.py").read_bytes() == b"code"
        assert (destination / "LICENSE").read_bytes() == b"license"
        assert not (destination / "docs").exists()
