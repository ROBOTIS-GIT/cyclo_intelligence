import json

import pytest

from lingbot_vla_training.artifacts import complete_checkpoint, prune_checkpoints, sha256, verify_assets, verify_checkpoint


def staging(root, name=".partial"):
    path = root / name
    path.mkdir()
    for file in ("train_config.json", "cyclo_io_mapping.json", "norm_stats.json", "config.json", "probe.safetensors", "model.safetensors"):
        (path / file).write_text("test")
    return path


def test_checkpoint_completion_and_mutation_detection(tmp_path):
    pending = staging(tmp_path)
    final = tmp_path / "step-1"
    complete_checkpoint(pending, final, 1)
    assert not pending.exists()
    assert verify_checkpoint(final)["optimizer_step"] == 1
    (final / "model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(final)


def test_incomplete_checkpoint_not_published(tmp_path):
    pending = staging(tmp_path)
    (pending / "probe.safetensors").unlink()
    with pytest.raises(ValueError):
        complete_checkpoint(pending, tmp_path / "step-1", 1)
    assert not (tmp_path / "step-1").exists()
    assert not (pending / "complete.json").exists()


def test_retention_keeps_incomplete_and_unrelated_paths(tmp_path):
    incomplete = tmp_path / "step-3"
    incomplete.mkdir()
    other = tmp_path / "user-data"
    other.mkdir()
    for step in (1, 2):
        complete_checkpoint(staging(tmp_path), tmp_path / f"step-{step}", step)
    prune_checkpoints(tmp_path, 1)
    assert not (tmp_path / "step-1").exists()
    assert (tmp_path / "step-2").exists()
    assert incomplete.exists() and other.exists()
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path, 0)


def test_existing_destination_never_overwritten(tmp_path):
    pending = staging(tmp_path)
    final = tmp_path / "step-1"
    final.mkdir()
    with pytest.raises(FileExistsError):
        complete_checkpoint(pending, final, 1)


@pytest.fixture
def assets(tmp_path):
    all_files = {
        "pretrained": {"model.safetensors": "weights", "model.safetensors.index.json": json.dumps({"weight_map": {"weight": "model.safetensors"}})},
        "processor": {name: "{}" for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")},
        **{key: {"model.pt": "teacher"} for key in ("moge", "morgbd", "video", "video_config")},
    }
    manifest = {"version": 1, "assets": {}}
    for role, files in all_files.items():
        root = tmp_path / role
        root.mkdir()
        for name, text in files.items():
            (root / name).write_text(text)
        manifest["assets"][role] = {"path": str(root), "repo_id": "org/model", "revision": "a" * 40,
                                     "files": [{"path": name, "sha256": sha256(root / name)} for name in files]}
    return manifest


def test_asset_roles_and_hashes(assets):
    paths = verify_assets(assets)
    assert paths["moge"].name == "model.pt"
    assert paths["processor"].is_dir()


@pytest.mark.parametrize("failure", ["missing", "revision", "hash", "duplicate", "unindexed", "processor"])
def test_bad_assets_fail_before_model_loading(assets, failure):
    if failure == "missing":
        del assets["assets"]["moge"]
    elif failure == "revision":
        assets["assets"]["processor"]["revision"] = "main"
    elif failure == "hash":
        assets["assets"]["video"]["files"][0]["sha256"] = "0" * 64
    elif failure == "duplicate":
        assets["assets"]["moge"]["files"] *= 2
    elif failure == "unindexed":
        assets["assets"]["pretrained"]["files"].pop(0)
    else:
        assets["assets"]["processor"]["files"].pop(0)
    with pytest.raises(ValueError):
        verify_assets(assets)
