import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml

from test_training_artifacts import assets
from test_training_data import prepared
from test_training_inputs import input_profile
from lingbot_vla_training.runtime import read_run, typed_native_arguments, training_probe_context


@dataclass
class NativeNumbers:
    coefficient: float
    path: str


def test_scientific_yaml_numbers_follow_native_cli_types():
    values = yaml.safe_load("coefficient: 1e-4\npath: 1e-4\n")
    assert typed_native_arguments(values, NativeNumbers) == {"coefficient": 0.0001, "path": "1e-4"}
    assert values["coefficient"] == "1e-4"


@pytest.mark.parametrize("value", [True, "nan", "inf", "invalid"])
def test_invalid_native_coefficients_are_rejected(value):
    with pytest.raises(ValueError):
        typed_native_arguments({"coefficient": value}, NativeNumbers)


@pytest.mark.parametrize("fail", [False, True])
def test_training_probe_preserves_router_counters_and_modes(fail):
    module = torch.nn.Sequential(torch.nn.Linear(1, 1))
    module[0].register_buffer("tokens_per_expert", torch.tensor([3.]), persistent=False)
    module[0].register_buffer("cache", None, persistent=False)
    module[0].eval()
    original = module[0].tokens_per_expert
    try:
        with training_probe_context(module):
            assert module[0].training
            assert not torch.is_grad_enabled()
            module[0].tokens_per_expert.add_(10)
            module[0].cache = torch.ones(1)
            if fail:
                raise RuntimeError("probe failed")
    except RuntimeError:
        assert fail
    assert module.training and not module[0].training
    assert module[0].tokens_per_expert is original
    assert original.item() == 3
    assert module[0].cache is None


@pytest.fixture
def run_json(tmp_path, assets, prepared):
    root, profile = prepared
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile))
    asset_path = tmp_path / "assets.json"
    asset_path.write_text(json.dumps(assets))
    recipe = Path(__file__).resolve().parents[1] / "lingbot-vla-v2/configs/vla/real_robot/real_robot.yaml"
    if not recipe.exists():
        pytest.skip("Pinned upstream is not initialized")
    run = {"version": 1, "profile": str(profile_path), "assets_manifest": str(asset_path),
           "dataset": str(root), "recipe": str(recipe), "output_dir": str(tmp_path / "output")}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    return path


def test_validate_phase_needs_no_gpu_and_has_explicit_retention(run_json):
    _, settings, _, _, _, _, episodes, _ = read_run(run_json)
    assert settings["save_steps"] == 1
    assert settings["save_total_limit"] == 1
    assert len(episodes) == 2


@pytest.mark.parametrize("settings", [
    {"save_total_limit": 0}, {"save_steps": 0}, {"max_steps": False}, {"num_workers": -1},
    {"chunk_size": 1}, {"lr": 0}, {"use_compile": True}, {"image_augment": 1}, {"unknown": 1},
])
def test_invalid_settings_fail_before_gpu_allocation(run_json, settings):
    run = json.loads(run_json.read_text())
    run["settings"] = settings
    run_json.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        read_run(run_json)


@pytest.mark.parametrize("change", ["optimizer", "teacher", "freeze", "cls", "frames", "weight"])
def test_recipe_cannot_silently_disable_native_objectives(run_json, tmp_path, change):
    run = json.loads(run_json.read_text())
    recipe = yaml.safe_load(Path(run["recipe"]).read_text())
    train = recipe["train"]
    if change == "optimizer":
        train["optimizer"] = "adamw"
    elif change == "teacher":
        train["align_params"]["use_future_video"] = False
    elif change == "freeze":
        train["train_expert_only"] = True
    elif change == "cls":
        train["align_params"]["video"]["use_cls_loss"] = True
    elif change == "frames":
        train["align_params"]["video"]["num_future_frames"] = 2
    else:
        train["align_params"]["depth_loss_weight"] = 0
    path = tmp_path / "bad-recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    run["recipe"] = str(path)
    run_json.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        read_run(run_json)
