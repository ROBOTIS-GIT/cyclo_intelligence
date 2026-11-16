from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from lerobot.common.training_adapter import TrainingAdapterConfig, prepare_training_data
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig

from cyclo_lerobot_io.mapping import FILENAME, read_mapping
from cyclo_lerobot_io.selection import ChannelSelectionAdapter


class Metadata:
    def __init__(self):
        self.info = {
            "features": {
                key: {"dtype": "float32", "shape": (22,), "names": [f"j{i}" for i in range(22)]}
                for key in ("observation.state", "action")
            }
        }
        self.info["features"]["observation.images.head"] = {
            "dtype": "image",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        }
        self.stats = {
            key: {
                "mean": np.arange(22, dtype=np.float32),
                "std": np.ones(22),
                "min": np.zeros(22),
                "max": np.full(22, 30.0),
                "count": np.array([8]),
            }
            for key in ("observation.state", "action")
        }

    @property
    def features(self):
        return self.info["features"]


class Samples(Dataset):
    def __init__(self):
        self.meta = Metadata()
        self.sample = {
            "observation.state": torch.arange(44.0).reshape(2, 22),
            "action": torch.arange(88.0).reshape(4, 22),
            "observation.images.head": torch.zeros(3, 32, 32),
            "action_is_pad": torch.tensor([False, False, True, True]),
            "task": "pick",
        }

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return self.sample


def make_config(tmp_path, selection=None):
    path = tmp_path / "channels.yaml"
    path.write_text(
        yaml.safe_dump(
            selection
            or {
                "state_names": [f"j{i}" for i in range(16)],
                "action_names": [f"j{i}" for i in reversed(range(16))],
            }
        )
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="fixture/data"),
        policy=ACTConfig(device="cpu", pretrained_backbone_weights=None),
        training_adapter=TrainingAdapterConfig(name="cyclo_channels", config_path=path),
    )
    accelerator = SimpleNamespace(num_processes=1, is_main_process=True)
    return cfg, accelerator


def test_select_values_metadata_stats_and_history_without_mutating_source(tmp_path):
    cfg, acc = make_config(tmp_path)
    source, evaluation = Samples(), Samples()
    original = deepcopy(source.meta)
    selected, evaluated = prepare_training_data(ChannelSelectionAdapter(), source, evaluation, cfg, acc)
    for view in (selected, evaluated):
        sample = view[0]
        assert sample["observation.state"].shape == (2, 16)
        assert sample["action"].shape == (4, 16)
        torch.testing.assert_close(sample["action"], source.sample["action"][..., list(reversed(range(16)))])
        assert sample["observation.images.head"] is view.dataset.sample["observation.images.head"]
        assert sample["action_is_pad"] is view.dataset.sample["action_is_pad"]
        assert view.meta.features["action"]["shape"] == (16,)
        np.testing.assert_equal(view.meta.stats["action"]["mean"], list(reversed(range(16))))
        np.testing.assert_equal(view.meta.stats["action"]["count"], [8])
    assert source.meta.info == original.info
    np.testing.assert_equal(source.meta.stats["action"]["mean"], original.stats["action"]["mean"])
    assert source[0]["action"].shape[-1] == 22
    assert not hasattr(selected, "_ensure_reader")
    batch = next(iter(DataLoader(selected, batch_size=2, num_workers=2, multiprocessing_context="spawn")))
    assert batch["action"].shape == (2, 4, 16)


@pytest.mark.parametrize("state,action", [("all", ["j8", "j2"]), (["j3"], "all"), ("all", "all")])
def test_all_and_independent_channel_counts(tmp_path, state, action):
    cfg, acc = make_config(tmp_path, {"state_names": state, "action_names": action})
    selected, _ = prepare_training_data(ChannelSelectionAdapter(), Samples(), None, cfg, acc)
    assert selected[0]["observation.state"].shape[-1] == (22 if state == "all" else len(state))
    assert selected[0]["action"].shape[-1] == (22 if action == "all" else len(action))


@pytest.mark.parametrize("names", [[], ["missing"], ["j1", "j1"], [1], "j1"])
def test_invalid_selection(tmp_path, names):
    cfg, acc = make_config(tmp_path, {"state_names": names, "action_names": "all"})
    with pytest.raises(ValueError):
        prepare_training_data(ChannelSelectionAdapter(), Samples(), None, cfg, acc)


@pytest.mark.parametrize("change", ["missing_names", "duplicate", "dimension", "nan_stats", "stats_shape"])
def test_invalid_source_metadata(tmp_path, change):
    cfg, acc = make_config(tmp_path)
    source = Samples()
    ft = source.meta.features["action"]
    if change == "missing_names":
        del ft["names"]
    elif change == "duplicate":
        ft["names"][1] = ft["names"][0]
    elif change == "dimension":
        ft["shape"] = (21,)
    elif change == "nan_stats":
        source.meta.stats["action"]["std"][0] = np.nan
    else:
        source.meta.stats["action"]["mean"] = np.zeros(21)
    with pytest.raises(ValueError):
        prepare_training_data(ChannelSelectionAdapter(), source, None, cfg, acc)


def test_portable_resume_and_changed_selection_rejected(tmp_path):
    cfg, acc = make_config(tmp_path)
    prepare_training_data(ChannelSelectionAdapter(), Samples(), None, cfg, acc)
    cfg.save_pretrained(tmp_path)
    expected = read_mapping(tmp_path)
    restored = TrainPipelineConfig.from_pretrained(tmp_path, cli_args=[])
    restored.resume = True
    restored.training_adapter.config_path.unlink()
    restored.dataset.root = "/a/different/server"
    prepare_training_data(ChannelSelectionAdapter(), Samples(), None, restored, acc)
    assert restored._training_adapter_artifacts[FILENAME] == expected
    restored.training_adapter.config_path.write_text(
        yaml.safe_dump({"state_names": "all", "action_names": "all"})
    )
    with pytest.raises(ValueError, match="selection changed"):
        prepare_training_data(ChannelSelectionAdapter(), Samples(), None, restored, acc)
    restored.training_adapter.config_path.unlink()
    source = Samples()
    source.meta.features["action"]["names"].reverse()
    with pytest.raises(ValueError, match="Source channel"):
        prepare_training_data(ChannelSelectionAdapter(), source, None, restored, acc)


@pytest.mark.parametrize("option", ["streaming", "env", "reward", "groot_relative", "rename", "weighting"])
def test_unsupported_paths_are_explicit(tmp_path, option):
    cfg, _ = make_config(tmp_path)
    if option == "streaming":
        cfg.dataset.streaming = True
    elif option == "env":
        cfg.env = object()
    elif option == "reward":
        cfg.reward_model = object()
    elif option == "groot_relative":
        cfg.policy = SimpleNamespace(type="groot", use_relative_actions=True)
    elif option == "rename":
        cfg.rename_map = {"observation.state": "another.state"}
    else:
        cfg.sample_weighting = object()
    with pytest.raises(ValueError, match="does not support"):
        ChannelSelectionAdapter().validate_config(cfg)


def test_wall_x_real_processors_use_selected_16_channels(tmp_path):
    from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
    from lerobot.policies.wall_x.processor_wall_x import make_wall_x_pre_post_processors

    cfg, acc = make_config(tmp_path)
    cfg.policy = WallXConfig(device="cpu")
    adapter = ChannelSelectionAdapter()
    selected, _ = prepare_training_data(adapter, Samples(), None, cfg, acc)
    pre, post = make_wall_x_pre_post_processors(cfg.policy, dataset_stats=selected.meta.stats)
    adapter.validate_policy(cfg, pre, post)
    batch = next(iter(DataLoader(selected, batch_size=2)))
    result = pre(batch)
    assert result["observation.state"].shape[-1] == 16
    assert result["action"].shape[-1] == 16
    assert post(torch.zeros(2, 4, 16)).shape == (2, 4, 16)


def test_relative_prefix_mismatch(tmp_path):
    from lerobot.policies.pi0.configuration_pi0 import PI0Config

    cfg, acc = make_config(tmp_path)
    cfg.policy = PI0Config(device="cpu", use_relative_actions=True)
    with pytest.raises(ValueError, match="prefix mismatch"):
        prepare_training_data(ChannelSelectionAdapter(), Samples(), None, cfg, acc)


@pytest.mark.parametrize("problem", ["missing", "wrong_dim", "relative_names", "relative_pair"])
def test_real_processor_validation_rejects_incompatible_boundary(tmp_path, problem):
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    cfg, acc = make_config(tmp_path)
    adapter = ChannelSelectionAdapter()
    selected, _ = prepare_training_data(adapter, Samples(), None, cfg, acc)
    pre, post = make_act_pre_post_processors(cfg.policy, dataset_stats=selected.meta.stats)
    if problem in ("missing", "wrong_dim"):
        normalizer = next(step for step in pre.steps if type(step).__name__ == "NormalizerProcessorStep")
        if problem == "missing":
            normalizer.stats["action"].pop("std")
        else:
            normalizer.stats["action"]["std"] = np.ones(22)
    else:
        adapter.mapping["action_names"] = list(adapter.mapping["state_names"])
        names = ["wrong"] * 16 if problem == "relative_names" else adapter.mapping["action_names"]
        pre.steps.append(RelativeActionsProcessorStep(enabled=True, action_names=names))
    with pytest.raises(ValueError):
        adapter.validate_policy(cfg, pre, post)


def test_legacy_label_only_option_requires_explicit_migration(tmp_path):
    cfg, _ = make_config(tmp_path)
    cfg.training_adapter = None
    cfg.save_pretrained(tmp_path)
    path = tmp_path / "train_config.json"
    payload = json.loads(path.read_text())
    payload["io_mapping_path"] = None
    path.write_text(json.dumps(payload))
    assert TrainPipelineConfig.from_pretrained(tmp_path, cli_args=[]).training_adapter is None
    payload["io_mapping_path"] = "/old/metadata.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only labelled"):
        TrainPipelineConfig.from_pretrained(tmp_path, cli_args=[])


def test_export_tool_draft_and_no_overwrite(tmp_path, monkeypatch, capsys):
    from cyclo_lerobot_io.cli import main

    info = tmp_path / "info.json"
    info.write_text(json.dumps({"features": Samples().meta.features}))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"shape": [22]}},
                "output_features": {"action": {"shape": [22]}},
            }
        )
    )
    args = ["cyclo-io-mapping", "--checkpoint", str(tmp_path), "--dataset-info", str(info)]
    monkeypatch.setattr("sys.argv", args)
    main()
    assert "draft only" in capsys.readouterr().out
    assert not (tmp_path / FILENAME).exists()
    monkeypatch.setattr("sys.argv", args + ["--write"])
    main()
    assert read_mapping(tmp_path)["state_names"] == [f"j{i}" for i in range(22)]
    with pytest.raises(FileExistsError):
        main()
