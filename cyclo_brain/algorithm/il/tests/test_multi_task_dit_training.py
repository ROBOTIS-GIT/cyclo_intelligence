from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cyclo_brain.algorithm.il.act_bc.dataset import (
    LeRobotDatasetDependencies,
    RootSelection,
)
from cyclo_brain.algorithm.il.multi_task_dit.training import (
    MULTI_TASK_DIT_HORIZON,
    MultiTaskDiTILConfig,
    OfficialTrainingDependencies,
    run_training,
)
from cyclo_brain.model.multi_task_dit import (
    CYCLO_SG2_ACTION_NAMES,
    CYCLO_SG2_CAMERA_KEYS,
)


def _features() -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [22],
            "names": list(CYCLO_SG2_ACTION_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [22],
            "names": list(CYCLO_SG2_ACTION_NAMES),
        },
    }
    for camera in CYCLO_SG2_CAMERA_KEYS:
        features[camera] = {
            "dtype": "video",
            "shape": [3, 256, 256],
            "names": ["channels", "height", "width"],
        }
    return features


def _episode_row() -> dict:
    row = {}
    for feature, spec in _features().items():
        shape = (3, 1, 1) if spec["dtype"] == "video" else tuple(spec["shape"])
        for statistic in ("min", "max", "mean"):
            row[f"stats/{feature}/{statistic}"] = np.ones(shape)
        row[f"stats/{feature}/std"] = np.ones(shape)
        row[f"stats/{feature}/count"] = np.asarray([1])
    return row


class _Meta:
    def __init__(self):
        self.info = SimpleNamespace(codebase_version="v3.0")
        self.features = _features()
        self.fps = 15
        self.episodes = {"episode_index": [0, 1]}
        self.camera_keys = list(CYCLO_SG2_CAMERA_KEYS)
        self.stats = {}


class _Dataset(torch.utils.data.Dataset):
    constructed = []

    def __init__(self, _repo_id, **kwargs):
        type(self).constructed.append(kwargs)
        self.meta = _Meta()
        self.episodes = kwargs["episodes"]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        sample = {
            "observation.state": torch.zeros(1, 22),
            "action": torch.ones(MULTI_TASK_DIT_HORIZON, 22),
            "action_is_pad": torch.zeros(MULTI_TASK_DIT_HORIZON, dtype=torch.bool),
            "task": torch.tensor(0),
            "frame_index": torch.tensor(index),
        }
        for camera in CYCLO_SG2_CAMERA_KEYS:
            sample[camera] = torch.zeros(3, 256, 256, dtype=torch.uint8)
        return sample


class _Object:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _OptimizerConfig:
    def __init__(self):
        self.lr = 2e-5
        self.grad_clip_norm = 10.0

    def build(self, parameters):
        return torch.optim.SGD(parameters, lr=0.05)


class _PolicyConfig(_Object):
    def get_optimizer_preset(self):
        return _OptimizerConfig()

    def get_scheduler_preset(self):
        return None


class _TinyPolicy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.observation_encoder = torch.nn.Linear(1, 1)
        self.noise_predictor = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.noise_predictor.weight)

    def forward(self, batch):
        prediction = self.noise_predictor(
            torch.ones(1, 1, device=self.noise_predictor.weight.device)
        ).squeeze()
        target = batch["action"].float().mean()
        return (prediction - target).square(), None


class _Processor:
    def __call__(self, batch):
        return batch


def _cycle(iterable):
    while True:
        yield from iterable


def _fake_dependencies() -> OfficialTrainingDependencies:
    metadata = _Meta()

    def save_checkpoint(*, checkpoint_dir, step, **_kwargs):
        model = checkpoint_dir / "pretrained_model"
        state = checkpoint_dir / "training_state"
        model.mkdir(parents=True)
        state.mkdir()
        for name in (
            "config.json",
            "model.safetensors",
            "train_config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ):
            (model / name).write_text("{}", encoding="utf-8")
        (state / "training_step.json").write_text(
            json.dumps({"step": step}),
            encoding="utf-8",
        )

    def update_last(checkpoint_dir):
        last = checkpoint_dir.parent / "last"
        if last.is_symlink():
            last.unlink()
        last.symlink_to(checkpoint_dir.name)

    return OfficialTrainingDependencies(
        dataset=LeRobotDatasetDependencies(
            metadata_cls=lambda *_args, **_kwargs: metadata,
            dataset_cls=_Dataset,
            resolve_delta_timestamps=lambda policy, meta: {
                "action": [index / meta.fps for index in range(policy.horizon)]
            },
            aggregate_stats=lambda values: values[0],
            load_episode_with_stats=lambda *_args: _episode_row(),
        ),
        policy_config_cls=_PolicyConfig,
        policy_feature_cls=_Object,
        feature_type=SimpleNamespace(VISUAL="visual", STATE="state", ACTION="action"),
        normalization_mode=SimpleNamespace(
            MEAN_STD="mean_std",
            MIN_MAX="min_max",
        ),
        dataset_config_cls=_Object,
        train_config_cls=_Object,
        policy_cls=_TinyPolicy,
        make_pre_post_processors=lambda *_args, **_kwargs: (_Processor(), _Processor()),
        get_step_checkpoint_dir=lambda output, total, step: output
        / "checkpoints"
        / f"{step:0{max(6, len(str(total)))}d}",
        save_checkpoint=save_checkpoint,
        update_last_checkpoint=update_last,
        cycle=_cycle,
        resize_factory=lambda size: ("resize", size),
        imagenet_stats={
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.25]], [[0.25]], [[0.25]]],
        },
    )


def _layout(root: Path) -> None:
    (root / "meta" / "episodes").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": _features()}),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")


class MultiTaskDiTTrainingTest(unittest.TestCase):
    def setUp(self):
        _Dataset.constructed.clear()

    def test_unlabeled_complete_run_exports_supervisor_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            dataset_root = temporary / "showroom_test_3"
            _layout(dataset_root)
            config = MultiTaskDiTILConfig(
                selections=(RootSelection(dataset_root, (0, 1)),),
                output_dir=temporary / "output",
                steps=3,
                batch_size=2,
                save_freq=2,
                progress_interval=1,
                num_workers=0,
                device="cpu",
                task_instruction="pick up jelly",
            )
            progress = []
            result = run_training(
                config,
                dependencies=_fake_dependencies(),
                progress_callback=progress.append,
            )

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.step, 3)
            expected = config.output_dir / "checkpoints" / "000003"
            self.assertEqual(
                Path(result.model_path),
                (expected / "pretrained_model").resolve(),
            )
            self.assertEqual(
                Path(result.checkpoint_path),
                (expected / "training_state").resolve(),
            )
            self.assertEqual(_Dataset.constructed[0]["episodes"], [0, 1])
            self.assertEqual(
                _Dataset.constructed[0]["image_transforms"],
                ("resize", (256, 256)),
            )
            contract = json.loads(
                (expected / "pretrained_model" / "cyclo_training_contract.json").read_text()
            )
            self.assertEqual(contract["episode_indices"], [[0, 1]])
            self.assertEqual(contract["horizon"], 16)
            self.assertEqual(progress[-1].status, "complete")
            self.assertIsNone(progress[-1].l1_loss)
            self.assertEqual(progress[-1].flow_matching_loss, progress[-1].loss)

    def test_config_rejects_non_sg2_horizon_and_blank_instruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            selection = RootSelection(root, (0,))
            with self.assertRaisesRegex(ValueError, "chunk_size=16"):
                MultiTaskDiTILConfig(
                    selections=(selection,),
                    output_dir=Path(temporary) / "output",
                    steps=1,
                    batch_size=1,
                    save_freq=1,
                    chunk_size=30,
                )
            with self.assertRaisesRegex(ValueError, "task_instruction"):
                MultiTaskDiTILConfig(
                    selections=(selection,),
                    output_dir=Path(temporary) / "output",
                    steps=1,
                    batch_size=1,
                    save_freq=1,
                    task_instruction="  ",
                )


if __name__ == "__main__":
    unittest.main()
