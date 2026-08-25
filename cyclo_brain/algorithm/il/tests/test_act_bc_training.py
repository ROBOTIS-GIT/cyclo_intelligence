from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cyclo_brain.algorithm.il.act_bc.dataset import LeRobotDatasetDependencies, RootSelection
from cyclo_brain.algorithm.il.act_bc.training import (
    ACTBCTrainingConfig,
    OfficialTrainingDependencies,
    run_training,
)

from cyclo_brain.algorithm.il.tests.test_act_bc_dataset import (
    _FakeMeta,
    _episode_row,
    _features,
)


class _ConfigObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _PolicyConfig(_ConfigObject):
    def get_optimizer_preset(self):
        return SimpleNamespace(grad_clip_norm=10.0)

    def get_scheduler_preset(self):
        return None


class _TrainConfig(_ConfigObject):
    pass


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, _repo_id, **kwargs):
        self.meta = _FakeMeta(episodes=(0,))
        self.episodes = kwargs["episodes"]
        self.delta_timestamps = kwargs["delta_timestamps"]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        sample = {
            "observation.state": torch.zeros(2),
            "action": torch.ones(30, 2),
            "action_is_pad": torch.zeros(30, dtype=torch.bool),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(index),
        }
        for camera in self.meta.camera_keys:
            sample[camera] = torch.full((3, 8, 8), index, dtype=torch.uint8)
        return sample


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        valid = (~batch["action_is_pad"]).unsqueeze(-1)
        absolute_error = torch.abs(batch["action"] - self.value)
        l1 = (absolute_error * valid).sum() / (valid.sum() * batch["action"].shape[-1])
        kld = self.value.square()
        return l1 + 10.0 * kld, {"l1_loss": l1.item(), "kld_loss": kld.item()}


class _Processor:
    def __call__(self, batch):
        return batch


def _cycle(iterable):
    while True:
        yield from iterable


def _fake_dependencies() -> OfficialTrainingDependencies:
    metadata = _FakeMeta(episodes=(0,))

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
            json.dumps({"step": step}), encoding="utf-8"
        )

    def update_last(checkpoint_dir):
        last = checkpoint_dir.parent / "last"
        if last.is_symlink():
            last.unlink()
        last.symlink_to(checkpoint_dir.name)

    return OfficialTrainingDependencies(
        dataset=LeRobotDatasetDependencies(
            metadata_cls=lambda *_args, **_kwargs: metadata,
            dataset_cls=_TinyDataset,
            resolve_delta_timestamps=lambda policy, meta: {
                "action": [index / meta.fps for index in range(policy.chunk_size)]
            },
            aggregate_stats=lambda values: values[0],
            load_episode_with_stats=lambda *_args: _episode_row(),
        ),
        act_config_cls=_PolicyConfig,
        dataset_config_cls=_ConfigObject,
        train_config_cls=_TrainConfig,
        make_policy=lambda **_kwargs: _TinyPolicy(),
        make_pre_post_processors=lambda **_kwargs: (_Processor(), _Processor()),
        make_optimizer_and_scheduler=lambda _config, policy: (
            torch.optim.SGD(policy.parameters(), lr=0.01),
            None,
        ),
        get_step_checkpoint_dir=lambda output, total, step: output
        / "checkpoints"
        / f"{step:0{max(6, len(str(total)))}d}",
        save_checkpoint=save_checkpoint,
        update_last_checkpoint=update_last,
        cycle=_cycle,
        apply_trainable_groups=lambda _policy, groups: tuple(groups),
    )


def _layout(root: Path):
    (root / "meta" / "episodes").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": _features()}),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")


class ACTBCTrainingTest(unittest.TestCase):
    def test_complete_run_writes_progress_and_actual_final_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            dataset_root = temporary / "dataset"
            _layout(dataset_root)
            config = ACTBCTrainingConfig(
                selections=(RootSelection(dataset_root, (0,)),),
                output_dir=temporary / "output",
                steps=3,
                batch_size=2,
                save_freq=2,
                progress_interval=1,
                num_workers=0,
                device="cpu",
            )
            progress = []
            result = run_training(
                config,
                dependencies=_fake_dependencies(),
                progress_callback=progress.append,
            )
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.step, 3)
            self.assertEqual(result.percentage, 100.0)
            expected = config.output_dir / "checkpoints" / "000003"
            self.assertEqual(Path(result.model_path), (expected / "pretrained_model").resolve())
            self.assertEqual(
                Path(result.checkpoint_path), (expected / "training_state").resolve()
            )
            self.assertTrue((config.output_dir / "checkpoints" / "last").is_symlink())
            self.assertTrue((config.output_dir / "manifest.json").is_file())
            self.assertTrue((config.output_dir / "progress.json").is_file())
            persisted = json.loads((config.output_dir / "result.json").read_text())
            self.assertEqual(persisted["model_path"], result.model_path)
            self.assertTrue(all(item.loss is None or np.isfinite(item.loss) for item in progress))
            self.assertEqual(progress[-1].status, "complete")

    def test_cooperative_stop_saves_state_but_does_not_publish_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            dataset_root = temporary / "dataset"
            _layout(dataset_root)
            config = ACTBCTrainingConfig(
                selections=(RootSelection(dataset_root, (0,)),),
                output_dir=temporary / "output",
                steps=5,
                batch_size=2,
                save_freq=5,
                progress_interval=1,
                num_workers=0,
                device="cpu",
            )
            stop = {"requested": False}

            def progress(item):
                if item.step == 1:
                    stop["requested"] = True

            result = run_training(
                config,
                dependencies=_fake_dependencies(),
                should_stop=lambda: stop["requested"],
                progress_callback=progress,
            )
            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.step, 1)
            self.assertIsNone(result.model_path)
            self.assertEqual(
                Path(result.checkpoint_path),
                (config.output_dir / "checkpoints" / "000001" / "training_state").resolve(),
            )
            persisted = json.loads((config.output_dir / "result.json").read_text())
            self.assertEqual(persisted["status"], "stopped")

    def test_config_fixes_chunk_size_and_rejects_output_dataset_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            selection = RootSelection(root, (0,))
            with self.assertRaisesRegex(ValueError, "chunk_size=30"):
                ACTBCTrainingConfig(
                    selections=(selection,),
                    output_dir=Path(temporary) / "output",
                    steps=1,
                    batch_size=1,
                    save_freq=1,
                    chunk_size=20,
                )
            with self.assertRaisesRegex(ValueError, "must not contain"):
                ACTBCTrainingConfig(
                    selections=(selection,),
                    output_dir=root / "run",
                    steps=1,
                    batch_size=1,
                    save_freq=1,
                )


if __name__ == "__main__":
    unittest.main()
