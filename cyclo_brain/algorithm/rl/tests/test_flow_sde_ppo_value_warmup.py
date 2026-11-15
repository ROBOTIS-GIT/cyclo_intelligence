"""Focused contracts for offline MultiTaskDiT PPO value warm-up."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup import (
    VALUE_WARMUP_FORMAT,
    EpisodeBalancedChunkBoundaryDataset,
    MultiTaskDiTValueWarmupRunner,
    ValueWarmupConfig,
    module_sha256,
)
from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup_cli import build_parser


class _Table:
    def __init__(self, columns):
        self.columns = columns
        self.length = len(next(iter(columns.values())))

    def __getitem__(self, key):
        return self.columns[key]

    def __len__(self):
        return self.length


class _Dataset:
    def __init__(self, episodes):
        episode_indices = []
        frame_indices = []
        successes = []
        self.items = []
        for episode_index, (length, successful) in enumerate(episodes):
            for frame in range(length):
                episode_indices.append(episode_index)
                frame_indices.append(frame)
                successes.append(successful)
                state = torch.tensor(
                    [1.0 if successful else 0.0, frame / max(1, length - 1)],
                    dtype=torch.float32,
                )
                self.items.append(
                    {
                        "observation.state": state,
                        "observation.images.rgb.cam_left_wrist": torch.zeros(3, 4, 4),
                        "observation.images.rgb.cam_left_head": torch.zeros(3, 4, 4),
                        "observation.images.rgb.cam_right_wrist": torch.zeros(3, 4, 4),
                    }
                )
        self.hf_dataset = _Table(
            {
                "episode_index": episode_indices,
                "frame_index": frame_indices,
                "episode_success": successes,
            }
        )
        self.features = {
            "episode_index": {},
            "frame_index": {},
            "episode_success": {},
            "observation.state": {},
            "observation.images.rgb.cam_left_wrist": {},
            "observation.images.rgb.cam_left_head": {},
            "observation.images.rgb.cam_right_wrist": {},
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.observation_encoder = nn.Linear(2, 2)
        self.noise_predictor = nn.Linear(2, 2)


OBSERVATION_KEYS = (
    "observation.state",
    "observation.images.rgb.cam_left_wrist",
    "observation.images.rgb.cam_left_head",
    "observation.images.rgb.cam_right_wrist",
)


def _dataset():
    return EpisodeBalancedChunkBoundaryDataset(
        (_Dataset(((5, True), (3, False))),),
        observation_keys=OBSERVATION_KEYS,
        n_action_steps=2,
        gamma=0.5,
    )


class FlowSDEValueWarmupTest(unittest.TestCase):
    def test_chunk_boundaries_returns_and_episode_balanced_sampling(self):
        dataset = _dataset()
        self.assertEqual(
            [
                (
                    record.episode_index,
                    record.start_frame_index,
                    record.successful,
                    record.target_return,
                )
                for record in dataset.records
            ],
            [
                (0, 0, True, 0.25),
                (0, 2, True, 0.5),
                (0, 4, True, 1.0),
                (1, 0, False, 0.0),
                (1, 2, False, 0.0),
            ],
        )
        generator = torch.Generator().manual_seed(5)
        indices, cursor = dataset.sample_indices(
            generator=generator, batch_size=9, sampling_cursor=0
        )
        self.assertEqual(cursor, 9)
        self.assertEqual(
            [dataset.records[index].successful for index in indices],
            [True, False, True, False, True, False, True, False, True],
        )

    def test_rejects_non_boolean_episode_labels(self):
        source = _Dataset(((2, True), (2, False)))
        source.hf_dataset.columns["episode_success"][0] = 1
        with self.assertRaisesRegex(TypeError, "explicit boolean"):
            EpisodeBalancedChunkBoundaryDataset(
                (source,),
                observation_keys=OBSERVATION_KEYS,
                n_action_steps=2,
                gamma=0.99,
            )

    def test_only_value_head_updates_and_stop_checkpoint_is_atomic(self):
        torch.manual_seed(3)
        policy = _TinyPolicy()
        policy_hash = module_sha256(policy)
        value_head = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))
        value_before = {
            name: tensor.detach().clone() for name, tensor in value_head.state_dict().items()
        }

        def encode(observations, _task):
            return observations["observation.state"]

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "training_state" / "value_warmup.pt"
            config = ValueWarmupConfig(
                steps=20,
                batch_size=4,
                value_lr=5.0e-2,
                gamma=0.5,
                task_instruction="pick up the jelly bag",
                checkpoint_interval=3,
            )
            runner = MultiTaskDiTValueWarmupRunner(
                policy,
                value_head,
                _dataset(),
                encode,
                config=config,
                checkpoint_path=checkpoint,
            )
            progress = []
            result = runner.run(
                progress=progress.append,
                should_stop=lambda: runner.completed_steps >= 5,
            )

            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.completed_steps, 5)
            self.assertEqual(module_sha256(policy), policy_hash)
            self.assertFalse(any(p.requires_grad for p in policy.observation_encoder.parameters()))
            self.assertFalse(any(p.requires_grad for p in policy.noise_predictor.parameters()))
            self.assertFalse(any(p.requires_grad for p in policy.parameters()))
            self.assertTrue(
                any(
                    not torch.equal(value_before[name], tensor)
                    for name, tensor in value_head.state_dict().items()
                )
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(payload["format"], VALUE_WARMUP_FORMAT)
            self.assertEqual(payload["status"], "stopped")
            self.assertEqual(payload["completed_steps"], 5)
            self.assertEqual(payload["policy_sha256_before"], payload["policy_sha256_after"])
            self.assertTrue(payload["value_optimizer"]["state"])
            self.assertFalse(list(checkpoint.parent.glob("*.tmp")))
            self.assertEqual(progress[-1].status, "stopped")
            self.assertEqual(progress[-1].step, 5)

    def test_cli_repeatable_roots_and_ui_defaults(self):
        args = build_parser().parse_args(
            [
                "--base-checkpoint",
                "/model",
                "--dataset-root",
                "/data/epoch-0",
                "--dataset-root",
                "/data/epoch-1",
                "--output-dir",
                "/output",
            ]
        )
        self.assertEqual(args.dataset_root, [Path("/data/epoch-0"), Path("/data/epoch-1")])
        self.assertEqual(args.steps, 2000)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.value_lr, 1.0e-4)
        self.assertEqual(args.gamma, 0.99)
        self.assertEqual(args.task_instruction, "pick up the jelly bag")
        self.assertEqual(args.seed, 17)


if __name__ == "__main__":
    unittest.main()
