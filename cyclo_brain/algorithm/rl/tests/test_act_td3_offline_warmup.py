"""Tests for the actor-frozen ACT-TD3 offline warm-up runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

import cyclo_brain.algorithm.rl.act_td3.offline_warmup as warmup_module
from cyclo_brain.algorithm.rl.act_td3 import (
    ACTTD3Config,
    ACTTD3CriticWarmupRunner,
    ACTTD3Learner,
    ACTTD3LeRobotCollator,
    FixedHorizonLeRobotACTTD3Dataset,
)
from cyclo_brain.algorithm.rl.tests.test_act_td3_lerobot_offline import (
    _FakeLeRobotDataset,
    _OffsetPreprocessor,
)
from cyclo_brain.model.act import (
    ACTTwinChunkCritic,
    create_act_model,
)


def _learner() -> ACTTD3Learner:
    torch.manual_seed(101)
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
        chunk_size=3,
        n_action_steps=3,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
        dropout=0.0,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    actor = create_act_model(config)
    critic = ACTTwinChunkCritic(
        actor.config,
        observation_feature_dim=8,
        action_feature_dim=8,
        hidden_dims=(16, 8),
    )
    algorithm = ACTTD3Config(
        discount=0.9,
        discount_reference_hz=2.0,
        target_update_rate=0.1,
        target_policy_noise=0.1,
        target_policy_noise_clip=0.2,
        policy_update_period=2,
        critic_warmup_updates=4,
        actor_learning_rate=1.0e-3,
        critic_learning_rate=1.0e-3,
        actor_gradient_clip_norm=10.0,
        critic_gradient_clip_norm=10.0,
        q_weight_max=0.25,
        q_weight_ramp_actor_updates=2,
    )
    return ACTTD3Learner(
        actor,
        critic,
        algorithm,
        random_seed=17,
    )


def _dataset(*, fps: int = 2) -> FixedHorizonLeRobotACTTD3Dataset:
    source = _FakeLeRobotDataset()
    source.fps = fps
    source.features[OBS_ENV_STATE] = {"dtype": "float32", "shape": [2]}
    for row in source._rows:
        row[OBS_ENV_STATE] = row[OBS_STATE].clone()
    return FixedHorizonLeRobotACTTD3Dataset(
        source,
        execution_horizon=3,
        observation_keys=(OBS_STATE, OBS_ENV_STATE),
    )


def _runner(
    checkpoint: Path,
    *,
    resume: bool = False,
    identity: str = "test-data-sha256",
    fps: int = 2,
) -> ACTTD3CriticWarmupRunner:
    return ACTTD3CriticWarmupRunner(
        _learner(),
        _dataset(fps=fps),
        ACTTD3LeRobotCollator(_OffsetPreprocessor()),
        batch_size=2,
        sampling_seed=19,
        training_data_identity=identity,
        checkpoint_path=checkpoint,
        checkpoint_interval=2,
        progress_interval=1,
        resume=resume,
    )


def _assert_tree_equal(test: unittest.TestCase, first, second) -> None:
    if isinstance(first, torch.Tensor):
        test.assertIsInstance(second, torch.Tensor)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    elif isinstance(first, dict):
        test.assertIsInstance(second, dict)
        test.assertEqual(set(first), set(second))
        for key in first:
            _assert_tree_equal(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertIsInstance(second, type(first))
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second, strict=True):
            _assert_tree_equal(test, left, right)
    else:
        test.assertEqual(first, second)


class ACTTD3CriticWarmupRunnerTest(unittest.TestCase):
    def test_exact_boundary_progress_and_frozen_actor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            runner = _runner(checkpoint)
            actor_before = {
                name: value.detach().clone()
                for name, value in runner.learner.actor.state_dict().items()
            }
            progress = []

            result = runner.run(progress_callback=progress.append)

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.completed_critic_updates, 4)
            self.assertEqual(result.total_critic_updates, 4)
            self.assertEqual(result.percentage, 100.0)
            self.assertEqual(result.durable_checkpoint_updates, 4)
            self.assertTrue(result.actor_exactly_unchanged)
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(progress[0].completed_critic_updates, 0)
            self.assertEqual(progress[0].percentage, 0.0)
            self.assertEqual(progress[-1], result)
            self.assertEqual(runner.learner.completed_actor_updates, 0)
            self.assertEqual(len(set(runner.last_sampled_indices)), 2)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            learner_contract = state["contract"]["learner"]
            self.assertNotIn("passthrough_mask", learner_contract)
            self.assertEqual(
                learner_contract["target_policy_smoothing"],
                "clipped_noise_all_dimensions_no_action_clamp",
            )
            self.assertIs(learner_contract["action_clamp"], False)
            for name, value in runner.learner.actor.state_dict().items():
                torch.testing.assert_close(
                    value,
                    actor_before[name],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_split_resume_matches_continuous_updates_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = _runner(root / "continuous.pt")
            continuous.run()

            first_segment = _runner(root / "split.pt")
            partial = first_segment.run(max_critic_updates=2)
            self.assertEqual(partial.status, "segment_complete")
            self.assertEqual(partial.percentage, 50.0)
            self.assertEqual(
                first_segment.learner.config.critic_warmup_updates,
                4,
            )
            resumed = _runner(root / "split.pt", resume=True)
            resumed.run()

            _assert_tree_equal(
                self,
                continuous.learner.state_dict(),
                resumed.learner.state_dict(),
            )
            continuous_state = torch.load(
                root / "continuous.pt",
                map_location="cpu",
                weights_only=True,
            )
            resumed_state = torch.load(
                root / "split.pt",
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(
                continuous_state["sampler_state"],
                resumed_state["sampler_state"],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                continuous_state["last_sampled_indices"],
                resumed_state["last_sampled_indices"],
            )

    def test_resume_rejects_data_contract_and_fps_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            _runner(checkpoint).run(max_critic_updates=1)

            with self.assertRaisesRegex(ValueError, "contract disagrees"):
                _runner(checkpoint, resume=True, identity="different-data")
            with self.assertRaisesRegex(ValueError, "dataset fps"):
                _runner(Path(directory) / "other.pt", fps=3)

    def test_failed_atomic_save_preserves_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            runner = _runner(checkpoint)
            runner.run(max_critic_updates=1)
            previous = checkpoint.read_bytes()

            with mock.patch.object(
                warmup_module.torch,
                "save",
                side_effect=OSError("simulated write failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    runner.run(max_critic_updates=2)

            self.assertEqual(checkpoint.read_bytes(), previous)
            self.assertFalse(any(checkpoint.parent.glob(".latest.pt.*.tmp")))

    def test_corrupt_checkpoint_and_actor_mutation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "corrupt.pt"
            corrupt.write_bytes(b"not a torch checkpoint")
            with self.assertRaisesRegex(ValueError, "cannot be read"):
                _runner(corrupt, resume=True)

            runner = _runner(root / "mutated.pt")
            with torch.no_grad():
                next(runner.learner.actor_target.parameters()).add_(1.0)
            progress = []
            with self.assertRaisesRegex(RuntimeError, "actor tensors changed"):
                runner.run(
                    max_critic_updates=1,
                    progress_callback=progress.append,
                )
            self.assertEqual(runner.learner.completed_critic_updates, 0)
            self.assertEqual(progress, [])

    def test_nonzero_learner_requires_runner_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _runner(root / "first.pt")
            first.run(max_critic_updates=1)

            with self.assertRaisesRegex(ValueError, "requires a fresh learner"):
                ACTTD3CriticWarmupRunner(
                    first.learner,
                    _dataset(),
                    ACTTD3LeRobotCollator(_OffsetPreprocessor()),
                    batch_size=2,
                    sampling_seed=19,
                    training_data_identity="test-data-sha256",
                    checkpoint_path=root / "invalid.pt",
                )

    def test_immediate_stop_saves_a_resumable_zero_step_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stopped.pt"
            runner = _runner(checkpoint)

            stopped = runner.run(should_stop=lambda: True)

            self.assertEqual(stopped.status, "stopped")
            self.assertEqual(stopped.completed_critic_updates, 0)
            self.assertTrue(checkpoint.is_file())
            resumed = _runner(checkpoint, resume=True)
            self.assertEqual(resumed.learner.completed_critic_updates, 0)
            self.assertEqual(
                resumed.run(max_critic_updates=1).completed_critic_updates,
                1,
            )


if __name__ == "__main__":
    unittest.main()
