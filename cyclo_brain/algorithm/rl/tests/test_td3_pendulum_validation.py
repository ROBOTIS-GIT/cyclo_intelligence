"""Boundary checks for the standard TD3 Pendulum validation harness."""

from __future__ import annotations

import unittest

import torch

from cyclo_brain.algorithm.rl.td3.validate_pendulum import (
    PendulumValidationConfig,
    _reference_action_scale,
    td3_batch_from_lerobot_sample,
)


class TD3PendulumValidationTest(unittest.TestCase):
    def test_replay_adapter_preserves_only_mdp_terminal_mask(self) -> None:
        sample = {
            "state": {"observation.state": torch.zeros(2, 3)},
            "action": torch.zeros(2, 1),
            "reward": torch.tensor([1.0, 2.0]),
            "next_state": {"observation.state": torch.ones(2, 3)},
            "done": torch.tensor([0.0, 1.0]),
            # A truncation is intentionally not folded into ``terminated``.
            "truncated": torch.tensor([1.0, 0.0]),
        }

        batch = td3_batch_from_lerobot_sample(sample)

        self.assertEqual(batch.rewards.shape, (2, 1))
        self.assertEqual(batch.terminated.dtype, torch.bool)
        self.assertTrue(
            torch.equal(batch.terminated, torch.tensor([[False], [True]]))
        )

    def test_validation_config_rejects_a_run_without_updates(self) -> None:
        with self.assertRaises(ValueError):
            PendulumValidationConfig(training_steps=10, random_action_steps=10)

    def test_validation_config_rejects_replay_smaller_than_batch(self) -> None:
        with self.assertRaises(ValueError):
            PendulumValidationConfig(replay_capacity=32, batch_size=64)

    def test_reference_noise_scale_requires_sfujim_action_contract(self) -> None:
        self.assertEqual(
            _reference_action_scale(
                torch.tensor([-2.0]).numpy(),
                torch.tensor([2.0]).numpy(),
            ),
            2.0,
        )
        with self.assertRaises(ValueError):
            _reference_action_scale(
                torch.tensor([-1.0, -2.0]).numpy(),
                torch.tensor([1.0, 2.0]).numpy(),
            )


if __name__ == "__main__":
    unittest.main()
