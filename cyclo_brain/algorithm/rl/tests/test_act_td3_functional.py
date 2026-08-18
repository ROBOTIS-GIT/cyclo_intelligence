"""Exact numerical contracts for chunk-SMDP ACT-TD3."""

from __future__ import annotations

import unittest

import torch

from cyclo_brain.algorithm.rl.act_td3 import (
    ACTTD3Batch,
    actor_update_is_due,
    build_smdp_returns,
    masked_deterministic_bc_l1,
    q_weight_for_actor_update,
    smooth_target_action_chunks,
)
from cyclo_brain.algorithm.rl.td3 import bellman_target


class ACTTD3FunctionalTest(unittest.TestCase):
    def test_duration_aware_chunk_return_and_bootstrap(self) -> None:
        rewards = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        durations = torch.tensor([[0.05, 0.15, 0.0], [0.1, 0.2, 0.1]])

        result = build_smdp_returns(
            rewards,
            mask,
            durations,
            torch.tensor([True, False]),
            discount=0.9,
            discount_reference_hz=10.0,
        )

        torch.testing.assert_close(
            result.discounted_returns,
            torch.tensor(
                [
                    [1.0 + 0.9**0.5 * 2.0],
                    [3.0 + 0.9 * 4.0 + 0.9**3 * 5.0],
                ]
            ),
        )
        torch.testing.assert_close(
            result.bootstrap_discounts,
            torch.tensor([[0.9**2], [0.0]]),
        )

    def test_one_step_smdp_reduces_to_standard_td3_target(self) -> None:
        reward = torch.tensor([[1.25]])
        smdp = build_smdp_returns(
            reward,
            torch.tensor([[True]]),
            torch.tensor([[0.1]]),
            torch.tensor([True]),
            discount=0.9,
            discount_reference_hz=10.0,
        )
        q1 = torch.tensor([[4.0]])
        q2 = torch.tensor([[3.0]])
        chunk_target = smdp.discounted_returns + smdp.bootstrap_discounts * torch.minimum(
            q1,
            q2,
        )
        scalar_target = bellman_target(
            reward,
            torch.tensor([[False]]),
            q1,
            q2,
            discount=0.9,
        )
        torch.testing.assert_close(chunk_target, scalar_target, rtol=0.0, atol=0.0)

    def test_target_smoothing_masks_binary_and_fixed_dimensions(self) -> None:
        actions = torch.zeros(1, 2, 3)
        noise = torch.tensor([[[3.0, 3.0, -3.0], [-3.0, -3.0, 3.0]]])
        result = smooth_target_action_chunks(
            actions,
            noise,
            torch.tensor([True, False, False]),
            noise_standard_deviation=0.2,
            noise_clip=0.5,
        )
        torch.testing.assert_close(
            result,
            torch.tensor([[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]]),
        )

    def test_warmup_gate_ramp_and_masked_anchor(self) -> None:
        self.assertFalse(
            actor_update_is_due(
                4,
                critic_warmup_updates=4,
                policy_update_period=2,
            )
        )
        self.assertFalse(
            actor_update_is_due(
                5,
                critic_warmup_updates=4,
                policy_update_period=2,
            )
        )
        self.assertTrue(
            actor_update_is_due(
                6,
                critic_warmup_updates=4,
                policy_update_period=2,
            )
        )
        self.assertEqual(
            q_weight_for_actor_update(1, maximum=0.25, ramp_updates=1000),
            0.0,
        )
        self.assertEqual(
            q_weight_for_actor_update(501, maximum=0.25, ramp_updates=1000),
            0.125,
        )
        policy = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
        behavior = torch.tensor([[[0.0, 1.0], [-100.0, -100.0]]])
        loss = masked_deterministic_bc_l1(
            policy,
            behavior,
            torch.tensor([[True, False]]),
        )
        torch.testing.assert_close(loss, torch.tensor(1.5))

    def test_batch_rejects_non_prefix_and_ambiguous_bootstrap(self) -> None:
        common = {
            "observations": {"observation.environment_state": torch.zeros(1, 2)},
            "next_observations": {
                "observation.environment_state": torch.zeros(1, 2)
            },
            "behavior_action_chunks": torch.zeros(1, 3, 2),
            "rewards": torch.zeros(1, 3),
            "step_durations_s": torch.tensor([[0.1, 0.0, 0.0]]),
            "terminated": torch.tensor([False]),
            "truncated": torch.tensor([True]),
            "next_observation_valid": torch.tensor([False]),
            "bootstrap_allowed": torch.tensor([False]),
        }
        with self.assertRaisesRegex(ValueError, "exact non-empty prefix"):
            ACTTD3Batch(
                **common,
                executed_mask=torch.tensor([[True, False, True]]),
            )
        with self.assertRaisesRegex(ValueError, "valid next observation"):
            ACTTD3Batch(
                **{
                    **common,
                    "bootstrap_allowed": torch.tensor([True]),
                },
                executed_mask=torch.tensor([[True, False, False]]),
            )
        with self.assertRaisesRegex(ValueError, "termination or truncation"):
            ACTTD3Batch(
                **{
                    **common,
                    "terminated": torch.tensor([False]),
                    "truncated": torch.tensor([False]),
                    "next_observation_valid": torch.tensor([True]),
                    "bootstrap_allowed": torch.tensor([True]),
                },
                executed_mask=torch.tensor([[True, False, False]]),
            )


if __name__ == "__main__":
    unittest.main()
