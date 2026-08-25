"""Numerical tests for the model-independent Flow-SDE PPO equations."""

import math
import unittest

import torch

from cyclo_brain.algorithm.rl.flow_sde_ppo import (
    FlowSDEPPOConfig,
    clipped_value_loss,
    flow_sde_transition_stats,
    gaussian_log_prob,
    generalized_advantage_estimate,
    ppo_clipped_actor_loss,
)


class FlowSDEPPOFunctionalTest(unittest.TestCase):
    def test_four_step_noise_schedule_matches_flow_sde_reference(self):
        current = torch.zeros(4, 1, 1)
        velocity = torch.ones_like(current)
        indices = torch.arange(4, dtype=torch.long)

        _mean, std = flow_sde_transition_stats(
            current,
            velocity,
            indices,
            num_steps=4,
            noise_level=0.5,
        )

        expected = torch.tensor([0.5, 0.4330127, 0.25, 0.1443376])
        torch.testing.assert_close(std[:, 0, 0], expected, rtol=1.0e-6, atol=1.0e-6)
        self.assertTrue(bool(torch.isfinite(std).all()))

    def test_unselected_steps_reduce_exactly_to_euler_ode(self):
        current = torch.tensor([[[1.0, -2.0]], [[0.5, 3.0]]])
        velocity = torch.tensor([[[0.8, -0.4]], [[-1.2, 2.0]]])
        indices = torch.tensor([0, 3], dtype=torch.long)
        mean, std = flow_sde_transition_stats(
            current,
            velocity,
            indices,
            num_steps=4,
            noise_level=0.5,
            stochastic_mask=torch.zeros(2, dtype=torch.bool),
        )

        torch.testing.assert_close(mean, current + 0.25 * velocity, rtol=0.0, atol=0.0)
        torch.testing.assert_close(std, torch.zeros_like(std), rtol=0.0, atol=0.0)

    def test_gaussian_log_probability_matches_closed_form(self):
        sample = torch.tensor([[[1.0, -1.0]]])
        mean = torch.tensor([[[0.5, -0.25]]])
        std = torch.tensor([[[0.25, 0.5]]])
        actual = gaussian_log_prob(sample, mean, std)
        expected = -torch.log(std) - 0.5 * math.log(2.0 * math.pi) - 0.5 * (
            (sample - mean) / std
        ).square()
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            gaussian_log_prob(sample, mean, torch.zeros_like(std))

    def test_unchanged_log_probability_has_unit_ratio_and_zero_loss_for_zero_advantage(self):
        old = torch.tensor([[[-1.0, -2.0], [-3.0, -4.0]]], dtype=torch.float32)
        mask = torch.ones_like(old, dtype=torch.bool)
        loss, metrics = ppo_clipped_actor_loss(
            old.clone(),
            old,
            torch.zeros(1),
            mask,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
        )
        torch.testing.assert_close(loss, torch.tensor(0.0))
        torch.testing.assert_close(metrics["ratio"], torch.tensor(1.0))
        torch.testing.assert_close(metrics["approx_kl"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["clip_fraction"], torch.tensor(0.0))

    def test_invalid_action_coordinates_do_not_change_actor_loss(self):
        old = torch.zeros(1, 2, 2, dtype=torch.float32)
        mask = torch.tensor([[[True, False], [True, False]]])
        baseline = torch.tensor([[[0.1, 0.0], [-0.1, 0.0]]], requires_grad=True)
        modified = baseline.detach().clone()
        modified[..., 1] = 100.0
        loss_a, _ = ppo_clipped_actor_loss(
            baseline,
            old,
            torch.ones(1),
            mask,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
        )
        loss_b, _ = ppo_clipped_actor_loss(
            modified,
            old,
            torch.ones(1),
            mask,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
        )
        torch.testing.assert_close(loss_a.detach(), loss_b)
        loss_a.backward()
        self.assertEqual(baseline.grad[..., 1].abs().sum().item(), 0.0)
        self.assertGreater(baseline.grad[..., 0].abs().sum().item(), 0.0)

    def test_gae_stops_at_true_terminal_and_bootstraps_otherwise(self):
        rewards = torch.tensor([[1.0], [2.0]])
        values = torch.tensor([[0.5], [0.25], [10.0]])
        terminated = torch.tensor([[False], [True]])
        advantages, returns = generalized_advantage_estimate(
            rewards,
            values,
            terminated,
            discount=0.9,
            gae_lambda=0.8,
        )
        last_advantage = 2.0 - 0.25
        first_delta = 1.0 + 0.9 * 0.25 - 0.5
        expected = torch.tensor([[first_delta + 0.9 * 0.8 * last_advantage], [last_advantage]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(returns, expected + values[:-1])

    def test_value_loss_uses_larger_clipped_or_unclipped_error(self):
        values = torch.tensor([1.0, 0.0])
        old_values = torch.tensor([0.0, 0.0])
        returns = torch.tensor([1.0, 1.0])
        actual = clipped_value_loss(values, old_values, returns, value_clip=0.2)
        expected = 0.5 * torch.tensor([(0.2 - 1.0) ** 2, (0.0 - 1.0) ** 2]).mean()
        torch.testing.assert_close(actual, expected)

    def test_config_rejects_non_stochastic_or_one_step_contracts(self):
        config = FlowSDEPPOConfig()
        self.assertEqual(config.num_denoising_steps, 4)
        self.assertEqual(config.noise_level, 0.5)
        with self.assertRaises(ValueError):
            FlowSDEPPOConfig(num_denoising_steps=1)
        with self.assertRaises(ValueError):
            FlowSDEPPOConfig(noise_level=0.0)


if __name__ == "__main__":
    unittest.main()
