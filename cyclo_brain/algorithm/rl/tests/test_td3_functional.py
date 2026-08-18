"""Numerical checks against the equations used by the official sfujim/TD3."""

import copy
import unittest

import torch
from torch import nn

from cyclo_brain.algorithm.rl.td3 import (
    TD3Config,
    bellman_target,
    clipped_target_action,
    critic_loss,
    deterministic_actor_loss,
    policy_update_is_due,
    polyak_update_,
)


class _ModuleWithBuffer(nn.Module):
    def __init__(self, weight: float, buffer: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([weight]))
        self.register_buffer("running", torch.tensor([buffer]))


class TD3FunctionalTest(unittest.TestCase):
    def test_target_smoothing_matches_reference_equations_with_vector_bounds(self):
        target_action = torch.tensor([[0.9, -1.8, 0.1], [-0.9, 1.8, -0.1]])
        noise = torch.tensor([[0.4, -0.7, 0.1], [-0.4, 0.7, -0.1]])
        low = torch.tensor([-1.0, -2.0, -0.15])
        high = torch.tensor([1.0, 2.0, 0.15])

        reference_noise = noise.clamp(-0.2, 0.2)
        reference = torch.maximum(
            torch.minimum(target_action + reference_noise, high),
            low,
        )
        actual = clipped_target_action(
            target_action,
            noise,
            noise_clip=0.2,
            action_low=low,
            action_high=high,
        )

        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
        self.assertTrue(bool((actual >= low).all()))
        self.assertTrue(bool((actual <= high).all()))

    def test_bellman_and_losses_match_reference_equations(self):
        rewards = torch.tensor([[1.0], [-0.5], [0.25]])
        terminated = torch.tensor([[False], [True], [False]])
        target_q1 = torch.tensor([[2.0], [10.0], [-1.0]])
        target_q2 = torch.tensor([[3.0], [9.0], [-2.0]])

        reference_target = rewards + (~terminated).float() * 0.99 * torch.min(
            target_q1, target_q2
        )
        actual_target = bellman_target(
            rewards,
            terminated,
            target_q1,
            target_q2,
            discount=0.99,
        )
        torch.testing.assert_close(actual_target, reference_target)

        q1 = torch.tensor([[1.5], [-0.25], [-1.25]])
        q2 = torch.tensor([[2.5], [0.0], [-2.25]])
        reference_critic_loss = ((q1 - reference_target) ** 2).mean() + (
            (q2 - reference_target) ** 2
        ).mean()
        torch.testing.assert_close(
            critic_loss(q1, q2, actual_target), reference_critic_loss
        )
        torch.testing.assert_close(deterministic_actor_loss(q1), -q1.mean())

    def test_policy_delay_counts_completed_critic_updates(self):
        self.assertFalse(policy_update_is_due(1, period=2))
        self.assertTrue(policy_update_is_due(2, period=2))
        self.assertFalse(policy_update_is_due(3, period=2))
        self.assertTrue(policy_update_is_due(4, period=2))

    def test_targets_can_be_hard_initialized_then_polyak_updated(self):
        online = _ModuleWithBuffer(2.0, 7.0)
        target = _ModuleWithBuffer(-4.0, -3.0)

        polyak_update_(online, target, tau=1.0)
        self.assertIsNot(target, online)
        torch.testing.assert_close(target.weight, online.weight)
        torch.testing.assert_close(target.running, online.running)

        previous_target = copy.deepcopy(target)
        with torch.no_grad():
            online.weight.fill_(6.0)
            online.running.fill_(11.0)
        polyak_update_(online, target, tau=0.25)

        expected_weight = previous_target.weight * 0.75 + online.weight * 0.25
        torch.testing.assert_close(target.weight, expected_weight)
        torch.testing.assert_close(target.running, online.running)

    def test_config_rejects_invalid_algorithm_values(self):
        defaults = TD3Config()
        self.assertEqual(defaults.policy_update_period, 2)
        self.assertEqual(defaults.actor_learning_rate, 3.0e-4)
        self.assertEqual(defaults.critic_learning_rate, 3.0e-4)
        with self.assertRaises(ValueError):
            TD3Config(policy_update_period=0)
        with self.assertRaises(ValueError):
            TD3Config(target_update_rate=0.0)
        with self.assertRaises(ValueError):
            TD3Config(discount=1.1)
        with self.assertRaises(ValueError):
            TD3Config(actor_learning_rate=0.0)
        with self.assertRaises(ValueError):
            TD3Config(critic_learning_rate=-1.0)


if __name__ == "__main__":
    unittest.main()
