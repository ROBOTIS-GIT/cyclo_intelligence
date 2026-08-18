"""Deterministic end-to-end checks for the standard one-step TD3 learner."""

import copy
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from cyclo_brain.algorithm.rl.td3 import (
    TD3Batch,
    TD3Config,
    TD3Learner,
    bellman_target,
    clipped_target_action,
    critic_loss,
    deterministic_actor_loss,
    polyak_update_,
)
from cyclo_brain.model.mlp import TD3MLPActor, TD3MLPTwinCritic


def _config() -> TD3Config:
    return TD3Config(
        discount=0.9,
        target_update_rate=0.25,
        target_policy_noise=0.2,
        target_policy_noise_clip=0.1,
        policy_update_period=2,
        actor_learning_rate=1.0e-2,
        critic_learning_rate=1.0e-2,
    )


def _learner(seed: int = 17) -> TD3Learner:
    torch.manual_seed(seed)
    actor = TD3MLPActor(
        3,
        action_low=torch.tensor([-2.0, 1.0]),
        action_high=torch.tensor([4.0, 5.0]),
        hidden_dims=(8, 8),
    )
    critic = TD3MLPTwinCritic(3, 2, hidden_dims=(8, 8))
    return TD3Learner(actor, critic, _config())


def _batch(*, next_requires_grad: bool = False) -> TD3Batch:
    next_observations = torch.tensor(
        [[0.4, -0.3, 0.2], [-0.2, 0.5, -0.6], [0.7, 0.1, -0.4]],
        requires_grad=next_requires_grad,
    )
    return TD3Batch(
        observations=torch.tensor(
            [[0.1, 0.2, -0.1], [-0.4, 0.3, 0.6], [0.8, -0.5, 0.2]]
        ),
        actions=torch.tensor([[-1.5, 1.5], [0.5, 3.0], [3.5, 4.5]]),
        rewards=torch.tensor([[1.0], [-0.5], [0.25]]),
        next_observations=next_observations,
        terminated=torch.tensor([[False], [True], [False]]),
    )


def _noise(*, requires_grad: bool = False) -> Tensor:
    return torch.tensor(
        [[0.5, -0.5], [0.05, -0.05], [-0.25, 0.25]],
        requires_grad=requires_grad,
    )


def _tensor_state(module: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_tensor_state_equal(
    test: unittest.TestCase,
    actual: torch.nn.Module,
    expected: Mapping[str, Tensor],
) -> None:
    test.assertEqual(set(actual.state_dict()), set(expected))
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0.0, atol=0.0)


def _assert_tree_equal(test: unittest.TestCase, actual: Any, expected: Any) -> None:
    if isinstance(expected, Tensor):
        test.assertIsInstance(actual, Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    elif isinstance(expected, Mapping):
        test.assertIsInstance(actual, Mapping)
        test.assertEqual(set(actual), set(expected))
        for key in expected:
            _assert_tree_equal(test, actual[key], expected[key])
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        test.assertIsInstance(actual, Sequence)
        test.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_tree_equal(test, actual_value, expected_value)
    else:
        test.assertEqual(actual, expected)


class TD3LearnerTest(unittest.TestCase):
    def test_targets_are_exact_frozen_nonaliased_copies(self):
        learner = _learner()

        for online, target in (
            (learner.actor, learner.actor_target),
            (learner.critic, learner.critic_target),
        ):
            for name, online_value in online.state_dict().items():
                torch.testing.assert_close(
                    target.state_dict()[name], online_value, rtol=0.0, atol=0.0
                )
            self.assertEqual(
                {id(parameter) for parameter in online.parameters()}
                & {id(parameter) for parameter in target.parameters()},
                set(),
            )
            for online_value, target_value in zip(
                online.state_dict().values(), target.state_dict().values(), strict=True
            ):
                self.assertNotEqual(online_value.data_ptr(), target_value.data_ptr())
            self.assertTrue(
                all(not parameter.requires_grad for parameter in target.parameters())
            )
            self.assertFalse(target.training)

        optimizer_parameters = {
            id(parameter)
            for optimizer in (learner.actor_optimizer, learner.critic_optimizer)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        target_parameters = {
            id(parameter)
            for module in (learner.actor_target, learner.critic_target)
            for parameter in module.parameters()
        }
        self.assertEqual(optimizer_parameters & target_parameters, set())

    def test_fixed_noise_target_path_matches_td3_equation(self):
        learner = _learner()
        batch = _batch()
        noise = _noise()

        with torch.no_grad():
            unsmoothed = learner.actor_target(batch.next_observations)
            expected_actions = clipped_target_action(
                unsmoothed,
                noise,
                noise_clip=learner.config.target_policy_noise_clip,
                action_low=learner.action_low,
                action_high=learner.action_high,
            )
            expected_q1, expected_q2 = learner.critic_target(
                batch.next_observations, expected_actions
            )
            expected = bellman_target(
                batch.rewards,
                batch.terminated,
                expected_q1,
                expected_q2,
                discount=learner.config.discount,
            )

        actual = learner.compute_bellman_targets(batch, target_noise=noise)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[1], batch.rewards[1], rtol=0.0, atol=0.0)
        self.assertTrue(bool((expected_actions >= learner.action_low).all()))
        self.assertTrue(bool((expected_actions <= learner.action_high).all()))

    def test_first_step_updates_only_online_critic(self):
        learner = _learner()
        actor_before = _tensor_state(learner.actor)
        critic_before = _tensor_state(learner.critic)
        actor_target_before = _tensor_state(learner.actor_target)
        critic_target_before = _tensor_state(learner.critic_target)

        result = learner.update(_batch(), target_noise=_noise())

        self.assertEqual(result.completed_critic_updates, 1)
        self.assertFalse(result.actor_updated)
        self.assertIsNone(result.actor_loss)
        _assert_tensor_state_equal(self, learner.actor, actor_before)
        _assert_tensor_state_equal(self, learner.actor_target, actor_target_before)
        _assert_tensor_state_equal(self, learner.critic_target, critic_target_before)
        self.assertTrue(
            any(
                not torch.equal(value, critic_before[name])
                for name, value in learner.critic.state_dict().items()
            )
        )

    def test_second_step_matches_independent_reference_update_order(self):
        actual = _learner()
        actual.update(_batch(), target_noise=_noise())
        reference = _learner(seed=999)
        reference.load_state_dict(copy.deepcopy(actual.state_dict()))
        batch = _batch()
        noise = _noise()

        with torch.no_grad():
            target_action = reference.actor_target(batch.next_observations)
            target_action = clipped_target_action(
                target_action,
                noise,
                noise_clip=reference.config.target_policy_noise_clip,
                action_low=reference.action_low,
                action_high=reference.action_high,
            )
            target_q1, target_q2 = reference.critic_target(
                batch.next_observations, target_action
            )
            targets = bellman_target(
                batch.rewards,
                batch.terminated,
                target_q1,
                target_q2,
                discount=reference.config.discount,
            )

        q1, q2 = reference.critic(batch.observations, batch.actions)
        expected_critic_loss = critic_loss(q1, q2, targets)
        reference.critic_optimizer.zero_grad(set_to_none=True)
        expected_critic_loss.backward()
        reference.critic_optimizer.step()
        reference.critic_optimizer.zero_grad(set_to_none=True)
        reference.completed_critic_updates += 1

        critic_flags = [
            (parameter, parameter.requires_grad)
            for parameter in reference.critic.parameters()
        ]
        for parameter, _flag in critic_flags:
            parameter.requires_grad_(False)
        policy_action = reference.actor(batch.observations)
        expected_actor_loss = deterministic_actor_loss(
            reference.critic.q1(batch.observations, policy_action)
        )
        reference.actor_optimizer.zero_grad(set_to_none=True)
        expected_actor_loss.backward()
        reference.actor_optimizer.step()
        reference.actor_optimizer.zero_grad(set_to_none=True)
        for parameter, flag in critic_flags:
            parameter.requires_grad_(flag)

        polyak_update_(
            reference.actor,
            reference.actor_target,
            tau=reference.config.target_update_rate,
        )
        polyak_update_(
            reference.critic,
            reference.critic_target,
            tau=reference.config.target_update_rate,
        )

        result = actual.update(batch, target_noise=noise)

        self.assertTrue(result.actor_updated)
        self.assertEqual(result.completed_critic_updates, 2)
        self.assertEqual(result.critic_loss, float(expected_critic_loss.detach()))
        self.assertEqual(result.actor_loss, float(expected_actor_loss.detach()))
        _assert_tree_equal(self, actual.state_dict(), reference.state_dict())

    def test_target_branch_never_builds_gradients(self):
        learner = _learner()
        batch = _batch(next_requires_grad=True)
        noise = _noise(requires_grad=True)

        learner.update(batch, target_noise=noise)

        self.assertIsNone(batch.next_observations.grad)
        self.assertIsNone(noise.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in learner.actor_target.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in learner.critic_target.parameters())
        )

    def test_training_state_round_trip_preserves_exact_fixed_noise_resume(self):
        learner = _learner()
        for _ in range(3):
            learner.update(_batch(), target_noise=_noise())
        saved = copy.deepcopy(learner.state_dict())
        restored = _learner(seed=1234)

        restored.load_state_dict(saved)

        _assert_tree_equal(self, restored.state_dict(), saved)
        original_result = learner.update(_batch(), target_noise=_noise())
        restored_result = restored.update(_batch(), target_noise=_noise())
        self.assertEqual(original_result, restored_result)
        _assert_tree_equal(self, restored.state_dict(), learner.state_dict())


if __name__ == "__main__":
    unittest.main()
