"""Model-level contracts for the reference MLP TD3 policy."""

import unittest

import torch
from torch import nn

from cyclo_brain.model.mlp import TD3MLPActor, TD3MLPTwinCritic


class TD3MLPModelTest(unittest.TestCase):
    def test_default_topology_is_the_reference_two_layer_relu_mlp(self):
        actor = TD3MLPActor(3, [-1.0, -1.0], [1.0, 1.0])
        critic = TD3MLPTwinCritic(3, 2)

        actor_linears = [
            module for module in actor.network if isinstance(module, nn.Linear)
        ]
        q1_linears = [
            module for module in critic.q1.network if isinstance(module, nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in actor_linears],
            [(3, 256), (256, 256), (256, 2)],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in q1_linears],
            [(5, 256), (256, 256), (256, 1)],
        )
        self.assertEqual(
            sum(isinstance(module, nn.ReLU) for module in actor.network), 2
        )

    def test_actor_maps_tanh_to_asymmetric_vector_bounds(self):
        actor = TD3MLPActor(
            3,
            action_low=torch.tensor([-2.0, 1.0]),
            action_high=torch.tensor([4.0, 5.0]),
            hidden_dims=(8, 8),
        )
        for module in actor.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

        midpoint = actor(torch.randn(5, 3))
        torch.testing.assert_close(
            midpoint,
            torch.tensor([[1.0, 3.0]]).expand_as(midpoint),
            rtol=0.0,
            atol=0.0,
        )

        final_layer = actor.network[-1]
        assert isinstance(final_layer, nn.Linear)
        with torch.no_grad():
            final_layer.bias.copy_(torch.tensor([20.0, -20.0]))
        saturated = actor(torch.full((5, 3), 1.0e6))
        torch.testing.assert_close(
            saturated,
            torch.tensor([[4.0, 1.0]]).expand_as(saturated),
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(bool(torch.isfinite(saturated).all()))
        self.assertTrue(bool((saturated >= actor.action_low).all()))
        self.assertTrue(bool((saturated <= actor.action_high).all()))
        self.assertEqual(set(dict(actor.named_buffers())), {"action_low", "action_high"})
        self.assertIn("action_low", actor.state_dict())
        self.assertIn("action_high", actor.state_dict())

    def test_twin_critics_are_disjoint_and_q1_path_is_identical(self):
        torch.manual_seed(5)
        critic = TD3MLPTwinCritic(3, 2, hidden_dims=(8, 8))
        observations = torch.randn(4, 3)
        actions = torch.randn(4, 2, requires_grad=True)

        q1, q2 = critic(observations, actions)
        direct_q1 = critic.q1(observations, actions)

        torch.testing.assert_close(q1, direct_q1, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(q1.shape), (4, 1))
        self.assertEqual(tuple(q2.shape), (4, 1))
        self.assertEqual(
            {id(parameter) for parameter in critic.q1.parameters()}
            & {id(parameter) for parameter in critic.q2.parameters()},
            set(),
        )
        (q1.mean() + q2.mean()).backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in critic.q1.parameters())
        )
        self.assertTrue(
            all(parameter.grad is not None for parameter in critic.q2.parameters())
        )
        self.assertIsNotNone(actions.grad)


if __name__ == "__main__":
    unittest.main()
