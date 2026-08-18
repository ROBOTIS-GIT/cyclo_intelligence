"""Model contracts for ACT execution projection and chunk critics."""

from __future__ import annotations

import unittest

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from cyclo_brain.model.act import (
    ACTExecutionProjector,
    ACTTwinChunkCritic,
    create_act_model,
    differentiable_act_action_chunk,
)


def _tiny_config(*, n_action_steps: int = 3) -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
        chunk_size=3,
        n_action_steps=n_action_steps,
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


def _tiny_visual_config(
    *,
    left_shape: tuple[int, int, int] = (3, 32, 32),
    right_shape: tuple[int, int, int] = (3, 32, 32),
) -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            f"{OBS_IMAGES}.left": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=left_shape,
            ),
            f"{OBS_IMAGES}.right": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=right_shape,
            ),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
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


class ACTExecutionProjectorTest(unittest.TestCase):
    def test_hard_forward_and_physical_binary_ste(self) -> None:
        projector = ACTExecutionProjector(
            action_mean=torch.zeros(3),
            action_std=torch.tensor([1.0, 0.5, 0.0]),
            physical_low=torch.tensor([-2.0, -1.0, -1.0]),
            physical_high=torch.tensor([2.0, 1.0, 1.0]),
            normalizer_eps=1.0e-6,
            binary_mask=torch.tensor([False, True, True]),
            binary_threshold=torch.zeros(3),
            binary_low=torch.tensor([0.0, -1.0, -1.0]),
            binary_high=torch.tensor([0.0, 1.0, 1.0]),
        )
        actions = torch.tensor(
            [[[0.25, 0.2, 100.0], [100.0, -0.2, -100.0]]],
            requires_grad=True,
        )

        projected = projector(actions, straight_through_binary=True)

        expected_physical = torch.tensor([[[0.25, 1.0, 1.0], [2.0, -1.0, 1.0]]])
        expected = expected_physical / torch.tensor([1.000001, 0.500001, 0.000001])
        torch.testing.assert_close(projected, expected)
        projected[0, 0].sum().backward()
        assert actions.grad is not None
        torch.testing.assert_close(actions.grad[0, 0, 0], torch.tensor(1.0 / 1.000001))
        torch.testing.assert_close(actions.grad[0, 0, 1], torch.tensor(0.5 / 0.500001))
        self.assertEqual(float(actions.grad[0, 0, 2]), 0.0)
        self.assertTrue(torch.equal(projector.noise_mask, torch.tensor([True, False, False])))

    def test_passthrough_is_exact_and_excluded_from_target_noise(self) -> None:
        projector = ACTExecutionProjector(
            action_mean=torch.tensor([0.5, -0.5, 100.0]),
            action_std=torch.tensor([2.0, 0.5, 0.25]),
            physical_low=torch.tensor([-1.0, -2.0, 0.0]),
            physical_high=torch.tensor([1.0, 2.0, 0.0]),
            normalizer_eps=1.0e-6,
            passthrough_mask=torch.tensor([False, False, True]),
        )
        actions = torch.tensor(
            [[[0.0, 10.0, 123_456.25], [-10.0, 0.0, -987_654.5]]]
        )

        projected = projector(actions, straight_through_binary=False)

        self.assertTrue(torch.equal(projected[..., 2], actions[..., 2]))
        self.assertTrue(
            torch.equal(projector.noise_mask, torch.tensor([True, True, False]))
        )

    def test_detached_passthrough_has_no_direct_actor_gradient(self) -> None:
        projector = ACTExecutionProjector(
            action_mean=torch.zeros(3),
            action_std=torch.ones(3),
            physical_low=torch.tensor([-2.0, -2.0, 0.0]),
            physical_high=torch.tensor([2.0, 2.0, 0.0]),
            normalizer_eps=1.0e-6,
            passthrough_mask=torch.tensor([False, False, True]),
        )
        actions = torch.tensor([[[0.25, -0.5, 1.75]]], requires_grad=True)

        projected = projector(
            actions,
            straight_through_binary=True,
            detach_passthrough=True,
        )

        self.assertTrue(torch.equal(projected[..., 2], actions[..., 2]))
        projected.sum().backward()
        assert actions.grad is not None
        self.assertGreater(float(actions.grad[0, 0, 0]), 0.0)
        self.assertGreater(float(actions.grad[0, 0, 1]), 0.0)
        self.assertEqual(float(actions.grad[0, 0, 2]), 0.0)

    def test_passthrough_contract_rejects_bounds_binary_and_noise(self) -> None:
        common = dict(
            action_mean=torch.zeros(2),
            action_std=torch.ones(2),
            normalizer_eps=1.0e-6,
            passthrough_mask=torch.tensor([False, True]),
        )
        with self.assertRaisesRegex(ValueError, "zero physical-limit placeholders"):
            ACTExecutionProjector(
                **common,
                physical_low=torch.tensor([-1.0, -1.0]),
                physical_high=torch.tensor([1.0, 1.0]),
            )
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            ACTExecutionProjector(
                **common,
                physical_low=torch.tensor([-1.0, 0.0]),
                physical_high=torch.tensor([1.0, 0.0]),
                binary_mask=torch.tensor([False, True]),
                binary_threshold=torch.zeros(2),
                binary_low=torch.zeros(2),
                binary_high=torch.zeros(2),
            )
        with self.assertRaisesRegex(ValueError, "cannot include.*passthrough"):
            ACTExecutionProjector(
                **common,
                physical_low=torch.tensor([-1.0, 0.0]),
                physical_high=torch.tensor([1.0, 0.0]),
                trainable_noise_mask=torch.tensor([True, True]),
            )


class ACTTwinChunkCriticTest(unittest.TestCase):
    def test_multicamera_observations_may_have_different_spatial_shapes(self) -> None:
        critic = ACTTwinChunkCritic(
            _tiny_visual_config(
                left_shape=(3, 32, 48),
                right_shape=(3, 48, 32),
            ),
            observation_feature_dim=8,
            action_feature_dim=8,
            hidden_dims=(16, 8),
            require_visual_initialization=False,
        )
        left = torch.randn(2, 3, 32, 48, requires_grad=True)
        right = torch.randn(2, 3, 48, 32, requires_grad=True)
        observations = {
            OBS_STATE: torch.randn(2, 2),
            f"{OBS_IMAGES}.left": left,
            f"{OBS_IMAGES}.right": right,
        }

        q1, q2 = critic(
            observations,
            torch.randn(2, 3, 3),
            torch.ones(2, 3, dtype=torch.bool),
        )

        self.assertEqual(q1.shape, (2, 1))
        self.assertEqual(q2.shape, (2, 1))
        self.assertTrue(bool(torch.isfinite(q1).all()))
        self.assertTrue(bool(torch.isfinite(q2).all()))
        (q1.mean() + q2.mean()).backward()
        for image in (left, right):
            self.assertIsNotNone(image.grad)
            assert image.grad is not None
            self.assertTrue(bool(torch.isfinite(image.grad).all()))
            self.assertGreater(torch.count_nonzero(image.grad).item(), 0)

    def test_critic_uses_execution_not_prediction_horizon(self) -> None:
        critic = ACTTwinChunkCritic(
            _tiny_config(n_action_steps=2),
            observation_feature_dim=8,
            action_feature_dim=8,
            hidden_dims=(16, 8),
        )
        observations = {
            OBS_STATE: torch.zeros(1, 2),
            OBS_ENV_STATE: torch.zeros(1, 2),
        }

        q1, q2 = critic(
            observations,
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
        )

        self.assertEqual(critic.prediction_horizon, 3)
        self.assertEqual(critic.execution_horizon, 2)
        self.assertEqual(q1.shape, (1, 1))
        self.assertEqual(q2.shape, (1, 1))
        with self.assertRaisesRegex(ValueError, r"finite \(B, T, A\)"):
            critic(
                observations,
                torch.zeros(1, 3, 3),
                torch.ones(1, 3, dtype=torch.bool),
            )

    def test_twins_are_independent_and_ignore_padded_actions(self) -> None:
        torch.manual_seed(37)
        critic = ACTTwinChunkCritic(
            _tiny_config(),
            observation_feature_dim=8,
            action_feature_dim=8,
            hidden_dims=(16, 8),
        )
        observations = {
            OBS_STATE: torch.randn(2, 2),
            OBS_ENV_STATE: torch.randn(2, 2),
        }
        actions = torch.randn(2, 3, 3, requires_grad=True)
        mask = torch.tensor([[True, True, True], [True, True, False]])

        q1, q2 = critic(observations, actions, mask)

        self.assertEqual(q1.shape, (2, 1))
        self.assertGreater(float((q1 - q2).abs().max().detach()), 0.0)
        self.assertFalse(
            bool(
                {id(parameter) for parameter in critic.q1.parameters()}
                & {id(parameter) for parameter in critic.q2.parameters()}
            )
        )
        changed = actions.detach().clone()
        changed[1, 2] = 10_000.0
        changed_q1, changed_q2 = critic(observations, changed, mask)
        torch.testing.assert_close(changed_q1, q1, rtol=0.0, atol=0.0)
        torch.testing.assert_close(changed_q2, q2, rtol=0.0, atol=0.0)
        (q1.mean() + q2.mean()).backward()
        assert actions.grad is not None
        self.assertEqual(torch.count_nonzero(actions.grad[1, 2]).item(), 0)
        self.assertGreater(torch.count_nonzero(actions.grad[mask]).item(), 0)

    def test_multicamera_backbones_are_initialized_from_official_act(self) -> None:
        torch.manual_seed(41)
        actor = create_act_model(_tiny_visual_config()).eval()
        critic = ACTTwinChunkCritic(
            actor.config,
            observation_feature_dim=8,
            action_feature_dim=8,
            hidden_dims=(16, 8),
        )
        observations = {
            OBS_STATE: torch.tensor([[0.1, -0.2]]),
            f"{OBS_IMAGES}.left": torch.zeros(1, 3, 32, 32),
            f"{OBS_IMAGES}.right": torch.ones(1, 3, 32, 32),
        }
        actions = torch.zeros(1, 3, 3)
        mask = torch.ones(1, 3, dtype=torch.bool)

        with self.assertRaisesRegex(RuntimeError, "initialized from ACT"):
            critic(observations, actions, mask)
        critic.initialize_visual_backbones_from_actor(actor)

        actor_backbone = actor.model.backbone.state_dict()
        for q_function in (critic.q1, critic.q2):
            q_backbone = q_function.observation_encoder.backbone.state_dict()
            self.assertEqual(q_backbone.keys(), actor_backbone.keys())
            for name, value in actor_backbone.items():
                torch.testing.assert_close(
                    q_backbone[name],
                    value,
                    rtol=0.0,
                    atol=0.0,
                )
        q1, q2 = critic(observations, actions, mask)
        self.assertEqual(q1.shape, (1, 1))
        self.assertEqual(q2.shape, (1, 1))

    def test_official_act_gradient_flows_through_projector_and_chunk_q(self) -> None:
        torch.manual_seed(43)
        actor = create_act_model(_tiny_config()).eval()
        critic = ACTTwinChunkCritic(
            actor.config,
            observation_feature_dim=8,
            action_feature_dim=8,
            hidden_dims=(16, 8),
        ).eval()
        critic.requires_grad_(False)
        projector = ACTExecutionProjector(
            action_mean=torch.zeros(3),
            action_std=torch.ones(3),
            physical_low=torch.full((3,), -2.0),
            physical_high=torch.full((3,), 2.0),
            normalizer_eps=1.0e-8,
        )
        observations = {
            OBS_STATE: torch.tensor([[0.1, -0.2], [0.3, 0.4]]),
            OBS_ENV_STATE: torch.tensor([[-0.4, 0.5], [0.6, -0.7]]),
        }
        policy_actions = differentiable_act_action_chunk(actor, observations)
        executed_actions = projector(
            policy_actions,
            straight_through_binary=True,
        )
        q1 = critic.q1(
            observations,
            executed_actions,
            torch.ones(2, 3, dtype=torch.bool),
        )

        (-q1.mean()).backward()

        gradient = actor.model.action_head.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(torch.count_nonzero(gradient).item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))


if __name__ == "__main__":
    unittest.main()
