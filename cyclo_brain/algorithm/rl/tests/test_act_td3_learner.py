"""Official ACT integration tests for the separate chunk-SMDP learner."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from unittest import mock

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

import cyclo_brain.algorithm.rl.act_td3.learner as learner_module
from cyclo_brain.algorithm.rl.act_td3 import ACTTD3Batch, ACTTD3Config
from cyclo_brain.algorithm.rl.act_td3.learner import ACTTD3Learner
from cyclo_brain.model.act import (
    ACTTwinChunkCritic,
    act_parameter_group,
    create_act_model,
)


def _act_config(*, n_action_steps: int = 3, action_dim: int = 2) -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
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


def _algorithm_config() -> ACTTD3Config:
    return ACTTD3Config(
        discount=0.9,
        discount_reference_hz=10.0,
        target_update_rate=0.1,
        target_policy_noise=0.1,
        target_policy_noise_clip=0.2,
        policy_update_period=2,
        critic_warmup_updates=2,
        actor_learning_rate=1.0e-3,
        critic_learning_rate=1.0e-3,
        actor_gradient_clip_norm=10.0,
        critic_gradient_clip_norm=10.0,
        q_weight_max=0.25,
        q_weight_ramp_actor_updates=2,
    )


def _learner(
    seed: int = 17,
    *,
    n_action_steps: int = 3,
    action_dim: int = 2,
    actor_trainable_groups: tuple[str, ...] | None = None,
) -> ACTTD3Learner:
    torch.manual_seed(101)
    actor = create_act_model(
        _act_config(n_action_steps=n_action_steps, action_dim=action_dim)
    )
    critic = ACTTwinChunkCritic(
        actor.config,
        observation_feature_dim=8,
        action_feature_dim=8,
        hidden_dims=(16, 8),
    )
    algorithm_config = _algorithm_config()
    if actor_trainable_groups is not None:
        algorithm_config = replace(
            algorithm_config,
            actor_trainable_groups=actor_trainable_groups,
        )
    return ACTTD3Learner(
        actor,
        critic,
        algorithm_config,
        random_seed=seed,
    )


def _batch() -> ACTTD3Batch:
    return ACTTD3Batch(
        observations={
            OBS_STATE: torch.tensor([[0.1, -0.2], [0.3, 0.4]]),
            OBS_ENV_STATE: torch.tensor([[-0.4, 0.5], [0.6, -0.7]]),
        },
        next_observations={
            OBS_STATE: torch.tensor([[0.2, -0.1], [0.0, 0.0]]),
            OBS_ENV_STATE: torch.tensor([[-0.3, 0.6], [0.0, 0.0]]),
        },
        behavior_action_chunks=torch.tensor(
            [
                [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
                [[-0.1, 0.0], [0.1, 0.2], [0.0, 0.0]],
            ]
        ),
        rewards=torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 0.0]]),
        executed_mask=torch.tensor(
            [[True, True, True], [True, True, False]]
        ),
        step_durations_s=torch.tensor(
            [[0.1, 0.1, 0.1], [0.1, 0.1, 0.0]]
        ),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
        next_observation_valid=torch.tensor([True, False]),
        bootstrap_allowed=torch.tensor([True, False]),
    )


def _short_execution_batch() -> ACTTD3Batch:
    return ACTTD3Batch(
        observations={
            OBS_STATE: torch.tensor([[0.1, -0.2], [0.3, 0.4]]),
            OBS_ENV_STATE: torch.tensor([[-0.4, 0.5], [0.6, -0.7]]),
        },
        next_observations={
            OBS_STATE: torch.tensor([[0.2, -0.1], [0.0, 0.0]]),
            OBS_ENV_STATE: torch.tensor([[-0.3, 0.6], [0.0, 0.0]]),
        },
        behavior_action_chunks=torch.tensor(
            [
                [[0.0, 0.1], [0.2, 0.3]],
                [[-0.1, 0.0], [0.0, 0.0]],
            ]
        ),
        rewards=torch.tensor([[0.1, 0.2], [1.0, 0.0]]),
        executed_mask=torch.tensor([[True, True], [True, False]]),
        step_durations_s=torch.tensor([[0.1, 0.1], [0.1, 0.0]]),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
        next_observation_valid=torch.tensor([True, False]),
        bootstrap_allowed=torch.tensor([True, False]),
    )


def _zero_action_batch(action_dim: int) -> ACTTD3Batch:
    base = _batch()
    return replace(
        base,
        behavior_action_chunks=torch.zeros(
            base.batch_size,
            base.execution_horizon,
            action_dim,
        ),
    )


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(before[name], value)
        for name, value in module.state_dict().items()
    )


def _legacy_v3_state(learner: ACTTD3Learner) -> dict[str, object]:
    state = learner.state_dict()
    state["format"] = learner.LEGACY_ALL_TRAINABLE_STATE_FORMAT
    del state["contract"]["actor_trainable_groups"]
    del state["config"]["actor_trainable_groups"]
    return state


class ACTTD3LearnerTest(unittest.TestCase):
    def test_target_noise_covers_all_22_dimensions_without_action_clamp(self) -> None:
        learner = _learner(action_dim=22)
        batch = _zero_action_batch(22)
        policy_actions = torch.full((1, 3, 22), 5.0)
        noise = torch.ones(1, 3, 22)
        noise[..., -3:] = torch.tensor([100.0, -100.0, 2.0])
        captured: dict[str, torch.Tensor] = {}

        def capture_target(_observations, actions, _mask):
            captured["actions"] = actions.detach().clone()
            zeros = actions.new_zeros((actions.shape[0], 1))
            return zeros, zeros.clone()

        with (
            mock.patch.object(
                learner_module,
                "differentiable_act_action_chunk",
                return_value=policy_actions,
            ),
            mock.patch.object(
                learner.critic_target,
                "forward",
                side_effect=capture_target,
            ),
        ):
            learner.compute_bellman_targets(
                batch,
                target_standard_normal_noise=noise,
            )

        expected_noise = (noise * learner.config.target_policy_noise).clamp(
            -learner.config.target_policy_noise_clip,
            learner.config.target_policy_noise_clip,
        )
        torch.testing.assert_close(
            captured["actions"],
            policy_actions + expected_noise,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(bool((captured["actions"] > 2.0).all()))

    def test_actor_q_gradient_reaches_all_22_outputs_including_mobile(self) -> None:
        learner = _learner(action_dim=22)
        learner.completed_actor_updates = 1
        before = learner.actor.model.action_head.bias.detach().clone()
        behavior = before.view(1, 1, -1).expand(2, 3, -1).clone()
        behavior[1, 2] = 0.0
        batch = replace(
            _zero_action_batch(22),
            behavior_action_chunks=behavior,
        )

        def actor_chunk(actor, observations):
            bias = actor.model.action_head.bias
            return bias.view(1, 1, -1).expand(
                next(iter(observations.values())).shape[0],
                actor.config.chunk_size,
                -1,
            )

        def zero_bc(actor, _batch):
            return next(actor.parameters()).sum() * 0.0, {}

        def sum_action_q(_observations, actions, _mask):
            return actions.sum(dim=(1, 2)).unsqueeze(-1)

        with (
            mock.patch.object(
                learner_module,
                "differentiable_act_action_chunk",
                side_effect=actor_chunk,
            ),
            mock.patch.object(
                learner_module,
                "compute_act_bc_loss",
                side_effect=zero_bc,
            ),
            mock.patch.object(
                learner.critic.q1,
                "forward",
                side_effect=sum_action_q,
            ),
        ):
            learner._actor_step(batch)

        after = learner.actor.model.action_head.bias.detach()
        self.assertTrue(bool((after != before).all()))
        self.assertTrue(bool((after[-3:] != before[-3:]).all()))

    def test_checkpoint_declares_normalized_unbounded_action_contract(self) -> None:
        state = _learner(action_dim=22).state_dict()

        self.assertEqual(state["format"], "cyclo_brain.act_td3_learner/v4")
        self.assertNotIn("action_projector", state)
        self.assertEqual(
            state["contract"]["action_domain"],
            "saved_act_preprocessor_mean_std_normalized",
        )
        self.assertEqual(
            state["contract"]["target_policy_smoothing"],
            "clipped_noise_all_dimensions_no_action_clamp",
        )
        self.assertEqual(
            state["contract"]["actor_q_gradient"],
            "all_action_dimensions",
        )
        self.assertIs(state["contract"]["action_clamp"], False)
        self.assertEqual(
            state["contract"]["actor_trainable_groups"],
            (
                "visual_backbone",
                "cvae_encoder",
                "transformer_encoder",
                "action_decoder",
            ),
        )

    def test_actor_optimizer_and_resume_preserve_selected_freeze_mask(self) -> None:
        groups = ("action_decoder",)
        learner = _learner(actor_trainable_groups=groups)
        expected_names = {
            name
            for name, _parameter in learner.actor.named_parameters()
            if act_parameter_group(name) == "action_decoder"
        }
        actual_names = {
            name
            for name, parameter in learner.actor.named_parameters()
            if parameter.requires_grad
        }
        optimizer_ids = {
            id(parameter)
            for group in learner.actor_optimizer.param_groups
            for parameter in group["params"]
        }

        self.assertEqual(actual_names, expected_names)
        self.assertEqual(
            optimizer_ids,
            {
                id(parameter)
                for parameter in learner.actor.parameters()
                if parameter.requires_grad
            },
        )
        state = learner.state_dict()
        learner.actor.requires_grad_(True)
        learner.load_state_dict(state)
        self.assertEqual(
            {
                name
                for name, parameter in learner.actor.named_parameters()
                if parameter.requires_grad
            },
            expected_names,
        )

    def test_config_rejects_empty_cvae_only_duplicate_and_unknown_groups(self) -> None:
        for groups, message in (
            ((), "cannot be empty"),
            (("cvae_encoder",), "deterministic inference-path"),
            (("action_decoder", "action_decoder"), "duplicates"),
            (("not_an_act_group",), "Unknown"),
        ):
            with self.subTest(groups=groups), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                replace(_algorithm_config(), actor_trainable_groups=groups)

    def test_checkpoint_rejects_different_trainable_group_contract(self) -> None:
        state = _learner(actor_trainable_groups=("action_decoder",)).state_dict()
        restored = _learner(actor_trainable_groups=("transformer_encoder",))

        with self.assertRaisesRegex(ValueError, "tensor contract"):
            restored.load_state_dict(state)

    def test_legacy_v3_checkpoint_resumes_with_default_all_groups(self) -> None:
        source = _learner(seed=29, n_action_steps=2)
        batch = _short_execution_batch()
        for _ in range(4):
            source.update(batch)
        legacy = _legacy_v3_state(source)
        restored = _learner(seed=29, n_action_steps=2)

        restored.load_state_dict(legacy)

        self.assertEqual(
            legacy["format"],
            source.LEGACY_ALL_TRAINABLE_STATE_FORMAT,
        )
        self.assertNotIn("actor_trainable_groups", legacy["contract"])
        self.assertNotIn("actor_trainable_groups", legacy["config"])
        self.assertEqual(source.update(batch), restored.update(batch))
        for expected, actual in (
            (source.actor, restored.actor),
            (source.actor_target, restored.actor_target),
            (source.critic, restored.critic),
            (source.critic_target, restored.critic_target),
        ):
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(
                    value,
                    actual.state_dict()[name],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_legacy_v3_checkpoint_rejects_partial_freeze_request(self) -> None:
        legacy = _legacy_v3_state(_learner())
        restored = _learner(actor_trainable_groups=("action_decoder",))

        with self.assertRaisesRegex(
            ValueError,
            "only with all ACT actor trainable groups",
        ):
            restored.load_state_dict(legacy)

    def test_v4_checkpoint_does_not_normalize_missing_group_contract(self) -> None:
        state = _learner().state_dict()
        del state["contract"]["actor_trainable_groups"]

        with self.assertRaisesRegex(ValueError, "tensor contract"):
            _learner().load_state_dict(state)

    def test_prediction_horizon_can_exceed_execution_horizon(self) -> None:
        learner = _learner(n_action_steps=2)
        batch = _short_execution_batch()
        noise = torch.zeros(1, 2, 2)

        self.assertEqual(learner.prediction_horizon, 3)
        self.assertEqual(learner.execution_horizon, 2)
        self.assertEqual(learner.critic.prediction_horizon, 3)
        self.assertEqual(learner.critic.execution_horizon, 2)
        for _ in range(4):
            result = learner.update(
                batch,
                target_standard_normal_noise=noise,
            )

        self.assertTrue(result.actor_updated)
        self.assertEqual(result.actor_q_full_row_count, 1)

    def test_prediction_suffix_cannot_affect_target_q(self) -> None:
        learner = _learner(n_action_steps=2)
        batch = _short_execution_batch()
        prefix = torch.tensor([[[0.1, -0.2], [0.3, -0.4]]])
        first_chunk = torch.cat((prefix, torch.tensor([[[10.0, 20.0]]])), dim=1)
        second_chunk = torch.cat((prefix, torch.tensor([[[-30.0, 40.0]]])), dim=1)
        noise = torch.zeros(1, 2, 2)

        with mock.patch.object(
            learner_module,
            "differentiable_act_action_chunk",
            return_value=first_chunk,
        ):
            first_target = learner.compute_bellman_targets(
                batch,
                target_standard_normal_noise=noise,
            )
        with mock.patch.object(
            learner_module,
            "differentiable_act_action_chunk",
            return_value=second_chunk,
        ):
            second_target = learner.compute_bellman_targets(
                batch,
                target_standard_normal_noise=noise,
            )

        torch.testing.assert_close(first_target, second_target, rtol=0.0, atol=0.0)

    def test_target_noise_uses_execution_horizon(self) -> None:
        learner = _learner(n_action_steps=2)

        with self.assertRaisesRegex(ValueError, "target noise"):
            learner.compute_bellman_targets(
                _short_execution_batch(),
                target_standard_normal_noise=torch.zeros(1, 3, 2),
            )

    def test_cvae_bc_receives_executed_prefix_and_padded_suffix(self) -> None:
        learner = _learner(n_action_steps=2)
        batch = _short_execution_batch()
        noise = torch.zeros(1, 2, 2)
        for _ in range(3):
            learner.update(batch, target_standard_normal_noise=noise)
        captured: dict[str, torch.Tensor] = {}
        original_compute_bc = learner_module.compute_act_bc_loss

        def capture_bc(actor, bc_batch):
            captured[ACTION] = bc_batch[ACTION].detach().clone()
            captured["action_is_pad"] = bc_batch["action_is_pad"].detach().clone()
            return original_compute_bc(actor, bc_batch)

        with mock.patch.object(
            learner_module,
            "compute_act_bc_loss",
            side_effect=capture_bc,
        ):
            result = learner.update(batch, target_standard_normal_noise=noise)

        self.assertTrue(result.actor_updated)
        self.assertEqual(captured[ACTION].shape, (2, 3, 2))
        torch.testing.assert_close(
            captured[ACTION][:, :2],
            batch.behavior_action_chunks,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(torch.count_nonzero(captured[ACTION][:, 2:]).item(), 0)
        self.assertTrue(torch.equal(
            captured["action_is_pad"][:, :2],
            ~batch.executed_mask,
        ))
        self.assertTrue(bool(captured["action_is_pad"][:, 2:].all()))

    def test_warmup_delays_actor_and_updates_target_critic(self) -> None:
        learner = _learner()
        batch = _batch()
        actor_before = _state(learner.actor)
        target_critic_before = _state(learner.critic_target)
        noise = torch.zeros(1, 3, 2)

        first = learner.update(batch, target_standard_normal_noise=noise)
        second = learner.update(batch, target_standard_normal_noise=noise)

        self.assertFalse(first.actor_updated)
        self.assertFalse(first.target_critic_updated)
        self.assertFalse(second.actor_updated)
        self.assertTrue(second.target_critic_updated)
        self.assertFalse(_changed(actor_before, learner.actor))
        self.assertTrue(_changed(target_critic_before, learner.critic_target))

    def test_official_bc_and_delayed_q_update_actor_on_first_due_step(self) -> None:
        learner = _learner()
        batch = _batch()
        noise = torch.zeros(1, 3, 2)
        for _ in range(3):
            result = learner.update(batch, target_standard_normal_noise=noise)
            self.assertFalse(result.actor_updated)
        actor_before = _state(learner.actor)
        target_actor_before = _state(learner.actor_target)

        result = learner.update(batch, target_standard_normal_noise=noise)

        self.assertTrue(result.actor_updated)
        self.assertEqual(result.completed_critic_updates, 4)
        self.assertEqual(result.completed_actor_updates, 1)
        self.assertEqual(result.actor_q_full_row_count, 1)
        self.assertEqual(result.actor_q_weight, 0.0)
        self.assertIsNotNone(result.cvae_bc_loss)
        self.assertIsNotNone(result.deterministic_bc_loss)
        self.assertTrue(_changed(actor_before, learner.actor))
        self.assertTrue(_changed(target_actor_before, learner.actor_target))
        self.assertTrue(all(parameter.grad is None for parameter in learner.critic.parameters()))
        self.assertFalse(learner.actor.training)
        self.assertFalse(learner.actor_target.training)

    def test_checkpoint_resume_preserves_owned_rng_and_next_updates(self) -> None:
        learner = _learner(seed=29, n_action_steps=2)
        batch = _short_execution_batch()
        for _ in range(4):
            learner.update(batch)
        state = learner.state_dict()
        restored = _learner(seed=29, n_action_steps=2)
        restored.load_state_dict(state)

        original_results = [learner.update(batch), learner.update(batch)]
        restored_results = [restored.update(batch), restored.update(batch)]

        self.assertEqual(original_results, restored_results)
        for original, actual in (
            (learner.actor, restored.actor),
            (learner.actor_target, restored.actor_target),
            (learner.critic, restored.critic),
            (learner.critic_target, restored.critic_target),
        ):
            for name, value in original.state_dict().items():
                torch.testing.assert_close(
                    value,
                    actual.state_dict()[name],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_checkpoint_rejects_contract_before_loading_weights(self) -> None:
        source = _learner(n_action_steps=2)
        state = source.state_dict()
        state["contract"]["execution_horizon"] = 3
        restored = _learner(n_action_steps=2)
        actor_before = copy.deepcopy(restored.actor.state_dict())

        with self.assertRaisesRegex(ValueError, "tensor contract"):
            restored.load_state_dict(state)

        for name, value in restored.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[name], rtol=0.0, atol=0.0)

    def test_state_dict_is_an_alias_free_snapshot(self) -> None:
        learner = _learner()
        batch = _batch()
        snapshot = learner.state_dict()
        saved_action_head = snapshot["actor"]["model.action_head.weight"].clone()

        for _ in range(4):
            learner.update(
                batch,
                target_standard_normal_noise=torch.zeros(1, 3, 2),
            )

        torch.testing.assert_close(
            snapshot["actor"]["model.action_head.weight"],
            saved_action_head,
            rtol=0.0,
            atol=0.0,
        )
        self.assertNotEqual(
            snapshot["actor"]["model.action_head.weight"].data_ptr(),
            learner.actor.state_dict()["model.action_head.weight"].data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
