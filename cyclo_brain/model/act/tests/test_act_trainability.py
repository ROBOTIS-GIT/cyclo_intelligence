"""ACT freeze-mask contract tests."""

from __future__ import annotations

import unittest

from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from cyclo_brain.model.act import (
    ACT_TRAINABLE_GROUPS,
    act_parameter_group,
    apply_act_trainable_groups,
    canonicalize_act_trainable_groups,
    classify_act_parameters,
    create_act_model,
)


def _tiny_config() -> ACTConfig:
    return ACTConfig(
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


class ACTTrainabilityTest(unittest.TestCase):
    def test_every_official_parameter_is_classified_once(self) -> None:
        policy = create_act_model(_tiny_config())
        classified = classify_act_parameters(policy)
        classified_names = [
            name for group in ACT_TRAINABLE_GROUPS for name in classified[group]
        ]

        self.assertEqual(set(classified), set(ACT_TRAINABLE_GROUPS))
        self.assertEqual(
            set(classified_names),
            {name for name, _parameter in policy.named_parameters()},
        )
        self.assertEqual(len(classified_names), len(set(classified_names)))
        self.assertTrue(classified["cvae_encoder"])
        self.assertTrue(classified["transformer_encoder"])
        self.assertTrue(classified["action_decoder"])
        self.assertFalse(classified["visual_backbone"])
        for group, names in classified.items():
            self.assertTrue(all(act_parameter_group(name) == group for name in names))

    def test_apply_mask_only_enables_selected_parameters(self) -> None:
        policy = create_act_model(_tiny_config())
        canonical = apply_act_trainable_groups(
            policy,
            ("action_decoder", "transformer_encoder"),
        )

        self.assertEqual(canonical, ("transformer_encoder", "action_decoder"))
        for name, parameter in policy.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                act_parameter_group(name)
                in {"transformer_encoder", "action_decoder"},
                msg=name,
            )

        optimizer_parameter_ids = {
            id(parameter)
            for group in policy.get_optim_params()
            for parameter in group["params"]
        }
        trainable_parameter_ids = {
            id(parameter)
            for parameter in policy.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(optimizer_parameter_ids, trainable_parameter_ids)

    def test_invalid_and_non_deployed_selections_are_rejected(self) -> None:
        for groups, message in (
            ((), "cannot be empty"),
            (("cvae_encoder",), "deterministic inference-path"),
            (("action_decoder", "action_decoder"), "duplicates"),
            (("unknown",), "Unknown"),
        ):
            with self.subTest(groups=groups), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                canonicalize_act_trainable_groups(groups)

        with self.assertRaisesRegex(TypeError, "iterable"):
            canonicalize_act_trainable_groups("action_decoder")
        with self.assertRaisesRegex(TypeError, "official LeRobot ACTPolicy"):
            classify_act_parameters(nn.Linear(2, 2))

    def test_selected_group_must_exist_on_the_loaded_actor(self) -> None:
        policy = create_act_model(_tiny_config())

        with self.assertRaisesRegex(ValueError, "no model parameters"):
            apply_act_trainable_groups(policy, ("visual_backbone",))


if __name__ == "__main__":
    unittest.main()
