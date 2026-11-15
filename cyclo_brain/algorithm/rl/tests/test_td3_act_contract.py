"""ACT-to-TD3 autograd contract smoke tests.

These tests deliberately use a test-local Q-function. They verify that the
official ACT graph can participate in TD3's deterministic actor objective,
but do not define the production chunk critic, action projection, or SMDP
transition contract.
"""

import copy
import unittest

import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from cyclo_brain.algorithm.rl.td3 import deterministic_actor_loss
from cyclo_brain.model.act import (
    create_act_model,
    differentiable_act_action_chunk,
)


def _act_config() -> ACTConfig:
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


def _observation() -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.tensor([[0.1, -0.2], [0.3, 0.4]]),
        OBS_ENV_STATE: torch.tensor([[-0.4, 0.5], [0.6, -0.7]]),
    }


class _TestQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Linear(2 + 3 * 2, 1)

    def forward(
        self,
        observation_state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((observation_state, action_chunk.flatten(1)), dim=-1)
        return self.value(inputs)


class _TestTwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = _TestQ()
        self.q2 = _TestQ()

    def forward(
        self,
        observation_state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(observation_state, action_chunk),
            self.q2(observation_state, action_chunk),
        )


class ACTTD3AutogradContractTest(unittest.TestCase):
    def test_td3_actor_step_reaches_official_act_action_head(self):
        torch.manual_seed(23)
        actor = create_act_model(_act_config()).eval()
        critic = _TestTwinQ().eval()
        critic.requires_grad_(False)
        optimizer = torch.optim.SGD(actor.parameters(), lr=1.0e-3)
        observation = _observation()
        action_head_before = actor.model.action_head.weight.detach().clone()

        action_chunk = differentiable_act_action_chunk(actor, observation)
        q1 = critic.q1(observation[OBS_STATE], action_chunk)
        loss = deterministic_actor_loss(q1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        action_head_gradient = actor.model.action_head.weight.grad
        self.assertIsNotNone(action_head_gradient)
        assert action_head_gradient is not None
        self.assertTrue(bool(torch.isfinite(action_head_gradient).all()))
        self.assertGreater(torch.count_nonzero(action_head_gradient).item(), 0)
        optimizer.step()

        self.assertFalse(
            torch.equal(actor.model.action_head.weight, action_head_before)
        )
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))

    def test_target_act_is_an_independent_frozen_exact_copy(self):
        torch.manual_seed(29)
        actor = create_act_model(_act_config()).eval()
        target_actor = copy.deepcopy(actor).eval().requires_grad_(False)
        observation = _observation()

        online_chunk = differentiable_act_action_chunk(actor, observation)
        with torch.no_grad():
            target_chunk = differentiable_act_action_chunk(target_actor, observation)

        for name, online_value in actor.state_dict().items():
            torch.testing.assert_close(
                target_actor.state_dict()[name], online_value, rtol=0.0, atol=0.0
            )
        torch.testing.assert_close(
            target_chunk, online_chunk.detach(), rtol=1.0e-6, atol=2.0e-7
        )
        self.assertFalse(target_chunk.requires_grad)
        self.assertEqual(
            {id(parameter) for parameter in actor.parameters()}
            & {id(parameter) for parameter in target_actor.parameters()},
            set(),
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in target_actor.parameters())
        )

    def test_twin_q_parameters_are_independent(self):
        critic = _TestTwinQ()

        self.assertEqual(
            {id(parameter) for parameter in critic.q1.parameters()}
            & {id(parameter) for parameter in critic.q2.parameters()},
            set(),
        )


if __name__ == "__main__":
    unittest.main()
