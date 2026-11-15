"""End-to-end CPU checks for Flow-SDE chain sampling and replay."""

import unittest

import torch
from torch import nn

from cyclo_brain.algorithm.rl.flow_sde_ppo import (
    FlowSDEPPOConfig,
    ppo_clipped_actor_loss,
    recompute_flow_sde_log_probs,
    sample_flow_sde_chunk,
)


class _TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, latent, progress, conditioning):
        bias = conditioning[:, :1, None]
        return self.scale * latent + bias + progress[:, None, None]


class FlowSDESamplerTest(unittest.TestCase):
    def test_rollout_replay_matches_and_backpropagates_into_velocity(self):
        model = _TinyVelocity()
        conditioning = torch.tensor([[0.1, 0.0], [-0.2, 0.5]])
        initial_noise = torch.tensor(
            [
                [[0.0, 0.5], [1.0, -0.5]],
                [[-1.0, 0.25], [0.75, 0.0]],
            ]
        )
        indices = torch.tensor([0, 3], dtype=torch.long)
        mask = torch.tensor(
            [
                [[True, True], [True, False]],
                [[True, True], [True, True]],
            ]
        )
        generator = torch.Generator().manual_seed(17)
        config = FlowSDEPPOConfig(num_denoising_steps=4, noise_level=0.5)

        rollout = sample_flow_sde_chunk(
            model,
            conditioning,
            horizon=2,
            action_dim=2,
            config=config,
            action_mask=mask,
            initial_noise=initial_noise,
            denoise_indices=indices,
            generator=generator,
        )
        self.assertEqual(rollout.chains.shape, (2, 5, 2, 2))
        self.assertEqual(rollout.actions.shape, (2, 2, 2))
        self.assertFalse(rollout.chains.requires_grad)
        self.assertTrue(bool(torch.isfinite(rollout.chains).all()))

        new_log_probs = recompute_flow_sde_log_probs(
            model,
            conditioning,
            rollout,
            config=config,
        )
        torch.testing.assert_close(new_log_probs, rollout.old_log_probs, rtol=1.0e-5, atol=1.0e-5)
        loss, metrics = ppo_clipped_actor_loss(
            new_log_probs,
            rollout.old_log_probs,
            torch.ones(2),
            rollout.action_mask,
            clip_ratio_low=config.clip_ratio_low,
            clip_ratio_high=config.clip_ratio_high,
        )
        torch.testing.assert_close(metrics["ratio"], torch.tensor(1.0), rtol=1.0e-5, atol=1.0e-5)
        loss.backward()
        self.assertIsNotNone(model.scale.grad)
        self.assertTrue(bool(torch.isfinite(model.scale.grad)))
        self.assertNotEqual(model.scale.grad.item(), 0.0)

    def test_each_batch_item_can_select_a_different_denoising_step(self):
        model = _TinyVelocity()
        conditioning = torch.zeros(4, 1)
        indices = torch.arange(4, dtype=torch.long)
        rollout = sample_flow_sde_chunk(
            model,
            conditioning,
            horizon=1,
            action_dim=1,
            config=FlowSDEPPOConfig(),
            initial_noise=torch.zeros(4, 1, 1),
            denoise_indices=indices,
            generator=torch.Generator().manual_seed(5),
        )
        torch.testing.assert_close(rollout.denoise_indices, indices)
        current, following = rollout.selected_transition()
        self.assertEqual(current.shape, (4, 1, 1))
        self.assertEqual(following.shape, (4, 1, 1))


if __name__ == "__main__":
    unittest.main()
