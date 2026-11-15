"""Host-only contract tests for the upstream-independent MultiTaskDiT adapter."""

import unittest

import torch
from torch import nn

from cyclo_brain.model.multi_task_dit import (
    CYCLO_SG2_CAMERA_KEYS,
    MultiTaskDiTFlowAdapter,
    MultiTaskDiTValueHead,
    with_default_task_instruction,
)


class _Feature:
    def __init__(self, shape):
        self.shape = shape


class _Config:
    objective = "flow_matching"
    sigma_min = 0.0
    horizon = 4
    n_obs_steps = 1
    n_action_steps = 3
    image_features = {key: _Feature((3, 8, 8)) for key in CYCLO_SG2_CAMERA_KEYS}
    robot_state_feature = _Feature((2,))
    action_feature = _Feature((2,))


class _ObservationEncoder(nn.Module):
    conditioning_dim = 7

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def encode(self, batch):
        state = batch["observation.state"][:, 0]
        images = batch["observation.images"][:, 0]
        camera_means = images.mean(dim=(2, 3, 4))
        tokens = batch["observation.language.tokens"].float().mean(dim=1, keepdim=True)
        attention = batch["observation.language.attention_mask"].float().mean(dim=1, keepdim=True)
        return self.scale * torch.cat([state, camera_means, tokens, attention], dim=1)


class _NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(0.5)

    def forward(self, latent, progress, conditioning):
        bias = conditioning[:, :1, None]
        return self.dropout(self.scale * latent) + progress[:, None, None] + bias


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.observation_encoder = _ObservationEncoder()
        self.noise_predictor = _NoisePredictor()

    def _prepare_batch(self, batch):
        result = dict(batch)
        result["observation.images"] = torch.stack(
            [result[key] for key in self.config.image_features], dim=-4
        )
        return result


def _processed_batch(batch_size=2):
    batch = {
        "observation.state": torch.randn(batch_size, 1, 2),
        "observation.language.tokens": torch.ones(batch_size, 4, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
    }
    for index, key in enumerate(CYCLO_SG2_CAMERA_KEYS):
        batch[key] = torch.full((batch_size, 1, 3, 8, 8), float(index + 1))
    return batch


class MultiTaskDiTFlowAdapterTest(unittest.TestCase):
    def test_three_cameras_state_and_language_form_conditioning(self):
        policy = _Policy()
        adapter = MultiTaskDiTFlowAdapter(policy)
        conditioning = adapter.encode_conditioning(_processed_batch())

        self.assertEqual(conditioning.shape, (2, 7))
        self.assertFalse(conditioning.requires_grad)
        self.assertFalse(policy.observation_encoder.scale.requires_grad)
        torch.testing.assert_close(conditioning[:, 2:5], torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1))

    def test_language_tokens_are_required_by_current_upstream_contract(self):
        adapter = MultiTaskDiTFlowAdapter(_Policy())
        batch = _processed_batch()
        batch.pop("observation.language.tokens")
        with self.assertRaisesRegex(ValueError, "language.tokens"):
            adapter.encode_conditioning(batch)

    def test_velocity_is_repeatable_and_action_head_receives_gradients(self):
        policy = _Policy()
        adapter = MultiTaskDiTFlowAdapter(policy)
        conditioning = adapter.encode_conditioning(_processed_batch())
        latent = torch.randn(2, 4, 2)
        progress = torch.tensor([0.0, 0.75])
        first = adapter.velocity(latent, progress, conditioning)
        second = adapter.velocity(latent, progress, conditioning)
        torch.testing.assert_close(first, second)
        first.sum().backward()
        self.assertIsNotNone(policy.noise_predictor.scale.grad)
        self.assertFalse(policy.noise_predictor.training)

    def test_executed_mask_matches_upstream_action_slice(self):
        adapter = MultiTaskDiTFlowAdapter(_Policy())
        mask = adapter.executed_action_mask(2, device="cpu")
        self.assertEqual(mask.shape, (2, 4, 2))
        self.assertTrue(bool(mask[:, :3].all()))
        self.assertFalse(bool(mask[:, 3:].any()))
        chunk = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        torch.testing.assert_close(adapter.executed_actions(chunk), chunk[:, :3])

    def test_blank_task_uses_cyclo_default_before_tokenization(self):
        raw = {"observation.state": torch.zeros(2, 2), "task": ["", "pick jelly"]}
        resolved = with_default_task_instruction(raw)
        self.assertEqual(resolved["task"], ["ACT_dataset", "pick jelly"])

    def test_value_head_detaches_frozen_conditioning(self):
        head = MultiTaskDiTValueHead(7, hidden_dims=(4,))
        conditioning = torch.randn(2, 7, requires_grad=True)
        values = head(conditioning)
        self.assertEqual(values.shape, (2,))
        values.sum().backward()
        self.assertIsNone(conditioning.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))


if __name__ == "__main__":
    unittest.main()
