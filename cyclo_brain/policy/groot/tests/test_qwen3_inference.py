"""Cache invalidation and wrapper semantics without a GPU or model weights."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    'qwen3_inference_under_test',
    Path(__file__).resolve().parents[1] / 'runtime/qwen3_inference.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Registry(dict):
    def register(self, name, function):
        self[name] = function


class Visual:
    training = False
    spatial_merge_size = 2
    num_grid_per_side = 4

    def __init__(self, registry):
        self.registry = registry
        self.blocks = [types.SimpleNamespace(attn=object())]
        self.pos_embed = types.SimpleNamespace(weight=torch.ones(1))
        self.rotary_pos_emb = types.SimpleNamespace(inv_freq=torch.ones(1))
        self.calls = {'rot': 0, 'pos': 0}
        self.fail = False

    def rot_pos_emb(self, grid):
        self.calls['rot'] += 1
        return grid.sum() + self.rotary_pos_emb.inv_freq.clone()

    def fast_pos_embed_interpolate(self, grid):
        self.calls['pos'] += 1
        return grid.sum() + self.pos_embed.weight.clone()

    def forward(self, pixels, grid_thw, **kwargs):
        self.registry['flash_attention_2'](self.blocks[0].attn, max_length_q=999, max_length_k=999)
        if self.fail:
            raise RuntimeError('vision failed')
        return pixels + self.rot_pos_emb(grid_thw) + self.fast_pos_embed_interpolate(grid_thw)


class Qwen3VLModel:
    def __init__(self):
        self.rope_calls = 0

    def get_rope_index(self, ids, grid, attention_mask):
        self.rope_calls += 1
        return ids.clone(), grid.sum().clone()

    def __call__(self, **kwargs):
        raise AssertionError('must retain the conditional wrapper hidden-state recorder')


class Qwen3VLForConditionalGeneration:
    def __init__(self, registry):
        self.model = Qwen3VLModel()
        self.visual = Visual(registry)
        self.config = types.SimpleNamespace(image_token_id=2)
        self.calls = []

    def __call__(self, **kwargs):
        assert kwargs['output_hidden_states'] is True
        assert kwargs['use_cache'] is False
        assert kwargs['logits_to_keep'].numel() == 0
        self.calls.append(kwargs)
        result = self.visual.forward(kwargs['pixel_values'], kwargs['image_grid_thw'])
        return types.SimpleNamespace(hidden_states=[result])


class Qwen3Backbone:
    __module__ = 'gr00t.model.modules.qwen3_backbone'
    training = False

    def __init__(self, registry):
        self.model = Qwen3VLForConditionalGeneration(registry)
        self.original_calls = 0

    def set_frozen_modules_to_eval_mode(self):
        pass

    def forward(self, inputs):
        self.original_calls += 1
        return 'original'


def inputs():
    return dict(input_ids=torch.tensor([[1, 2, 3, 4]]),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                pixel_values=torch.ones(1))


class Qwen3InferenceTests(unittest.TestCase):
    def setUp(self):
        self.flash_calls = []
        self.registry = Registry(flash_attention_2=lambda obj, **kw: self.flash_calls.append((obj, kw)))
        feature = types.ModuleType('transformers.feature_extraction_utils')
        feature.BatchFeature = lambda data: data
        modeling = types.ModuleType('transformers.models.qwen3_vl.modeling_qwen3_vl')
        modeling.ALL_ATTENTION_FUNCTIONS = self.registry
        self.stub = patch.dict(sys.modules, {
            'transformers.feature_extraction_utils': feature,
            'transformers.models.qwen3_vl.modeling_qwen3_vl': modeling,
        })
        self.stub.start()
        self.addCleanup(self.stub.stop)
        self.backbone = Qwen3Backbone(self.registry)
        self.policy = types.SimpleNamespace(model=types.SimpleNamespace(backbone=self.backbone))
        self.assertTrue(module.optimize_qwen3_backbone(self.policy))

    def run_forward(self, value=None):
        with torch.no_grad():
            return self.backbone.forward(inputs() if value is None else value)

    def test_pixels_are_always_fresh_and_wrapper_is_preserved(self):
        first = self.run_forward()
        changed = inputs()
        changed['pixel_values'] += 5
        second = self.run_forward(changed)
        torch.testing.assert_close(second['backbone_features'] - first['backbone_features'], torch.tensor([5.]))
        self.assertEqual(self.backbone.model.model.rope_calls, 1)
        self.assertEqual(self.backbone.model.visual.calls, {'rot': 1, 'pos': 1})
        self.assertTrue(torch.equal(second['image_mask'], torch.tensor([[False, True, False, False]])))

    def test_same_shape_changed_content_invalidates_structural_cache(self):
        self.run_forward()
        for key in ('input_ids', 'attention_mask', 'image_grid_thw'):
            value = inputs()
            value[key].reshape(-1)[-1] += 1
            self.run_forward(value)
        self.assertEqual(self.backbone.model.model.rope_calls, 4)

    def test_parameter_updates_invalidate_only_affected_positions(self):
        self.run_forward()
        visual = self.backbone.model.visual
        with torch.no_grad():
            visual.pos_embed.weight.add_(1)
        second = self.run_forward()
        self.assertEqual(visual.calls, {'rot': 1, 'pos': 2})
        with torch.no_grad():
            visual.rotary_pos_emb.inv_freq.add_(1)
        third = self.run_forward()
        self.assertEqual(visual.calls, {'rot': 2, 'pos': 2})
        torch.testing.assert_close(third['backbone_features'] - second['backbone_features'], torch.ones(1))

    def test_training_and_gradients_use_original_path(self):
        self.assertEqual(self.backbone.forward(inputs()), 'original')
        self.backbone.training = True
        self.assertEqual(self.run_forward(), 'original')
        self.assertEqual(self.backbone.original_calls, 2)
        self.assertEqual(self.backbone.model.calls, [])

    def test_attention_lengths_are_scoped_and_reset_after_exception(self):
        self.run_forward()
        self.assertEqual(self.flash_calls[-1][1]['max_length_q'], 4)
        visual = self.backbone.model.visual
        visual.fail = True
        with self.assertRaisesRegex(RuntimeError, 'vision failed'):
            self.run_forward()
        self.assertIsNone(self.backbone._cyclo_inference_path.flash_context.get())
        self.assertIsNone(self.backbone._cyclo_inference_path.grid_context.get())
        self.registry['flash_attention_2'](visual.blocks[0].attn, max_length_q=123)
        self.assertEqual(self.flash_calls[-1][1]['max_length_q'], 123)

    def test_cache_is_bounded_and_installation_idempotent(self):
        path = self.backbone._cyclo_inference_path
        self.assertTrue(module.optimize_qwen3_backbone(self.policy))
        self.assertIs(self.backbone._cyclo_inference_path, path)
        for i in range(10):
            value = inputs()
            value['image_grid_thw'][0, 2] += i
            self.run_forward(value)
        self.assertEqual(set(path.cache), {'rope', 'rot_pos_emb', 'fast_pos_embed_interpolate'})

    def test_unrelated_models_are_unchanged(self):
        other = types.SimpleNamespace(forward=object())
        forward = other.forward
        self.assertFalse(module.optimize_qwen3_backbone(types.SimpleNamespace(model=types.SimpleNamespace(backbone=other))))
        self.assertIs(other.forward, forward)


if __name__ == '__main__':
    unittest.main()
