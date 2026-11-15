"""Inference-only scheduling optimizations for GR00T's Qwen3 backbone.

Keep the conditional wrapper's hidden-state recorder: calling the inner model
directly changes the last-layer normalization semantics of this checkpoint.
Only unused language logits and repeated structural metadata are skipped.
Camera pixels, vision features, and language features are never cached.
"""

from contextvars import ContextVar
from functools import wraps
from types import MethodType

import torch


def _tensor_key(value):
    return (str(value.device), str(value.dtype), tuple(value.shape),
            tuple(value.detach().cpu().reshape(-1).tolist()))


def _parameter_key(value):
    try:
        version = value._version
    except RuntimeError:  # Inference-created tensors have no version counter.
        return None
    return (str(value.device), str(value.dtype), tuple(value.shape),
            value.data_ptr(), version)


def _stream_key(value):
    return (torch.cuda.current_stream(value.device).cuda_stream
            if value.is_cuda else None)


def _flash_context(registry):
    original = registry["flash_attention_2"]
    context = getattr(original, "_cyclo_vision_context", None)
    if context is not None:
        return context
    context = ContextVar("cyclo_vision_attention", default=None)

    @wraps(original)
    def dispatch(module, *args, **kwargs):
        active = context.get()
        if active is not None and id(module) in active[0]:
            # HF otherwise converts a GPU scalar to a Python length at every
            # vision layer, forcing repeated GPU/CPU synchronization.
            kwargs["max_length_q"] = active[1]
            kwargs["max_length_k"] = active[1]
        return original(module, *args, **kwargs)

    dispatch._cyclo_vision_context = context
    registry.register("flash_attention_2", dispatch)
    return context


class _InferencePath:
    def __init__(self, backbone, batch_feature, registry):
        self.backbone = backbone
        self.visual = backbone.model.visual
        self.batch_feature = batch_feature
        self.original_forward = backbone.forward
        self.original_visual = self.visual.forward
        self.original_positions = {
            name: getattr(self.visual, name)
            for name in ("rot_pos_emb", "fast_pos_embed_interpolate")
        }
        self.attention_ids = frozenset(id(block.attn) for block in self.visual.blocks)
        self.flash_context = _flash_context(registry)
        self.grid_context = ContextVar("cyclo_vision_grid", default=None)
        # One entry per operation: changing instructions/shapes cannot grow
        # memory indefinitely. Values are never mutated after publication.
        self.cache = {}

    def _cached(self, name, key, calculate):
        entry = self.cache.get(name)
        if entry is None or entry[0] != key:
            entry = (key, calculate())
            self.cache[name] = entry
        return entry[1]

    def forward(self, vl_input):
        if self.backbone.training or torch.is_grad_enabled():
            return self.original_forward(vl_input)
        self.backbone.set_frozen_modules_to_eval_mode()
        inputs = {k: vl_input[k] for k in (
            "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
        )}
        body = self.backbone.model.model
        key = (_stream_key(inputs["input_ids"]),) + tuple(_tensor_key(inputs[k]) for k in (
            "input_ids", "attention_mask", "image_grid_thw",
        ))
        positions, deltas = self._cached("rope", key, lambda: body.get_rope_index(
            inputs["input_ids"], inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
        ))
        inputs["position_ids"] = positions
        body.rope_deltas = deltas
        # Empty sequence selection avoids the unused vocabulary projection
        # while preserving the original hidden-state capture path exactly.
        output = self.backbone.model(
            **inputs, output_hidden_states=True, use_cache=False,
            logits_to_keep=torch.empty(0, dtype=torch.long, device=inputs["input_ids"].device),
        )
        return self.batch_feature(data={
            "backbone_features": output.hidden_states[-1],
            "backbone_attention_mask": inputs["attention_mask"] == 1,
            "image_mask": inputs["input_ids"] == self.backbone.model.config.image_token_id,
        })

    def vision_forward(self, hidden_states, grid_thw, **kwargs):
        if self.visual.training or torch.is_grad_enabled():
            return self.original_visual(hidden_states, grid_thw, **kwargs)
        grid = tuple(tuple(row) for row in grid_thw.detach().cpu().tolist())
        key = (str(grid_thw.device), str(grid_thw.dtype), grid)
        # Cache tensors are owned by their producing CUDA stream. Recompute
        # when the caller changes streams rather than sharing pending work.
        stream = _stream_key(hidden_states)
        grid_token = self.grid_context.set((key, stream))
        flash_token = self.flash_context.set((self.attention_ids, max(h * w for _, h, w in grid)))
        try:
            return self.original_visual(hidden_states, grid_thw, **kwargs)
        finally:
            self.flash_context.reset(flash_token)
            self.grid_context.reset(grid_token)

    def positions(self, name, grid_thw):
        active = self.grid_context.get()
        original = self.original_positions[name]
        if active is None:
            return original(grid_thw)
        tensor = (self.visual.pos_embed.weight if name == "fast_pos_embed_interpolate"
                  else self.visual.rotary_pos_emb.inv_freq)
        weights = _parameter_key(tensor)
        if weights is None:
            return original(grid_thw)
        key = (active, weights, self.visual.spatial_merge_size,
               self.visual.num_grid_per_side)
        return self._cached(name, key, lambda: original(grid_thw))


def optimize_qwen3_backbone(policy) -> bool:
    """Install only on the known GR00T/Qwen3 interface; leave other models alone."""
    backbone = getattr(getattr(policy, "model", None), "backbone", None)
    if (type(backbone).__name__ != "Qwen3Backbone"
            or type(backbone).__module__ != "gr00t.model.modules.qwen3_backbone"):
        return False
    model = getattr(backbone, "model", None)
    if (type(model).__name__ != "Qwen3VLForConditionalGeneration"
            or type(getattr(model, "model", None)).__name__ != "Qwen3VLModel"):
        return False
    if getattr(backbone, "_cyclo_inference_path", None) is not None:
        return True
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.models.qwen3_vl.modeling_qwen3_vl import ALL_ATTENTION_FUNCTIONS

    path = _InferencePath(backbone, BatchFeature, ALL_ATTENTION_FUNCTIONS)
    backbone._cyclo_inference_path = path
    backbone.forward = path.forward
    path.visual.forward = path.vision_forward
    for name in path.original_positions:
        def cached(_self, grid_thw, _name=name):
            return path.positions(_name, grid_thw)
        setattr(path.visual, name, MethodType(cached, path.visual))
    return True
