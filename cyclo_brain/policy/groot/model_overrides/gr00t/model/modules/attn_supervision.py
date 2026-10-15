"""Last prompt-query to image-key attention, supervised by reviewed object masks."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv


def vision_token_order(grid_h, grid_w, merge_size=2):
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Grid must be divisible by spatial_merge_size")
    return (
        torch.arange(grid_h * grid_w)
        .reshape(grid_h // merge_size, merge_size, grid_w // merge_size, merge_size)
        .permute(0, 2, 1, 3)
        .reshape(-1)
    )


def mask_to_targets(masks, grid_thw, merge_size=2):
    if len(masks) != len(grid_thw):
        raise ValueError("One mask is required for each image")
    rows = []
    for mask, grid in zip(masks, grid_thw):
        t, gh, gw = map(int, grid)
        if t != 1:
            raise ValueError("Only single-frame image grids (t=1) are supported")
        order = vision_token_order(gh, gw, merge_size).to(mask.device)
        coverage = F.interpolate(mask.float()[None, None], size=(gh, gw), mode="area")[0, 0]
        rows.append(coverage.flatten()[order].reshape(-1, merge_size**2).mean(-1).clamp(0, 1))
    return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=-1)


def last_valid_queries(attention_mask):
    valid = attention_mask.bool()
    if not valid.any(-1).all():
        raise ValueError("Empty prompt")
    return (
        torch.arange(valid.shape[1], device=valid.device)
        .expand_as(valid)
        .masked_fill(~valid, -1)
        .max(-1)
        .values
    )


class LanguageAttnSupervisor:
    def __init__(self, language_model, layers=(-1,), image_token_id=None, merge_size=2):
        self.layers = [i % len(language_model.layers) for i in layers]
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Duplicate attention layers")
        if any(not -len(language_model.layers) <= i < len(language_model.layers) for i in layers):
            raise ValueError("Attention layer outside retained decoder")
        self.image_token_id = image_token_id
        self.merge_size = merge_size
        self.active = False
        self.received = {}
        self._handles = [
            language_model.layers[i].self_attn.register_forward_pre_hook(
                self._hook(i), with_kwargs=True
            )
            for i in self.layers
        ]

    def begin(self, inputs):
        ids, valid = inputs["input_ids"], inputs["attention_mask"].bool()
        self.queries = last_valid_queries(valid)
        self.keys = [
            (ids[b].eq(self.image_token_id) & valid[b]).nonzero().flatten() for b in range(len(ids))
        ]
        grids = inputs["image_grid_thw"]
        if len(grids) != len(ids):
            raise ValueError("Baseline requires exactly one image per prompt")
        for b, (t, gh, gw) in enumerate(grids.tolist()):
            if t != 1 or gh % self.merge_size or gw % self.merge_size:
                raise ValueError("Unsupported temporal/spatial image grid")
            keys = self.keys[b]
            if len(keys) != gh * gw // self.merge_size**2 or not len(keys):
                raise ValueError("Image token count/grid mismatch")
            if not torch.all(keys[1:] == keys[:-1] + 1) or keys[-1] >= self.queries[b]:
                raise ValueError("Image keys must form one contiguous group before the query")
        self.received = {}
        self.active = True

    def _hook(self, layer):
        def hook(module, args, kwargs):
            if not self.active:
                return
            if kwargs.get("past_key_values") is not None:
                raise ValueError("Attention supervision does not support KV cache")
            hidden = args[0] if args else kwargs["hidden_states"]
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            q = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            k = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
            cos, sin = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, module.num_key_value_groups)
            rows = []
            for b, keys in enumerate(self.keys):
                scores = (
                    q[b, :, self.queries[b]].float().unsqueeze(1) * k[b, :, keys].float()
                ).sum(-1) * module.scaling
                rows.append(scores.softmax(-1).mean(0))
            self.received[layer] = rows

        return hook

    def end(self):
        self.active = False

    def loss(self, targets, loss_type="kl", margin=None, eps=1e-8):
        if loss_type != "kl":
            raise ValueError("This pipeline implements KL(GT || attention) only")
        terms = []
        if set(self.received) != set(self.layers):
            raise RuntimeError("Supervised decoder hooks did not run")
        for layer in self.layers:
            if len(self.received[layer]) != len(targets):
                raise ValueError("Attention/target image count mismatch")
            for a, row in zip(self.received[layer], targets):
                cov = row[row >= 0].to(a.device).float()
                if cov.numel() != a.numel():
                    raise ValueError("Mask coverage/image token count mismatch")
                if cov.sum() <= 0:
                    continue
                p = cov / cov.sum()
                terms.append((p * (p.clamp_min(eps).log() - a.float().clamp_min(eps).log())).sum())
        if not terms:
            raise ValueError("Entire batch has empty target masks")
        return torch.stack(terms).mean()

    def close(self):
        self.end()
        for handle in self._handles:
            handle.remove()
        self._handles = []

    remove = close
