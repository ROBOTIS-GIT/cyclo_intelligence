"""Differentiable last-prompt-query -> image attention for Qwen3.5.

Conditional softmax is over image keys PER HEAD, followed by head averaging.
Qwen3.5's query gate and partial RoPE differ from the Qwen3-VL reference.
"""
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def query_and_images(input_ids, attention_mask, grid_thw, image_token_id, merge_size):
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError('Expected matching [batch, sequence] prompt IDs and mask')
    if grid_thw.shape != (input_ids.shape[0], 3):
        raise ValueError('Exactly one image per sample is required')
    result = []
    for ids, valid, grid in zip(input_ids, attention_mask.bool(), grid_thw):
        positions = torch.where(valid)[0]
        if not positions.numel():
            raise ValueError('Empty prompt')
        query = int(positions[-1])  # works for left AND right padding
        images = torch.where((ids == image_token_id) & valid)[0]
        t, h, w = map(int, grid)
        if t != 1 or h % merge_size or w % merge_size:
            raise ValueError('Unsupported still-image grid')
        if images.numel() != h * w // merge_size**2:
            raise ValueError('Image token count does not match merged grid')
        if images.numel() == 0 or int(images[-1]) >= query:
            raise ValueError('Prompt query must follow every image token')
        if images.numel() > 1 and not torch.all(images.diff() == 1):
            raise ValueError('Multiple image groups are not supported')
        result.append((query, images))
    return result


def conditional_image_attention(module, hidden, position_embeddings, locations):
    shape = (*hidden.shape[:-1], -1, module.head_dim)
    # The second half of EACH Qwen3.5 query head is an output gate, not a query.
    query, _gate = module.q_proj(hidden).view(
        *hidden.shape[:-1], -1, 2 * module.head_dim).chunk(2, dim=-1)
    query = module.q_norm(query.reshape(shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    result = []
    for batch, (position, images) in enumerate(locations):
        q = query[batch, :, position].float()
        k = key[batch].index_select(1, images.to(key.device)).float()
        # Casting inputs alone is insufficient: autocast would lower einsum again.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            scores = torch.einsum('hd,hkd->hk', q, k) * module.scaling
            result.append(scores.softmax(dim=-1).mean(dim=0))
    return result


class PreviewAttention:
    """Capture the last prompt query once during generate's prefill, ignore decode."""
    def __init__(self, model, inputs):
        base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        layers = base.model.language_model.layers
        self.layer = max(i for i, layer in enumerate(layers) if layer.layer_type == 'full_attention')
        self.locations = query_and_images(inputs['input_ids'], inputs['attention_mask'],
            inputs['image_grid_thw'], base.config.image_token_id,
            base.config.vision_config.spatial_merge_size)
        t,h,w = map(int, inputs['image_grid_thw'][0])
        m = base.config.vision_config.spatial_merge_size
        self.shape = (h//m,w//m)
        self.grid = None
        self.error = None
        self.attempted = False
        self.handle = layers[self.layer].self_attn.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if self.attempted: return
        self.attempted = True
        try:
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            self.grid = conditional_image_attention(module, hidden, kwargs['position_embeddings'],
                self.locations)[0].detach().float().cpu().reshape(self.shape).numpy()
        except Exception as exc:
            self.error = str(exc)

    def remove(self):
        self.handle.remove()

    def render(self, image):
        import base64, io, json
        from pathlib import Path
        import numpy as np
        from PIL import Image
        if self.grid is None: raise ValueError(self.error or 'No prefill attention captured')
        # Same fixed scale and TURBO palette as reasoning_tuning/video.py.
        # Uniform attention stays at 0.25; never normalize each frame to its peak.
        density = np.clip(self.grid*self.grid.size/4.,0,1)
        density = np.asarray(Image.fromarray(density).resize(image.size, Image.Resampling.BILINEAR))
        palette = np.asarray(json.loads(Path(__file__).with_name('turbo_palette.json').read_text()),dtype=np.uint8)
        heat = Image.fromarray(palette[(density*255).astype(np.uint8)])
        overlay = Image.blend(image.convert('RGB'),heat,.5)
        overlay.thumbnail((640,480))
        output=io.BytesIO();overlay.save(output,format='JPEG',quality=85)
        return {'image_base64':base64.b64encode(output.getvalue()).decode(), 'mime_type':'image/jpeg',
            'layer':self.layer,'method':'last_prompt_query_to_image','scale':'fixed_density_4',
            'grid_height':self.shape[0],'grid_width':self.shape[1]}
