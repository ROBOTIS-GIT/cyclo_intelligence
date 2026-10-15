"""Validated vision-only overlay carried alongside a language LoRA adapter."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def apply_vision_checkpoint(visual,directory,model_id,revision,required=True):
    root=Path(directory).resolve()
    manifest=root/'vision_manifest.json'
    weights=root/'vision.safetensors'
    config=root/'vision_config.json'
    assets=(manifest,weights,config)
    if not any(p.exists() for p in assets):
        if required:raise ValueError('Vision checkpoint is missing')
        return None
    if not all(p.resolve().is_relative_to(root) and p.is_file() for p in assets):
        raise ValueError('Incomplete or unsafe vision checkpoint')
    metadata=json.loads(manifest.read_text())
    if metadata.get('schema')!='qwen_vision_tuning_v1' or metadata.get('base_model')!=model_id or metadata.get('revision')!=revision:
        raise ValueError('Vision checkpoint base model/revision mismatch')
    if file_hash(weights)!=metadata.get('sha256') or file_hash(config)!=metadata.get('config_sha256'):
        raise ValueError('Vision checkpoint hash mismatch')
    expected=json.loads(config.read_text())
    for key in ('hidden_size','depth','num_heads','patch_size','spatial_merge_size','temporal_patch_size','out_hidden_size'):
        if expected.get(key)!=getattr(visual.config,key,None):raise ValueError(f'Vision configuration mismatch: {key}')
    # Quantized visual Linear4bit parameters cannot accept full floating point weights.
    if any(type(m).__name__=='Linear4bit' for m in visual.modules()):
        raise ValueError('Load a vision overlay with visual modules excluded from quantization')
    from safetensors.torch import load_file
    state=load_file(str(weights),device='cpu')
    visual.load_state_dict(state,strict=True)
    visual.requires_grad_(False)
    return metadata
