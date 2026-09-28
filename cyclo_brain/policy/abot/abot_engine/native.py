"""Load unmodified upstream modules; only relocate the declared VLM asset."""

import os
from pathlib import Path


def load_policy(bundle):
    import torch
    from omegaconf import OmegaConf
    from ABot.model.framework.ABot_M0 import ABot_M0

    config = OmegaConf.create(bundle.config)
    base = os.environ.get("ABOT_BASE_VLM") or config.framework.qwenvl.base_vlm
    if base.startswith(("/", "./", "../")):
        path = Path(base)
        if not path.is_absolute():
            path = bundle.path / path
        if not path.is_dir():
            raise ValueError("Base VLM path is unavailable; set ABOT_BASE_VLM to its local path or Hub ID")
        base = str(path.resolve())
    config.framework.qwenvl.base_vlm = base
    config.trainer.pretrained_checkpoint = None
    policy = ABot_M0(config)
    weights = torch.load(bundle.checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(weights, strict=True)
    del weights
    return policy.to(dtype=torch.bfloat16, device="cuda").eval()
