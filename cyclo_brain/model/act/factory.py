"""Construction and checkpoint loading for the upstream LeRobot ACT model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import device as TorchDevice

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy


def create_act_model(config: "ACTConfig") -> "ACTPolicy":
    """Create the official LeRobot ``ACTPolicy`` without altering its graph."""

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    if not isinstance(config, ACTConfig):
        raise TypeError("ACT model config must be a LeRobot ACTConfig")
    return ACTPolicy(config).to(config.device)


def load_act_model(
    checkpoint: str | Path,
    *,
    device: str | "TorchDevice" | None = None,
    strict: bool = True,
) -> "ACTPolicy":
    """Load an official LeRobot ACT checkpoint in inference mode.

    Loading the config explicitly lets the requested device take effect before
    safetensors are read. The returned object remains an unwrapped
    ``ACTPolicy``, so its state-dict and ``save_pretrained`` format stay fully
    compatible with LeRobot training and the existing Cyclo inference engine.
    """

    # LeRobot 0.3 imports only the packaging root but accesses its ``version``
    # submodule during safetensors loading. Import it here so local checkpoint
    # loading is independent of whether another dependency happened to import
    # that submodule first.
    import packaging.version  # noqa: F401
    import torch

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, ACTConfig):
        raise TypeError("ACT checkpoint config must have policy type 'act'")
    if device is not None:
        config.device = str(torch.device(device))
    # Local checkpoints strictly restore the complete visual backbone. Avoid
    # torchvision downloading an initialization that would immediately be
    # overwritten, while retaining the saved config value on the loaded actor.
    saved_backbone_weights = config.pretrained_backbone_weights
    config.pretrained_backbone_weights = None
    try:
        policy = ACTPolicy.from_pretrained(
            checkpoint,
            config=config,
            strict=strict,
        )
    finally:
        config.pretrained_backbone_weights = saved_backbone_weights
    policy.eval()
    return policy
