"""Stable operations over the unmodified upstream LeRobot ACT policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

    from lerobot.policies.act.modeling_act import ACTPolicy


def _require_act_policy(policy: Any) -> "ACTPolicy":
    from lerobot.policies.act.modeling_act import ACTPolicy

    if not isinstance(policy, ACTPolicy):
        raise TypeError("Expected the official LeRobot ACTPolicy")
    return policy


def _prepare_model_batch(
    policy: "ACTPolicy",
    batch: Mapping[str, "Tensor"],
) -> dict[str, Any]:
    """Mirror ACTPolicy's public image-key packing for its inner model."""

    from lerobot.utils.constants import OBS_IMAGES

    model_batch = dict(batch)
    if policy.config.image_features:
        model_batch[OBS_IMAGES] = [
            model_batch[key] for key in policy.config.image_features
        ]
    return model_batch


def compute_act_bc_loss(
    policy: "ACTPolicy",
    batch: Mapping[str, "Tensor"],
) -> tuple["Tensor", dict[str, float]]:
    """Delegate BC loss computation to LeRobot's official ACT forward pass."""

    policy = _require_act_policy(policy)
    if not policy.training:
        raise RuntimeError("ACT BC loss requires policy.train() mode")
    return policy(dict(batch))


def predict_act_action_chunk(
    policy: "ACTPolicy",
    batch: Mapping[str, "Tensor"],
) -> "Tensor":
    """Run the official no-gradient ACT inference path."""

    policy = _require_act_policy(policy)
    return policy.predict_action_chunk(dict(batch))


def differentiable_act_action_chunk(
    policy: "ACTPolicy",
    batch: Mapping[str, "Tensor"],
) -> "Tensor":
    """Return the deployed zero-latent ACT chunk while preserving gradients.

    LeRobot's public inference method is intentionally decorated with
    ``torch.no_grad``. TD3 needs the same deterministic policy action with an
    autograd graph. ACT itself defines inference as an eval-mode forward with
    no action target, which selects an all-zero latent. This adapter invokes
    that existing inner model directly; it does not reproduce the ACT network.

    The caller must put the policy in eval mode explicitly. This keeps the
    VAE/dropout mode transition visible at the BC/TD3 orchestration boundary
    instead of silently changing it inside a model operation.
    """

    policy = _require_act_policy(policy)
    if policy.training:
        raise RuntimeError("Differentiable ACT inference requires policy.eval() mode")
    action_chunk, _latent_parameters = policy.model(
        _prepare_model_batch(policy, batch)
    )
    return action_chunk
