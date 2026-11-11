"""Episode-boundary masking without changing native loss weights or heads."""

from dataclasses import dataclass
from types import MethodType

import torch


@dataclass(frozen=True)
class MaskedTarget:
    values: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self):
        if self.valid.dtype != torch.bool or self.valid.shape != self.values.shape[:1]:
            raise ValueError("Teacher validity must be one boolean per sample")
        if self.values.ndim < 2 or not torch.isfinite(self.values).all():
            raise ValueError("Teacher targets must be finite batched features")
        if self.values.requires_grad:
            raise ValueError("Teacher targets must be detached")


def _masked_loss(original, predictions, target, *args, **kwargs):
    if not isinstance(target, MaskedTarget):
        return original(predictions, target, *args, **kwargs)
    if predictions.shape != target.values.shape:
        raise ValueError("Teacher and prediction shapes differ")
    valid = target.valid.to(device=predictions.device)
    values = target.values.to(device=predictions.device)
    if valid.any():
        return original(predictions[valid], values[valid], *args, **kwargs)
    # Keep the same metric names and autograd connection, but never take a mean
    # over an empty tensor. Clamped future frames must contribute no gradient.
    result = original(predictions[:1], values[:1], *args, **kwargs)
    if isinstance(result, tuple):
        loss, metrics = result
        return loss * 0, {name: value * 0 for name, value in metrics.items()}
    return result * 0


def install_boundary_losses(flow_model):
    """Adapt only this model instance, preserving native parameter names.

    Current targets remain tensors; only future targets carry MaskedTarget.
    The pinned upstream calls these two loss methods after producing features.
    """
    if getattr(flow_model, "_cyclo_boundary_losses", False):
        raise ValueError("Boundary losses are already installed")
    if getattr(flow_model, "use_future_video_cls", False):
        raise ValueError("Future CLS masking is not qualified; use the native patch recipe")
    emb_loss = flow_model._emb_loss
    video_loss = flow_model._video_emb_loss

    def emb(self, predictions, targets):
        return _masked_loss(emb_loss, predictions, targets)

    def video(self, predictions, targets, metric_prefix="future_video"):
        return _masked_loss(video_loss, predictions, targets, metric_prefix=metric_prefix)

    flow_model._emb_loss = MethodType(emb, flow_model)
    flow_model._video_emb_loss = MethodType(video, flow_model)
    flow_model._cyclo_boundary_losses = True


def prepare_loss_inputs(batch, *, depth, future_depth, video):
    """Attach frozen teachers and combine action time/channel padding masks."""
    result = dict(batch)
    future_valid = result.pop("future_valid")
    mask = result["joint_mask"]
    padding = result["action_is_pad"]
    actions = result["actions"]
    if (
        mask.dtype != torch.bool or padding.dtype != torch.bool
        or mask.shape != actions.shape or padding.shape != actions.shape[:2]
        or future_valid.dtype != torch.bool or future_valid.shape != actions.shape[:1]
    ):
        raise ValueError("Invalid action or future padding mask")
    result["joint_mask"] = mask & ~padding[..., None]
    if not result["joint_mask"].flatten(1).any(dim=1).all():
        raise ValueError("Every sample requires at least one valid action element")
    if not isinstance(video, dict) or "patch" not in video or "current_patch" not in video:
        raise ValueError("Native future and current video patch targets are required")
    for target in (depth, future_depth, video["patch"], video["current_patch"]):
        if target.requires_grad or not torch.isfinite(target).all():
            raise ValueError("Teachers must produce finite detached targets")
    result.update(
        depth_targets=depth,
        future_depth_targets=MaskedTarget(future_depth, future_valid),
        future_video_targets=MaskedTarget(video["patch"], future_valid),
        future_video_current_patch=video["current_patch"],
    )
    return result
