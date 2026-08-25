"""Pure Flow-SDE transition and PPO equations.

The transition is expressed in the Cyclo/GR00T time direction: ``t=0`` is
Gaussian noise and ``t=1`` is the normalized action chunk. It is the
noise-to-data form of the ODE-to-SDE construction used by RLinf Flow-SDE PPO.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _require_floating_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"Flow-SDE {name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Flow-SDE {name} must be finite")


def flow_sde_transition_stats(
    current: Tensor,
    velocity: Tensor,
    step_indices: Tensor,
    *,
    num_steps: int,
    noise_level: float,
    stochastic_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return Gaussian mean/std for one noise-to-action denoising step.

    The straight flow path must use ``sigma_min=0``. For entries where
    ``stochastic_mask`` is false, the function returns the exact Euler ODE
    step and zero standard deviation.
    """

    _require_floating_tensor(current, "current latent")
    _require_floating_tensor(velocity, "velocity")
    if current.shape != velocity.shape or current.ndim != 3:
        raise ValueError("Flow-SDE current and velocity must share shape (B, H, A)")
    if current.device != velocity.device:
        raise ValueError("Flow-SDE current and velocity must share one device")
    batch_size = current.shape[0]
    if (
        not isinstance(step_indices, Tensor)
        or step_indices.shape != (batch_size,)
        or step_indices.dtype != torch.long
        or step_indices.device != current.device
    ):
        raise ValueError("Flow-SDE step_indices must be int64 with shape (B,) on the latent device")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 2:
        raise ValueError("Flow-SDE num_steps must be an integer >= 2")
    if bool((step_indices < 0).any()) or bool((step_indices >= num_steps).any()):
        raise ValueError("Flow-SDE step index is outside the denoising schedule")
    if (
        isinstance(noise_level, bool)
        or not isinstance(noise_level, (int, float))
        or not math.isfinite(float(noise_level))
        or noise_level < 0.0
    ):
        raise ValueError("Flow-SDE noise_level must be finite and non-negative")
    if stochastic_mask is None:
        resolved_stochastic_mask = torch.ones(batch_size, dtype=torch.bool, device=current.device)
    elif (
        not isinstance(stochastic_mask, Tensor)
        or stochastic_mask.shape != (batch_size,)
        or stochastic_mask.dtype != torch.bool
        or stochastic_mask.device != current.device
    ):
        raise ValueError("Flow-SDE stochastic_mask must be boolean with shape (B,)")
    else:
        resolved_stochastic_mask = stochastic_mask

    # Log-probability arithmetic is deliberately fp32 even when the velocity
    # model itself runs in bf16.
    current_f = current.float()
    velocity_f = velocity.float()
    dt = 1.0 / float(num_steps)
    progress = step_indices.to(torch.float32) * dt
    next_progress = progress + dt
    progress_3d = progress[:, None, None]
    next_progress_3d = next_progress[:, None, None]

    noise_prediction = current_f - progress_3d * velocity_f
    action_prediction = current_f + (1.0 - progress_3d) * velocity_f

    # The t=0 SDE coefficient has a removable singularity. Matching RLinf's
    # discrete schedule, use the first positive time (dt) as its denominator.
    progress_denominator = torch.where(progress == 0.0, next_progress, progress)
    diffusion_ratio = (1.0 - progress) / progress_denominator
    diffusion = float(noise_level) * torch.sqrt(diffusion_ratio.clamp_min(0.0))
    diffusion_3d = diffusion[:, None, None]
    remaining = 1.0 - progress_3d

    noise_weight = (
        1.0
        - next_progress_3d
        - diffusion_3d.square() * dt / (2.0 * remaining)
    )
    stochastic_mean = noise_weight * noise_prediction + next_progress_3d * action_prediction
    stochastic_std = math.sqrt(dt) * diffusion_3d.expand_as(current_f)

    deterministic_mean = current_f + dt * velocity_f
    selector = resolved_stochastic_mask[:, None, None]
    mean = torch.where(selector, stochastic_mean, deterministic_mean)
    std = torch.where(selector, stochastic_std, torch.zeros_like(stochastic_std))
    return mean, std


def gaussian_log_prob(sample: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Return per-coordinate Gaussian log-probability in float32.

    This function is only for the selected stochastic transition. Deterministic
    ODE steps have no density and are intentionally rejected.
    """

    for name, value in (("sample", sample), ("mean", mean), ("std", std)):
        _require_floating_tensor(value, name)
    if sample.shape != mean.shape or sample.shape != std.shape:
        raise ValueError("Flow-SDE Gaussian tensors must have identical shapes")
    if sample.device != mean.device or sample.device != std.device:
        raise ValueError("Flow-SDE Gaussian tensors must share one device")
    if bool((std <= 0.0).any()):
        raise ValueError("Flow-SDE Gaussian std must be strictly positive")

    sample_f = sample.float()
    mean_f = mean.float()
    std_f = std.float()
    return (
        -torch.log(std_f)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((sample_f - mean_f) / std_f).square()
    )


def masked_chunk_mean(value: Tensor, mask: Tensor) -> Tensor:
    """Mean within each chunk, then mean across the batch.

    Equal per-chunk weighting avoids giving longer or less-padded chunks more
    influence than shorter valid chunks.
    """

    _require_floating_tensor(value, "masked value")
    if not isinstance(mask, Tensor) or mask.shape != value.shape or mask.dtype != torch.bool:
        raise ValueError("Flow-SDE mask must be boolean and match the value shape")
    if mask.device != value.device:
        raise ValueError("Flow-SDE mask and value must share one device")
    flattened_value = value.reshape(value.shape[0], -1)
    flattened_mask = mask.reshape(mask.shape[0], -1)
    counts = flattened_mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every Flow-SDE chunk must contain at least one valid coordinate")
    per_chunk = (flattened_value * flattened_mask).sum(dim=1) / counts
    return per_chunk.mean()


def ppo_clipped_actor_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return RLinf-style per-coordinate PPO loss for one action chunk."""

    for name, value in (("new_log_probs", new_log_probs), ("old_log_probs", old_log_probs)):
        _require_floating_tensor(value, name)
        if value.dtype != torch.float32:
            raise ValueError(f"Flow-SDE {name} must be float32")
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.ndim != 3:
        raise ValueError("Flow-SDE log-probabilities must share shape (B, H, A)")
    if new_log_probs.device != old_log_probs.device:
        raise ValueError("Flow-SDE log-probabilities must share one device")
    _require_floating_tensor(advantages, "advantages")
    if advantages.shape not in ((new_log_probs.shape[0],), (new_log_probs.shape[0], 1)):
        raise ValueError("Flow-SDE advantages must have shape (B,) or (B, 1)")
    if advantages.device != new_log_probs.device:
        raise ValueError("Flow-SDE advantages must share the log-probability device")
    if (
        not isinstance(action_mask, Tensor)
        or action_mask.shape != new_log_probs.shape
        or action_mask.dtype != torch.bool
        or action_mask.device != new_log_probs.device
    ):
        raise ValueError("Flow-SDE action_mask must be boolean and match log-probabilities")
    for name, value in (("clip_ratio_low", clip_ratio_low), ("clip_ratio_high", clip_ratio_high)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0.0
        ):
            raise ValueError(f"Flow-SDE {name} must be finite and non-negative")

    broadcast_advantages = advantages.reshape(-1, 1, 1).float()
    # Remove invalid coordinates before exponentiation. Merely masking the
    # final mean is too late: a padded log-ratio can overflow to inf and poison
    # otherwise valid metrics or gradients.
    log_ratio = torch.where(
        action_mask,
        new_log_probs - old_log_probs.detach(),
        torch.zeros_like(new_log_probs),
    )
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - float(clip_ratio_low), 1.0 + float(clip_ratio_high))
    objective = torch.minimum(
        ratio * broadcast_advantages,
        clipped_ratio * broadcast_advantages,
    )
    loss = -masked_chunk_mean(objective, action_mask)
    metrics = {
        "ratio": masked_chunk_mean(ratio.detach(), action_mask),
        "approx_kl": masked_chunk_mean((-log_ratio).detach(), action_mask),
        "clip_fraction": masked_chunk_mean((ratio != clipped_ratio).float(), action_mask),
    }
    return loss, metrics


def clipped_value_loss(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    *,
    value_clip: float,
) -> Tensor:
    """Return PPO's clipped scalar value loss."""

    for name, value in (("values", values), ("old_values", old_values), ("returns", returns)):
        _require_floating_tensor(value, name)
    if values.shape != old_values.shape or values.shape != returns.shape or values.ndim not in (1, 2):
        raise ValueError("Flow-SDE value tensors must share shape (B,) or (B, 1)")
    if values.device != old_values.device or values.device != returns.device:
        raise ValueError("Flow-SDE value tensors must share one device")
    if (
        isinstance(value_clip, bool)
        or not isinstance(value_clip, (int, float))
        or not math.isfinite(float(value_clip))
        or value_clip < 0.0
    ):
        raise ValueError("Flow-SDE value_clip must be finite and non-negative")
    clipped = old_values + (values - old_values).clamp(-float(value_clip), float(value_clip))
    raw_error = (values - returns).square()
    clipped_error = (clipped - returns).square()
    return 0.5 * torch.maximum(raw_error, clipped_error).mean()


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute chunk-level GAE for tensors shaped ``(T, B)``.

    ``values`` includes the bootstrap value and therefore has shape
    ``(T + 1, B)``. ``terminated`` denotes a true MDP terminal; time-limit
    truncations remain false so they can bootstrap.
    """

    _require_floating_tensor(rewards, "rewards")
    _require_floating_tensor(values, "values")
    if rewards.ndim != 2 or values.shape != (rewards.shape[0] + 1, rewards.shape[1]):
        raise ValueError("Flow-SDE rewards/values must have shapes (T, B) and (T+1, B)")
    if terminated.shape != rewards.shape or terminated.dtype != torch.bool:
        raise ValueError("Flow-SDE terminated must be boolean with shape (T, B)")
    if values.device != rewards.device or terminated.device != rewards.device:
        raise ValueError("Flow-SDE GAE tensors must share one device")
    for name, value in (("discount", discount), ("gae_lambda", gae_lambda)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"Flow-SDE {name} must be finite and in [0, 1]")

    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros_like(rewards[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_terminal = (~terminated[step]).to(rewards.dtype)
        temporal_difference = (
            rewards[step]
            + float(discount) * not_terminal * values[step + 1]
            - values[step]
        )
        running_advantage = (
            temporal_difference
            + float(discount) * float(gae_lambda) * not_terminal * running_advantage
        )
        advantages[step] = running_advantage
    return advantages, advantages + values[:-1]
