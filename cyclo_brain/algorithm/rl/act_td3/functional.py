"""Pure chunk-SMDP equations for ACT-TD3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ACTSMDPReturns:
    discounted_returns: Tensor
    bootstrap_discounts: Tensor
    executed_mask: Tensor


def build_smdp_returns(
    rewards: Tensor,
    executed_mask: Tensor,
    step_durations_s: Tensor,
    bootstrap_allowed: Tensor,
    *,
    discount: float,
    discount_reference_hz: float,
) -> ACTSMDPReturns:
    """Aggregate primitive rewards using elapsed-time discount exponents."""

    if (
        not isinstance(rewards, Tensor)
        or rewards.ndim != 2
        or rewards.shape[0] < 1
        or rewards.shape[1] < 1
        or not rewards.is_floating_point()
        or not bool(torch.isfinite(rewards).all())
    ):
        raise ValueError("ACT-TD3 rewards must be finite floating (B, T)")
    if (
        not isinstance(executed_mask, Tensor)
        or executed_mask.shape != rewards.shape
        or executed_mask.dtype != torch.bool
        or executed_mask.device != rewards.device
    ):
        raise ValueError("ACT-TD3 executed_mask must be boolean and match rewards")
    if (
        not isinstance(step_durations_s, Tensor)
        or step_durations_s.shape != rewards.shape
        or not step_durations_s.is_floating_point()
        or step_durations_s.dtype != rewards.dtype
        or step_durations_s.device != rewards.device
        or not bool(torch.isfinite(step_durations_s).all())
    ):
        raise ValueError("ACT-TD3 durations must be finite and match rewards")
    batch_size, chunk_size = rewards.shape
    if (
        not isinstance(bootstrap_allowed, Tensor)
        or bootstrap_allowed.shape != (batch_size,)
        or bootstrap_allowed.dtype != torch.bool
        or bootstrap_allowed.device != rewards.device
    ):
        raise ValueError("ACT-TD3 bootstrap_allowed must be boolean (B,)")
    for name, value in (
        ("discount", discount),
        ("discount_reference_hz", discount_reference_hz),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"ACT-TD3 {name} must be finite and positive")
    if discount > 1.0:
        raise ValueError("ACT-TD3 discount must be at most 1")

    lengths = executed_mask.to(torch.long).sum(dim=1)
    expected_mask = torch.arange(chunk_size, device=rewards.device).unsqueeze(0) < (
        lengths.unsqueeze(1)
    )
    if bool((lengths < 1).any()) or not torch.equal(executed_mask, expected_mask):
        raise ValueError("ACT-TD3 executed_mask must be an exact non-empty prefix")
    padding = ~executed_mask
    if bool((rewards.masked_select(padding) != 0.0).any()):
        raise ValueError("ACT-TD3 padded rewards must be exactly zero")
    if bool((step_durations_s.masked_select(padding) != 0.0).any()):
        raise ValueError("ACT-TD3 padded durations must be exactly zero")
    if bool((step_durations_s.masked_select(executed_mask) <= 0.0).any()):
        raise ValueError("ACT-TD3 executed durations must be strictly positive")

    duration_ticks = step_durations_s * float(discount_reference_hz)
    elapsed_after_step = torch.cumsum(duration_ticks, dim=1)
    elapsed_before_step = elapsed_after_step - duration_ticks
    reward_discounts = torch.pow(
        rewards.new_tensor(float(discount)),
        elapsed_before_step,
    )
    discounted_returns = (
        rewards * reward_discounts * executed_mask.to(rewards.dtype)
    ).sum(dim=1, keepdim=True)
    total_ticks = duration_ticks.sum(dim=1, keepdim=True)
    bootstrap_discounts = torch.pow(
        rewards.new_tensor(float(discount)),
        total_ticks,
    )
    bootstrap_discounts = bootstrap_discounts.masked_fill(
        ~bootstrap_allowed.unsqueeze(1),
        0.0,
    )
    return ACTSMDPReturns(
        discounted_returns=discounted_returns,
        bootstrap_discounts=bootstrap_discounts,
        executed_mask=executed_mask,
    )


def smooth_target_action_chunks(
    policy_action_chunks: Tensor,
    standard_normal_noise: Tensor,
    noise_mask: Tensor,
    *,
    noise_standard_deviation: float,
    noise_clip: float,
) -> Tensor:
    """Apply TD3 target noise in the raw normalized ACT output domain."""

    if (
        not isinstance(policy_action_chunks, Tensor)
        or policy_action_chunks.ndim != 3
        or not policy_action_chunks.is_floating_point()
        or not bool(torch.isfinite(policy_action_chunks).all())
    ):
        raise ValueError("ACT-TD3 target actions must be finite floating (B, T, A)")
    if (
        not isinstance(standard_normal_noise, Tensor)
        or standard_normal_noise.shape != policy_action_chunks.shape
        or standard_normal_noise.dtype != policy_action_chunks.dtype
        or standard_normal_noise.device != policy_action_chunks.device
        or not bool(torch.isfinite(standard_normal_noise).all())
    ):
        raise ValueError("ACT-TD3 target noise must match target actions")
    action_dim = policy_action_chunks.shape[-1]
    if (
        not isinstance(noise_mask, Tensor)
        or noise_mask.shape != (action_dim,)
        or noise_mask.dtype != torch.bool
        or noise_mask.device != policy_action_chunks.device
    ):
        raise ValueError("ACT-TD3 noise mask must be boolean (action_dim,)")
    for name, value in (
        ("noise_standard_deviation", noise_standard_deviation),
        ("noise_clip", noise_clip),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"ACT-TD3 {name} must be finite and non-negative")
    noise = standard_normal_noise * float(noise_standard_deviation)
    noise = noise.clamp(-float(noise_clip), float(noise_clip))
    noise = noise * noise_mask.view(1, 1, -1).to(noise.dtype)
    return policy_action_chunks + noise


def actor_update_is_due(
    completed_critic_updates: int,
    *,
    critic_warmup_updates: int,
    policy_update_period: int,
) -> bool:
    """Gate the first actor update to warmup + one complete delay period."""

    for name, value, minimum in (
        ("completed_critic_updates", completed_critic_updates, 0),
        ("critic_warmup_updates", critic_warmup_updates, 0),
        ("policy_update_period", policy_update_period, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"ACT-TD3 {name} must be at least {minimum}")
    if completed_critic_updates <= critic_warmup_updates:
        return False
    return (
        completed_critic_updates - critic_warmup_updates
    ) % policy_update_period == 0


def q_weight_for_actor_update(
    actor_update_number: int,
    *,
    maximum: float,
    ramp_updates: int,
) -> float:
    """Return the conservative linear Q-weight ramp used by cyclo_lab."""

    if (
        isinstance(actor_update_number, bool)
        or not isinstance(actor_update_number, int)
        or actor_update_number < 1
    ):
        raise ValueError("ACT-TD3 actor update number must be positive")
    if (
        isinstance(ramp_updates, bool)
        or not isinstance(ramp_updates, int)
        or ramp_updates < 1
    ):
        raise ValueError("ACT-TD3 Q ramp updates must be positive")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or maximum < 0.0
    ):
        raise ValueError("ACT-TD3 maximum Q weight must be finite and non-negative")
    progress = min(max(actor_update_number - 1, 0) / ramp_updates, 1.0)
    return float(maximum) * progress


def masked_deterministic_bc_l1(
    policy_action_chunks: Tensor,
    behavior_action_chunks: Tensor,
    executed_mask: Tensor,
) -> Tensor:
    """Anchor the deployed zero-latent ACT path on each executed prefix."""

    if (
        not isinstance(policy_action_chunks, Tensor)
        or not isinstance(behavior_action_chunks, Tensor)
        or policy_action_chunks.ndim != 3
        or policy_action_chunks.shape != behavior_action_chunks.shape
        or not policy_action_chunks.is_floating_point()
        or policy_action_chunks.dtype != behavior_action_chunks.dtype
        or policy_action_chunks.device != behavior_action_chunks.device
        or not bool(torch.isfinite(policy_action_chunks).all())
        or not bool(torch.isfinite(behavior_action_chunks).all())
    ):
        raise ValueError("ACT-TD3 deterministic BC actions must align as finite (B, T, A)")
    batch_size, chunk_size, action_dim = policy_action_chunks.shape
    if (
        batch_size < 1
        or not isinstance(executed_mask, Tensor)
        or executed_mask.shape != (batch_size, chunk_size)
        or executed_mask.dtype != torch.bool
        or executed_mask.device != policy_action_chunks.device
    ):
        raise ValueError("ACT-TD3 deterministic BC mask must be boolean (B, T)")
    lengths = executed_mask.to(torch.long).sum(dim=1)
    expected = torch.arange(chunk_size, device=executed_mask.device).unsqueeze(0) < (
        lengths.unsqueeze(1)
    )
    if bool((lengths < 1).any()) or not torch.equal(executed_mask, expected):
        raise ValueError("ACT-TD3 deterministic BC mask must be an exact prefix")
    absolute_error = (policy_action_chunks - behavior_action_chunks).abs()
    denominator = executed_mask.sum() * action_dim
    return (
        absolute_error * executed_mask.unsqueeze(-1).to(absolute_error.dtype)
    ).sum() / denominator


__all__ = [
    "ACTSMDPReturns",
    "actor_update_is_due",
    "build_smdp_returns",
    "masked_deterministic_bc_l1",
    "q_weight_for_actor_update",
    "smooth_target_action_chunks",
]
