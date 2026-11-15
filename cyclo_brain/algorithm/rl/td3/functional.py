"""Pure TD3 equations, with no policy, replay, or worker dependencies.

The equations follow Fujimoto et al., "Addressing Function Approximation Error
in Actor-Critic Methods" (ICML 2018), and the authors' ``sfujim/TD3``
reference implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _require_floating_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"TD3 {name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"TD3 {name} must be finite")


def clipped_target_action(
    target_action: Tensor,
    noise: Tensor,
    *,
    noise_clip: float,
    action_low: Tensor,
    action_high: Tensor,
) -> Tensor:
    """Apply clipped target-policy noise, then enforce action bounds.

    ``action_low`` and ``action_high`` may be per-dimension vectors and must be
    broadcastable to ``target_action``. This avoids the equal symmetric bound
    assumption made by many compact TD3 examples.
    """

    _require_floating_tensor(target_action, "target_action")
    _require_floating_tensor(noise, "noise")
    _require_floating_tensor(action_low, "action_low")
    _require_floating_tensor(action_high, "action_high")
    if noise.shape != target_action.shape:
        raise ValueError("TD3 target action and noise shapes must match")
    if noise.device != target_action.device or noise.dtype != target_action.dtype:
        raise ValueError("TD3 target action and noise must share dtype and device")
    for bound_name, bound in (("action_low", action_low), ("action_high", action_high)):
        if bound.device != target_action.device or bound.dtype != target_action.dtype:
            raise ValueError(f"TD3 {bound_name} must share action dtype and device")
    if not isinstance(noise_clip, (int, float)) or isinstance(noise_clip, bool):
        raise TypeError("TD3 noise_clip must be a real number")
    if not math.isfinite(float(noise_clip)) or noise_clip < 0.0:
        raise ValueError("TD3 noise_clip must be finite and non-negative")
    try:
        low, high = torch.broadcast_tensors(action_low, action_high)
        torch.broadcast_shapes(target_action.shape, low.shape)
    except RuntimeError as exc:
        raise ValueError("TD3 action bounds are not broadcastable to actions") from exc
    if bool((low >= high).any().item()):
        raise ValueError("TD3 action_low must be strictly below action_high")

    clipped_noise = noise.clamp(-float(noise_clip), float(noise_clip))
    return torch.maximum(torch.minimum(target_action + clipped_noise, action_high), action_low)


def bellman_target(
    rewards: Tensor,
    terminated: Tensor,
    target_q1: Tensor,
    target_q2: Tensor,
    *,
    discount: float,
) -> Tensor:
    """Compute the clipped-double-Q TD target.

    ``terminated`` represents a true MDP terminal, not a time-limit
    truncation. Callers must resolve that distinction in their data contract.
    """

    for name, value in (("rewards", rewards), ("target_q1", target_q1), ("target_q2", target_q2)):
        _require_floating_tensor(value, name)
    if rewards.shape != target_q1.shape or rewards.shape != target_q2.shape:
        raise ValueError("TD3 rewards and target Q tensors must have identical shapes")
    if terminated.shape != rewards.shape or terminated.dtype != torch.bool:
        raise ValueError("TD3 terminated must be boolean and match reward shape")
    if any(
        value.device != rewards.device or value.dtype != rewards.dtype
        for value in (target_q1, target_q2)
    ):
        raise ValueError("TD3 rewards and target Q tensors must share dtype and device")
    if terminated.device != rewards.device:
        raise ValueError("TD3 terminated must share reward device")
    if not isinstance(discount, (int, float)) or isinstance(discount, bool):
        raise TypeError("TD3 discount must be a real number")
    if not math.isfinite(float(discount)) or not 0.0 <= discount <= 1.0:
        raise ValueError("TD3 discount must be finite and in [0, 1]")
    not_terminal = (~terminated).to(dtype=rewards.dtype)
    return rewards + not_terminal * float(discount) * torch.minimum(target_q1, target_q2)


def critic_loss(q1: Tensor, q2: Tensor, targets: Tensor) -> Tensor:
    """Return the sum of the two mean-squared Bellman errors."""

    for name, value in (("q1", q1), ("q2", q2), ("targets", targets)):
        _require_floating_tensor(value, name)
    if q1.shape != q2.shape or q1.shape != targets.shape:
        raise ValueError("TD3 Q predictions and targets must have identical shapes")
    return F.mse_loss(q1, targets) + F.mse_loss(q2, targets)


def deterministic_actor_loss(q1_for_policy: Tensor) -> Tensor:
    """Return the deterministic policy objective ``-E[Q1(s, policy(s))]``."""

    _require_floating_tensor(q1_for_policy, "q1_for_policy")
    if q1_for_policy.numel() == 0:
        raise ValueError("TD3 actor Q tensor must be non-empty")
    return -q1_for_policy.mean()


def policy_update_is_due(completed_critic_updates: int, *, period: int) -> bool:
    """Return whether a delayed actor/target update follows this critic step."""

    for name, value in (("completed_critic_updates", completed_critic_updates), ("period", period)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"TD3 {name} must be an integer")
    if completed_critic_updates < 1 or period < 1:
        raise ValueError("TD3 update count and period must be positive")
    return completed_critic_updates % period == 0


@torch.no_grad()
def polyak_update_(source: nn.Module, target: nn.Module, *, tau: float) -> None:
    """Update target parameters and copy buffers from the online network."""

    if source is target:
        raise ValueError("TD3 online and target modules must be distinct")
    if not isinstance(tau, (int, float)) or isinstance(tau, bool):
        raise TypeError("TD3 tau must be a real number")
    if not math.isfinite(float(tau)) or not 0.0 < tau <= 1.0:
        raise ValueError("TD3 tau must be finite and in (0, 1]")
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError("TD3 online and target parameter structures must match")
    for name, target_parameter in target_parameters.items():
        source_parameter = source_parameters[name]
        if source_parameter.shape != target_parameter.shape:
            raise ValueError(f"TD3 parameter shape mismatch for {name!r}")
        target_parameter.lerp_(source_parameter, float(tau))
    source_buffers = dict(source.named_buffers())
    target_buffers = dict(target.named_buffers())
    if source_buffers.keys() != target_buffers.keys():
        raise ValueError("TD3 online and target buffer structures must match")
    for name, target_buffer in target_buffers.items():
        source_buffer = source_buffers[name]
        if source_buffer.shape != target_buffer.shape:
            raise ValueError(f"TD3 buffer shape mismatch for {name!r}")
        target_buffer.copy_(source_buffer)
