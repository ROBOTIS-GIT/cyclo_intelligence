"""Execution-domain projection for normalized ACT action chunks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _action_vector(
    value: Tensor,
    name: str,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    result = torch.as_tensor(value, dtype=dtype, device=device).detach().clone().reshape(-1)
    if result.numel() < 1 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"ACT {name} must be a finite non-empty action vector")
    return result


class ACTExecutionProjector(nn.Module):
    """Map raw normalized ACT output to replay's executed-normalized domain.

    The required values come from two immutable sources: normalization
    statistics stored with the ACT checkpoint, and explicit physical robot
    limits. Dataset extrema are deliberately not accepted as implicit safety
    limits. Binary actions use exact hard forward values; the actor path may
    opt into a physical-domain straight-through estimator. Passthrough
    dimensions preserve the normalized ACT output exactly and are excluded
    from target-policy noise because no physical box bound is defined.
    """

    def __init__(
        self,
        *,
        action_mean: Tensor,
        action_std: Tensor,
        physical_low: Tensor,
        physical_high: Tensor,
        normalizer_eps: float,
        passthrough_mask: Tensor | None = None,
        binary_mask: Tensor | None = None,
        binary_threshold: Tensor | None = None,
        binary_low: Tensor | None = None,
        binary_high: Tensor | None = None,
        trainable_noise_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        mean = _action_vector(action_mean, "action_mean", dtype=torch.float32)
        std = _action_vector(
            action_std,
            "action_std",
            dtype=mean.dtype,
            device=mean.device,
        )
        low = _action_vector(
            physical_low,
            "physical_low",
            dtype=mean.dtype,
            device=mean.device,
        )
        high = _action_vector(
            physical_high,
            "physical_high",
            dtype=mean.dtype,
            device=mean.device,
        )
        action_dim = int(mean.numel())
        if any(value.shape != mean.shape for value in (std, low, high)):
            raise ValueError("ACT projection vectors must share shape (action_dim,)")
        if bool((std < 0.0).any()):
            raise ValueError("ACT action standard deviations must be non-negative")
        if passthrough_mask is None:
            resolved_passthrough_mask = torch.zeros(
                action_dim,
                dtype=torch.bool,
                device=mean.device,
            )
        else:
            raw_passthrough_mask = torch.as_tensor(
                passthrough_mask,
                device=mean.device,
            )
            if raw_passthrough_mask.dtype != torch.bool:
                raise TypeError("ACT passthrough_mask must be boolean")
            if raw_passthrough_mask.shape != mean.shape:
                raise ValueError("ACT passthrough_mask must have shape (action_dim,)")
            resolved_passthrough_mask = raw_passthrough_mask.detach().clone()
        bounded_mask = ~resolved_passthrough_mask
        if bool((low[bounded_mask] >= high[bounded_mask]).any()):
            raise ValueError(
                "ACT physical_low must be strictly below physical_high on bounded dimensions"
            )
        if bool(
            (
                (low[resolved_passthrough_mask] != 0.0)
                | (high[resolved_passthrough_mask] != 0.0)
            ).any()
        ):
            raise ValueError(
                "ACT passthrough dimensions require zero physical-limit placeholders"
            )
        if (
            isinstance(normalizer_eps, bool)
            or not isinstance(normalizer_eps, (int, float))
            or not math.isfinite(float(normalizer_eps))
            or float(normalizer_eps) <= 0.0
        ):
            raise ValueError("ACT normalizer_eps must be finite and positive")
        denominator = std + float(normalizer_eps)
        if not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0.0).any()):
            raise ValueError("ACT action std + eps must remain finite and positive")

        if binary_mask is None:
            resolved_binary_mask = torch.zeros(
                action_dim,
                dtype=torch.bool,
                device=mean.device,
            )
        else:
            resolved_binary_mask = torch.as_tensor(
                binary_mask,
                dtype=torch.bool,
                device=mean.device,
            ).reshape(-1)
            if resolved_binary_mask.shape != mean.shape:
                raise ValueError("ACT binary_mask must have shape (action_dim,)")
        if bool((resolved_binary_mask & resolved_passthrough_mask).any()):
            raise ValueError("ACT binary_mask and passthrough_mask cannot overlap")
        has_binary = bool(resolved_binary_mask.any())
        binary_values = (binary_threshold, binary_low, binary_high)
        if has_binary and any(value is None for value in binary_values):
            raise ValueError("ACT binary dimensions require threshold, low, and high vectors")
        if not has_binary and any(value is not None for value in binary_values):
            raise ValueError("ACT binary values require at least one binary dimension")
        if has_binary:
            assert binary_threshold is not None
            assert binary_low is not None
            assert binary_high is not None
            threshold = _action_vector(
                binary_threshold,
                "binary_threshold",
                dtype=mean.dtype,
                device=mean.device,
            )
            binary_low_value = _action_vector(
                binary_low,
                "binary_low",
                dtype=mean.dtype,
                device=mean.device,
            )
            binary_high_value = _action_vector(
                binary_high,
                "binary_high",
                dtype=mean.dtype,
                device=mean.device,
            )
            if any(
                value.shape != mean.shape
                for value in (threshold, binary_low_value, binary_high_value)
            ):
                raise ValueError("ACT binary vectors must have shape (action_dim,)")
            if bool(
                (
                    (binary_low_value[resolved_binary_mask] < low[resolved_binary_mask])
                    | (binary_low_value[resolved_binary_mask] > high[resolved_binary_mask])
                    | (binary_high_value[resolved_binary_mask] < low[resolved_binary_mask])
                    | (binary_high_value[resolved_binary_mask] > high[resolved_binary_mask])
                ).any()
            ):
                raise ValueError("ACT binary targets must lie inside physical limits")
        else:
            threshold = torch.zeros_like(mean)
            binary_low_value = torch.zeros_like(mean)
            binary_high_value = torch.zeros_like(mean)

        default_noise_mask = (
            (~resolved_binary_mask) & (~resolved_passthrough_mask) & (std > 0.0)
        )
        if trainable_noise_mask is None:
            resolved_noise_mask = default_noise_mask
        else:
            resolved_noise_mask = torch.as_tensor(
                trainable_noise_mask,
                dtype=torch.bool,
                device=mean.device,
            ).reshape(-1)
            if resolved_noise_mask.shape != mean.shape:
                raise ValueError("ACT trainable_noise_mask must have shape (action_dim,)")
            if bool((resolved_noise_mask & ~default_noise_mask).any()):
                raise ValueError(
                    "ACT target noise cannot include binary, passthrough, or zero-std dimensions"
                )

        self.register_buffer("action_mean", mean)
        self.register_buffer("action_std", std)
        self.register_buffer(
            "normalizer_eps",
            mean.new_tensor(float(normalizer_eps)),
        )
        self.register_buffer("physical_low", low)
        self.register_buffer("physical_high", high)
        self.register_buffer("passthrough_mask", resolved_passthrough_mask)
        self.register_buffer("binary_mask", resolved_binary_mask)
        self.register_buffer("binary_threshold", threshold)
        self.register_buffer("binary_low", binary_low_value)
        self.register_buffer("binary_high", binary_high_value)
        self.register_buffer("noise_mask", resolved_noise_mask)

    @property
    def action_dim(self) -> int:
        return int(self.action_mean.numel())

    def forward(
        self,
        policy_normalized_action_chunks: Tensor,
        *,
        straight_through_binary: bool,
        detach_passthrough: bool = False,
    ) -> Tensor:
        """Project a chunk, optionally blocking direct passthrough gradients.

        ``detach_passthrough`` changes only the actor-gradient path: the
        passthrough forward values remain exact normalized ACT outputs.
        """
        if not isinstance(straight_through_binary, bool):
            raise TypeError("ACT binary straight-through selection must be boolean")
        if not isinstance(detach_passthrough, bool):
            raise TypeError("ACT passthrough detach selection must be boolean")
        value = policy_normalized_action_chunks
        if (
            not isinstance(value, Tensor)
            or value.ndim != 3
            or value.shape[0] < 1
            or value.shape[1] < 1
            or value.shape[2] != self.action_dim
            or not value.is_floating_point()
            or value.dtype != self.action_mean.dtype
            or value.device != self.action_mean.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                "ACT policy actions must be finite (B, T, action_dim) on the projector domain"
            )

        passthrough_mask = self.passthrough_mask.view(1, 1, -1)
        projection_value = torch.where(
            passthrough_mask,
            torch.zeros_like(value),
            value,
        )
        proposed_physical = projection_value * self.action_std + self.action_mean
        executed_physical = torch.maximum(
            torch.minimum(proposed_physical, self.physical_high),
            self.physical_low,
        )
        if bool(self.binary_mask.any()):
            hard_binary = torch.where(
                proposed_physical < self.binary_threshold,
                self.binary_low,
                self.binary_high,
            )
            mask = self.binary_mask.view(1, 1, -1)
            executed_physical = torch.where(mask, hard_binary, executed_physical)
            if straight_through_binary:
                executed_physical = executed_physical + mask.to(value.dtype) * (
                    proposed_physical - proposed_physical.detach()
                )
        executed_normalized = (executed_physical - self.action_mean) / (
            self.action_std + self.normalizer_eps
        )
        passthrough_value = value.detach() if detach_passthrough else value
        executed_normalized = torch.where(
            passthrough_mask,
            passthrough_value,
            executed_normalized,
        )
        if not bool(torch.isfinite(executed_normalized).all()):
            raise RuntimeError("ACT projected actions contain NaN or Inf")
        return executed_normalized


__all__ = ["ACTExecutionProjector"]
