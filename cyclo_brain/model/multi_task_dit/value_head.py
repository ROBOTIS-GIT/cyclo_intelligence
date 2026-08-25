"""Value head for MultiTaskDiT Flow-SDE PPO."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class MultiTaskDiTValueHead(nn.Module):
    """Map a frozen observation-conditioning vector to one chunk value."""

    def __init__(self, conditioning_dim: int, hidden_dims: Sequence[int] = (512, 256)) -> None:
        super().__init__()
        if (
            isinstance(conditioning_dim, bool)
            or not isinstance(conditioning_dim, int)
            or conditioning_dim < 1
        ):
            raise ValueError("MultiTaskDiT value conditioning_dim must be positive")
        resolved_hidden_dims = tuple(hidden_dims)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in resolved_hidden_dims
        ):
            raise ValueError("MultiTaskDiT value hidden dimensions must be positive integers")

        dimensions = (conditioning_dim, *resolved_hidden_dims, 1)
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(nn.GELU())
        self.conditioning_dim = conditioning_dim
        self.network = nn.Sequential(*layers)

    def forward(self, conditioning: Tensor, *, detach_conditioning: bool = True) -> Tensor:
        if (
            not isinstance(conditioning, Tensor)
            or conditioning.ndim != 2
            or conditioning.shape[1] != self.conditioning_dim
            or not conditioning.is_floating_point()
        ):
            raise ValueError("MultiTaskDiT value input must have shape (B, conditioning_dim)")
        resolved_conditioning = conditioning.detach() if detach_conditioning else conditioning
        parameter = next(self.parameters())
        values = self.network(resolved_conditioning.to(parameter.dtype))[:, 0]
        return values.float()
