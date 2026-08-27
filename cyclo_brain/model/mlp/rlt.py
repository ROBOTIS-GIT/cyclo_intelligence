"""Small action MLP used by the PI RLT Stage-2 policy."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"RLT Action MLP {name} must be a positive integer")
    return value


def _hidden_dimensions(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in result
    ):
        raise ValueError("RLT Action MLP hidden dimensions must be positive integers")
    return result


def _finite_tensor(value: Tensor, shape: tuple[int, ...], name: str) -> None:
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"RLT {name} must be finite floating with shape {shape}")


class RLTGaussianChunkActor(nn.Module):
    """Direct Gaussian action-chunk actor from equation (4) of PI RLT.

    The deterministic inference action is the MLP mean.  ``reference_actions``
    is conditioning, not a residual that is added to the output.
    """

    def __init__(
        self,
        rl_token_dim: int,
        proprio_dim: int,
        chunk_length: int,
        action_dim: int,
        *,
        fixed_standard_deviation: float,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.rl_token_dim = _positive_dimension(rl_token_dim, "rl_token_dim")
        self.proprio_dim = _positive_dimension(proprio_dim, "proprio_dim")
        self.chunk_length = _positive_dimension(chunk_length, "chunk_length")
        self.action_dim = _positive_dimension(action_dim, "action_dim")
        self.hidden_dims = _hidden_dimensions(hidden_dims)
        if (
            isinstance(fixed_standard_deviation, bool)
            or not isinstance(fixed_standard_deviation, (int, float))
            or not math.isfinite(float(fixed_standard_deviation))
            or float(fixed_standard_deviation) <= 0.0
        ):
            raise ValueError("RLT fixed policy standard deviation must be positive")

        self.register_buffer(
            "fixed_standard_deviation",
            torch.tensor(float(fixed_standard_deviation), dtype=torch.float32),
        )
        flattened_action_dim = self.chunk_length * self.action_dim
        input_dim = self.rl_token_dim + self.proprio_dim + flattened_action_dim
        layers: list[nn.Module] = []
        previous = input_dim
        for width in self.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, flattened_action_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        reference_actions: Tensor,
    ) -> Tensor:
        if not isinstance(z_rl, Tensor) or z_rl.ndim != 2:
            raise ValueError("RLT actor z_rl must have shape (B, Z)")
        batch_size = int(z_rl.shape[0])
        if batch_size < 1:
            raise ValueError("RLT actor batch must be non-empty")
        _finite_tensor(z_rl, (batch_size, self.rl_token_dim), "actor z_rl")
        _finite_tensor(proprio, (batch_size, self.proprio_dim), "actor proprio")
        _finite_tensor(
            reference_actions,
            (batch_size, self.chunk_length, self.action_dim),
            "actor reference actions",
        )
        if len({z_rl.device, proprio.device, reference_actions.device}) != 1:
            raise ValueError("RLT actor inputs must share one device")
        if len({z_rl.dtype, proprio.dtype, reference_actions.dtype}) != 1:
            raise ValueError("RLT actor inputs must share one dtype")
        parameter = next(self.parameters())
        if parameter.device != z_rl.device or parameter.dtype != z_rl.dtype:
            raise ValueError("RLT actor inputs and parameters must share dtype and device")

        inputs = torch.cat(
            (z_rl, proprio, reference_actions.reshape(batch_size, -1)),
            dim=-1,
        )
        output = self.network(inputs).reshape(
            batch_size,
            self.chunk_length,
            self.action_dim,
        )
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("RLT actor mean became non-finite")
        return output


__all__ = ["RLTGaussianChunkActor"]
