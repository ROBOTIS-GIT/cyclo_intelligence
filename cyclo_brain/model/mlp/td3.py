"""Reference MLP models for Twin Delayed DDPG.

The default two-layer ReLU topology follows the authors' ``sfujim/TD3``
implementation. Algorithm state, targets, losses, and optimizers intentionally
live outside this model module.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"TD3 MLP {name} must be a positive integer")
    return value


def _hidden_dimensions(hidden_dims: Sequence[int]) -> tuple[int, ...]:
    result = tuple(hidden_dims)
    if not result or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in result
    ):
        raise ValueError("TD3 MLP hidden dimensions must be positive integers")
    return result


def _mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def _two_dimensional_floating_tensor(
    value: Tensor,
    *,
    width: int,
    name: str,
) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"TD3 {name} must have shape (batch, {width})")
    if value.shape[0] < 1 or not value.is_floating_point():
        raise ValueError(f"TD3 {name} must be a non-empty floating tensor")


class TD3MLPActor(nn.Module):
    """Deterministic bounded actor ``center + scale * tanh(MLP(s))``."""

    def __init__(
        self,
        observation_dim: int,
        action_low: Tensor | Sequence[float],
        action_high: Tensor | Sequence[float],
        *,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.observation_dim = _positive_dimension(observation_dim, "observation_dim")
        resolved_hidden_dims = _hidden_dimensions(hidden_dims)
        low = torch.as_tensor(action_low, dtype=torch.float32).detach().clone()
        high = torch.as_tensor(action_high, dtype=torch.float32).detach().clone()
        if low.ndim != 1 or low.numel() < 1 or high.shape != low.shape:
            raise ValueError("TD3 action bounds must be equal non-empty vectors")
        if not bool(torch.isfinite(low).all()) or not bool(torch.isfinite(high).all()):
            raise ValueError("TD3 action bounds must be finite")
        if bool((low >= high).any()):
            raise ValueError("TD3 action_low must be strictly below action_high")

        self.action_dim = int(low.numel())
        self.hidden_dims = resolved_hidden_dims
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.network = _mlp(
            self.observation_dim,
            resolved_hidden_dims,
            self.action_dim,
        )

    def forward(self, observations: Tensor) -> Tensor:
        _two_dimensional_floating_tensor(
            observations,
            width=self.observation_dim,
            name="actor observations",
        )
        center = (self.action_high + self.action_low) * 0.5
        scale = (self.action_high - self.action_low) * 0.5
        return center + scale * torch.tanh(self.network(observations))


class TD3MLPQFunction(nn.Module):
    """One scalar Q-function over a vector observation and action."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.observation_dim = _positive_dimension(observation_dim, "observation_dim")
        self.action_dim = _positive_dimension(action_dim, "action_dim")
        self.hidden_dims = _hidden_dimensions(hidden_dims)
        self.network = _mlp(
            self.observation_dim + self.action_dim,
            self.hidden_dims,
            1,
        )

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        _two_dimensional_floating_tensor(
            observations,
            width=self.observation_dim,
            name="critic observations",
        )
        _two_dimensional_floating_tensor(
            actions,
            width=self.action_dim,
            name="critic actions",
        )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("TD3 critic observation and action batches must match")
        if observations.device != actions.device or observations.dtype != actions.dtype:
            raise ValueError("TD3 critic inputs must share dtype and device")
        return self.network(torch.cat((observations, actions), dim=-1))


class TD3MLPTwinCritic(nn.Module):
    """Two parameter-independent MLP Q-functions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.observation_dim = _positive_dimension(observation_dim, "observation_dim")
        self.action_dim = _positive_dimension(action_dim, "action_dim")
        resolved_hidden_dims = _hidden_dimensions(hidden_dims)
        self.q1 = TD3MLPQFunction(
            self.observation_dim,
            self.action_dim,
            hidden_dims=resolved_hidden_dims,
        )
        self.q2 = TD3MLPQFunction(
            self.observation_dim,
            self.action_dim,
            hidden_dims=resolved_hidden_dims,
        )

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.q1(observations, actions), self.q2(observations, actions)


__all__ = ["TD3MLPActor", "TD3MLPQFunction", "TD3MLPTwinCritic"]
