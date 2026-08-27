"""Strict executed-prefix transition contract for ACT-TD3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


def _move_mapping(
    values: Mapping[str, Tensor],
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, Tensor]:
    return {
        name: value.to(device=device, non_blocking=non_blocking)
        for name, value in values.items()
    }


@dataclass(frozen=True)
class ACTTD3Batch:
    """Actor-ready observations and executed normalized action chunks.

    The tensor horizon is the policy's actual execution horizon
    (``n_action_steps``), not its potentially longer prediction horizon
    (``chunk_size``). Partial rows are exact prefixes caused by
    termination/truncation. Observations and actions must already be
    transformed by the immutable processors paired with the ACT checkpoint;
    this class never guesses normalization statistics.
    """

    observations: dict[str, Tensor]
    next_observations: dict[str, Tensor]
    behavior_action_chunks: Tensor
    rewards: Tensor
    executed_mask: Tensor
    step_durations_s: Tensor
    episode_success: Tensor
    terminated: Tensor
    truncated: Tensor
    next_observation_valid: Tensor
    bootstrap_allowed: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.observations, Mapping) or not self.observations:
            raise ValueError("ACT-TD3 observations must be a non-empty mapping")
        if not isinstance(self.next_observations, Mapping):
            raise TypeError("ACT-TD3 next observations must be a mapping")
        if set(self.observations) != set(self.next_observations):
            raise ValueError("ACT-TD3 current and next observation keys must match")
        if not isinstance(self.behavior_action_chunks, Tensor) or (
            self.behavior_action_chunks.ndim != 3
        ):
            raise ValueError("ACT-TD3 behavior actions must have shape (B, T, A)")
        batch_size, chunk_size, action_dim = self.behavior_action_chunks.shape
        if batch_size < 1 or chunk_size < 1 or action_dim < 1:
            raise ValueError("ACT-TD3 behavior action dimensions must be non-empty")
        device = self.behavior_action_chunks.device
        dtype = self.behavior_action_chunks.dtype
        if not self.behavior_action_chunks.is_floating_point() or not bool(
            torch.isfinite(self.behavior_action_chunks).all()
        ):
            raise ValueError("ACT-TD3 behavior actions must be finite floating tensors")

        for name, current in self.observations.items():
            next_value = self.next_observations[name]
            if (
                not isinstance(current, Tensor)
                or not isinstance(next_value, Tensor)
                or current.shape != next_value.shape
                or current.ndim < 2
                or current.shape[0] != batch_size
                or not current.is_floating_point()
                or not next_value.is_floating_point()
                or current.dtype != dtype
                or next_value.dtype != dtype
                or current.device != device
                or next_value.device != device
                or not bool(torch.isfinite(current).all())
                or not bool(torch.isfinite(next_value).all())
            ):
                raise ValueError(
                    f"ACT-TD3 observation {name!r} must be matching finite actor-ready tensors"
                )

        for name, value in (
            ("rewards", self.rewards),
            ("step_durations_s", self.step_durations_s),
        ):
            if (
                not isinstance(value, Tensor)
                or value.shape != (batch_size, chunk_size)
                or not value.is_floating_point()
                or value.dtype != dtype
                or value.device != device
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"ACT-TD3 {name} must be finite and have shape (B, T)"
                )
        if (
            not isinstance(self.executed_mask, Tensor)
            or self.executed_mask.shape != (batch_size, chunk_size)
            or self.executed_mask.dtype != torch.bool
            or self.executed_mask.device != device
        ):
            raise ValueError("ACT-TD3 executed_mask must be boolean (B, T)")
        for name, value in (
            ("episode_success", self.episode_success),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
            ("next_observation_valid", self.next_observation_valid),
            ("bootstrap_allowed", self.bootstrap_allowed),
        ):
            if (
                not isinstance(value, Tensor)
                or value.shape != (batch_size,)
                or value.dtype != torch.bool
                or value.device != device
            ):
                raise ValueError(f"ACT-TD3 {name} must be boolean (B,)")

        lengths = self.executed_mask.to(torch.long).sum(dim=1)
        expected_mask = torch.arange(chunk_size, device=device).unsqueeze(0) < (
            lengths.unsqueeze(1)
        )
        if bool((lengths < 1).any()) or not torch.equal(
            self.executed_mask,
            expected_mask,
        ):
            raise ValueError("ACT-TD3 executed_mask must be an exact non-empty prefix")
        padding = ~self.executed_mask
        if bool(
            (
                self.behavior_action_chunks.masked_select(padding.unsqueeze(-1))
                != 0.0
            ).any()
        ):
            raise ValueError("ACT-TD3 padded behavior actions must be exactly zero")
        if bool((self.rewards.masked_select(padding) != 0.0).any()):
            raise ValueError("ACT-TD3 padded rewards must be exactly zero")
        if bool((self.step_durations_s.masked_select(padding) != 0.0).any()):
            raise ValueError("ACT-TD3 padded durations must be exactly zero")
        if bool((self.step_durations_s.masked_select(self.executed_mask) <= 0.0).any()):
            raise ValueError("ACT-TD3 executed durations must be strictly positive")
        if bool((self.bootstrap_allowed & self.terminated).any()):
            raise ValueError("ACT-TD3 terminated rows cannot bootstrap")
        if bool((self.bootstrap_allowed & ~self.next_observation_valid).any()):
            raise ValueError("ACT-TD3 bootstrap requires a valid next observation")
        partial = lengths < chunk_size
        if bool((partial & ~(self.terminated | self.truncated)).any()):
            raise ValueError(
                "ACT-TD3 partial chunks require termination or truncation"
            )
        invalid_next = ~self.next_observation_valid
        for name, value in self.next_observations.items():
            if bool((value[invalid_next] != 0.0).any()):
                raise ValueError(
                    f"ACT-TD3 invalid next observation {name!r} must be an exact zero sentinel"
                )

    @property
    def batch_size(self) -> int:
        return int(self.behavior_action_chunks.shape[0])

    @property
    def execution_horizon(self) -> int:
        return int(self.behavior_action_chunks.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.behavior_action_chunks.shape[2])

    @property
    def lengths(self) -> Tensor:
        return self.executed_mask.to(torch.long).sum(dim=1)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "ACTTD3Batch":
        target = torch.device(device)

        def move(value: Tensor) -> Tensor:
            return value.to(device=target, non_blocking=non_blocking)

        return ACTTD3Batch(
            observations=_move_mapping(
                self.observations,
                target,
                non_blocking=non_blocking,
            ),
            next_observations=_move_mapping(
                self.next_observations,
                target,
                non_blocking=non_blocking,
            ),
            behavior_action_chunks=move(self.behavior_action_chunks),
            rewards=move(self.rewards),
            executed_mask=move(self.executed_mask),
            step_durations_s=move(self.step_durations_s),
            episode_success=move(self.episode_success),
            terminated=move(self.terminated),
            truncated=move(self.truncated),
            next_observation_valid=move(self.next_observation_valid),
            bootstrap_allowed=move(self.bootstrap_allowed),
        )


__all__ = ["ACTTD3Batch"]
