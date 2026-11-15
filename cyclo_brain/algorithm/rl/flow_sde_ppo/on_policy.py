"""On-policy episode and training-batch contracts for Flow-SDE PPO.

One transition represents one action-chunk policy decision, not one robot
control tick.  This keeps reward/termination semantics aligned with the
probability density stored by :class:`FlowSDERollout`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from .batch import FlowSDERollout
from .functional import generalized_advantage_estimate


def rollout_to(rollout: FlowSDERollout, device: torch.device | str) -> FlowSDERollout:
    """Copy a rollout to ``device`` without weakening its validation contract."""

    return FlowSDERollout(
        chains=rollout.chains.to(device),
        denoise_indices=rollout.denoise_indices.to(device),
        old_log_probs=rollout.old_log_probs.to(device),
        action_mask=rollout.action_mask.to(device),
    )


def index_rollout(rollout: FlowSDERollout, indices: Tensor) -> FlowSDERollout:
    """Select rollout batch entries for a PPO minibatch."""

    if not isinstance(indices, Tensor) or indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("Flow-SDE rollout indices must be a one-dimensional int64 tensor")
    if indices.device != rollout.chains.device:
        raise ValueError("Flow-SDE rollout indices must be on the rollout device")
    return FlowSDERollout(
        chains=rollout.chains.index_select(0, indices),
        denoise_indices=rollout.denoise_indices.index_select(0, indices),
        old_log_probs=rollout.old_log_probs.index_select(0, indices),
        action_mask=rollout.action_mask.index_select(0, indices),
    )


@dataclass(frozen=True)
class FlowSDETransition:
    """One environment response to one sampled action chunk.

    ``conditioning`` is the frozen observation embedding used by both actor
    and value head. Keeping it instead of images makes on-policy replay exact
    while avoiding a second frozen vision/language encoder pass during PPO.
    The embedded :class:`FlowSDERollout` must contain exactly one sample.
    """

    conditioning: Tensor
    rollout: FlowSDERollout
    reward: float
    terminated: bool
    truncated: bool
    old_value: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.conditioning, Tensor)
            or self.conditioning.ndim != 1
            or not self.conditioning.is_floating_point()
            or not bool(torch.isfinite(self.conditioning).all())
        ):
            raise ValueError("Flow-SDE transition conditioning must be finite with shape (C,)")
        if not isinstance(self.rollout, FlowSDERollout) or self.rollout.chains.shape[0] != 1:
            raise ValueError("Flow-SDE transition rollout must contain exactly one sample")
        if self.conditioning.device != self.rollout.chains.device:
            raise ValueError("Flow-SDE transition tensors must share one device")
        for name, value in (("reward", self.reward), ("old_value", self.old_value)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Flow-SDE transition {name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"Flow-SDE transition {name} must be finite")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise TypeError("Flow-SDE terminated/truncated flags must be boolean")
        if self.terminated and self.truncated:
            raise ValueError("Flow-SDE transition cannot be both terminated and truncated")


@dataclass(frozen=True)
class FlowSDEEpisode:
    """A complete on-policy episode with an optional truncation bootstrap."""

    transitions: tuple[FlowSDETransition, ...]
    bootstrap_value: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.transitions, tuple) or not self.transitions:
            raise ValueError("Flow-SDE episode requires at least one transition")
        if not all(isinstance(item, FlowSDETransition) for item in self.transitions):
            raise TypeError("Flow-SDE episode entries must be FlowSDETransition values")
        for transition in self.transitions[:-1]:
            if transition.terminated or transition.truncated:
                raise ValueError("Only the final Flow-SDE episode transition may be done")
        if not (self.transitions[-1].terminated or self.transitions[-1].truncated):
            raise ValueError("A Flow-SDE episode must end in termination or truncation")
        first = self.transitions[0]
        contract = (
            first.conditioning.shape,
            first.rollout.chains.shape[1:],
            first.conditioning.device,
        )
        for transition in self.transitions[1:]:
            candidate = (
                transition.conditioning.shape,
                transition.rollout.chains.shape[1:],
                transition.conditioning.device,
            )
            if candidate != contract:
                raise ValueError("Flow-SDE episode transitions have inconsistent tensor contracts")
        if isinstance(self.bootstrap_value, bool) or not isinstance(
            self.bootstrap_value, (int, float)
        ):
            raise TypeError("Flow-SDE bootstrap value must be a real scalar")
        if not math.isfinite(float(self.bootstrap_value)):
            raise ValueError("Flow-SDE bootstrap value must be finite")

    @property
    def episode_return(self) -> float:
        return float(sum(float(item.reward) for item in self.transitions))


@dataclass(frozen=True)
class FlowSDETrainingBatch:
    """Flattened complete episodes plus precomputed chunk-level GAE."""

    conditioning: Tensor
    rollout: FlowSDERollout
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    episode_returns: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.conditioning, Tensor) or self.conditioning.ndim != 2:
            raise ValueError("Flow-SDE training conditioning must have shape (B, C)")
        batch_size = self.conditioning.shape[0]
        if self.rollout.chains.shape[0] != batch_size:
            raise ValueError("Flow-SDE training rollout batch size mismatch")
        for name, value in (
            ("rewards", self.rewards),
            ("old_values", self.old_values),
            ("advantages", self.advantages),
            ("returns", self.returns),
        ):
            if not isinstance(value, Tensor) or value.shape != (batch_size,) or not value.is_floating_point():
                raise ValueError(f"Flow-SDE training {name} must have floating shape (B,)")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Flow-SDE training {name} must be finite")
        for name, value in (("terminated", self.terminated), ("truncated", self.truncated)):
            if not isinstance(value, Tensor) or value.shape != (batch_size,) or value.dtype != torch.bool:
                raise ValueError(f"Flow-SDE training {name} must have boolean shape (B,)")
        if not isinstance(self.episode_returns, Tensor) or self.episode_returns.ndim != 1:
            raise ValueError("Flow-SDE episode returns must be one-dimensional")
        devices = {
            self.conditioning.device,
            self.rollout.chains.device,
            self.rewards.device,
            self.terminated.device,
            self.truncated.device,
            self.old_values.device,
            self.advantages.device,
            self.returns.device,
            self.episode_returns.device,
        }
        if len(devices) != 1:
            raise ValueError("Flow-SDE training tensors must share one device")

    def to(self, device: torch.device | str) -> "FlowSDETrainingBatch":
        return FlowSDETrainingBatch(
            conditioning=self.conditioning.to(device),
            rollout=rollout_to(self.rollout, device),
            rewards=self.rewards.to(device),
            terminated=self.terminated.to(device),
            truncated=self.truncated.to(device),
            old_values=self.old_values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            episode_returns=self.episode_returns.to(device),
        )

class FlowSDEOnPolicyBuffer:
    """Complete-episode buffer; consumed and cleared after each PPO update."""

    def __init__(self) -> None:
        self._episodes: list[FlowSDEEpisode] = []

    def __len__(self) -> int:
        return sum(len(episode.transitions) for episode in self._episodes)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def add_episode(self, episode: FlowSDEEpisode) -> None:
        if not isinstance(episode, FlowSDEEpisode):
            raise TypeError("Flow-SDE buffer only accepts complete FlowSDEEpisode values")
        self._episodes.append(episode)

    def extend(self, episodes: Iterable[FlowSDEEpisode]) -> None:
        for episode in episodes:
            self.add_episode(episode)

    def clear(self) -> None:
        self._episodes.clear()

    def build_batch(self, *, discount: float, gae_lambda: float) -> FlowSDETrainingBatch:
        if not self._episodes:
            raise ValueError("Cannot build a Flow-SDE batch from an empty buffer")

        transitions: list[FlowSDETransition] = []
        advantages: list[Tensor] = []
        returns: list[Tensor] = []
        episode_returns: list[float] = []
        for episode in self._episodes:
            episode_transitions = list(episode.transitions)
            device = episode_transitions[0].conditioning.device
            rewards = torch.tensor(
                [[item.reward] for item in episode_transitions],
                device=device,
                dtype=torch.float32,
            )
            old_values = torch.tensor(
                [item.old_value for item in episode_transitions] + [episode.bootstrap_value],
                device=device,
                dtype=torch.float32,
            )[:, None]
            terminated = torch.tensor(
                [[item.terminated] for item in episode_transitions],
                device=device,
                dtype=torch.bool,
            )
            episode_advantages, episode_value_targets = generalized_advantage_estimate(
                rewards,
                old_values,
                terminated,
                discount=discount,
                gae_lambda=gae_lambda,
            )
            transitions.extend(episode_transitions)
            advantages.append(episode_advantages[:, 0])
            returns.append(episode_value_targets[:, 0])
            episode_returns.append(episode.episode_return)

        device = transitions[0].conditioning.device
        if any(item.conditioning.device != device for item in transitions):
            raise ValueError("All Flow-SDE buffer episodes must share one device")
        rollout = FlowSDERollout(
            chains=torch.cat([item.rollout.chains for item in transitions], dim=0),
            denoise_indices=torch.cat([item.rollout.denoise_indices for item in transitions], dim=0),
            old_log_probs=torch.cat([item.rollout.old_log_probs for item in transitions], dim=0),
            action_mask=torch.cat([item.rollout.action_mask for item in transitions], dim=0),
        )
        return FlowSDETrainingBatch(
            conditioning=torch.stack([item.conditioning for item in transitions]),
            rollout=rollout,
            rewards=torch.tensor([item.reward for item in transitions], device=device),
            terminated=torch.tensor([item.terminated for item in transitions], device=device),
            truncated=torch.tensor([item.truncated for item in transitions], device=device),
            old_values=torch.tensor([item.old_value for item in transitions], device=device),
            advantages=torch.cat(advantages),
            returns=torch.cat(returns),
            episode_returns=torch.tensor(episode_returns, device=device),
        )
