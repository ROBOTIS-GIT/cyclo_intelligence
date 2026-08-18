"""One-step TD3 learner independent of replay buffers and environments."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .config import TD3Config
from .functional import (
    bellman_target,
    clipped_target_action,
    critic_loss,
    deterministic_actor_loss,
    policy_update_is_due,
    polyak_update_,
)


@dataclass(frozen=True)
class TD3Batch:
    """One-step continuous-control transitions.

    ``terminated`` is true only for an MDP terminal. A time-limit truncation
    with a valid final observation must remain false so TD3 can bootstrap.
    """

    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor

    def __post_init__(self) -> None:
        floating = {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
        }
        if any(not isinstance(value, Tensor) for value in (*floating.values(), self.terminated)):
            raise TypeError("TD3 batch values must be torch tensors")
        if (
            self.observations.ndim != 2
            or self.observations.shape[0] < 1
            or self.observations.shape[1] < 1
        ):
            raise ValueError("TD3 observations must have non-empty shape (batch, obs_dim)")
        if self.next_observations.shape != self.observations.shape:
            raise ValueError("TD3 next observations must match observation shape")
        batch_size = self.observations.shape[0]
        if (
            self.actions.ndim != 2
            or self.actions.shape[0] != batch_size
            or self.actions.shape[1] < 1
        ):
            raise ValueError("TD3 actions must have shape (batch, action_dim)")
        if self.rewards.shape != (batch_size, 1):
            raise ValueError("TD3 rewards must have shape (batch, 1)")
        if self.terminated.shape != (batch_size, 1) or self.terminated.dtype != torch.bool:
            raise ValueError("TD3 terminated must be boolean with shape (batch, 1)")
        if any(not value.is_floating_point() for value in floating.values()):
            raise ValueError("TD3 observations, actions, and rewards must be floating tensors")
        reference = self.observations
        if any(
            value.device != reference.device or value.dtype != reference.dtype
            for value in floating.values()
        ):
            raise ValueError("TD3 floating batch values must share dtype and device")
        if self.terminated.device != reference.device:
            raise ValueError("TD3 terminated must share the batch device")
        if any(not bool(torch.isfinite(value).all()) for value in floating.values()):
            raise ValueError("TD3 batch floating values must be finite")


@dataclass(frozen=True)
class TD3UpdateResult:
    critic_loss: float
    actor_loss: float | None
    target_mean: float
    actor_updated: bool
    completed_critic_updates: int


def _module_parameters(module: nn.Module, name: str) -> tuple[nn.Parameter, ...]:
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError(f"TD3 {name} must have trainable parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or len(dtypes) != 1 or not next(iter(dtypes)).is_floating_point:
        raise ValueError(f"TD3 {name} parameters must share one floating dtype and device")
    return parameters


class TD3Learner:
    """Canonical TD3 update orchestration for vector observations/actions.

    The supplied actor must expose ``observation_dim``, ``action_dim``,
    ``action_low``, and ``action_high``. The twin critic must expose matching
    dimensions, return ``(q1, q2)``, and provide its first Q-function as
    ``critic.q1``. The reference :mod:`cyclo_brain.model.mlp` models satisfy
    this contract.
    """

    STATE_FORMAT = "cyclo_brain.td3_learner/v1"

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        config: TD3Config | None = None,
    ) -> None:
        if not isinstance(actor, nn.Module) or not isinstance(critic, nn.Module):
            raise TypeError("TD3 actor and critic must be torch modules")
        self.config = config or TD3Config()
        if not isinstance(self.config, TD3Config):
            raise TypeError("TD3 learner config must be TD3Config")
        self.actor = actor
        self.critic = critic
        actor_parameters = _module_parameters(actor, "actor")
        critic_parameters = _module_parameters(critic, "critic")
        if {id(value) for value in actor_parameters} & {
            id(value) for value in critic_parameters
        }:
            raise ValueError("TD3 actor and critic parameters must be independent")

        self.observation_dim = getattr(actor, "observation_dim", None)
        self.action_dim = getattr(actor, "action_dim", None)
        if (
            isinstance(self.observation_dim, bool)
            or not isinstance(self.observation_dim, int)
            or self.observation_dim < 1
            or isinstance(self.action_dim, bool)
            or not isinstance(self.action_dim, int)
            or self.action_dim < 1
        ):
            raise ValueError("TD3 actor must expose positive observation/action dimensions")
        if (
            getattr(critic, "observation_dim", None) != self.observation_dim
            or getattr(critic, "action_dim", None) != self.action_dim
            or not isinstance(getattr(critic, "q1", None), nn.Module)
            or not isinstance(getattr(critic, "q2", None), nn.Module)
        ):
            raise ValueError("TD3 twin critic contract does not match the actor")
        if {
            id(parameter) for parameter in critic.q1.parameters()
        } & {id(parameter) for parameter in critic.q2.parameters()}:
            raise ValueError("TD3 Q1 and Q2 parameters must be independent")

        action_low = getattr(actor, "action_low", None)
        action_high = getattr(actor, "action_high", None)
        if (
            not isinstance(action_low, Tensor)
            or not isinstance(action_high, Tensor)
            or action_low.shape != (self.action_dim,)
            or action_high.shape != (self.action_dim,)
            or not action_low.is_floating_point()
            or action_low.dtype != action_high.dtype
            or action_low.device != action_high.device
            or not bool(torch.isfinite(action_low).all())
            or not bool(torch.isfinite(action_high).all())
            or bool((action_low >= action_high).any())
        ):
            raise ValueError("TD3 actor action bounds are invalid")
        self.action_low = action_low.detach().clone()
        self.action_high = action_high.detach().clone()

        self.actor.train()
        self.critic.train()
        self.actor_target = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            actor_parameters,
            lr=self.config.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=self.config.critic_learning_rate,
        )
        self.completed_critic_updates = 0

    def _validate_batch(self, batch: TD3Batch) -> None:
        if not isinstance(batch, TD3Batch):
            raise TypeError("TD3 update requires TD3Batch")
        if batch.observations.shape[1] != self.observation_dim:
            raise ValueError("TD3 batch observation dimension does not match actor")
        if batch.actions.shape[1] != self.action_dim:
            raise ValueError("TD3 batch action dimension does not match actor")
        actor_parameter = next(self.actor.parameters())
        critic_parameter = next(self.critic.parameters())
        if (
            batch.observations.device != actor_parameter.device
            or batch.observations.dtype != actor_parameter.dtype
            or critic_parameter.device != actor_parameter.device
            or critic_parameter.dtype != actor_parameter.dtype
        ):
            raise ValueError("TD3 batch, actor, and critic must share dtype and device")

    @torch.no_grad()
    def compute_bellman_targets(
        self,
        batch: TD3Batch,
        *,
        target_noise: Tensor | None = None,
    ) -> Tensor:
        """Compute the clipped-double-Q target using frozen target networks.

        An injected ``target_noise`` is interpreted as pre-clipped additive
        action-space noise. When omitted, Gaussian noise with the configured
        standard deviation is sampled.
        """

        self._validate_batch(batch)
        target_actions = self.actor_target(batch.next_observations)
        if target_noise is None:
            resolved_noise = (
                torch.randn_like(target_actions) * self.config.target_policy_noise
            )
        else:
            if (
                not isinstance(target_noise, Tensor)
                or target_noise.shape != target_actions.shape
                or not target_noise.is_floating_point()
                or target_noise.device != target_actions.device
                or target_noise.dtype != target_actions.dtype
                or not bool(torch.isfinite(target_noise).all())
            ):
                raise ValueError("TD3 injected target noise must match target actions")
            resolved_noise = target_noise
        smoothed_actions = clipped_target_action(
            target_actions,
            resolved_noise,
            noise_clip=self.config.target_policy_noise_clip,
            action_low=self.action_low.to(target_actions),
            action_high=self.action_high.to(target_actions),
        )
        target_q1, target_q2 = self.critic_target(
            batch.next_observations,
            smoothed_actions,
        )
        return bellman_target(
            batch.rewards,
            batch.terminated,
            target_q1,
            target_q2,
            discount=self.config.discount,
        )

    def update(
        self,
        batch: TD3Batch,
        *,
        target_noise: Tensor | None = None,
    ) -> TD3UpdateResult:
        """Perform one critic update and a due delayed actor/target update."""

        targets = self.compute_bellman_targets(batch, target_noise=target_noise)
        q1, q2 = self.critic(batch.observations, batch.actions)
        loss_critic = critic_loss(q1, q2, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss_critic.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.completed_critic_updates += 1

        actor_updated = policy_update_is_due(
            self.completed_critic_updates,
            period=self.config.policy_update_period,
        )
        loss_actor: Tensor | None = None
        if actor_updated:
            critic_gradient_flags = tuple(
                (parameter, parameter.requires_grad)
                for parameter in self.critic.parameters()
            )
            for parameter, _requires_grad in critic_gradient_flags:
                parameter.requires_grad_(False)
            try:
                policy_actions = self.actor(batch.observations)
                q1_for_policy = self.critic.q1(batch.observations, policy_actions)
                loss_actor = deterministic_actor_loss(q1_for_policy)
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss_actor.backward()
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad(set_to_none=True)
            finally:
                for parameter, requires_grad in critic_gradient_flags:
                    parameter.requires_grad_(requires_grad)

            polyak_update_(
                self.actor,
                self.actor_target,
                tau=self.config.target_update_rate,
            )
            polyak_update_(
                self.critic,
                self.critic_target,
                tau=self.config.target_update_rate,
            )

        return TD3UpdateResult(
            critic_loss=float(loss_critic.detach().item()),
            actor_loss=(
                None if loss_actor is None else float(loss_actor.detach().item())
            ),
            target_mean=float(targets.mean().item()),
            actor_updated=actor_updated,
            completed_critic_updates=self.completed_critic_updates,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return network, optimizer, and update-schedule state.

        Replay data and random-number-generator state are infrastructure state
        and are deliberately not owned by this learner.
        """

        return {
            "format": self.STATE_FORMAT,
            "config": asdict(self.config),
            "action_low": self.action_low,
            "action_high": self.action_high,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "completed_critic_updates": self.completed_critic_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore compatible learner state without re-synchronizing targets."""

        if not isinstance(state, Mapping) or state.get("format") != self.STATE_FORMAT:
            raise ValueError("TD3 learner state format is invalid")
        if state.get("config") != asdict(self.config):
            raise ValueError("TD3 learner state config does not match")
        for name, expected in (
            ("action_low", self.action_low),
            ("action_high", self.action_high),
        ):
            value = state.get(name)
            if not isinstance(value, Tensor) or value.shape != expected.shape:
                raise ValueError(f"TD3 learner state {name} is invalid")
            if not torch.equal(value.to(expected), expected):
                raise ValueError(f"TD3 learner state {name} does not match")
        completed_updates = state.get("completed_critic_updates")
        if (
            isinstance(completed_updates, bool)
            or not isinstance(completed_updates, int)
            or completed_updates < 0
        ):
            raise ValueError("TD3 learner update counter is invalid")

        self.actor.load_state_dict(state["actor"], strict=True)
        self.actor_target.load_state_dict(state["actor_target"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.critic_target.load_state_dict(state["critic_target"], strict=True)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.completed_critic_updates = completed_updates
        self.actor.train()
        self.critic.train()
        self.actor_target.eval().requires_grad_(False)
        self.critic_target.eval().requires_grad_(False)


__all__ = ["TD3Batch", "TD3Learner", "TD3UpdateResult"]
