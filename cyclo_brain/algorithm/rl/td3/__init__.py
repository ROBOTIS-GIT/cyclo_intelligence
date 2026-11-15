"""Shared Twin Delayed Deep Deterministic Policy Gradient primitives."""

from .functional import (
    bellman_target,
    clipped_target_action,
    critic_loss,
    deterministic_actor_loss,
    policy_update_is_due,
    polyak_update_,
)

__all__ = [
    "bellman_target",
    "clipped_target_action",
    "critic_loss",
    "deterministic_actor_loss",
    "policy_update_is_due",
    "polyak_update_",
]
