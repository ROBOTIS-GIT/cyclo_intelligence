"""Twin Delayed Deep Deterministic Policy Gradient primitives."""

from .config import TD3Config
from .functional import (
    bellman_target,
    clipped_target_action,
    critic_loss,
    deterministic_actor_loss,
    policy_update_is_due,
    polyak_update_,
)
from .learner import TD3Batch, TD3Learner, TD3UpdateResult

__all__ = [
    "TD3Config",
    "TD3Batch",
    "TD3Learner",
    "TD3UpdateResult",
    "bellman_target",
    "clipped_target_action",
    "critic_loss",
    "deterministic_actor_loss",
    "policy_update_is_due",
    "polyak_update_",
]
