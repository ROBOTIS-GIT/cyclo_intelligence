"""Configuration for the Cyclo ACT chunk-SMDP TD3 recipe."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cyclo_brain.model.act import (
    ACT_TRAINABLE_GROUPS,
    canonicalize_act_trainable_groups,
)


ACT_TD3_ACTOR_OBJECTIVES = ("td3", "td3_bc")
"""Canonical actor objectives exposed by the ACT-TD3 training contract."""


def canonicalize_act_td3_actor_objective(value: str) -> str:
    """Validate one exact, checkpoint-stable ACT-TD3 actor objective ID."""

    if not isinstance(value, str):
        raise TypeError("ACT-TD3 actor_objective must be a string")
    if value not in ACT_TD3_ACTOR_OBJECTIVES:
        raise ValueError(
            "ACT-TD3 actor_objective must be one of: "
            + ", ".join(ACT_TD3_ACTOR_OBJECTIVES)
        )
    return value


def _finite_real(value: float, name: str, *, minimum: float, inclusive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"ACT-TD3 {name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or (
        numeric < minimum if inclusive else numeric <= minimum
    ):
        relation = "at least" if inclusive else "greater than"
        raise ValueError(f"ACT-TD3 {name} must be finite and {relation} {minimum}")


@dataclass(frozen=True)
class ACTTD3Config:
    """Algorithm settings for an ACT executed-prefix macro policy.

    This is not the scalar-action :class:`TD3Config`. ``td3`` uses only the
    delayed ``-Q1`` chunk objective. ``td3_bc`` adds the official ACT CVAE loss
    and a deterministic deployed-path BC anchor on successful episodes only;
    its Q coefficient is linearly ramped for conservative offline training.
    """

    actor_objective: str = "td3_bc"
    discount: float = 0.99
    discount_reference_hz: float = 10.0
    target_update_rate: float = 0.005
    target_policy_noise: float = 0.2
    target_policy_noise_clip: float = 0.5
    policy_update_period: int = 2
    critic_warmup_updates: int = 5_000
    actor_learning_rate: float = 1.0e-5
    critic_learning_rate: float = 3.0e-4
    actor_weight_decay: float = 1.0e-4
    actor_gradient_clip_norm: float | None = 10.0
    critic_gradient_clip_norm: float | None = 10.0
    cvae_bc_weight: float = 1.0
    deterministic_bc_weight: float = 1.0
    q_weight_max: float = 0.25
    q_weight_ramp_actor_updates: int = 1_000
    actor_trainable_groups: tuple[str, ...] = ACT_TRAINABLE_GROUPS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "actor_objective",
            canonicalize_act_td3_actor_objective(self.actor_objective),
        )
        actor_trainable_groups = canonicalize_act_trainable_groups(
            self.actor_trainable_groups
        )
        # The deployed zero-latent action path used by pure TD3 never traverses
        # the target-action-only CVAE encoder. Keep default/all-group configs
        # usable while ensuring checkpoints do not falsely advertise it as a
        # trainable pure-TD3 parameter group.
        if self.actor_objective == "td3":
            actor_trainable_groups = tuple(
                group
                for group in actor_trainable_groups
                if group != "cvae_encoder"
            )
        object.__setattr__(
            self,
            "actor_trainable_groups",
            actor_trainable_groups,
        )
        _finite_real(self.discount, "discount", minimum=0.0, inclusive=False)
        if self.discount > 1.0:
            raise ValueError("ACT-TD3 discount must be at most 1")
        _finite_real(
            self.discount_reference_hz,
            "discount_reference_hz",
            minimum=0.0,
            inclusive=False,
        )
        _finite_real(
            self.target_update_rate,
            "target_update_rate",
            minimum=0.0,
            inclusive=False,
        )
        if self.target_update_rate > 1.0:
            raise ValueError("ACT-TD3 target_update_rate must be at most 1")
        for name, value in (
            ("target_policy_noise", self.target_policy_noise),
            ("target_policy_noise_clip", self.target_policy_noise_clip),
            ("actor_weight_decay", self.actor_weight_decay),
            ("q_weight_max", self.q_weight_max),
        ):
            _finite_real(value, name, minimum=0.0, inclusive=True)
        for name, value in (
            ("actor_learning_rate", self.actor_learning_rate),
            ("critic_learning_rate", self.critic_learning_rate),
            ("cvae_bc_weight", self.cvae_bc_weight),
            ("deterministic_bc_weight", self.deterministic_bc_weight),
        ):
            _finite_real(value, name, minimum=0.0, inclusive=False)
        for name, value in (
            ("actor_gradient_clip_norm", self.actor_gradient_clip_norm),
            ("critic_gradient_clip_norm", self.critic_gradient_clip_norm),
        ):
            if value is not None:
                _finite_real(value, name, minimum=0.0, inclusive=False)
        for name, value, minimum in (
            ("policy_update_period", self.policy_update_period, 1),
            ("critic_warmup_updates", self.critic_warmup_updates, 0),
            ("q_weight_ramp_actor_updates", self.q_weight_ramp_actor_updates, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"ACT-TD3 {name} must be at least {minimum}")


__all__ = [
    "ACT_TD3_ACTOR_OBJECTIVES",
    "ACTTD3Config",
    "canonicalize_act_td3_actor_objective",
]
