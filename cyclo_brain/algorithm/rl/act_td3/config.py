"""Configuration for the Cyclo ACT chunk-SMDP TD3 recipe."""

from __future__ import annotations

import math
from dataclasses import dataclass


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

    This is not the scalar-action :class:`TD3Config`. The actor objective is
    the official ACT CVAE loss plus a deterministic deployed-path BC anchor
    and a delayed Q1 objective. The latter is linearly ramped after a frozen
    critic warm-up, matching the conservative recipe validated in cyclo_lab.
    """

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

    def __post_init__(self) -> None:
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


__all__ = ["ACTTD3Config"]
