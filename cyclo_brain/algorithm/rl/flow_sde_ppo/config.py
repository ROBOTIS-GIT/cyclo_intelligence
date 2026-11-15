"""Configuration shared by Flow-SDE sampling and PPO updates."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FlowSDEPPOConfig:
    """Minimal Flow-SDE PPO contract for action-chunk policies.

    The first implementation intentionally follows the stable ``joint_logprob=False``
    setup: one denoising transition is stochastic per policy decision, and PPO
    clips its per-coordinate probability ratios.
    """

    num_denoising_steps: int = 4
    noise_level: float = 0.5
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    discount: float = 0.99
    gae_lambda: float = 0.95
    value_clip: float = 0.2
    value_loss_coefficient: float = 0.5
    actor_learning_rate: float = 3.0e-5
    value_learning_rate: float = 1.0e-4
    ppo_epochs: int = 4
    minibatch_size: int = 32
    actor_max_grad_norm: float = 1.0
    value_max_grad_norm: float = 1.0
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_denoising_steps, bool)
            or not isinstance(self.num_denoising_steps, int)
            or self.num_denoising_steps < 2
        ):
            raise ValueError("Flow-SDE num_denoising_steps must be an integer >= 2")

        for name, value in (
            ("noise_level", self.noise_level),
            ("clip_ratio_low", self.clip_ratio_low),
            ("clip_ratio_high", self.clip_ratio_high),
            ("discount", self.discount),
            ("gae_lambda", self.gae_lambda),
            ("value_clip", self.value_clip),
            ("value_loss_coefficient", self.value_loss_coefficient),
            ("actor_learning_rate", self.actor_learning_rate),
            ("value_learning_rate", self.value_learning_rate),
            ("actor_max_grad_norm", self.actor_max_grad_norm),
            ("value_max_grad_norm", self.value_max_grad_norm),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Flow-SDE {name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"Flow-SDE {name} must be finite")

        if self.noise_level <= 0.0:
            raise ValueError("Flow-SDE noise_level must be positive")
        if self.clip_ratio_low < 0.0 or self.clip_ratio_high < 0.0:
            raise ValueError("Flow-SDE PPO clip ratios must be non-negative")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("Flow-SDE discount must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("Flow-SDE gae_lambda must be in [0, 1]")
        if self.value_clip < 0.0:
            raise ValueError("Flow-SDE value_clip must be non-negative")
        if self.value_loss_coefficient < 0.0:
            raise ValueError("Flow-SDE value_loss_coefficient must be non-negative")
        if self.actor_learning_rate <= 0.0 or self.value_learning_rate <= 0.0:
            raise ValueError("Flow-SDE learning rates must be positive")
        if self.actor_max_grad_norm <= 0.0 or self.value_max_grad_norm <= 0.0:
            raise ValueError("Flow-SDE gradient clip norms must be positive")
        for name, value in (
            ("ppo_epochs", self.ppo_epochs),
            ("minibatch_size", self.minibatch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Flow-SDE {name} must be a positive integer")
        if not isinstance(self.normalize_advantages, bool):
            raise TypeError("Flow-SDE normalize_advantages must be boolean")
