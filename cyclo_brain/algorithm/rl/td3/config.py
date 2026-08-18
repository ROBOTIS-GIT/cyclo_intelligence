"""Configuration for the TD3 algorithm defined by Fujimoto et al. (2018)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TD3Config:
    """Model-independent TD3 hyperparameters.

    Action bounds and noise scales belong to the model/action contract and are
    deliberately not represented as one scalar here.
    """

    discount: float = 0.99
    target_update_rate: float = 0.005
    target_policy_noise: float = 0.2
    target_policy_noise_clip: float = 0.5
    policy_update_period: int = 2
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4

    def __post_init__(self) -> None:
        for name, value in (
            ("discount", self.discount),
            ("target_update_rate", self.target_update_rate),
            ("target_policy_noise", self.target_policy_noise),
            ("target_policy_noise_clip", self.target_policy_noise_clip),
            ("actor_learning_rate", self.actor_learning_rate),
            ("critic_learning_rate", self.critic_learning_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"TD3 {name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"TD3 {name} must be finite")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("TD3 discount must be in [0, 1]")
        if not 0.0 < self.target_update_rate <= 1.0:
            raise ValueError("TD3 target_update_rate must be in (0, 1]")
        if self.target_policy_noise < 0.0:
            raise ValueError("TD3 target_policy_noise must be non-negative")
        if self.target_policy_noise_clip < 0.0:
            raise ValueError("TD3 target_policy_noise_clip must be non-negative")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("TD3 learning rates must be positive")
        if (
            isinstance(self.policy_update_period, bool)
            or not isinstance(self.policy_update_period, int)
            or self.policy_update_period < 1
        ):
            raise ValueError("TD3 policy_update_period must be a positive integer")
