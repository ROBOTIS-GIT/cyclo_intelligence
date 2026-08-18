"""Small multilayer-perceptron models for continuous-control baselines."""

from .td3 import TD3MLPActor, TD3MLPQFunction, TD3MLPTwinCritic

__all__ = [
    "TD3MLPActor",
    "TD3MLPQFunction",
    "TD3MLPTwinCritic",
]
