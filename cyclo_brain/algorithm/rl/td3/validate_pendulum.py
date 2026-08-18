"""End-to-end reference validation of the MLP TD3 baseline on Pendulum.

This module is deliberately a validation harness, not training infrastructure.
It reuses LeRobot's vendored replay buffer and keeps Gymnasium out of the core
TD3 learner. Run it in an environment that provides both dependencies:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m cyclo_brain.algorithm.rl.td3.validate_pendulum

The thread limits avoid severe CPU oversubscription in the project's LeRobot
container; they do not change the TD3 update equations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from cyclo_brain.model.mlp import TD3MLPActor, TD3MLPTwinCritic

from .config import TD3Config
from .learner import TD3Batch, TD3Learner


_OBSERVATION_KEY = "observation.state"


@dataclass(frozen=True)
class PendulumValidationConfig:
    """Canonical small-scale TD3 validation settings.

    The network and algorithm defaults follow the authors' reference TD3.
    ``exploration_noise_fraction`` is multiplied by each action dimension's
    half-range, matching ``expl_noise * max_action`` on symmetric bounds.
    """

    environment_id: str = "Pendulum-v1"
    training_steps: int = 25_000
    random_action_steps: int = 5_000
    replay_capacity: int = 200_000
    batch_size: int = 256
    exploration_noise_fraction: float = 0.1
    evaluation_episodes: int = 10
    device: str = "cpu"

    def __post_init__(self) -> None:
        integer_fields = (
            ("training_steps", self.training_steps, 1),
            ("random_action_steps", self.random_action_steps, 0),
            ("replay_capacity", self.replay_capacity, 1),
            ("batch_size", self.batch_size, 1),
            ("evaluation_episodes", self.evaluation_episodes, 1),
        )
        if not isinstance(self.environment_id, str) or not self.environment_id:
            raise ValueError("TD3 validation environment_id must be non-empty")
        for name, value, minimum in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"TD3 validation {name} must be at least {minimum}")
        if self.random_action_steps >= self.training_steps:
            raise ValueError("TD3 validation needs updates after random-action collection")
        if self.replay_capacity < self.batch_size:
            raise ValueError("TD3 validation replay capacity must cover one batch")
        if (
            isinstance(self.exploration_noise_fraction, bool)
            or not isinstance(self.exploration_noise_fraction, (int, float))
            or not math.isfinite(float(self.exploration_noise_fraction))
            or self.exploration_noise_fraction < 0.0
        ):
            raise ValueError("TD3 validation exploration noise must be finite and non-negative")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("TD3 validation device must be non-empty")


@dataclass(frozen=True)
class PendulumSeedResult:
    seed: int
    initial_mean_return: float
    final_mean_return: float
    critic_updates: int
    actor_updates: int
    elapsed_seconds: float

    @property
    def return_improvement(self) -> float:
        return self.final_mean_return - self.initial_mean_return


def td3_batch_from_lerobot_sample(sample: dict[str, Any]) -> TD3Batch:
    """Adapt one LeRobot replay sample without treating truncation as terminal."""

    try:
        observations = sample["state"][_OBSERVATION_KEY]
        actions = sample["action"]
        rewards = sample["reward"]
        next_observations = sample["next_state"][_OBSERVATION_KEY]
        terminated = sample["done"]
    except (KeyError, TypeError) as error:
        raise ValueError("LeRobot replay sample is missing a TD3 transition field") from error

    tensors = (observations, actions, rewards, next_observations, terminated)
    if any(not isinstance(value, Tensor) for value in tensors):
        raise TypeError("LeRobot replay TD3 fields must be torch tensors")
    if rewards.ndim != 1 or terminated.ndim != 1:
        raise ValueError("LeRobot replay reward and done fields must be one-dimensional")

    return TD3Batch(
        observations=observations,
        actions=actions,
        rewards=rewards.unsqueeze(-1),
        next_observations=next_observations,
        terminated=terminated.to(dtype=torch.bool).unsqueeze(-1),
    )


def _vector_box_dimensions(space: Any, name: str) -> int:
    shape = getattr(space, "shape", None)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 1
        or shape[0] < 1
        or not isinstance(low, np.ndarray)
        or not isinstance(high, np.ndarray)
        or low.shape != shape
        or high.shape != shape
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(low >= high)
    ):
        raise ValueError(f"TD3 validation requires a finite vector Box {name} space")
    return int(shape[0])


def _reference_action_scale(action_low: np.ndarray, action_high: np.ndarray) -> float:
    """Return the scalar action scale assumed by the authors' TD3 code."""

    if not np.allclose(action_low, -action_high):
        raise ValueError("TD3 reference validation requires symmetric action bounds")
    half_ranges = (action_high - action_low) * 0.5
    if not np.allclose(half_ranges, half_ranges[0]):
        raise ValueError("TD3 reference validation requires one shared action scale")
    return float(half_ranges[0])


@torch.no_grad()
def _deterministic_action(actor: TD3MLPActor, observation: np.ndarray) -> np.ndarray:
    parameter = next(actor.parameters())
    observation_tensor = torch.as_tensor(
        observation,
        dtype=parameter.dtype,
        device=parameter.device,
    ).unsqueeze(0)
    return actor(observation_tensor).squeeze(0).cpu().numpy()


def _evaluate_actor(
    actor: TD3MLPActor,
    *,
    environment_id: str,
    seed: int,
    episodes: int,
) -> float:
    import gymnasium as gym

    environment = gym.make(environment_id)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                action = _deterministic_action(actor, observation)
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        environment.close()
    return float(np.mean(returns))


def run_pendulum_seed(
    seed: int,
    config: PendulumValidationConfig | None = None,
) -> PendulumSeedResult:
    """Train and evaluate one independently seeded standard TD3 baseline."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("TD3 validation seed must be a non-negative integer")
    resolved_config = config or PendulumValidationConfig()

    import gymnasium as gym
    from lerobot.rl.buffer import ReplayBuffer

    device = torch.device(resolved_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TD3 validation requested CUDA, but CUDA is unavailable")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    noise_generator = np.random.default_rng(seed)

    environment = gym.make(resolved_config.environment_id)
    observation_dim = _vector_box_dimensions(environment.observation_space, "observation")
    action_dim = _vector_box_dimensions(environment.action_space, "action")
    action_low = np.asarray(environment.action_space.low, dtype=np.float32)
    action_high = np.asarray(environment.action_space.high, dtype=np.float32)
    action_scale = (action_high - action_low) * 0.5
    reference_action_scale = _reference_action_scale(action_low, action_high)
    environment.action_space.seed(seed)

    actor = TD3MLPActor(observation_dim, action_low, action_high).to(device)
    critic = TD3MLPTwinCritic(observation_dim, action_dim).to(device)
    learner = TD3Learner(
        actor,
        critic,
        TD3Config(
            target_policy_noise=0.2 * reference_action_scale,
            target_policy_noise_clip=0.5 * reference_action_scale,
        ),
    )
    replay = ReplayBuffer(
        capacity=resolved_config.replay_capacity,
        device=str(device),
        state_keys=[_OBSERVATION_KEY],
        use_drq=False,
        storage_device="cpu",
        optimize_memory=False,
    )

    evaluation_seed = seed + 100_000
    initial_return = _evaluate_actor(
        actor,
        environment_id=resolved_config.environment_id,
        seed=evaluation_seed,
        episodes=resolved_config.evaluation_episodes,
    )
    observation, _ = environment.reset(seed=seed)
    actor_updates = 0
    start_time = time.monotonic()

    try:
        for step in range(resolved_config.training_steps):
            if step < resolved_config.random_action_steps:
                action = environment.action_space.sample()
            else:
                action = _deterministic_action(actor, observation)
                action += noise_generator.normal(
                    loc=0.0,
                    scale=resolved_config.exploration_noise_fraction * action_scale,
                    size=action_dim,
                ).astype(np.float32)
                action = np.clip(action, action_low, action_high)

            next_observation, reward, terminated, truncated, _ = environment.step(action)
            replay.add(
                state={
                    _OBSERVATION_KEY: torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                },
                action=torch.as_tensor(action, dtype=torch.float32).unsqueeze(0),
                reward=float(reward),
                next_state={
                    _OBSERVATION_KEY: torch.as_tensor(
                        next_observation,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                },
                done=bool(terminated),
                truncated=bool(truncated),
            )

            if step >= resolved_config.random_action_steps:
                batch = td3_batch_from_lerobot_sample(
                    replay.sample(resolved_config.batch_size)
                )
                update_result = learner.update(batch)
                actor_updates += int(update_result.actor_updated)

            if terminated or truncated:
                observation, _ = environment.reset()
            else:
                observation = next_observation
    finally:
        environment.close()

    elapsed_seconds = time.monotonic() - start_time
    final_return = _evaluate_actor(
        actor,
        environment_id=resolved_config.environment_id,
        seed=evaluation_seed,
        episodes=resolved_config.evaluation_episodes,
    )
    return PendulumSeedResult(
        seed=seed,
        initial_mean_return=initial_return,
        final_mean_return=final_return,
        critic_updates=learner.completed_critic_updates,
        actor_updates=actor_updates,
        elapsed_seconds=elapsed_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--training-steps", type=int, default=25_000)
    parser.add_argument("--random-action-steps", type=int, default=5_000)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PendulumValidationConfig(
        training_steps=args.training_steps,
        random_action_steps=args.random_action_steps,
        evaluation_episodes=args.evaluation_episodes,
        device=args.device,
    )
    results = [run_pendulum_seed(seed, config) for seed in args.seeds]
    payload = {
        "config": asdict(config),
        "results": [
            {**asdict(result), "return_improvement": result.return_improvement}
            for result in results
        ],
        "median_final_mean_return": float(
            np.median([result.final_mean_return for result in results])
        ),
        "median_return_improvement": float(
            np.median([result.return_improvement for result in results])
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PendulumSeedResult",
    "PendulumValidationConfig",
    "run_pendulum_seed",
    "td3_batch_from_lerobot_sample",
]
