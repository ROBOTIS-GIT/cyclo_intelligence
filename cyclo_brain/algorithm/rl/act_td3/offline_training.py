"""Versioned cumulative-replay training rounds for offline ACT-TD3."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from cyclo_brain.model.act import ACT_TRAINABLE_GROUPS

from .learner import ACTTD3Learner, ACTTD3UpdateResult
from .lerobot_offline import (
    ACTTD3LeRobotCollator,
    FixedHorizonLeRobotACTTD3Dataset,
    VirtualCumulativeLeRobotACTTD3Dataset,
)
from .offline_warmup import _atomic_torch_save, _positive_integer
from .training_identity import ACTTD3TrainingDataIdentity


ProgressCallback = Callable[["ACTTD3OfflineTrainingProgress"], None]
StopPredicate = Callable[[], bool]


@dataclass(frozen=True)
class ACTTD3OfflineTrainingProgress:
    """One JSON-friendly progress snapshot for a cumulative replay round."""

    status: str
    round_index: int
    episode_count: int
    completed_epochs: int
    total_epochs: int
    completed_critic_updates: int
    total_critic_updates: int
    completed_actor_updates: int
    total_actor_updates: int
    percentage: float
    critic_loss: float | None
    actor_loss: float | None
    elapsed_seconds: float
    eta_seconds: float | None
    durable_critic_updates: int
    checkpoint_path: str


def _checked_episode_indices(identity: ACTTD3TrainingDataIdentity) -> tuple[int, ...]:
    raw = identity.virtual_contract.get("episode_indices")
    if (
        not isinstance(raw, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        or any(value < 0 for value in raw)
        or raw != sorted(set(raw))
    ):
        raise ValueError("ACT-TD3 training identity episode indices are invalid")
    return tuple(raw)


def _checked_data_roots(
    identity: ACTTD3TrainingDataIdentity | Mapping[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    """Validate and return the ordered immutable-root contract when present."""

    if isinstance(identity, ACTTD3TrainingDataIdentity):
        virtual = identity.virtual_contract
    elif isinstance(identity, Mapping):
        virtual = identity.get("virtual_contract")
        if not isinstance(virtual, Mapping):
            raise ValueError("ACT-TD3 training identity virtual contract is invalid")
    else:
        raise TypeError("ACT-TD3 training identity is invalid")
    raw = virtual.get("data_roots")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("ACT-TD3 ordered data root contract is invalid")
    roots: list[dict[str, Any]] = []
    expected_global = 0
    seen_paths: set[str] = set()
    for ordinal, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise ValueError("ACT-TD3 ordered data root entry is invalid")
        required = {
            "ordinal",
            "root",
            "name",
            "identity",
            "dataset_sha256",
            "episode_indices",
            "global_episode_indices",
            "file_count",
            "byte_count",
        }
        if set(value) != required or value.get("ordinal") != ordinal:
            raise ValueError("ACT-TD3 ordered data root entry fields disagree")
        root = value.get("root")
        name = value.get("name")
        identity_value = value.get("identity")
        dataset_sha256 = value.get("dataset_sha256")
        local_indices = value.get("episode_indices")
        global_indices = value.get("global_episode_indices")
        file_count = value.get("file_count")
        byte_count = value.get("byte_count")
        if (
            not isinstance(root, str)
            or not root
            or root in seen_paths
            or not isinstance(name, str)
            or not name
            or not isinstance(identity_value, str)
            or not identity_value.startswith("sha256:")
            or not isinstance(dataset_sha256, str)
            or not dataset_sha256.startswith("sha256:")
            or not isinstance(local_indices, list)
            or not local_indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in local_indices
            )
            or len(set(local_indices)) != len(local_indices)
            or not isinstance(global_indices, list)
            or global_indices
            != list(range(expected_global, expected_global + len(local_indices)))
            or isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count < 1
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise ValueError("ACT-TD3 ordered data root entry is invalid")
        roots.append(dict(value))
        seen_paths.add(root)
        expected_global += len(local_indices)
    episode_indices = virtual.get("episode_indices")
    if episode_indices != list(range(expected_global)):
        raise ValueError("ACT-TD3 data roots and global episode indices disagree")
    return tuple(roots)


def _single_identity_matches_root(
    single: Mapping[str, Any],
    root: Mapping[str, Any],
) -> bool:
    component_sha256 = single.get("component_sha256")
    return (
        isinstance(component_sha256, Mapping)
        and single.get("identity") == root.get("identity")
        and component_sha256.get("dataset") == root.get("dataset_sha256")
    )


def _training_identity_same_or_single_root_upgrade(
    previous: Any,
    current: ACTTD3TrainingDataIdentity,
) -> bool:
    if previous == current.to_dict():
        return True
    if not isinstance(previous, Mapping):
        return False
    previous_roots = _checked_data_roots(previous)
    current_roots = _checked_data_roots(current)
    return (
        previous_roots is None
        and current_roots is not None
        and len(current_roots) == 1
        and _single_identity_matches_root(previous, current_roots[0])
    )


class ACTTD3OfflineTrainingRunner:
    """Train one versioned round over the entire current cumulative replay.

    Every round makes ``critic_epochs`` exact passes over replay. One critic
    update is made for every batch, and the fixed TD3
    ``policy_update_period=2`` schedule interleaves one actor update for every
    two critic updates. ``actor_equivalent_epochs`` is therefore not an
    independent actor-only phase and must be exactly half ``critic_epochs``.

    A completed round may seed a new checkpoint after one or more immutable
    LeRobot data-epoch roots are appended. The ordered root identity must retain
    the exact prior prefix; parquet/video files are never physically merged.
    Legacy one-root checkpoints remain readable and can be upgraded at the
    resume boundary when the first virtual root has the same content identity.
    """

    STATE_FORMAT = "cyclo_brain.act_td3_offline_training/v2"
    SAMPLING = "one_random_permutation_per_replay_epoch"
    CRITIC_EPOCHS = 10
    POLICY_UPDATE_PERIOD = 2
    ACTOR_EQUIVALENT_EPOCHS = 5
    MAX_EPISODES = 200
    ROUND_EPISODES = 50

    def __init__(
        self,
        learner: ACTTD3Learner,
        dataset: (
            FixedHorizonLeRobotACTTD3Dataset
            | VirtualCumulativeLeRobotACTTD3Dataset
        ),
        collator: ACTTD3LeRobotCollator,
        *,
        batch_size: int,
        sampling_seed: int,
        training_data_identity: ACTTD3TrainingDataIdentity,
        checkpoint_path: str | Path,
        resume_from: str | Path | None = None,
        critic_epochs: int = CRITIC_EPOCHS,
        actor_equivalent_epochs: int = ACTOR_EQUIVALENT_EPOCHS,
        checkpoint_interval: int = 100,
        progress_interval: int = 1,
    ) -> None:
        if not isinstance(learner, ACTTD3Learner):
            raise TypeError("ACT-TD3 offline training requires ACTTD3Learner")
        if not isinstance(
            dataset,
            (
                FixedHorizonLeRobotACTTD3Dataset,
                VirtualCumulativeLeRobotACTTD3Dataset,
            ),
        ):
            raise TypeError(
                "ACT-TD3 offline training requires a fixed-horizon LeRobot replay"
            )
        if not isinstance(collator, ACTTD3LeRobotCollator):
            raise TypeError("ACT-TD3 offline training requires ACTTD3LeRobotCollator")
        if not isinstance(training_data_identity, ACTTD3TrainingDataIdentity):
            raise TypeError(
                "ACT-TD3 offline training requires ACTTD3TrainingDataIdentity"
            )
        self.batch_size = _positive_integer(batch_size, "batch_size")
        if self.batch_size > len(dataset):
            raise ValueError(
                "ACT-TD3 offline training batch_size cannot exceed replay size"
            )
        if (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, int)
            or not 0 <= sampling_seed < 2**63 - 1
        ):
            raise ValueError("ACT-TD3 offline training sampling_seed is invalid")
        self.checkpoint_interval = _positive_integer(
            checkpoint_interval, "checkpoint_interval"
        )
        self.progress_interval = _positive_integer(
            progress_interval, "progress_interval"
        )
        self.critic_epochs = _positive_integer(critic_epochs, "critic_epochs")
        self.actor_equivalent_epochs = _positive_integer(
            actor_equivalent_epochs,
            "actor_equivalent_epochs",
        )
        if (
            self.critic_epochs
            != self.POLICY_UPDATE_PERIOD * self.actor_equivalent_epochs
        ):
            raise ValueError(
                "ACT-TD3 critic_epochs must equal policy_update_period "
                "times actor_equivalent_epochs"
            )
        if learner.config.critic_warmup_updates != 0:
            raise ValueError(
                "ACT-TD3 staged training requires critic_warmup_updates=0; "
                "critic epochs already interleave delayed actor updates"
            )
        if learner.config.policy_update_period != self.POLICY_UPDATE_PERIOD:
            raise ValueError("ACT-TD3 staged training requires policy_update_period=2")
        if learner.completed_critic_updates != 0 or learner.completed_actor_updates != 0:
            raise ValueError(
                "ACT-TD3 staged runner requires a fresh learner; resume through "
                "a versioned runner checkpoint"
            )
        if dataset.execution_horizon != learner.execution_horizon:
            raise ValueError("ACT-TD3 staged dataset execution horizon disagrees")
        if dataset.action_dim != learner.action_dim:
            raise ValueError("ACT-TD3 staged dataset action dimension disagrees")
        if float(dataset.fps) != float(learner.config.discount_reference_hz):
            raise ValueError("ACT-TD3 discount_reference_hz must exactly match dataset fps")
        if not 1 <= dataset.num_episodes <= self.MAX_EPISODES:
            raise ValueError(
                f"ACT-TD3 cumulative replay must contain 1..{self.MAX_EPISODES} episodes"
            )
        identity_indices = _checked_episode_indices(training_data_identity)
        _checked_data_roots(training_data_identity)
        dataset_indices = tuple(record[0] for record in dataset.episode_records)
        if identity_indices != dataset_indices:
            raise ValueError(
                "ACT-TD3 training identity and replay episode indices disagree"
            )

        self.learner = learner
        self.dataset = dataset
        self.collator = collator
        self.sampling_seed = sampling_seed
        self.training_data_identity = training_data_identity
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if self.checkpoint_path.exists() and self.checkpoint_path.is_dir():
            raise IsADirectoryError(self.checkpoint_path)
        self.resume_from = (
            Path(resume_from).expanduser().resolve() if resume_from is not None else None
        )
        if self.resume_from is None:
            if self.checkpoint_path.exists():
                raise FileExistsError(
                    f"ACT-TD3 round checkpoint already exists: {self.checkpoint_path}"
                )
            if dataset.num_episodes > self.ROUND_EPISODES:
                raise ValueError(
                    "ACT-TD3 first cumulative replay round must contain 1..50 episodes"
                )
        elif not self.resume_from.is_file():
            raise FileNotFoundError(self.resume_from)

        self.batches_per_epoch = math.ceil(len(dataset) / self.batch_size)
        self.total_critic_updates = self.critic_epochs * self.batches_per_epoch
        if self.total_critic_updates % self.POLICY_UPDATE_PERIOD:
            raise RuntimeError(
                "ACT-TD3 round cannot realize the requested actor-equivalent epochs"
            )
        self.total_actor_updates = (
            self.actor_equivalent_epochs * self.batches_per_epoch
        )
        self._sampler = torch.Generator(device="cpu").manual_seed(sampling_seed)
        self._permutation = torch.empty(0, dtype=torch.long)
        self._cursor = 0
        self._completed_epochs = 0
        self._round_index = 1
        self._new_episode_count = dataset.num_episodes
        self._round_start_critic_updates = 0
        self._round_start_actor_updates = 0
        self._history: tuple[dict[str, Any], ...] = ()
        self._elapsed_seconds = 0.0
        self._last_update: ACTTD3UpdateResult | None = None
        self._last_sampled_indices: tuple[int, ...] = ()
        self._durable_critic_updates = 0

        if self.resume_from is not None:
            self._load_checkpoint()

    @property
    def round_index(self) -> int:
        return self._round_index

    @property
    def last_sampled_indices(self) -> tuple[int, ...]:
        return self._last_sampled_indices

    @property
    def new_episode_count(self) -> int:
        return self._new_episode_count

    def _learner_contract(self) -> dict[str, Any]:
        return {
            "config": asdict(self.learner.config),
            "prediction_horizon": self.learner.prediction_horizon,
            "execution_horizon": self.learner.execution_horizon,
            "action_dim": self.learner.action_dim,
            "observation_keys": tuple(self.learner.critic.observation_keys),
            "action_domain": self.learner.ACTION_DOMAIN,
            "target_policy_smoothing": self.learner.TARGET_POLICY_SMOOTHING,
            "actor_q_gradient": self.learner.ACTOR_Q_GRADIENT,
            "action_clamp": False,
            "device": str(self.learner.device),
            "dtype": str(self.learner.dtype),
        }

    def _base_contract(self) -> dict[str, Any]:
        return {
            "sampling": self.SAMPLING,
            "sampling_seed": self.sampling_seed,
            "batch_size": self.batch_size,
            **self._schedule_contract(),
            "policy_update_period": self.POLICY_UPDATE_PERIOD,
            "max_episodes": self.MAX_EPISODES,
            "round_episodes": self.ROUND_EPISODES,
            "fps": float(self.dataset.fps),
            "execution_horizon": self.dataset.execution_horizon,
            "action_dim": self.dataset.action_dim,
            "learner": self._learner_contract(),
        }

    def _schedule_contract(self) -> dict[str, int]:
        return {
            "critic_epochs": self.critic_epochs,
            "actor_equivalent_epochs": self.actor_equivalent_epochs,
        }

    def _dataset_contract(self) -> dict[str, Any]:
        return {
            "training_data": self.training_data_identity.to_dict(),
            "transition_count": len(self.dataset),
            "episode_count": self.dataset.num_episodes,
            "success_count": self.dataset.num_successes,
            "failure_count": self.dataset.num_failures,
            "episode_records": self.dataset.episode_records,
            "batches_per_epoch": self.batches_per_epoch,
            "total_critic_updates": self.total_critic_updates,
            "total_actor_updates": self.total_actor_updates,
        }

    def _completed_critic_updates(self) -> int:
        return self.learner.completed_critic_updates - self._round_start_critic_updates

    def _completed_actor_updates(self) -> int:
        return self.learner.completed_actor_updates - self._round_start_actor_updates

    def _current_round_summary(self) -> dict[str, Any]:
        return {
            "round_index": self._round_index,
            "new_episode_count": self._new_episode_count,
            "schedule": self._schedule_contract(),
            "dataset": self._dataset_contract(),
            "completed_critic_updates": self._completed_critic_updates(),
            "completed_actor_updates": self._completed_actor_updates(),
        }

    def _checkpoint_state(self, elapsed_seconds: float) -> dict[str, Any]:
        return {
            "format": self.STATE_FORMAT,
            "base_contract": self._base_contract(),
            "history": self._history,
            "current_round": self._current_round_summary(),
            "round_start_critic_updates": self._round_start_critic_updates,
            "round_start_actor_updates": self._round_start_actor_updates,
            "completed_epochs": self._completed_epochs,
            "permutation": self._permutation.cpu().clone(),
            "cursor": self._cursor,
            "learner": self.learner.state_dict(),
            "sampler_state": self._sampler.get_state().cpu().clone(),
            "elapsed_seconds": float(elapsed_seconds),
            "last_update": (
                asdict(self._last_update) if self._last_update is not None else None
            ),
            "last_sampled_indices": self._last_sampled_indices,
        }

    def _save_checkpoint(self, elapsed_seconds: float) -> None:
        _atomic_torch_save(
            self.checkpoint_path,
            self._checkpoint_state(elapsed_seconds),
        )
        self._durable_critic_updates = self._completed_critic_updates()

    @staticmethod
    def _checkpoint_mapping(path: Path) -> Mapping[str, Any]:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("ACT-TD3 round checkpoint cannot be read") from error
        expected = {
            "format",
            "base_contract",
            "history",
            "current_round",
            "round_start_critic_updates",
            "round_start_actor_updates",
            "completed_epochs",
            "permutation",
            "cursor",
            "learner",
            "sampler_state",
            "elapsed_seconds",
            "last_update",
            "last_sampled_indices",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("ACT-TD3 round checkpoint fields disagree")
        if state["format"] != ACTTD3OfflineTrainingRunner.STATE_FORMAT:
            raise ValueError("ACT-TD3 round checkpoint format disagrees")
        return state

    def _validate_restored_progress(self, state: Mapping[str, Any]) -> None:
        completed_epochs = state["completed_epochs"]
        cursor = state["cursor"]
        permutation = state["permutation"]
        if (
            isinstance(completed_epochs, bool)
            or not isinstance(completed_epochs, int)
            or not 0 <= completed_epochs <= self.critic_epochs
        ):
            raise ValueError("ACT-TD3 checkpoint epoch progress is invalid")
        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise ValueError("ACT-TD3 checkpoint replay cursor is invalid")
        if not isinstance(permutation, Tensor) or permutation.dtype != torch.long:
            raise ValueError("ACT-TD3 checkpoint replay permutation is invalid")
        if permutation.numel() == 0:
            if cursor != 0:
                raise ValueError("ACT-TD3 checkpoint empty permutation has a cursor")
        elif (
            permutation.ndim != 1
            or permutation.numel() != len(self.dataset)
            or not 0 < cursor < len(self.dataset)
            or cursor % self.batch_size != 0
            or completed_epochs >= self.critic_epochs
            or not torch.equal(
                permutation.sort().values,
                torch.arange(len(self.dataset), dtype=torch.long),
            )
        ):
            raise ValueError("ACT-TD3 checkpoint replay permutation is invalid")
        expected_updates = completed_epochs * self.batches_per_epoch
        if permutation.numel():
            expected_updates += cursor // self.batch_size
        if self._completed_critic_updates() != expected_updates:
            raise ValueError("ACT-TD3 checkpoint critic progress disagrees")
        expected_actor_updates = (
            (
                self._round_start_critic_updates
                + expected_updates
                - self.learner.config.critic_warmup_updates
            )
            // self.POLICY_UPDATE_PERIOD
            - self._round_start_actor_updates
        )
        if self._completed_actor_updates() != expected_actor_updates:
            raise ValueError("ACT-TD3 checkpoint actor progress disagrees")
        self._completed_epochs = completed_epochs
        self._cursor = cursor
        self._permutation = permutation.clone()

    def _normalize_legacy_base_contract(
        self,
        stored_base: Any,
        stored_learner: Any,
    ) -> Any:
        """Add the implicit v3 all-trainable group to the outer resume contract."""

        if (
            not isinstance(stored_learner, Mapping)
            or stored_learner.get("format")
            != self.learner.LEGACY_ALL_TRAINABLE_STATE_FORMAT
        ):
            return stored_base
        if self.learner.config.actor_trainable_groups != ACT_TRAINABLE_GROUPS:
            raise ValueError(
                "Legacy ACT-TD3 v3 checkpoints can resume only with all ACT "
                "actor trainable groups"
            )
        if not isinstance(stored_base, Mapping):
            raise ValueError("ACT-TD3 round checkpoint base contract disagrees")
        legacy_learner_contract = stored_base.get("learner")
        if not isinstance(legacy_learner_contract, Mapping):
            raise ValueError("ACT-TD3 round checkpoint learner contract disagrees")
        legacy_config = legacy_learner_contract.get("config")
        if (
            not isinstance(legacy_config, Mapping)
            or "actor_trainable_groups" in legacy_config
        ):
            raise ValueError("ACT-TD3 legacy round learner contract is invalid")

        normalized_config = dict(legacy_config)
        normalized_config["actor_trainable_groups"] = ACT_TRAINABLE_GROUPS
        normalized_learner_contract = dict(legacy_learner_contract)
        normalized_learner_contract["config"] = normalized_config
        normalized_base = dict(stored_base)
        normalized_base["learner"] = normalized_learner_contract
        return normalized_base

    def _load_checkpoint(self) -> None:
        assert self.resume_from is not None
        state = self._checkpoint_mapping(self.resume_from)
        stored_base = self._normalize_legacy_base_contract(
            state["base_contract"],
            state["learner"],
        )
        current_base = self._base_contract()
        if not isinstance(stored_base, Mapping):
            raise ValueError("ACT-TD3 round checkpoint base contract disagrees")
        stored_critic_epochs = stored_base.get("critic_epochs")
        stored_actor_epochs = stored_base.get("actor_equivalent_epochs")
        if (
            isinstance(stored_critic_epochs, bool)
            or not isinstance(stored_critic_epochs, int)
            or isinstance(stored_actor_epochs, bool)
            or not isinstance(stored_actor_epochs, int)
            or stored_critic_epochs < 1
            or stored_actor_epochs < 1
            or stored_critic_epochs
            != self.POLICY_UPDATE_PERIOD * stored_actor_epochs
        ):
            raise ValueError("ACT-TD3 round checkpoint schedule is invalid")
        schedule_fields = {"critic_epochs", "actor_equivalent_epochs"}
        if {
            key: value for key, value in stored_base.items()
            if key not in schedule_fields
        } != {
            key: value for key, value in current_base.items()
            if key not in schedule_fields
        }:
            raise ValueError("ACT-TD3 round checkpoint base contract disagrees")
        previous_round = state["current_round"]
        if not isinstance(previous_round, Mapping):
            raise ValueError("ACT-TD3 checkpoint current round is invalid")
        previous_dataset = previous_round.get("dataset")
        if not isinstance(previous_dataset, Mapping):
            raise ValueError("ACT-TD3 checkpoint dataset contract is invalid")
        current_dataset = self._dataset_contract()
        if set(previous_dataset) != set(current_dataset):
            raise ValueError("ACT-TD3 checkpoint dataset contract is invalid")
        schedule_derived_fields = {
            "total_critic_updates",
            "total_actor_updates",
        }
        previous_dataset_without_identity = {
            key: value for key, value in previous_dataset.items()
            if key not in (*schedule_derived_fields, "training_data")
        }
        current_dataset_without_identity = {
            key: value for key, value in current_dataset.items()
            if key not in (*schedule_derived_fields, "training_data")
        }
        same_dataset = (
            previous_dataset_without_identity == current_dataset_without_identity
            and _training_identity_same_or_single_root_upgrade(
                previous_dataset.get("training_data"),
                self.training_data_identity,
            )
        )
        recorded_schedule = previous_round.get("schedule")
        stored_schedule = {
            "critic_epochs": stored_critic_epochs,
            "actor_equivalent_epochs": stored_actor_epochs,
        }
        if recorded_schedule is not None and recorded_schedule != stored_schedule:
            raise ValueError("ACT-TD3 checkpoint round schedule disagrees")

        self.learner.load_state_dict(state["learner"])
        sampler_state = state["sampler_state"]
        if not isinstance(sampler_state, Tensor):
            raise ValueError("ACT-TD3 checkpoint sampler state is invalid")
        try:
            self._sampler.set_state(sampler_state)
        except RuntimeError as error:
            raise ValueError("ACT-TD3 checkpoint sampler state is invalid") from error
        history = state["history"]
        if not isinstance(history, tuple) or any(
            not isinstance(value, Mapping) for value in history
        ):
            raise ValueError("ACT-TD3 checkpoint round history is invalid")
        elapsed = state["elapsed_seconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise ValueError("ACT-TD3 checkpoint elapsed time is invalid")
        raw_update = state["last_update"]
        if raw_update is not None:
            update_fields = {field.name for field in fields(ACTTD3UpdateResult)}
            if not isinstance(raw_update, Mapping) or set(raw_update) != update_fields:
                raise ValueError("ACT-TD3 checkpoint last update is invalid")
        raw_indices = state["last_sampled_indices"]
        if (
            not isinstance(raw_indices, tuple)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_indices)
            or len(set(raw_indices)) != len(raw_indices)
            or any(not 0 <= value < int(previous_dataset["transition_count"]) for value in raw_indices)
        ):
            raise ValueError("ACT-TD3 checkpoint sampled indices are invalid")

        previous_index = previous_round.get("round_index")
        if isinstance(previous_index, bool) or not isinstance(previous_index, int):
            raise ValueError("ACT-TD3 checkpoint round index is invalid")
        round_start_critic = state["round_start_critic_updates"]
        round_start_actor = state["round_start_actor_updates"]
        for name, value, completed in (
            (
                "round_start_critic_updates",
                round_start_critic,
                self.learner.completed_critic_updates,
            ),
            (
                "round_start_actor_updates",
                round_start_actor,
                self.learner.completed_actor_updates,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > completed
            ):
                raise ValueError(f"ACT-TD3 checkpoint {name} is invalid")
        previous_completed = previous_round.get("completed_critic_updates")
        previous_actor_completed = previous_round.get("completed_actor_updates")
        if (
            isinstance(previous_completed, bool)
            or not isinstance(previous_completed, int)
            or previous_completed < 0
            or isinstance(previous_actor_completed, bool)
            or not isinstance(previous_actor_completed, int)
            or previous_actor_completed < 0
            or self.learner.completed_critic_updates - round_start_critic
            != previous_completed
            or self.learner.completed_actor_updates - round_start_actor
            != previous_actor_completed
        ):
            raise ValueError("ACT-TD3 checkpoint round counters disagree")
        if previous_completed == 0:
            if raw_update is not None or raw_indices:
                raise ValueError("ACT-TD3 empty round has update metadata")
        elif (
            raw_update is None
            or not raw_indices
            or len(raw_indices) > self.batch_size
            or raw_update["completed_critic_updates"]
            != self.learner.completed_critic_updates
            or raw_update["completed_actor_updates"]
            != self.learner.completed_actor_updates
            or raw_update["actor_updated"]
            != (
                self.learner.completed_critic_updates
                % self.POLICY_UPDATE_PERIOD
                == 0
            )
            or not math.isfinite(float(raw_update["critic_loss"]))
            or not math.isfinite(float(raw_update["target_mean"]))
        ):
            raise ValueError("ACT-TD3 checkpoint update metadata disagrees")
        self._history = tuple(dict(value) for value in history)
        self._last_update = (
            ACTTD3UpdateResult(**dict(raw_update)) if raw_update is not None else None
        )
        self._elapsed_seconds = float(elapsed)

        previous_episode_count = previous_dataset.get("episode_count")
        if (
            isinstance(previous_episode_count, bool)
            or not isinstance(previous_episode_count, int)
            or previous_episode_count < 1
        ):
            raise ValueError("ACT-TD3 checkpoint episode count is invalid")
        prior_episode_count = 0
        if self._history:
            prior_round = self._history[-1]
            prior_dataset = prior_round.get("dataset")
            if not isinstance(prior_dataset, Mapping):
                raise ValueError(
                    "ACT-TD3 checkpoint round history dataset is invalid"
                )
            prior_episode_count = prior_dataset.get("episode_count")
            if (
                isinstance(prior_episode_count, bool)
                or not isinstance(prior_episode_count, int)
                or prior_episode_count < 1
                or prior_episode_count >= previous_episode_count
            ):
                raise ValueError(
                    "ACT-TD3 checkpoint round history episode count is invalid"
                )
        inferred_previous_delta = previous_episode_count - prior_episode_count
        recorded_previous_delta = previous_round.get(
            "new_episode_count",
            inferred_previous_delta,
        )
        if (
            isinstance(recorded_previous_delta, bool)
            or not isinstance(recorded_previous_delta, int)
            or recorded_previous_delta != inferred_previous_delta
            or not 1 <= recorded_previous_delta <= self.ROUND_EPISODES
        ):
            raise ValueError("ACT-TD3 checkpoint new episode count is invalid")

        if same_dataset:
            if self._schedule_contract() != stored_schedule:
                raise ValueError(
                    "ACT-TD3 interrupted round must resume with the same schedule"
                )
            if previous_dataset_without_identity != current_dataset_without_identity:
                raise ValueError("ACT-TD3 checkpoint dataset contract disagrees")
            if self.checkpoint_path != self.resume_from:
                raise ValueError(
                    "ACT-TD3 interrupted round must resume into the same checkpoint"
                )
            self._round_index = previous_index
            self._new_episode_count = recorded_previous_delta
            self._round_start_critic_updates = round_start_critic
            self._round_start_actor_updates = round_start_actor
            self._last_sampled_indices = raw_indices
            self._validate_restored_progress(state)
            self._durable_critic_updates = self._completed_critic_updates()
            return

        if (
            previous_completed != previous_dataset.get("total_critic_updates")
            or previous_actor_completed != previous_dataset.get("total_actor_updates")
            or state["completed_epochs"] != stored_critic_epochs
            or state["permutation"].numel() != 0
            or state["cursor"] != 0
        ):
            raise ValueError("ACT-TD3 dataset cannot grow before a round completes")
        if self.checkpoint_path == self.resume_from or self.checkpoint_path.exists():
            raise FileExistsError(
                "ACT-TD3 next round requires a new versioned checkpoint path"
            )
        old_records = tuple(tuple(value) for value in previous_dataset["episode_records"])
        if previous_episode_count != len(old_records):
            raise ValueError("ACT-TD3 checkpoint episode records are invalid")
        new_records = self.dataset.episode_records
        added = len(new_records) - len(old_records)
        if added < 1 or added > self.ROUND_EPISODES or new_records[: len(old_records)] != old_records:
            raise ValueError(
                "ACT-TD3 cumulative replay must preserve prior episodes and add 1..50"
            )
        old_training = previous_dataset.get("training_data")
        if not isinstance(old_training, Mapping):
            raise ValueError("ACT-TD3 previous training identity is invalid")
        old_virtual = old_training.get("virtual_contract")
        new_virtual = self.training_data_identity.virtual_contract
        if not isinstance(old_virtual, Mapping):
            raise ValueError("ACT-TD3 previous virtual contract is invalid")
        if {
            key: value
            for key, value in old_virtual.items()
            if key not in {"episode_indices", "data_roots"}
        } != {
            key: value
            for key, value in new_virtual.items()
            if key not in {"episode_indices", "data_roots"}
        }:
            raise ValueError("ACT-TD3 cumulative replay virtual contract changed")
        for component in ("act_checkpoint", "robot"):
            if old_training["component_sha256"].get(component) != (
                self.training_data_identity.component_sha256.get(component)
            ):
                raise ValueError(
                    f"ACT-TD3 cumulative replay {component} identity changed"
                )

        old_roots = _checked_data_roots(old_training)
        new_roots = _checked_data_roots(self.training_data_identity)
        if new_roots is not None:
            if old_roots is not None:
                if (
                    len(new_roots) <= len(old_roots)
                    or new_roots[: len(old_roots)] != old_roots
                ):
                    raise ValueError(
                        "ACT-TD3 cumulative replay must preserve the ordered data-root prefix"
                    )
                root_added_episodes = sum(
                    len(root["episode_indices"])
                    for root in new_roots[len(old_roots) :]
                )
            else:
                if not new_roots or not _single_identity_matches_root(
                    old_training, new_roots[0]
                ):
                    raise ValueError(
                        "ACT-TD3 cumulative replay does not preserve the legacy data root"
                    )
                root_added_episodes = sum(
                    len(root["episode_indices"]) for root in new_roots[1:]
                )
            if root_added_episodes != added or not 1 <= root_added_episodes <= self.ROUND_EPISODES:
                raise ValueError(
                    "ACT-TD3 cumulative replay must append data roots containing 1..50 episodes"
                )
        elif old_roots is not None:
            raise ValueError(
                "ACT-TD3 cumulative replay cannot discard its ordered data-root contract"
            )

        completed_round = dict(previous_round)
        completed_round.setdefault("schedule", stored_schedule)
        self._history = (*self._history, completed_round)
        self._round_index = previous_index + 1
        self._new_episode_count = added
        self._round_start_critic_updates = self.learner.completed_critic_updates
        self._round_start_actor_updates = self.learner.completed_actor_updates
        self._elapsed_seconds = 0.0
        self._last_update = None
        self._last_sampled_indices = ()
        self._durable_critic_updates = 0

    def _next_batch(self):
        if self._permutation.numel() == 0:
            self._permutation = torch.randperm(
                len(self.dataset), generator=self._sampler
            )
            self._cursor = 0
        stop = min(self._cursor + self.batch_size, len(self.dataset))
        indices = self._permutation[self._cursor : stop]
        self._cursor = stop
        self._last_sampled_indices = tuple(int(value) for value in indices.tolist())
        batch = self.collator(
            [self.dataset[index] for index in self._last_sampled_indices]
        )
        if self._cursor == len(self.dataset):
            self._completed_epochs += 1
            self._permutation = torch.empty(0, dtype=torch.long)
            self._cursor = 0
        return batch

    def _progress(self, *, status: str, elapsed_seconds: float) -> ACTTD3OfflineTrainingProgress:
        completed = self._completed_critic_updates()
        eta = (
            None
            if completed == 0
            else elapsed_seconds / completed * (self.total_critic_updates - completed)
        )
        return ACTTD3OfflineTrainingProgress(
            status=status,
            round_index=self._round_index,
            episode_count=self.dataset.num_episodes,
            completed_epochs=self._completed_epochs,
            total_epochs=self.critic_epochs,
            completed_critic_updates=completed,
            total_critic_updates=self.total_critic_updates,
            completed_actor_updates=self._completed_actor_updates(),
            total_actor_updates=self.total_actor_updates,
            percentage=100.0 * completed / self.total_critic_updates,
            critic_loss=(
                self._last_update.critic_loss if self._last_update is not None else None
            ),
            actor_loss=(
                self._last_update.actor_loss if self._last_update is not None else None
            ),
            elapsed_seconds=float(elapsed_seconds),
            eta_seconds=float(eta) if eta is not None else None,
            durable_critic_updates=self._durable_critic_updates,
            checkpoint_path=str(self.checkpoint_path),
        )

    def run(
        self,
        *,
        max_round_critic_updates: int | None = None,
        progress_callback: ProgressCallback | None = None,
        should_stop: StopPredicate | None = None,
    ) -> ACTTD3OfflineTrainingProgress:
        """Run to the round boundary, or to an absolute in-round test boundary."""

        if max_round_critic_updates is None:
            stop_at = self.total_critic_updates
        else:
            stop_at = _positive_integer(
                max_round_critic_updates, "max_round_critic_updates"
            )
            if stop_at > self.total_critic_updates:
                raise ValueError("ACT-TD3 test boundary exceeds the round")
        if stop_at < self._completed_critic_updates():
            raise ValueError("ACT-TD3 test boundary precedes current round progress")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("ACT-TD3 progress_callback must be callable")
        if should_stop is not None and not callable(should_stop):
            raise TypeError("ACT-TD3 should_stop must be callable")

        started = time.monotonic()

        def elapsed() -> float:
            return self._elapsed_seconds + (time.monotonic() - started)

        if progress_callback is not None:
            progress_callback(self._progress(status="running", elapsed_seconds=elapsed()))
        stopped = False
        while self._completed_critic_updates() < stop_at:
            if should_stop is not None and should_stop():
                stopped = True
                break
            before_actor = self.learner.completed_actor_updates
            update = self.learner.update(self._next_batch())
            expected_actor = (
                self.learner.completed_critic_updates
                % self.POLICY_UPDATE_PERIOD
                == 0
            )
            if update.actor_updated != expected_actor or (
                self.learner.completed_actor_updates - before_actor
            ) != int(expected_actor):
                raise RuntimeError("ACT-TD3 learner violated the delayed actor schedule")
            self._last_update = update
            step = self._completed_critic_updates()
            checkpoint_due = step % self.checkpoint_interval == 0
            if checkpoint_due:
                self._save_checkpoint(elapsed())
            report_due = step % self.progress_interval == 0 or checkpoint_due
            if progress_callback is not None and report_due and step < stop_at:
                progress_callback(
                    self._progress(status="running", elapsed_seconds=elapsed())
                )

        final_elapsed = elapsed()
        if (
            not self.checkpoint_path.is_file()
            or self._durable_critic_updates != self._completed_critic_updates()
        ):
            self._save_checkpoint(final_elapsed)
        self._elapsed_seconds = final_elapsed
        if self._completed_critic_updates() == self.total_critic_updates:
            status = "complete"
        elif stopped:
            status = "stopped"
        else:
            status = "segment_complete"
        result = self._progress(status=status, elapsed_seconds=self._elapsed_seconds)
        if progress_callback is not None:
            progress_callback(result)
        return result


__all__ = ["ACTTD3OfflineTrainingProgress", "ACTTD3OfflineTrainingRunner"]
