"""Deterministic offline critic warm-up orchestration for ACT-TD3."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .learner import ACTTD3Learner, ACTTD3UpdateResult
from .lerobot_offline import (
    ACTTD3LeRobotCollator,
    FixedHorizonLeRobotACTTD3Dataset,
)


ProgressCallback = Callable[["ACTTD3CriticWarmupProgress"], None]
StopPredicate = Callable[[], bool]


@dataclass(frozen=True)
class ACTTD3CriticWarmupProgress:
    """One JSON-friendly progress snapshot from a completed update boundary."""

    status: str
    completed_critic_updates: int
    total_critic_updates: int
    percentage: float
    critic_loss: float | None
    target_mean: float | None
    elapsed_seconds: float
    eta_seconds: float | None
    durable_checkpoint_updates: int
    actor_exactly_unchanged: bool | None


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"ACT-TD3 warm-up {name} must be a positive integer")
    return value


def _module_sha256(module: nn.Module) -> str:
    """Hash exact parameter and buffer values without serializing locations."""

    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _atomic_torch_save(path: Path, state: Mapping[str, Any]) -> None:
    """Durably replace one checkpoint after its temporary file is complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class ACTTD3CriticWarmupRunner:
    """Run only the actor-frozen prefix of an :class:`ACTTD3Learner`.

    The algorithm's configured ``critic_warmup_updates`` is the absolute final
    boundary. ``max_critic_updates`` passed to :meth:`run` may stop an
    invocation earlier, but it never changes that algorithm contract.
    """

    STATE_FORMAT = "cyclo_brain.act_td3_critic_warmup/v2"
    SAMPLING = "uniform_without_replacement_within_batch"

    def __init__(
        self,
        learner: ACTTD3Learner,
        dataset: FixedHorizonLeRobotACTTD3Dataset,
        collator: ACTTD3LeRobotCollator,
        *,
        batch_size: int,
        sampling_seed: int,
        training_data_identity: str,
        checkpoint_path: str | Path,
        checkpoint_interval: int = 500,
        progress_interval: int = 1,
        resume: bool = False,
    ) -> None:
        if not isinstance(learner, ACTTD3Learner):
            raise TypeError("ACT-TD3 warm-up requires ACTTD3Learner")
        if not isinstance(dataset, FixedHorizonLeRobotACTTD3Dataset):
            raise TypeError(
                "ACT-TD3 warm-up requires FixedHorizonLeRobotACTTD3Dataset"
            )
        if not isinstance(collator, ACTTD3LeRobotCollator):
            raise TypeError("ACT-TD3 warm-up requires ACTTD3LeRobotCollator")
        self.batch_size = _positive_integer(batch_size, "batch_size")
        if self.batch_size > len(dataset):
            raise ValueError(
                "ACT-TD3 warm-up batch_size cannot exceed replay size when "
                "sampling without replacement"
            )
        if (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, int)
            or not 0 <= sampling_seed < 2**63 - 1
        ):
            raise ValueError("ACT-TD3 warm-up sampling_seed is invalid")
        if not isinstance(training_data_identity, str) or not training_data_identity:
            raise ValueError(
                "ACT-TD3 warm-up requires a non-empty training_data_identity"
            )
        if not isinstance(resume, bool):
            raise TypeError("ACT-TD3 warm-up resume selection must be boolean")

        self.checkpoint_interval = _positive_integer(
            checkpoint_interval,
            "checkpoint_interval",
        )
        self.progress_interval = _positive_integer(
            progress_interval,
            "progress_interval",
        )
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if self.checkpoint_path.exists() and self.checkpoint_path.is_dir():
            raise IsADirectoryError(self.checkpoint_path)
        if not resume and self.checkpoint_path.exists():
            raise FileExistsError(
                f"ACT-TD3 warm-up checkpoint already exists: {self.checkpoint_path}"
            )

        self.learner = learner
        self.dataset = dataset
        self.collator = collator
        self.sampling_seed = sampling_seed
        self.training_data_identity = training_data_identity
        self.total_critic_updates = learner.config.critic_warmup_updates
        if self.total_critic_updates < 1:
            raise ValueError("ACT-TD3 warm-up requires at least one configured update")
        if learner.completed_actor_updates != 0:
            raise ValueError("ACT-TD3 warm-up cannot start after an actor update")
        if learner.completed_critic_updates != 0:
            raise ValueError(
                "ACT-TD3 warm-up runner requires a fresh learner; nonzero progress "
                "must be restored from its runner checkpoint"
            )
        if dataset.execution_horizon != learner.execution_horizon:
            raise ValueError("ACT-TD3 warm-up dataset execution horizon disagrees")
        if dataset.action_dim != learner.action_dim:
            raise ValueError("ACT-TD3 warm-up dataset action dimension disagrees")
        if float(dataset.fps) != float(learner.config.discount_reference_hz):
            raise ValueError(
                "ACT-TD3 discount_reference_hz must exactly match dataset fps"
            )

        self._sampler = torch.Generator(device="cpu").manual_seed(sampling_seed)
        self._baseline_actor_sha256 = _module_sha256(learner.actor)
        self._baseline_target_actor_sha256 = _module_sha256(learner.actor_target)
        if self._baseline_actor_sha256 != self._baseline_target_actor_sha256:
            raise ValueError("ACT-TD3 warm-up actor and target actor must start equal")
        self._elapsed_seconds = 0.0
        self._last_update: ACTTD3UpdateResult | None = None
        self._last_sampled_indices: tuple[int, ...] = ()
        self._durable_checkpoint_updates = 0
        self._assert_actor_invariant(exact=True)

        if resume:
            self._load_checkpoint()
        elif self.checkpoint_path.exists():
            raise FileExistsError(self.checkpoint_path)

    @property
    def last_sampled_indices(self) -> tuple[int, ...]:
        return self._last_sampled_indices

    def _contract(self) -> dict[str, Any]:
        return {
            "training_data_identity": self.training_data_identity,
            "sampling": self.SAMPLING,
            "sampling_seed": self.sampling_seed,
            "batch_size": self.batch_size,
            "dataset": {
                "transition_count": len(self.dataset),
                "episode_count": self.dataset.num_episodes,
                "success_count": self.dataset.num_successes,
                "failure_count": self.dataset.num_failures,
                "fps": float(self.dataset.fps),
                "execution_horizon": self.dataset.execution_horizon,
                "action_dim": self.dataset.action_dim,
            },
            "learner": {
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
            },
        }

    def _assert_actor_invariant(self, *, exact: bool) -> None:
        if self.learner.completed_actor_updates != 0:
            raise RuntimeError("ACT-TD3 actor updated during critic warm-up")
        if any(parameter.grad is not None for parameter in self.learner.actor.parameters()):
            raise RuntimeError("ACT-TD3 actor accumulated gradients during warm-up")
        if self.learner.actor.training or self.learner.actor_target.training:
            raise RuntimeError("ACT-TD3 warm-up actors must remain in evaluation mode")
        if exact and (
            _module_sha256(self.learner.actor) != self._baseline_actor_sha256
            or _module_sha256(self.learner.actor_target)
            != self._baseline_target_actor_sha256
        ):
            raise RuntimeError("ACT-TD3 actor tensors changed during critic warm-up")

    def _sample_batch(self):
        indices = torch.randperm(len(self.dataset), generator=self._sampler)[
            : self.batch_size
        ]
        self._last_sampled_indices = tuple(int(index) for index in indices.tolist())
        return self.collator(
            [self.dataset[index] for index in self._last_sampled_indices]
        )

    def _checkpoint_state(self, elapsed_seconds: float) -> dict[str, Any]:
        return {
            "format": self.STATE_FORMAT,
            "contract": self._contract(),
            "learner": self.learner.state_dict(),
            "sampler_state": self._sampler.get_state().cpu().clone(),
            "baseline_actor_sha256": self._baseline_actor_sha256,
            "baseline_target_actor_sha256": self._baseline_target_actor_sha256,
            "elapsed_seconds": float(elapsed_seconds),
            "last_update": (
                asdict(self._last_update) if self._last_update is not None else None
            ),
            "last_sampled_indices": self._last_sampled_indices,
        }

    def _save_checkpoint(self, elapsed_seconds: float) -> None:
        self._assert_actor_invariant(exact=True)
        _atomic_torch_save(
            self.checkpoint_path,
            self._checkpoint_state(elapsed_seconds),
        )
        self._durable_checkpoint_updates = self.learner.completed_critic_updates

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        try:
            state = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise ValueError("ACT-TD3 warm-up checkpoint cannot be read") from error
        expected_keys = {
            "format",
            "contract",
            "learner",
            "sampler_state",
            "baseline_actor_sha256",
            "baseline_target_actor_sha256",
            "elapsed_seconds",
            "last_update",
            "last_sampled_indices",
        }
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise ValueError("ACT-TD3 warm-up checkpoint fields disagree")
        if state["format"] != self.STATE_FORMAT or state["contract"] != self._contract():
            raise ValueError("ACT-TD3 warm-up checkpoint contract disagrees")
        if (
            state["baseline_actor_sha256"] != self._baseline_actor_sha256
            or state["baseline_target_actor_sha256"]
            != self._baseline_target_actor_sha256
        ):
            raise ValueError("ACT-TD3 warm-up base actor identity disagrees")
        elapsed = state["elapsed_seconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise ValueError("ACT-TD3 warm-up checkpoint elapsed time is invalid")
        sampler_state = state["sampler_state"]
        if not isinstance(sampler_state, Tensor):
            raise ValueError("ACT-TD3 warm-up sampler state is invalid")
        checked_sampler = torch.Generator(device="cpu")
        try:
            checked_sampler.set_state(sampler_state)
        except RuntimeError as error:
            raise ValueError("ACT-TD3 warm-up sampler state is invalid") from error
        raw_indices = state["last_sampled_indices"]
        if (
            not isinstance(raw_indices, tuple)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_indices)
            or len(raw_indices) not in {0, self.batch_size}
            or len(set(raw_indices)) != len(raw_indices)
            or any(not 0 <= value < len(self.dataset) for value in raw_indices)
        ):
            raise ValueError("ACT-TD3 warm-up sampled indices are invalid")
        raw_update = state["last_update"]
        if raw_update is not None:
            update_fields = {field.name for field in fields(ACTTD3UpdateResult)}
            if not isinstance(raw_update, Mapping) or set(raw_update) != update_fields:
                raise ValueError("ACT-TD3 warm-up last update is invalid")

        self.learner.load_state_dict(state["learner"])
        if self.learner.completed_critic_updates > self.total_critic_updates:
            raise ValueError("ACT-TD3 checkpoint is past the warm-up boundary")
        if self.learner.completed_actor_updates != 0:
            raise ValueError("ACT-TD3 checkpoint contains actor updates")
        completed = self.learner.completed_critic_updates
        if completed == 0:
            if raw_update is not None or raw_indices:
                raise ValueError("ACT-TD3 empty warm-up checkpoint has update metadata")
        else:
            if raw_update is None or len(raw_indices) != self.batch_size:
                raise ValueError("ACT-TD3 warm-up checkpoint lacks update metadata")
            if (
                raw_update["completed_critic_updates"] != completed
                or raw_update["completed_actor_updates"] != 0
                or raw_update["actor_updated"] is not False
                or not math.isfinite(float(raw_update["critic_loss"]))
                or not math.isfinite(float(raw_update["target_mean"]))
            ):
                raise ValueError("ACT-TD3 warm-up checkpoint update metadata disagrees")
        self._sampler.set_state(sampler_state)
        self._elapsed_seconds = float(elapsed)
        self._last_sampled_indices = raw_indices
        self._last_update = (
            ACTTD3UpdateResult(**dict(raw_update)) if raw_update is not None else None
        )
        self._durable_checkpoint_updates = self.learner.completed_critic_updates
        self._assert_actor_invariant(exact=True)

    def _progress(
        self,
        *,
        status: str,
        elapsed_seconds: float,
        actor_exactly_unchanged: bool | None,
    ) -> ACTTD3CriticWarmupProgress:
        completed = self.learner.completed_critic_updates
        eta = (
            None
            if completed == 0
            else elapsed_seconds / completed * (self.total_critic_updates - completed)
        )
        return ACTTD3CriticWarmupProgress(
            status=status,
            completed_critic_updates=completed,
            total_critic_updates=self.total_critic_updates,
            percentage=100.0 * completed / self.total_critic_updates,
            critic_loss=(
                self._last_update.critic_loss if self._last_update is not None else None
            ),
            target_mean=(
                self._last_update.target_mean if self._last_update is not None else None
            ),
            elapsed_seconds=float(elapsed_seconds),
            eta_seconds=float(eta) if eta is not None else None,
            durable_checkpoint_updates=self._durable_checkpoint_updates,
            actor_exactly_unchanged=actor_exactly_unchanged,
        )

    def run(
        self,
        *,
        max_critic_updates: int | None = None,
        progress_callback: ProgressCallback | None = None,
        should_stop: StopPredicate | None = None,
    ) -> ACTTD3CriticWarmupProgress:
        """Run to an absolute update boundary and atomically save the result."""

        if max_critic_updates is None:
            stop_at = self.total_critic_updates
        else:
            stop_at = _positive_integer(max_critic_updates, "max_critic_updates")
            if stop_at > self.total_critic_updates:
                raise ValueError(
                    "ACT-TD3 max_critic_updates exceeds the warm-up boundary"
                )
        if stop_at < self.learner.completed_critic_updates:
            raise ValueError("ACT-TD3 max_critic_updates precedes current progress")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("ACT-TD3 progress_callback must be callable")
        if should_stop is not None and not callable(should_stop):
            raise TypeError("ACT-TD3 should_stop must be callable")

        started = time.monotonic()

        # Fail before reporting progress or consuming replay if either actor was
        # mutated between runner construction and this invocation.
        self._assert_actor_invariant(exact=True)

        def elapsed() -> float:
            return self._elapsed_seconds + (time.monotonic() - started)

        if progress_callback is not None:
            progress_callback(
                self._progress(
                    status="running",
                    elapsed_seconds=elapsed(),
                    actor_exactly_unchanged=True,
                )
            )

        stopped = False
        while self.learner.completed_critic_updates < stop_at:
            if should_stop is not None and should_stop():
                stopped = True
                break
            update = self.learner.update(self._sample_batch())
            if update.actor_updated or update.completed_actor_updates != 0:
                raise RuntimeError("ACT-TD3 warm-up reached an actor update")
            self._last_update = update
            self._assert_actor_invariant(exact=False)
            step = self.learner.completed_critic_updates
            checkpoint_due = step % self.checkpoint_interval == 0
            if checkpoint_due:
                self._save_checkpoint(elapsed())
            report_due = step % self.progress_interval == 0 or checkpoint_due
            if progress_callback is not None and report_due and step < stop_at:
                progress_callback(
                    self._progress(
                        status="running",
                        elapsed_seconds=elapsed(),
                        actor_exactly_unchanged=(True if checkpoint_due else None),
                    )
                )

        final_elapsed = elapsed()
        self._assert_actor_invariant(exact=True)
        if (
            not self.checkpoint_path.is_file()
            or self._durable_checkpoint_updates
            != self.learner.completed_critic_updates
        ):
            self._save_checkpoint(final_elapsed)
        self._elapsed_seconds = final_elapsed
        if self.learner.completed_critic_updates == self.total_critic_updates:
            status = "complete"
        elif stopped:
            status = "stopped"
        else:
            status = "segment_complete"
        result = self._progress(
            status=status,
            elapsed_seconds=self._elapsed_seconds,
            actor_exactly_unchanged=True,
        )
        if progress_callback is not None:
            progress_callback(result)
        return result


__all__ = [
    "ACTTD3CriticWarmupProgress",
    "ACTTD3CriticWarmupRunner",
]
