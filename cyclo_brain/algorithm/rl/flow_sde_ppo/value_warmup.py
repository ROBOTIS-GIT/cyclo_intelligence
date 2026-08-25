"""Offline value warm-up for MultiTaskDiT Flow-SDE PPO.

The warm-up consumes only chunk-boundary observations from explicitly labelled
LeRobot v3 episodes.  It never optimizes the policy: both the observation
encoder and Flow-Matching noise predictor are frozen, while a
``MultiTaskDiTValueHead`` learns episode-level Monte-Carlo returns.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VALUE_WARMUP_FORMAT = "cyclo.flow_sde_ppo.value_warmup.v1"
SAMPLING_CONTRACT = "alternate_outcome_then_uniform_episode_then_uniform_chunk"


def module_sha256(module: nn.Module) -> str:
    """Hash exact parameter and buffer values independently of device."""

    if not isinstance(module, nn.Module):
        raise TypeError("module_sha256 requires a torch module")
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
    return f"sha256:{digest.hexdigest()}"


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Durably replace a torch checkpoint at an optimizer-step boundary."""

    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
        directory = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def _scalar_index(value: Any, *, name: str) -> int:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.item()
    elif hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _scalar_success(value: Any) -> bool:
    if isinstance(value, Tensor):
        if value.numel() != 1 or value.dtype != torch.bool:
            raise TypeError("episode_success must be an explicit scalar boolean")
        return bool(value.item())
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, bool):
        return value
    if type(value).__module__.startswith("numpy") and type(value).__name__ == "bool_":
        return bool(value)
    raise TypeError("episode_success must be an explicit boolean, not an integer")


def _column(table: Any, name: str, expected_length: int) -> Sequence[Any]:
    try:
        values = table[name]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"LeRobot dataset is missing required column {name!r}") from error
    if len(values) != expected_length:
        raise ValueError(f"LeRobot column {name!r} length does not match the dataset")
    return values


@dataclass(frozen=True)
class ValueWarmupConfig:
    steps: int
    batch_size: int
    value_lr: float
    gamma: float
    task_instruction: str
    seed: int = 17
    checkpoint_interval: int = 100
    progress_interval: int = 1

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "checkpoint_interval", "progress_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"value warm-up {name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("value warm-up seed must be a non-negative integer")
        if not math.isfinite(self.value_lr) or self.value_lr <= 0.0:
            raise ValueError("value warm-up value_lr must be finite and positive")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("value warm-up gamma must be in [0, 1]")
        if not isinstance(self.task_instruction, str) or not self.task_instruction.strip():
            raise ValueError("value warm-up task_instruction must be non-empty")


@dataclass(frozen=True)
class ChunkBoundaryRecord:
    dataset_index: int
    episode_index: int
    row_index: int
    start_frame_index: int
    chunk_index: int
    chunk_count: int
    successful: bool
    target_return: float


@dataclass(frozen=True)
class ValueWarmupProgress:
    status: str
    phase: str
    step: int
    total_steps: int
    percentage: float
    value_loss: float | None
    mean_value_loss: float | None
    elapsed_seconds: float
    eta_seconds: float | None


@dataclass(frozen=True)
class ValueWarmupResult:
    status: str
    completed_steps: int
    final_value_loss: float | None
    mean_value_loss: float | None
    elapsed_seconds: float
    policy_sha256_before: str
    policy_sha256_after: str
    checkpoint_path: str


class EpisodeBalancedChunkBoundaryDataset:
    """Expose non-overlapping policy-decision states from labelled episodes.

    Sampling first alternates success/failure, then chooses an episode
    uniformly within that outcome and finally a chunk boundary uniformly
    within that episode.  Long episodes and large dataset roots therefore do
    not dominate the value target distribution.
    """

    def __init__(
        self,
        datasets: Sequence[Any],
        *,
        observation_keys: Sequence[str],
        n_action_steps: int,
        gamma: float,
        dataset_names: Sequence[str] | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("value warm-up requires at least one LeRobot dataset")
        if isinstance(n_action_steps, bool) or not isinstance(n_action_steps, int) or n_action_steps < 1:
            raise ValueError("n_action_steps must be a positive integer")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.datasets = tuple(datasets)
        self.observation_keys = tuple(observation_keys)
        if not self.observation_keys or len(set(self.observation_keys)) != len(self.observation_keys):
            raise ValueError("observation_keys must be unique and non-empty")
        self.n_action_steps = n_action_steps
        self.gamma = float(gamma)
        self.dataset_names = tuple(dataset_names or [f"dataset-{i}" for i in range(len(datasets))])
        if len(self.dataset_names) != len(self.datasets):
            raise ValueError("dataset_names must match datasets")

        records: list[ChunkBoundaryRecord] = []
        episode_records: dict[bool, list[tuple[int, ...]]] = {True: [], False: []}
        episode_counts = {True: 0, False: 0}
        for dataset_index, dataset in enumerate(self.datasets):
            features = getattr(dataset, "features", None)
            if not isinstance(features, Mapping):
                raise TypeError("LeRobot dataset must expose a features mapping")
            required = {"episode_index", "frame_index", "episode_success", *self.observation_keys}
            missing = required.difference(features)
            if missing:
                raise ValueError("LeRobot dataset is missing features: " + ", ".join(sorted(missing)))
            table = getattr(dataset, "hf_dataset", None)
            if table is None:
                raise TypeError("LeRobot dataset must expose hf_dataset")
            row_count = len(table)
            if row_count < 1 or len(dataset) != row_count:
                raise ValueError("LeRobot dataset and hf_dataset must have the same non-zero length")
            episode_values = _column(table, "episode_index", row_count)
            frame_values = _column(table, "frame_index", row_count)
            success_values = _column(table, "episode_success", row_count)
            grouped: dict[int, list[tuple[int, int, bool]]] = {}
            for row, (episode, frame, success) in enumerate(
                zip(episode_values, frame_values, success_values, strict=True)
            ):
                grouped.setdefault(_scalar_index(episode, name="episode_index"), []).append(
                    (_scalar_index(frame, name="frame_index"), row, _scalar_success(success))
                )
            for episode_index in sorted(grouped):
                values = sorted(grouped[episode_index])
                if [frame for frame, _, _ in values] != list(range(len(values))):
                    raise ValueError(
                        f"dataset {self.dataset_names[dataset_index]!r} episode {episode_index} "
                        "must contain contiguous frames starting at zero"
                    )
                labels = {successful for _, _, successful in values}
                if len(labels) != 1:
                    raise ValueError(f"episode {episode_index} has mixed episode_success labels")
                successful = labels.pop()
                chunk_count = math.ceil(len(values) / n_action_steps)
                indices: list[int] = []
                for chunk_index, start in enumerate(range(0, len(values), n_action_steps)):
                    frame, row, _ = values[start]
                    target = self.gamma ** (chunk_count - chunk_index - 1) if successful else 0.0
                    indices.append(len(records))
                    records.append(
                        ChunkBoundaryRecord(
                            dataset_index=dataset_index,
                            episode_index=episode_index,
                            row_index=row,
                            start_frame_index=frame,
                            chunk_index=chunk_index,
                            chunk_count=chunk_count,
                            successful=successful,
                            target_return=float(target),
                        )
                    )
                episode_records[successful].append(tuple(indices))
                episode_counts[successful] += 1
        if not episode_records[True] or not episode_records[False]:
            raise ValueError("episode-balanced value warm-up requires both success and fail episodes")
        self.records = tuple(records)
        self._episode_records = {
            outcome: tuple(groups) for outcome, groups in episode_records.items()
        }
        self.success_episode_count = episode_counts[True]
        self.failure_episode_count = episode_counts[False]

    def __len__(self) -> int:
        return len(self.records)

    def sample_indices(
        self,
        *,
        generator: torch.Generator,
        batch_size: int,
        sampling_cursor: int,
    ) -> tuple[tuple[int, ...], int]:
        if not isinstance(generator, torch.Generator):
            raise TypeError("value warm-up sampler requires a torch.Generator")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if isinstance(sampling_cursor, bool) or not isinstance(sampling_cursor, int) or sampling_cursor < 0:
            raise ValueError("sampling_cursor must be non-negative")
        result: list[int] = []
        for offset in range(batch_size):
            successful = (sampling_cursor + offset) % 2 == 0
            episodes = self._episode_records[successful]
            episode_slot = int(torch.randint(len(episodes), (), generator=generator).item())
            chunks = episodes[episode_slot]
            chunk_slot = int(torch.randint(len(chunks), (), generator=generator).item())
            result.append(chunks[chunk_slot])
        return tuple(result), sampling_cursor + batch_size

    def collate(self, indices: Sequence[int]) -> tuple[dict[str, Tensor], Tensor]:
        if not indices:
            raise ValueError("cannot collate an empty value warm-up batch")
        observations: dict[str, list[Tensor]] = {key: [] for key in self.observation_keys}
        targets: list[float] = []
        for index in indices:
            record = self.records[index]
            item = self.datasets[record.dataset_index][record.row_index]
            if not isinstance(item, Mapping):
                raise TypeError("LeRobot item must be a mapping")
            for key in self.observation_keys:
                if key not in item:
                    raise ValueError(f"LeRobot item is missing observation {key!r}")
                value = item[key]
                tensor = value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
                if not (tensor.dtype == torch.uint8 or tensor.is_floating_point()):
                    raise TypeError(f"observation {key!r} must be uint8 or floating point")
                if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"observation {key!r} contains non-finite values")
                observations[key].append(tensor.clone())
            targets.append(record.target_return)
        return (
            {key: torch.stack(values) for key, values in observations.items()},
            torch.tensor(targets, dtype=torch.float32),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "sampling": SAMPLING_CONTRACT,
            "n_action_steps": self.n_action_steps,
            "gamma": self.gamma,
            "discount_unit": "chunk_decision",
            "chunk_boundary_count": len(self.records),
            "success_episode_count": self.success_episode_count,
            "failure_episode_count": self.failure_episode_count,
            "dataset_names": list(self.dataset_names),
            "observation_keys": list(self.observation_keys),
        }


ConditioningEncoder = Callable[[Mapping[str, Tensor], str], Tensor]
ProgressCallback = Callable[[ValueWarmupProgress], None]
StopPredicate = Callable[[], bool]


class MultiTaskDiTValueWarmupRunner:
    """Fit only a value MLP while enforcing a bitwise-fixed policy invariant."""

    def __init__(
        self,
        policy: nn.Module,
        value_head: nn.Module,
        dataset: EpisodeBalancedChunkBoundaryDataset,
        conditioning_encoder: ConditioningEncoder,
        *,
        config: ValueWarmupConfig,
        checkpoint_path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        base_identity: Mapping[str, Any] | None = None,
        dataset_identities: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not isinstance(policy, nn.Module):
            raise TypeError("value warm-up policy must be a torch module")
        policy.requires_grad_(False)
        for name in ("observation_encoder", "noise_predictor"):
            module = getattr(policy, name, None)
            if not isinstance(module, nn.Module):
                raise TypeError(f"value warm-up policy is missing {name}")
            module.requires_grad_(False)
            module.eval()
        policy.eval()
        if not isinstance(value_head, nn.Module):
            raise TypeError("value warm-up value_head must be a torch module")
        if not callable(conditioning_encoder):
            raise TypeError("conditioning_encoder must be callable")
        self.policy = policy
        self.value_head = value_head
        self.dataset = dataset
        self.conditioning_encoder = conditioning_encoder
        self.config = config
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.value_parameters = tuple(p for p in value_head.parameters() if p.requires_grad)
        if not self.value_parameters:
            raise ValueError("value warm-up value head has no trainable parameters")
        self.device = self.value_parameters[0].device
        if any(p.device != self.device for p in self.value_parameters):
            raise ValueError("value warm-up parameters must share one device")
        self.optimizer = optimizer or torch.optim.AdamW(self.value_parameters, lr=config.value_lr)
        optimizer_parameters = {
            id(p) for group in self.optimizer.param_groups for p in group["params"]
        }
        if optimizer_parameters != {id(p) for p in self.value_parameters}:
            raise ValueError("value warm-up optimizer must contain only value-head parameters")
        self.base_identity = dict(base_identity or {})
        self.dataset_identities = tuple(dict(value) for value in dataset_identities)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.sampling_cursor = 0
        self.completed_steps = 0
        self.policy_sha256_before = module_sha256(policy)
        self._losses: list[float] = []

    def _checkpoint(
        self,
        *,
        status: str,
        elapsed_seconds: float,
        verify_policy: bool,
    ) -> Path:
        # Hashing the 225M-parameter policy is intentionally reserved for the
        # durable terminal boundary. Per-step gradient and requires_grad checks
        # enforce immutability between the exact initial/final hashes without
        # repeatedly copying the frozen model to CPU.
        policy_after = module_sha256(self.policy) if verify_policy else None
        if policy_after is not None and policy_after != self.policy_sha256_before:
            raise RuntimeError("MultiTaskDiT policy changed during value warm-up")
        return atomic_torch_save(
            self.checkpoint_path,
            {
                "format": VALUE_WARMUP_FORMAT,
                "status": status,
                "config": asdict(self.config),
                "completed_steps": self.completed_steps,
                "sampling_cursor": self.sampling_cursor,
                "dataset_contract": self.dataset.contract(),
                "base_identity": self.base_identity,
                "dataset_identities": list(self.dataset_identities),
                "value_head": self.value_head.state_dict(),
                "value_optimizer": self.optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "sampler_rng_state": self.generator.get_state(),
                "policy_sha256_before": self.policy_sha256_before,
                "policy_sha256_after": policy_after,
                "elapsed_seconds": float(elapsed_seconds),
                "final_value_loss": self._losses[-1] if self._losses else None,
                "mean_value_loss": sum(self._losses) / len(self._losses) if self._losses else None,
            },
        )

    def run(
        self,
        *,
        progress: ProgressCallback | None = None,
        should_stop: StopPredicate | None = None,
    ) -> ValueWarmupResult:
        started = time.monotonic()
        final_status = "complete"
        while self.completed_steps < self.config.steps:
            if should_stop is not None and should_stop():
                final_status = "stopped"
                break
            indices, self.sampling_cursor = self.dataset.sample_indices(
                generator=self.generator,
                batch_size=self.config.batch_size,
                sampling_cursor=self.sampling_cursor,
            )
            observations, targets = self.dataset.collate(indices)
            with torch.no_grad():
                conditioning = self.conditioning_encoder(
                    observations, self.config.task_instruction.strip()
                )
            if not isinstance(conditioning, Tensor) or conditioning.ndim != 2:
                raise ValueError("conditioning_encoder must return a tensor shaped (B, C)")
            if conditioning.shape[0] != targets.shape[0] or not bool(torch.isfinite(conditioning).all()):
                raise ValueError("value warm-up conditioning batch is invalid")
            self.value_head.train()
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.value_head(conditioning.detach().to(self.device)).float()
            if predictions.shape == (targets.shape[0], 1):
                predictions = predictions[:, 0]
            if predictions.shape != targets.shape:
                raise ValueError("value head must return one scalar per chunk-boundary state")
            targets = targets.to(self.device)
            loss = F.mse_loss(predictions, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("value warm-up produced a non-finite loss")
            loss.backward()
            if any(parameter.grad is not None for parameter in self.policy.parameters()):
                raise RuntimeError("frozen MultiTaskDiT policy received a value warm-up gradient")
            torch.nn.utils.clip_grad_norm_(self.value_parameters, 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.completed_steps += 1
            self._losses.append(float(loss.detach().cpu()))
            elapsed = time.monotonic() - started
            mean_step = elapsed / self.completed_steps
            eta = mean_step * (self.config.steps - self.completed_steps)
            if (
                self.completed_steps % self.config.checkpoint_interval == 0
                or self.completed_steps == self.config.steps
            ):
                self._checkpoint(
                    status="running",
                    elapsed_seconds=elapsed,
                    verify_policy=False,
                )
            if progress is not None and (
                self.completed_steps % self.config.progress_interval == 0
                or self.completed_steps == self.config.steps
            ):
                progress(
                    ValueWarmupProgress(
                        status="running",
                        phase="value_warmup",
                        step=self.completed_steps,
                        total_steps=self.config.steps,
                        percentage=100.0 * self.completed_steps / self.config.steps,
                        value_loss=self._losses[-1],
                        mean_value_loss=sum(self._losses) / len(self._losses),
                        elapsed_seconds=elapsed,
                        eta_seconds=eta,
                    )
                )
        elapsed = time.monotonic() - started
        checkpoint = self._checkpoint(
            status=final_status,
            elapsed_seconds=elapsed,
            verify_policy=True,
        )
        policy_after = self.policy_sha256_before
        if progress is not None:
            progress(
                ValueWarmupProgress(
                    status=final_status,
                    phase="complete" if final_status == "complete" else "stopped",
                    step=self.completed_steps,
                    total_steps=self.config.steps,
                    percentage=100.0 * self.completed_steps / self.config.steps,
                    value_loss=self._losses[-1] if self._losses else None,
                    mean_value_loss=sum(self._losses) / len(self._losses) if self._losses else None,
                    elapsed_seconds=elapsed,
                    eta_seconds=0.0 if final_status == "complete" else None,
                )
            )
        return ValueWarmupResult(
            status=final_status,
            completed_steps=self.completed_steps,
            final_value_loss=self._losses[-1] if self._losses else None,
            mean_value_loss=sum(self._losses) / len(self._losses) if self._losses else None,
            elapsed_seconds=elapsed,
            policy_sha256_before=self.policy_sha256_before,
            policy_sha256_after=policy_after,
            checkpoint_path=str(checkpoint),
        )


__all__ = [
    "VALUE_WARMUP_FORMAT",
    "SAMPLING_CONTRACT",
    "ValueWarmupConfig",
    "ChunkBoundaryRecord",
    "ValueWarmupProgress",
    "ValueWarmupResult",
    "EpisodeBalancedChunkBoundaryDataset",
    "MultiTaskDiTValueWarmupRunner",
    "module_sha256",
    "atomic_torch_save",
]
