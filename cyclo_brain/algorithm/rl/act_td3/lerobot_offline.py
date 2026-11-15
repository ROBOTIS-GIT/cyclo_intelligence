"""LeRobot v3 episodes adapted to the strict ACT-TD3 batch contract.

The adapter intentionally builds *fixed*, non-overlapping macro transitions.
For an ACT execution horizon ``H``, every episode is partitioned from frame zero
as ``[0:H], [H:2H], ...``.  The final short prefix is zero padded and masked;
the final block is terminal even when its length is exactly ``H``.

Only an episode-level success label is used for reward construction.  A
successful episode receives ``+1`` on its final executed primitive step and a
failed episode receives zero everywhere.  Request/chunk identifiers are neither
read nor inferred.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

import torch
from torch import Tensor

from .batch import ACTTD3Batch


_ACTION_KEY = "action"
_ACTION_PAD_KEY = "action_is_pad"
_EPISODE_INDEX_KEY = "episode_index"
_FRAME_INDEX_KEY = "frame_index"
_SUCCESS_KEY = "episode_success"


def _as_index(value: Any, *, name: str) -> int:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return _as_index(value.item(), name=name)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _as_success(value: Any) -> bool:
    if isinstance(value, Tensor):
        if value.numel() != 1 or value.dtype != torch.bool:
            raise TypeError("episode_success must be a scalar boolean")
        return bool(value.item())
    if isinstance(value, bool):
        return value
    # Hugging Face columns can expose numpy.bool_ without returning a tensor.
    value_type = type(value)
    if value_type.__module__.startswith("numpy") and value_type.__name__ == "bool_":
        return bool(value)
    raise TypeError("episode_success must be boolean, not an integer label")


def _read_column(table: Any, key: str, expected_length: int) -> Sequence[Any]:
    try:
        values = table[key]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"LeRobot dataset is missing column {key!r}") from error
    if len(values) != expected_length:
        raise ValueError(
            f"LeRobot column {key!r} has {len(values)} rows; expected {expected_length}"
        )
    return values


def _stack_actions(values: Sequence[Any]) -> Tensor:
    actions: list[Tensor] = []
    action_shape: tuple[int, ...] | None = None
    for row_index, value in enumerate(values):
        action = value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
        if action.ndim != 1 or action.numel() < 1 or not action.is_floating_point():
            raise ValueError(
                f"LeRobot action at row {row_index} must be a non-empty floating vector"
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError(f"LeRobot action at row {row_index} contains non-finite values")
        if action_shape is None:
            action_shape = tuple(action.shape)
        elif tuple(action.shape) != action_shape:
            raise ValueError("LeRobot action vectors must all have the same shape")
        actions.append(action.to(dtype=torch.float32).clone())
    if not actions:
        raise ValueError("LeRobot dataset must contain at least one action")
    return torch.stack(actions, dim=0)


def _freeze_feature_contract(value: Any, *, source: str) -> Any:
    """Convert LeRobot feature metadata into a deterministic comparable value."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_feature_contract(item, source=source))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_feature_contract(item, source=source) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # LeRobot occasionally exposes enum-like dtype/feature objects.  Their
    # string form is the public schema representation used in info.json.
    rendered = str(value)
    if not rendered or rendered.startswith("<"):
        raise TypeError(f"{source} contains unsupported feature metadata")
    return rendered


@dataclass(frozen=True)
class LeRobotACTTD3Transition:
    """One raw fixed-horizon transition before checkpoint preprocessing."""

    observations: dict[str, Tensor]
    next_observations: dict[str, Tensor]
    behavior_action_chunk: Tensor
    rewards: Tensor
    executed_mask: Tensor
    step_durations_s: Tensor
    episode_success: bool
    terminated: bool
    truncated: bool
    next_observation_valid: bool
    bootstrap_allowed: bool
    episode_index: int
    start_frame_index: int

    def __post_init__(self) -> None:
        if not self.observations or set(self.observations) != set(self.next_observations):
            raise ValueError("raw current and next observation keys must match and be non-empty")
        if (
            self.behavior_action_chunk.ndim != 2
            or not self.behavior_action_chunk.is_floating_point()
        ):
            raise ValueError("raw behavior action chunk must have shape (T, A)")
        horizon, action_dim = self.behavior_action_chunk.shape
        if horizon < 1 or action_dim < 1 or not bool(
            torch.isfinite(self.behavior_action_chunk).all()
        ):
            raise ValueError("raw behavior action chunk must be finite and non-empty")
        for name, value in (
            ("rewards", self.rewards),
            ("step_durations_s", self.step_durations_s),
        ):
            if (
                value.shape != (horizon,)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"raw {name} must be finite and have shape (T,)")
        if self.executed_mask.shape != (horizon,) or self.executed_mask.dtype != torch.bool:
            raise ValueError("raw executed_mask must be boolean (T,)")
        length = int(self.executed_mask.to(torch.long).sum().item())
        expected = torch.arange(horizon) < length
        if length < 1 or not torch.equal(self.executed_mask.cpu(), expected):
            raise ValueError("raw executed_mask must be an exact non-empty prefix")
        padding = ~self.executed_mask
        if bool((self.behavior_action_chunk[padding] != 0.0).any()):
            raise ValueError("raw padded behavior actions must be exactly zero")
        if bool((self.rewards[padding] != 0.0).any()):
            raise ValueError("raw padded rewards must be exactly zero")
        if bool((self.step_durations_s[padding] != 0.0).any()):
            raise ValueError("raw padded durations must be exactly zero")
        if bool((self.step_durations_s[self.executed_mask] <= 0.0).any()):
            raise ValueError("raw executed durations must be positive")
        if not isinstance(self.episode_success, bool):
            raise TypeError("raw episode_success must be boolean")
        if length < horizon and not (self.terminated or self.truncated):
            raise ValueError("raw partial chunks must terminate or truncate")
        if self.bootstrap_allowed and (self.terminated or not self.next_observation_valid):
            raise ValueError("raw bootstrap requires a valid non-terminal next observation")
        for key, current in self.observations.items():
            next_value = self.next_observations[key]
            if (
                not isinstance(current, Tensor)
                or not isinstance(next_value, Tensor)
                or current.shape != next_value.shape
                or not current.is_floating_point()
                or not next_value.is_floating_point()
                or not bool(torch.isfinite(current).all())
                or not bool(torch.isfinite(next_value).all())
            ):
                raise ValueError(f"raw observation {key!r} must be matching finite tensors")
            if not self.next_observation_valid and bool((next_value != 0.0).any()):
                raise ValueError(f"invalid raw next observation {key!r} must be zero")
        _as_index(self.episode_index, name="episode_index")
        _as_index(self.start_frame_index, name="start_frame_index")


@dataclass(frozen=True)
class _Block:
    episode_index: int
    start_frame_index: int
    executed_rows: tuple[int, ...]
    start_row: int
    next_row: int | None
    successful: bool


class FixedHorizonLeRobotACTTD3Dataset(torch.utils.data.Dataset):
    """Partition complete LeRobot episodes into non-overlapping ACT transitions.

    ``dataset`` must be an unwindowed ``LeRobotDataset`` constructed with
    ``delta_timestamps=None`` and ``return_uint8=False``.  Observations stay in
    raw dataset space here.  Use :class:`ACTTD3LeRobotCollator` with the
    immutable preprocessor loaded from the ACT checkpoint to obtain actor-ready
    :class:`ACTTD3Batch` objects.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        execution_horizon: int,
        observation_keys: Sequence[str],
    ) -> None:
        if isinstance(execution_horizon, bool) or not isinstance(execution_horizon, Integral):
            raise TypeError("execution_horizon must be an integer")
        self._execution_horizon = int(execution_horizon)
        if self._execution_horizon < 1:
            raise ValueError("execution_horizon must be positive")
        if isinstance(observation_keys, (str, bytes)):
            raise TypeError("observation_keys must be a sequence of feature names")
        self._observation_keys = tuple(observation_keys)
        if (
            not self._observation_keys
            or any(not isinstance(key, str) or not key for key in self._observation_keys)
            or len(set(self._observation_keys)) != len(self._observation_keys)
        ):
            raise ValueError("observation_keys must be unique non-empty strings")
        if getattr(dataset, "delta_timestamps", None) is not None:
            raise ValueError(
                "ACT-TD3 requires an unwindowed LeRobotDataset; delta_timestamps must be None"
            )
        if getattr(dataset, "_return_uint8", False) is not False:
            raise ValueError(
                "ACT-TD3 requires return_uint8=False so checkpoint image preprocessing is valid"
            )

        features = getattr(dataset, "features", None)
        if not isinstance(features, Mapping):
            raise TypeError("LeRobot dataset must expose a features mapping")
        required_features = {
            _ACTION_KEY,
            _EPISODE_INDEX_KEY,
            _FRAME_INDEX_KEY,
            _SUCCESS_KEY,
            *self._observation_keys,
        }
        missing_features = required_features.difference(features)
        if missing_features:
            raise ValueError(
                "LeRobot dataset is missing required features: "
                + ", ".join(sorted(missing_features))
            )

        fps_value = getattr(dataset, "fps", None)
        if isinstance(fps_value, bool):
            raise TypeError("LeRobot fps must be numeric")
        try:
            self._fps = float(fps_value)
        except (TypeError, ValueError) as error:
            raise TypeError("LeRobot dataset must expose numeric fps metadata") from error
        if not math.isfinite(self._fps) or self._fps <= 0.0:
            raise ValueError("LeRobot fps must be finite and positive")

        table = getattr(dataset, "hf_dataset", None)
        if table is None:
            raise TypeError("LeRobot dataset must expose hf_dataset")
        row_count = len(table)
        if row_count < 1 or len(dataset) != row_count:
            raise ValueError("LeRobot dataset and hf_dataset must have the same non-zero length")

        episode_values = _read_column(table, _EPISODE_INDEX_KEY, row_count)
        frame_values = _read_column(table, _FRAME_INDEX_KEY, row_count)
        success_values = _read_column(table, _SUCCESS_KEY, row_count)
        action_values = _read_column(table, _ACTION_KEY, row_count)
        self._actions = _stack_actions(action_values)
        self._dataset = dataset
        self._feature_shapes: dict[str, tuple[int, ...] | None] = {}
        for key in self._observation_keys:
            feature = features[key]
            shape = feature.get("shape") if isinstance(feature, Mapping) else None
            self._feature_shapes[key] = tuple(shape) if shape is not None else None
        contract_keys = (_ACTION_KEY, *self._observation_keys)
        self._schema_contract = tuple(
            (
                key,
                _freeze_feature_contract(
                    features[key], source=f"LeRobot feature {key!r}"
                ),
            )
            for key in contract_keys
        )
        self._camera_keys = tuple(
            key
            for key in self._observation_keys
            if key.startswith("observation.images.")
            or key.startswith("observation.image.")
        )

        episode_rows: dict[int, list[tuple[int, int]]] = {}
        episode_outcomes: dict[int, bool] = {}
        for row_index, (episode_value, frame_value, success_value) in enumerate(
            zip(episode_values, frame_values, success_values, strict=True)
        ):
            episode_index = _as_index(episode_value, name="episode_index")
            frame_index = _as_index(frame_value, name="frame_index")
            successful = _as_success(success_value)
            episode_rows.setdefault(episode_index, []).append((frame_index, row_index))
            previous = episode_outcomes.setdefault(episode_index, successful)
            if previous != successful:
                raise ValueError(
                    f"episode {episode_index} has inconsistent episode_success labels"
                )

        blocks: list[_Block] = []
        for episode_index in sorted(episode_rows):
            indexed_rows = sorted(episode_rows[episode_index])
            frame_indices = [frame for frame, _ in indexed_rows]
            if frame_indices != list(range(len(indexed_rows))):
                raise ValueError(
                    f"episode {episode_index} must contain every frame once, starting at frame 0"
                )
            rows = tuple(row for _, row in indexed_rows)
            for start in range(0, len(rows), self._execution_horizon):
                stop = min(start + self._execution_horizon, len(rows))
                next_row = (
                    rows[start + self._execution_horizon]
                    if start + self._execution_horizon < len(rows)
                    else None
                )
                blocks.append(
                    _Block(
                        episode_index=episode_index,
                        start_frame_index=start,
                        executed_rows=rows[start:stop],
                        start_row=rows[start],
                        next_row=next_row,
                        successful=episode_outcomes[episode_index],
                    )
                )
        self._blocks = tuple(blocks)
        self._episode_records = tuple(
            (
                episode_index,
                len(episode_rows[episode_index]),
                episode_outcomes[episode_index],
            )
            for episode_index in sorted(episode_rows)
        )
        self._episode_count = len(self._episode_records)
        self._success_count = sum(episode_outcomes.values())

    @property
    def execution_horizon(self) -> int:
        return self._execution_horizon

    @property
    def action_dim(self) -> int:
        return int(self._actions.shape[1])

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def num_episodes(self) -> int:
        return self._episode_count

    @property
    def num_successes(self) -> int:
        return self._success_count

    @property
    def num_failures(self) -> int:
        return self._episode_count - self._success_count

    @property
    def episode_records(self) -> tuple[tuple[int, int, bool], ...]:
        """Return immutable ``(index, frame_count, success)`` replay metadata."""

        return self._episode_records

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return self._observation_keys

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return self._camera_keys

    @property
    def schema_contract(self) -> tuple[tuple[str, Any], ...]:
        return self._schema_contract

    def __len__(self) -> int:
        return len(self._blocks)

    def _extract_observations(self, item: Any) -> dict[str, Tensor]:
        if not isinstance(item, Mapping):
            raise TypeError("LeRobot dataset items must be mappings")
        observations: dict[str, Tensor] = {}
        for key in self._observation_keys:
            if key not in item:
                raise ValueError(f"LeRobot item is missing observation {key!r}")
            value = item[key]
            tensor = value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
            expected_shape = self._feature_shapes[key]
            if (
                not tensor.is_floating_point()
                or not bool(torch.isfinite(tensor).all())
                or (expected_shape is not None and tuple(tensor.shape) != expected_shape)
            ):
                raise ValueError(
                    f"LeRobot observation {key!r} must be a finite floating tensor"
                    + (f" with shape {expected_shape}" if expected_shape is not None else "")
                )
            observations[key] = tensor.clone()
        return observations

    def __getitem__(self, index: int) -> LeRobotACTTD3Transition:
        block = self._blocks[index]
        observations = self._extract_observations(self._dataset[block.start_row])
        next_valid = block.next_row is not None
        if next_valid:
            next_observations = self._extract_observations(self._dataset[block.next_row])
        else:
            next_observations = {
                key: torch.zeros_like(value) for key, value in observations.items()
            }

        horizon = self._execution_horizon
        length = len(block.executed_rows)
        actions = torch.zeros((horizon, self.action_dim), dtype=self._actions.dtype)
        row_indices = torch.tensor(block.executed_rows, dtype=torch.long)
        actions[:length] = self._actions.index_select(0, row_indices)
        mask = torch.arange(horizon) < length
        rewards = torch.zeros(horizon, dtype=self._actions.dtype)
        terminated = not next_valid
        if terminated and block.successful:
            rewards[length - 1] = 1.0
        durations = mask.to(dtype=self._actions.dtype) * (1.0 / self._fps)

        return LeRobotACTTD3Transition(
            observations=observations,
            next_observations=next_observations,
            behavior_action_chunk=actions,
            rewards=rewards,
            executed_mask=mask,
            step_durations_s=durations,
            episode_success=block.successful,
            terminated=terminated,
            truncated=False,
            next_observation_valid=next_valid,
            bootstrap_allowed=next_valid,
            episode_index=block.episode_index,
            start_frame_index=block.start_frame_index,
        )


class VirtualCumulativeLeRobotACTTD3Dataset(torch.utils.data.Dataset):
    """Expose ordered immutable LeRobot roots as one logical replay dataset.

    No parquet, video, or metadata file is copied or rewritten.  Transition
    indices are mapped to their root-local adapter and local episode identifiers
    are remapped to a deterministic global sequence in root order.  This keeps
    training/checkpoint contracts independent from colliding local episode 0s.
    """

    def __init__(
        self,
        datasets: Sequence[FixedHorizonLeRobotACTTD3Dataset],
    ) -> None:
        if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
            raise TypeError("virtual replay datasets must be a sequence")
        self._datasets = tuple(datasets)
        if not self._datasets:
            raise ValueError("virtual replay requires at least one LeRobot root")
        if any(
            not isinstance(dataset, FixedHorizonLeRobotACTTD3Dataset)
            for dataset in self._datasets
        ):
            raise TypeError(
                "virtual replay roots must be FixedHorizonLeRobotACTTD3Dataset objects"
            )

        reference = self._datasets[0]
        for root_index, dataset in enumerate(self._datasets[1:], start=1):
            mismatches: list[str] = []
            if dataset.execution_horizon != reference.execution_horizon:
                mismatches.append("execution horizon")
            if dataset.action_dim != reference.action_dim:
                mismatches.append("action dimension")
            if float(dataset.fps) != float(reference.fps):
                mismatches.append("fps")
            if dataset.observation_keys != reference.observation_keys:
                mismatches.append("observation keys")
            if dataset.camera_keys != reference.camera_keys:
                mismatches.append("camera keys")
            if dataset.schema_contract != reference.schema_contract:
                mismatches.append("feature schema")
            if mismatches:
                raise ValueError(
                    f"LeRobot data root {root_index} disagrees with root 0: "
                    + ", ".join(mismatches)
                )

        transition_map: list[tuple[int, int]] = []
        episode_records: list[tuple[int, int, bool]] = []
        episode_maps: list[dict[int, int]] = []
        root_episode_ranges: list[tuple[int, int]] = []
        global_episode = 0
        for root_index, dataset in enumerate(self._datasets):
            local_to_global: dict[int, int] = {}
            range_start = global_episode
            for local_episode, frame_count, successful in dataset.episode_records:
                local_to_global[local_episode] = global_episode
                episode_records.append((global_episode, frame_count, successful))
                global_episode += 1
            episode_maps.append(local_to_global)
            root_episode_ranges.append((range_start, global_episode))
            transition_map.extend(
                (root_index, local_index) for local_index in range(len(dataset))
            )

        self._transition_map = tuple(transition_map)
        self._episode_maps = tuple(episode_maps)
        self._episode_records = tuple(episode_records)
        self._root_episode_ranges = tuple(root_episode_ranges)

    @property
    def execution_horizon(self) -> int:
        return self._datasets[0].execution_horizon

    @property
    def action_dim(self) -> int:
        return self._datasets[0].action_dim

    @property
    def fps(self) -> float:
        return self._datasets[0].fps

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return self._datasets[0].observation_keys

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return self._datasets[0].camera_keys

    @property
    def schema_contract(self) -> tuple[tuple[str, Any], ...]:
        return self._datasets[0].schema_contract

    @property
    def num_roots(self) -> int:
        return len(self._datasets)

    @property
    def num_episodes(self) -> int:
        return len(self._episode_records)

    @property
    def num_successes(self) -> int:
        return sum(record[2] for record in self._episode_records)

    @property
    def num_failures(self) -> int:
        return self.num_episodes - self.num_successes

    @property
    def episode_records(self) -> tuple[tuple[int, int, bool], ...]:
        return self._episode_records

    @property
    def root_episode_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return half-open global episode ranges for the ordered roots."""

        return self._root_episode_ranges

    @property
    def root_transition_counts(self) -> tuple[int, ...]:
        """Return each immutable root's macro-transition count in root order."""

        return tuple(len(dataset) for dataset in self._datasets)

    def __len__(self) -> int:
        return len(self._transition_map)

    def __getitem__(self, index: int) -> LeRobotACTTD3Transition:
        root_index, local_index = self._transition_map[index]
        transition = self._datasets[root_index][local_index]
        try:
            global_episode = self._episode_maps[root_index][transition.episode_index]
        except KeyError as error:
            raise RuntimeError("virtual replay local episode mapping is inconsistent") from error
        return replace(transition, episode_index=global_episode)


class ACTTD3LeRobotCollator:
    """Apply the saved ACT preprocessor and construct a strict TD3 batch.

    The saved pipeline owns the checkpoint's normalization and device transfer
    steps.  Invoke this collator in the training process, normally with
    ``DataLoader(num_workers=0)``; do not execute it in forked loader workers.
    """

    def __init__(self, preprocessor: Callable[[dict[str, Tensor]], Mapping[str, Tensor]]) -> None:
        if not callable(preprocessor):
            raise TypeError("preprocessor must be the callable loaded with the ACT checkpoint")
        self._preprocessor = preprocessor

    def __call__(self, transitions: Sequence[LeRobotACTTD3Transition]) -> ACTTD3Batch:
        if not transitions:
            raise ValueError("cannot collate an empty ACT-TD3 transition sequence")
        observation_keys = tuple(transitions[0].observations)
        if any(tuple(transition.observations) != observation_keys for transition in transitions):
            raise ValueError("all ACT-TD3 transitions must use the same observation keys")

        raw_observations = {
            key: torch.stack([transition.observations[key] for transition in transitions])
            for key in observation_keys
        }
        raw_next_observations = {
            key: torch.stack([transition.next_observations[key] for transition in transitions])
            for key in observation_keys
        }
        raw_actions = torch.stack(
            [transition.behavior_action_chunk for transition in transitions]
        )
        raw_mask = torch.stack([transition.executed_mask for transition in transitions])

        actor_input = dict(raw_observations)
        actor_input[_ACTION_KEY] = raw_actions
        actor_input[_ACTION_PAD_KEY] = ~raw_mask
        processed = self._preprocessor(actor_input)
        if not isinstance(processed, Mapping) or _ACTION_KEY not in processed:
            raise ValueError("ACT checkpoint preprocessor must return the action feature")
        actions = processed[_ACTION_KEY]
        if not isinstance(actions, Tensor) or actions.shape != raw_actions.shape:
            raise ValueError("preprocessed ACT actions must preserve shape (B, T, A)")
        mask = raw_mask.to(device=actions.device)
        actions = actions.clone().masked_fill(~mask.unsqueeze(-1), 0.0)

        observations: dict[str, Tensor] = {}
        for key in observation_keys:
            value = processed.get(key)
            if not isinstance(value, Tensor) or value.shape[0] != len(transitions):
                raise ValueError(
                    f"ACT checkpoint preprocessor must return batched observation {key!r}"
                )
            observations[key] = value

        valid_cpu = torch.tensor(
            [transition.next_observation_valid for transition in transitions],
            dtype=torch.bool,
        )
        next_observations = {
            key: torch.zeros_like(value) for key, value in observations.items()
        }
        if bool(valid_cpu.any()):
            raw_valid_indices = valid_cpu.nonzero(as_tuple=False).squeeze(1)
            next_input = {
                key: value.index_select(0, raw_valid_indices.to(value.device))
                for key, value in raw_next_observations.items()
            }
            processed_next = self._preprocessor(next_input)
            if not isinstance(processed_next, Mapping):
                raise ValueError("ACT checkpoint preprocessor must return a mapping")
            for key, current in observations.items():
                value = processed_next.get(key)
                expected_shape = (int(raw_valid_indices.numel()), *current.shape[1:])
                if (
                    not isinstance(value, Tensor)
                    or tuple(value.shape) != expected_shape
                    or value.dtype != current.dtype
                    or value.device != current.device
                ):
                    raise ValueError(
                        f"preprocessed next observation {key!r} must match current observations"
                    )
                next_observations[key].index_copy_(
                    0,
                    raw_valid_indices.to(current.device),
                    value,
                )

        device = actions.device
        dtype = actions.dtype

        def floats(values: Sequence[Tensor]) -> Tensor:
            return torch.stack(list(values)).to(device=device, dtype=dtype)

        def booleans(values: Sequence[bool]) -> Tensor:
            return torch.tensor(list(values), dtype=torch.bool, device=device)

        return ACTTD3Batch(
            observations=observations,
            next_observations=next_observations,
            behavior_action_chunks=actions,
            rewards=floats([transition.rewards for transition in transitions]),
            executed_mask=mask,
            step_durations_s=floats(
                [transition.step_durations_s for transition in transitions]
            ),
            episode_success=booleans(
                [transition.episode_success for transition in transitions]
            ),
            terminated=booleans([transition.terminated for transition in transitions]),
            truncated=booleans([transition.truncated for transition in transitions]),
            next_observation_valid=valid_cpu.to(device=device),
            bootstrap_allowed=booleans(
                [transition.bootstrap_allowed for transition in transitions]
            ),
        )


__all__ = [
    "ACTTD3LeRobotCollator",
    "FixedHorizonLeRobotACTTD3Dataset",
    "LeRobotACTTD3Transition",
    "VirtualCumulativeLeRobotACTTD3Dataset",
]
