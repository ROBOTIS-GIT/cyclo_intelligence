"""Discover consistently labelled success episodes in a LeRobot v3 dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


_OUTCOME_COLUMNS = (
    "episode_index",
    "length",
    "stats/episode_success/min",
    "stats/episode_success/max",
    "stats/episode_success/mean",
    "stats/episode_success/count",
)


@dataclass(frozen=True)
class EpisodeOutcomeSplit:
    """Deterministic success/failure split and its frame counts."""

    success_episodes: tuple[int, ...]
    failure_episodes: tuple[int, ...]
    success_frames: int
    failure_frames: int

    @property
    def success_episode_count(self) -> int:
        return len(self.success_episodes)

    @property
    def failure_episode_count(self) -> int:
        return len(self.failure_episodes)

    @property
    def total_episode_count(self) -> int:
        return self.success_episode_count + self.failure_episode_count

    @property
    def total_frames(self) -> int:
        return self.success_frames + self.failure_frames


def _unwrap_singleton(value: Any, *, field: str) -> Any:
    """Unwrap Arrow, tensor, ndarray, and Python singleton containers."""

    for _ in range(16):
        as_py = getattr(value, "as_py", None)
        if callable(as_py):
            value = as_py()
            continue

        # torch.Tensor and numpy.ndarray both expose numel/size plus item.
        numel = getattr(value, "numel", None)
        if callable(numel):
            if int(numel()) != 1:
                raise ValueError(f"{field} must contain exactly one value")
            value = value.item()
            continue
        size = getattr(value, "size", None)
        if isinstance(size, int):
            if size != 1:
                raise ValueError(f"{field} must contain exactly one value")
            value = value.item()
            continue

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{field} must contain exactly one value")
            value = value[0]
            continue

        # numpy scalar values expose item() but no integer-valued ``size``.
        item = getattr(value, "item", None)
        if callable(item) and type(value).__module__.startswith("numpy"):
            value = item()
            continue
        return value

    raise ValueError(f"{field} has too many nested singleton containers")


def _as_non_negative_int(value: Any, *, field: str, positive: bool = False) -> int:
    value = _unwrap_singleton(value, field=field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        relation = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {relation}")
    return value


def _as_bool(value: Any, *, field: str) -> bool:
    value = _unwrap_singleton(value, field=field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _as_finite_float(value: Any, *, field: str) -> float:
    value = _unwrap_singleton(value, field=field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def classify_episode_outcome_rows(
    rows: Iterable[Mapping[str, Any]],
) -> EpisodeOutcomeSplit:
    """Validate and classify rows read from ``meta/episodes/*.parquet``.

    An episode is accepted only when every frame had one outcome: ``min`` and
    ``max`` must agree, ``mean`` must be exactly 0 or 1 accordingly, and the
    recorded outcome count must equal the episode length.  This prevents a
    partially or inconsistently labelled episode from entering pretraining.
    """

    episodes: dict[int, tuple[bool, int]] = {}
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"episode outcome row {row_number} must be a mapping")
        missing = [column for column in _OUTCOME_COLUMNS if column not in row]
        if missing:
            raise ValueError(
                f"episode outcome row {row_number} is missing columns: {', '.join(missing)}"
            )

        episode_id = _as_non_negative_int(row["episode_index"], field="episode_index")
        if episode_id in episodes:
            raise ValueError(f"duplicate episode_index {episode_id} in episode metadata")
        length = _as_non_negative_int(row["length"], field="length", positive=True)
        minimum = _as_bool(
            row["stats/episode_success/min"],
            field="stats/episode_success/min",
        )
        maximum = _as_bool(
            row["stats/episode_success/max"],
            field="stats/episode_success/max",
        )
        mean = _as_finite_float(
            row["stats/episode_success/mean"],
            field="stats/episode_success/mean",
        )
        count = _as_non_negative_int(
            row["stats/episode_success/count"],
            field="stats/episode_success/count",
        )

        if minimum != maximum:
            raise ValueError(f"episode {episode_id} contains mixed success labels")
        expected_mean = 1.0 if minimum else 0.0
        if mean != expected_mean:
            raise ValueError(
                f"episode {episode_id} outcome mean {mean} contradicts label {minimum}"
            )
        if count != length:
            raise ValueError(
                f"episode {episode_id} outcome count {count} does not match length {length}"
            )
        episodes[episode_id] = (minimum, length)

    if not episodes:
        raise ValueError("LeRobot episode metadata contains no episodes")

    success_ids = tuple(sorted(index for index, (success, _) in episodes.items() if success))
    failure_ids = tuple(sorted(index for index, (success, _) in episodes.items() if not success))
    return EpisodeOutcomeSplit(
        success_episodes=success_ids,
        failure_episodes=failure_ids,
        success_frames=sum(episodes[index][1] for index in success_ids),
        failure_frames=sum(episodes[index][1] for index in failure_ids),
    )


def discover_episode_outcomes(dataset_root: str | Path) -> EpisodeOutcomeSplit:
    """Read LeRobot v3 episode parquet files and return a validated split.

    ``pyarrow`` is imported lazily so importing the Cyclo model package does
    not require dataset tooling in inference-only environments.
    """

    root = Path(dataset_root).expanduser()
    episodes_root = root if root.name == "episodes" else root / "meta" / "episodes"
    parquet_files = sorted(episodes_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no LeRobot episode parquet files under {episodes_root}")

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to discover LeRobot success episodes"
        ) from error

    rows: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        try:
            table = parquet.read_table(parquet_file, columns=list(_OUTCOME_COLUMNS))
        except Exception as error:
            raise ValueError(f"failed to read episode outcomes from {parquet_file}: {error}") from error
        rows.extend(table.to_pylist())
    return classify_episode_outcome_rows(rows)


__all__ = [
    "EpisodeOutcomeSplit",
    "classify_episode_outcome_rows",
    "discover_episode_outcomes",
]
