"""LeRobot v3 data-epoch adapter for ACT behavior cloning.

LeRobot 0.5.2 intentionally disables its public multi-dataset factory.  This
module keeps every converted data epoch immutable and exposes the selected
demonstration episodes as one logical PyTorch dataset.  Labeled replay roots
remain success-filtered, while conventional unlabeled imitation datasets use
all selected episodes.  No parquet, video, or metadata file is copied or
rewritten.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch


LEROBOT_CODEBASE_VERSION = "v3.0"
EPISODE_SUCCESS_FEATURE = "episode_success"
_REQUIRED_STATS = frozenset({"min", "max", "mean", "std", "count"})


def parse_success_episode_csv(value: str) -> tuple[int, ...]:
    """Parse one CLI ``--success-episodes`` value without accepting ambiguity."""

    if not isinstance(value, str):
        raise TypeError("success episode CSV must be a string")
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("success episode CSV must contain comma-separated indices")
    try:
        indices = tuple(int(token, 10) for token in tokens)
    except ValueError as error:
        raise ValueError("success episode indices must be base-10 integers") from error
    if any(index < 0 for index in indices):
        raise ValueError("success episode indices must be non-negative")
    if len(indices) != len(set(indices)):
        raise ValueError("success episode indices cannot contain duplicates")
    return indices


@dataclass(frozen=True)
class RootSelection:
    """One immutable LeRobot root and its selected demonstration episodes.

    ``success_episodes`` retains its original field name for checkpoint/CLI
    compatibility.  For an unlabeled imitation root it simply contains every
    selected root-local episode index.
    """

    root: Path
    success_episodes: tuple[int, ...]

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        episodes = tuple(self.success_episodes)
        if not episodes:
            raise ValueError(f"LeRobot root {root} has no selected success episodes")
        if any(isinstance(index, bool) or not isinstance(index, Integral) for index in episodes):
            raise TypeError("success episode indices must be integers")
        episodes = tuple(int(index) for index in episodes)
        if any(index < 0 for index in episodes):
            raise ValueError("success episode indices must be non-negative")
        if len(episodes) != len(set(episodes)):
            raise ValueError("success episode indices cannot contain duplicates")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "success_episodes", episodes)


@dataclass(frozen=True)
class LeRobotDatasetDependencies:
    """Pinned LeRobot APIs injected lazily by the training process."""

    metadata_cls: type
    dataset_cls: type
    resolve_delta_timestamps: Callable[[Any, Any], dict[str, list[float]] | None]
    aggregate_stats: Callable[[list[dict[str, dict]]], dict[str, dict]]
    load_episode_with_stats: Callable[[Any, int], Mapping[str, Any]]


class _AggregatedMetadata:
    """Read-only view of root-zero metadata with selected-episode statistics."""

    def __init__(self, base: Any, stats: Mapping[str, Mapping[str, Any]]) -> None:
        self._base = base
        self.stats = dict(stats)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _version(meta: Any) -> str:
    info = getattr(meta, "info", None)
    value = getattr(info, "codebase_version", None)
    if value is None and isinstance(info, Mapping):
        value = info.get("codebase_version")
    return str(value) if value is not None else ""


def _feature_schema(features: Mapping[str, Mapping[str, Any]]) -> tuple[Any, ...]:
    """Return the model-facing feature contract, excluding outcome metadata."""

    contract = []
    for key in sorted(features):
        # Outcome labels are replay metadata, not ACT inputs.  Excluding them
        # lets a labeled RL data epoch and an unlabeled demonstration epoch
        # share one otherwise-identical BC feature contract.
        if key == EPISODE_SUCCESS_FEATURE:
            continue
        feature = features[key]
        shape = tuple(int(value) for value in feature.get("shape", ()))
        names_value = feature.get("names")
        names = None if names_value is None else tuple(str(value) for value in names_value)
        contract.append((str(key), str(feature.get("dtype")), shape, names))
    return tuple(contract)


def _validate_metadata(meta: Any, *, source: Path) -> tuple[float, tuple[Any, ...]]:
    version = _version(meta)
    if version != LEROBOT_CODEBASE_VERSION:
        raise ValueError(
            f"LeRobot root {source} uses {version or 'an unknown version'}; "
            f"expected {LEROBOT_CODEBASE_VERSION}"
        )
    fps = float(getattr(meta, "fps"))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"LeRobot root {source} has invalid fps {fps!r}")
    features = getattr(meta, "features", None)
    if not isinstance(features, Mapping) or "action" not in features:
        raise ValueError(f"LeRobot root {source} is missing an action feature schema")
    if "observation.state" not in features:
        raise ValueError(f"LeRobot root {source} is missing observation.state")
    camera_keys = [
        key
        for key, feature in features.items()
        if str(key).startswith("observation.")
        and str(feature.get("dtype")) in {"image", "video"}
    ]
    if len(camera_keys) != 3:
        raise ValueError(
            f"LeRobot root {source} must provide exactly three image/video observations; "
            f"got {len(camera_keys)}"
        )
    return fps, _feature_schema(features)


def _episode_indices(meta: Any) -> set[int]:
    episodes = getattr(meta, "episodes", None)
    if episodes is None:
        return set()
    try:
        values = episodes["episode_index"]
    except (KeyError, TypeError, ValueError):
        values = []
        try:
            values = [row["episode_index"] for row in episodes]
        except (KeyError, TypeError):
            pass
    return {int(value.item() if hasattr(value, "item") else value) for value in values}


def _as_stat_array(value: Any, *, image: bool, stat_name: str) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if image and stat_name != "count" and array.shape != (3, 1, 1):
        if array.size != 3:
            raise ValueError(
                f"image statistic {stat_name!r} must contain three channels; got {array.shape}"
            )
        array = np.asarray(array.tolist(), dtype=np.float64).reshape(3, 1, 1)
    return array


def _episode_stats(
    row: Mapping[str, Any],
    features: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, value in row.items():
        if not str(key).startswith("stats/"):
            continue
        feature_name, separator, stat_name = str(key)[len("stats/") :].rpartition("/")
        if (
            not separator
            or feature_name not in features
            or feature_name == EPISODE_SUCCESS_FEATURE
        ):
            continue
        is_image = str(features[feature_name].get("dtype")) in {"image", "video"}
        result.setdefault(feature_name, {})[stat_name] = _as_stat_array(
            value,
            image=is_image,
            stat_name=stat_name,
        )
    if not result:
        raise ValueError("selected episode metadata contains no statistics")
    incomplete = sorted(
        feature
        for feature, values in result.items()
        if not _REQUIRED_STATS.issubset(values)
    )
    if incomplete:
        raise ValueError(
            "selected episode metadata has incomplete statistics for: " + ", ".join(incomplete)
        )
    return result


def _require_success(row: Mapping[str, Any], *, root: Path, episode: int) -> None:
    candidates = (
        "stats/episode_success/mean",
        "stats/episode_success/min",
        "stats/episode_success/max",
    )
    values = [row[key] for key in candidates if key in row]
    if not values:
        raise ValueError(
            f"LeRobot root {root} episode {episode} has no episode_success statistics"
        )
    if any(not bool(np.asarray(value).astype(np.float64).min() == 1.0) for value in values):
        raise ValueError(
            f"LeRobot root {root} episode {episode} is not labeled successful"
        )


def _require_local_v3_layout(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset root does not exist: {root}")
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset is missing {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read LeRobot metadata {info_path}: {error}") from error
    version = str(info.get("codebase_version", ""))
    if version != LEROBOT_CODEBASE_VERSION:
        raise ValueError(
            f"LeRobot root {root} uses {version or 'an unknown version'}; "
            f"expected {LEROBOT_CODEBASE_VERSION}"
        )
    for relative in (Path("meta/stats.json"), Path("meta/episodes"), Path("data")):
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"LeRobot dataset is missing {path}")
    features = info.get("features", {})
    if any(feature.get("dtype") == "video" for feature in features.values()):
        videos = root / "videos"
        if not videos.is_dir():
            raise FileNotFoundError(f"LeRobot video dataset is missing {videos}")


class VirtualACTBCDataset(torch.utils.data.Dataset):
    """Expose compatible, success-filtered roots as one logical ACT dataset."""

    def __init__(
        self,
        datasets: Sequence[Any],
        selections: Sequence[RootSelection],
        selected_episode_stats: Sequence[Mapping[str, Mapping[str, Any]]],
        *,
        aggregate_stats: Callable[[list[dict[str, dict]]], dict[str, dict]],
    ) -> None:
        if not datasets:
            raise ValueError("ACT behavior cloning requires at least one LeRobot root")
        if len(datasets) != len(selections):
            raise ValueError("dataset and root-selection counts must match")
        if len(selected_episode_stats) != sum(
            len(selection.success_episodes) for selection in selections
        ):
            raise ValueError("selected episode-stat count does not match success selections")

        reference_fps: float | None = None
        reference_schema: tuple[Any, ...] | None = None
        offsets = [0]
        for root_index, (dataset, selection) in enumerate(zip(datasets, selections, strict=True)):
            if len(dataset) < 1:
                raise ValueError(f"LeRobot root {selection.root} selected no frames")
            fps, schema = _validate_metadata(dataset.meta, source=selection.root)
            if reference_fps is None:
                reference_fps, reference_schema = fps, schema
            else:
                mismatches = []
                if fps != reference_fps:
                    mismatches.append("fps")
                if schema != reference_schema:
                    mismatches.append("feature schema")
                if mismatches:
                    raise ValueError(
                        f"LeRobot data root {root_index} disagrees with root 0: "
                        + ", ".join(mismatches)
                    )
            offsets.append(offsets[-1] + len(dataset))

        stats = aggregate_stats([dict(value) for value in selected_episode_stats])
        if not stats:
            raise ValueError("aggregated selected-episode statistics are empty")
        self._datasets = tuple(datasets)
        self._selections = tuple(selections)
        self._offsets = tuple(offsets)
        self.meta = _AggregatedMetadata(self._datasets[0].meta, stats)
        self.num_frames = offsets[-1]
        self.num_episodes = sum(len(selection.success_episodes) for selection in selections)
        self.fps = float(reference_fps)

    @property
    def selections(self) -> tuple[RootSelection, ...]:
        return self._selections

    @property
    def datasets(self) -> tuple[Any, ...]:
        return self._datasets

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("ACT dataset index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"ACT dataset index {index} is out of range")
        for dataset_index, stop in enumerate(self._offsets[1:]):
            if index < stop:
                return self._datasets[dataset_index][index - self._offsets[dataset_index]]
        raise AssertionError("validated virtual index was not mapped")


def load_virtual_act_bc_dataset(
    selections: Sequence[RootSelection],
    *,
    policy_config: Any,
    dependencies: LeRobotDatasetDependencies,
    video_backend: str = "pyav",
    tolerance_s: float = 1e-4,
    image_transforms: Any | None = None,
) -> VirtualACTBCDataset:
    """Load selected demonstrations and construct a virtual ACT dataset.

    When a root declares ``episode_success`` the selected indices are checked
    as successful so known failures cannot be imitated accidentally.  When the
    feature is absent, the root is a normal unlabeled imitation dataset and no
    outcome label is required.
    """

    selections = tuple(selections)
    if not selections:
        raise ValueError("ACT behavior cloning requires at least one LeRobot root")
    datasets = []
    all_stats: list[dict[str, dict[str, np.ndarray]]] = []
    for root_index, selection in enumerate(selections):
        _require_local_v3_layout(selection.root)
        repo_id = f"cyclo-local/act-bc-{root_index}-{selection.root.name}"
        metadata = dependencies.metadata_cls(
            repo_id,
            root=selection.root,
            revision=LEROBOT_CODEBASE_VERSION,
        )
        _validate_metadata(metadata, source=selection.root)
        features = metadata.features
        require_success = EPISODE_SUCCESS_FEATURE in features
        available = _episode_indices(metadata)
        missing = sorted(set(selection.success_episodes).difference(available))
        if missing:
            raise ValueError(
                f"LeRobot root {selection.root} is missing episode indices: "
                + ", ".join(str(value) for value in missing)
            )
        delta_timestamps = dependencies.resolve_delta_timestamps(policy_config, metadata)
        dataset_kwargs = {
            "root": selection.root,
            "episodes": list(selection.success_episodes),
            "delta_timestamps": delta_timestamps,
            "revision": LEROBOT_CODEBASE_VERSION,
            "video_backend": video_backend,
            "return_uint8": True,
            "tolerance_s": tolerance_s,
            "download_videos": False,
        }
        # MultiTaskDiT requires every camera to share one spatial shape before
        # DataLoader collation.  ACT keeps the previous no-transform behavior,
        # while other IL policies may opt into a per-frame transform here.
        if image_transforms is not None:
            dataset_kwargs["image_transforms"] = image_transforms
        dataset = dependencies.dataset_cls(repo_id, **dataset_kwargs)
        for episode in selection.success_episodes:
            row = dependencies.load_episode_with_stats(dataset, episode)
            if require_success:
                _require_success(row, root=selection.root, episode=episode)
            all_stats.append(_episode_stats(row, dataset.meta.features))
        datasets.append(dataset)
    return VirtualACTBCDataset(
        datasets,
        selections,
        all_stats,
        aggregate_stats=dependencies.aggregate_stats,
    )


__all__ = [
    "LEROBOT_CODEBASE_VERSION",
    "LeRobotDatasetDependencies",
    "RootSelection",
    "VirtualACTBCDataset",
    "load_virtual_act_bc_dataset",
    "parse_success_episode_csv",
]
