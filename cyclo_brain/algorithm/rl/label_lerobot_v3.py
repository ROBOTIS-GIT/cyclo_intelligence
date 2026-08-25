"""Materialize one immutable LeRobot v3 dataset with episode outcomes.

The source dataset is never edited.  LeRobot's public ``add_features`` helper
creates a new dataset in a temporary directory, this module fills the
episode-level statistics required by Cyclo's offline-RL inventory, and the
validated result is atomically renamed into place.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EPISODE_SUCCESS = "episode_success"
DEFAULT_DATASET_LOCK = Path("/workspace/.cyclo_dataset.lock")


@dataclass(frozen=True)
class LabeledDatasetSummary:
    destination: Path
    episodes: int
    frames: int
    success: bool


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def _copy_optional_artifacts(source: Path, destination: Path) -> None:
    """Keep Cyclo annotations and fork metadata omitted by LeRobot editing."""

    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        if ".cache" in relative.parts:
            continue
        destination_path = destination / relative
        if source_path.is_symlink():
            raise ValueError(f"Source dataset contains a symbolic link: {source_path}")
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif not destination_path.exists():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def _restore_source_dataset_info(source: Path, destination: Path) -> None:
    """Preserve Cyclo's v3 metadata and add only the new outcome feature.

    LeRobot's dataset editing helper rebuilds ``meta/info.json`` and drops
    extension fields it does not know about.  In particular, Cyclo datasets
    carry ``annotation_path`` and per-feature ``fps`` values used by the data
    tooling.  The source is the authority for every existing field; the
    generated metadata is the authority only for ``episode_success``.
    """

    source_path = source / "meta" / "info.json"
    destination_path = destination / "meta" / "info.json"
    source_info = _read_json(source_path)
    generated_info = _read_json(destination_path)
    source_features = source_info.get("features")
    generated_features = generated_info.get("features")
    if not isinstance(source_features, dict) or not isinstance(generated_features, dict):
        raise ValueError("LeRobot meta/info.json has no feature mapping")
    success_feature = generated_features.get(EPISODE_SUCCESS)
    if not isinstance(success_feature, dict):
        raise ValueError("Generated dataset has no episode_success feature")

    merged_info = dict(source_info)
    merged_features = dict(source_features)
    merged_features[EPISODE_SUCCESS] = success_feature
    merged_info["features"] = merged_features
    _write_json(destination_path, merged_info)


def _patch_episode_statistics(dataset_root: Path, success: bool) -> int:
    episode_paths = sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    )
    if not episode_paths:
        raise ValueError("LeRobot v3 dataset has no episode metadata parquet")

    total_episodes = 0
    for path in episode_paths:
        table = pq.read_table(path)
        if "length" not in table.column_names:
            raise ValueError(f"Episode metadata is missing length: {path}")
        lengths = table["length"].to_pylist()
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in lengths):
            raise ValueError(f"Episode metadata has an invalid length: {path}")

        names = (
            "stats/episode_success/min",
            "stats/episode_success/max",
            "stats/episode_success/mean",
            "stats/episode_success/std",
            "stats/episode_success/count",
        )
        if any(name in table.column_names for name in names):
            raise ValueError(f"Destination already has episode_success statistics: {path}")

        row_count = table.num_rows
        table = table.append_column(
            names[0], pa.array([[success]] * row_count, type=pa.list_(pa.bool_()))
        )
        table = table.append_column(
            names[1], pa.array([[success]] * row_count, type=pa.list_(pa.bool_()))
        )
        table = table.append_column(
            names[2],
            pa.array([[float(success)]] * row_count, type=pa.list_(pa.float64())),
        )
        table = table.append_column(
            names[3], pa.array([[0.0]] * row_count, type=pa.list_(pa.float64()))
        )
        table = table.append_column(
            names[4],
            pa.array(
                [[int(length)] for length in lengths],
                type=pa.list_(pa.int64()),
            ),
        )
        _write_parquet(path, table)
        total_episodes += row_count
    return total_episodes


def _patch_global_statistics(dataset_root: Path, success: bool) -> None:
    path = dataset_root / "meta" / "stats.json"
    stats = _read_json(path)
    if EPISODE_SUCCESS in stats:
        raise ValueError("Destination already has global episode_success statistics")
    value = float(success)
    stats[EPISODE_SUCCESS] = {
        "mean": [value],
        "std": [0.001],
        "min": [value],
        "max": [value],
    }
    _write_json(path, stats)


def _patch_derivation_metadata(
    dataset_root: Path,
    *,
    source: Path,
    success: bool,
    episodes: int,
) -> None:
    path = dataset_root / "info.json"
    if not path.is_file():
        return
    payload = _read_json(path)
    payload["derived_dataset"] = {
        "source_dataset": str(source),
        "episode_success": success,
        "episode_count": episodes,
    }
    _write_json(path, payload)


def _validate_labeled_dataset(
    dataset_root: Path,
    *,
    expected_success: bool,
) -> tuple[int, int]:
    info = _read_json(dataset_root / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Outcome materialization requires LeRobot v3.0")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot meta/info.json has no feature mapping")
    success_feature = features.get(EPISODE_SUCCESS)
    if not isinstance(success_feature, dict) or success_feature.get("dtype") != "bool":
        raise ValueError("LeRobot episode_success feature is not boolean")

    expected_frames = int(info.get("total_frames", -1))
    expected_episodes = int(info.get("total_episodes", -1))
    if expected_frames < 1 or expected_episodes < 1:
        raise ValueError("LeRobot dataset has invalid frame or episode totals")

    frame_count = 0
    for path in sorted((dataset_root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=[EPISODE_SUCCESS])
        values = table[EPISODE_SUCCESS]
        if not pa.types.is_boolean(values.type):
            raise ValueError(f"episode_success is not boolean in {path}")
        if any(value.as_py() is not expected_success for value in values):
            raise ValueError(f"episode_success contains an unexpected value in {path}")
        frame_count += table.num_rows
    if frame_count != expected_frames:
        raise ValueError(
            f"Frame total changed during outcome materialization: {frame_count} != {expected_frames}"
        )

    episode_count = 0
    for path in sorted(
        (dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    ):
        table = pq.read_table(
            path,
            columns=[
                "length",
                "stats/episode_success/min",
                "stats/episode_success/max",
                "stats/episode_success/mean",
                "stats/episode_success/std",
                "stats/episode_success/count",
            ],
        )
        payload = table.to_pydict()
        for row, length in enumerate(payload["length"]):
            expected_bool = [expected_success]
            expected_float = [float(expected_success)]
            if payload["stats/episode_success/min"][row] != expected_bool:
                raise ValueError("Episode success minimum is inconsistent")
            if payload["stats/episode_success/max"][row] != expected_bool:
                raise ValueError("Episode success maximum is inconsistent")
            if payload["stats/episode_success/mean"][row] != expected_float:
                raise ValueError("Episode success mean is inconsistent")
            if payload["stats/episode_success/std"][row] != [0.0]:
                raise ValueError("Episode success standard deviation is inconsistent")
            if payload["stats/episode_success/count"][row] != [length]:
                raise ValueError("Episode success count does not match episode length")
        episode_count += table.num_rows
    if episode_count != expected_episodes:
        raise ValueError(
            "Episode total changed during outcome materialization: "
            f"{episode_count} != {expected_episodes}"
        )

    stats = _read_json(dataset_root / "meta" / "stats.json")
    expected_value = float(expected_success)
    if stats.get(EPISODE_SUCCESS) != {
        "mean": [expected_value],
        "std": [0.001],
        "min": [expected_value],
        "max": [expected_value],
    }:
        raise ValueError("Global episode_success statistics are inconsistent")
    return episode_count, frame_count


@contextmanager
def _dataset_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def derive_labeled_v3_dataset(
    source: str | Path,
    destination: str | Path,
    *,
    success: bool,
    lock_path: str | Path = DEFAULT_DATASET_LOCK,
) -> LabeledDatasetSummary:
    """Create and atomically publish a fully labeled derivative dataset."""

    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_path = Path(source).expanduser().resolve(strict=True)
    destination_path = Path(destination).expanduser().absolute()
    if not source_path.is_dir():
        raise ValueError(f"Source dataset is not a directory: {source_path}")
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path == destination_path or source_path in destination_path.parents:
        raise ValueError("Destination must not be the source or a child of the source")

    source_info = _read_json(source_path / "meta" / "info.json")
    source_features = source_info.get("features")
    if not isinstance(source_features, dict) or EPISODE_SUCCESS in source_features:
        raise ValueError("Source must be an unlabeled LeRobot v3 dataset")
    total_frames = int(source_info.get("total_frames", -1))
    if total_frames < 1:
        raise ValueError("Source dataset has no frames")

    with _dataset_lock(Path(lock_path)):
        temporary_parent = Path(
            tempfile.mkdtemp(
                prefix=f".{destination_path.name}.",
                dir=destination_path.parent,
            )
        )
        temporary_dataset = temporary_parent / "dataset"
        try:
            source_dataset = LeRobotDataset(
                repo_id="cyclo-local/source-outcome-materialization",
                root=source_path,
                download_videos=False,
            )
            add_features(
                source_dataset,
                {
                    EPISODE_SUCCESS: (
                        np.full(total_frames, success, dtype=np.bool_),
                        {
                            "dtype": "bool",
                            "shape": (1,),
                            "names": None,
                            "fps": int(source_info["fps"]),
                        },
                    )
                },
                output_dir=temporary_dataset,
                repo_id="cyclo-local/labeled-outcome-materialization",
            )
            _copy_optional_artifacts(source_path, temporary_dataset)
            _restore_source_dataset_info(source_path, temporary_dataset)
            episodes = _patch_episode_statistics(temporary_dataset, success)
            _patch_global_statistics(temporary_dataset, success)
            _patch_derivation_metadata(
                temporary_dataset,
                source=source_path,
                success=success,
                episodes=episodes,
            )
            validated_episodes, validated_frames = _validate_labeled_dataset(
                temporary_dataset,
                expected_success=success,
            )
            os.replace(temporary_dataset, destination_path)
        finally:
            shutil.rmtree(temporary_parent, ignore_errors=True)

    return LabeledDatasetSummary(
        destination=destination_path,
        episodes=validated_episodes,
        frames=validated_frames,
        success=success,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--outcome",
        choices=("success", "fail"),
        required=True,
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_DATASET_LOCK)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = derive_labeled_v3_dataset(
        args.source,
        args.destination,
        success=args.outcome == "success",
        lock_path=args.lock_path,
    )
    print(
        json.dumps(
            {
                "destination": str(summary.destination),
                "episodes": summary.episodes,
                "frames": summary.frames,
                "outcome": "success" if summary.success else "fail",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
