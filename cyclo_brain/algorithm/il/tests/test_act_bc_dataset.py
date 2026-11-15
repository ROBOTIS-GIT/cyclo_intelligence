from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cyclo_brain.algorithm.il.act_bc.dataset import (
    LeRobotDatasetDependencies,
    RootSelection,
    VirtualACTBCDataset,
    load_virtual_act_bc_dataset,
    parse_success_episode_csv,
)


def _features(*, action_size: int = 2, include_success: bool = True) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [action_size],
            "names": [f"joint_{index}" for index in range(action_size)],
        },
        "action": {
            "dtype": "float32",
            "shape": [action_size],
            "names": [f"joint_{index}" for index in range(action_size)],
        },
        "observation.images.left": {
            "dtype": "video",
            "shape": [3, 8, 8],
            "names": ["channels", "height", "width"],
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": [3, 8, 8],
            "names": ["channels", "height", "width"],
        },
        "observation.images.right": {
            "dtype": "video",
            "shape": [3, 8, 8],
            "names": ["channels", "height", "width"],
        },
    }
    if include_success:
        features["episode_success"] = {"dtype": "bool", "shape": [1], "names": None}
    return features


class _FakeMeta:
    def __init__(self, *, features: dict | None = None, fps: int = 15, episodes=(0, 1)):
        self.info = SimpleNamespace(codebase_version="v3.0")
        self.features = features or _features()
        self.fps = fps
        self.episodes = {"episode_index": list(episodes)}
        self.stats = {"unselected": {}}
        self.camera_keys = [
            key
            for key, feature in self.features.items()
            if feature["dtype"] in {"image", "video"}
        ]


class _FakeDataset:
    def __init__(self, label: str, meta: _FakeMeta, length: int):
        self.label = label
        self.meta = meta
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self.label, index


def _episode_row(
    value: float = 1.0,
    *,
    successful: bool = True,
    include_success: bool = True,
) -> dict:
    row = {}
    for feature, spec in _features(include_success=include_success).items():
        image = spec["dtype"] in {"image", "video"}
        shape = (3, 1, 1) if image else tuple(spec["shape"])
        for stat in ("min", "max", "mean"):
            row[f"stats/{feature}/{stat}"] = np.full(shape, value)
        row[f"stats/{feature}/std"] = np.zeros(shape)
        row[f"stats/{feature}/count"] = np.asarray([1])
    if include_success:
        success = np.asarray([1.0 if successful else 0.0])
        for stat in ("min", "max", "mean"):
            row[f"stats/episode_success/{stat}"] = success
    return row


def _nested_stats(value: float = 1.0) -> dict:
    result = {}
    for feature, spec in _features().items():
        image = spec["dtype"] in {"image", "video"}
        shape = (3, 1, 1) if image else tuple(spec["shape"])
        result[feature] = {
            "min": np.full(shape, value),
            "max": np.full(shape, value),
            "mean": np.full(shape, value),
            "std": np.zeros(shape),
            "count": np.asarray([1]),
        }
    return result


class ACTBCDatasetTest(unittest.TestCase):
    def test_success_csv_is_strict(self):
        self.assertEqual(parse_success_episode_csv("0, 2,5"), (0, 2, 5))
        for invalid in ("", "0,", "-1", "1,1", "x"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_success_episode_csv(invalid)

    def test_virtual_concat_preserves_root_order_and_aggregates_selected_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = (
                RootSelection(root / "epoch0", (0,)),
                RootSelection(root / "epoch1", (2,)),
            )
            seen = []

            def aggregate(values):
                seen.extend(values)
                return {"action": values[0]["action"]}

            dataset = VirtualACTBCDataset(
                (
                    _FakeDataset("epoch0", _FakeMeta(), 2),
                    _FakeDataset("epoch1", _FakeMeta(), 3),
                ),
                selections,
                (_nested_stats(1.0), _nested_stats(3.0)),
                aggregate_stats=aggregate,
            )
            self.assertEqual(len(dataset), 5)
            self.assertEqual(dataset.num_episodes, 2)
            self.assertEqual(dataset[1], ("epoch0", 1))
            self.assertEqual(dataset[2], ("epoch1", 0))
            self.assertEqual(dataset[-1], ("epoch1", 2))
            self.assertEqual(len(seen), 2)
            self.assertNotIn("unselected", dataset.meta.stats)

    def test_virtual_concat_rejects_fps_schema_and_camera_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selections = (
                RootSelection(root / "epoch0", (0,)),
                RootSelection(root / "epoch1", (0,)),
            )
            stats = (_nested_stats(), _nested_stats())
            with self.assertRaisesRegex(ValueError, "fps"):
                VirtualACTBCDataset(
                    (
                        _FakeDataset("a", _FakeMeta(fps=15), 1),
                        _FakeDataset("b", _FakeMeta(fps=10), 1),
                    ),
                    selections,
                    stats,
                    aggregate_stats=lambda values: values[0],
                )
            changed = _features(action_size=3)
            with self.assertRaisesRegex(ValueError, "feature schema"):
                VirtualACTBCDataset(
                    (
                        _FakeDataset("a", _FakeMeta(), 1),
                        _FakeDataset("b", _FakeMeta(features=changed), 1),
                    ),
                    selections,
                    stats,
                    aggregate_stats=lambda values: values[0],
                )
            missing_camera = _features()
            missing_camera.pop("observation.images.right")
            with self.assertRaisesRegex(ValueError, "exactly three"):
                VirtualACTBCDataset(
                    (_FakeDataset("a", _FakeMeta(features=missing_camera), 1),),
                    (selections[0],),
                    (stats[0],),
                    aggregate_stats=lambda values: values[0],
                )

    def test_loader_selects_only_declared_successes_and_builds_action_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "epoch"
            (root / "meta" / "episodes").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "videos").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps({"codebase_version": "v3.0", "features": _features()}),
                encoding="utf-8",
            )
            (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
            metadata = _FakeMeta(episodes=(0, 1))
            constructed = []
            loaded_episodes = []

            class MetadataFactory:
                def __new__(cls, *_args, **_kwargs):
                    return metadata

            class DatasetFactory(_FakeDataset):
                def __init__(self, _repo_id, **kwargs):
                    constructed.append(kwargs)
                    super().__init__("selected", metadata, len(kwargs["episodes"]))

            def resolve(policy, meta):
                self.assertIs(meta, metadata)
                return {"action": [index / meta.fps for index in range(policy.chunk_size)]}

            def load_episode(_dataset, episode):
                loaded_episodes.append(episode)
                return _episode_row(float(episode + 1), successful=episode == 1)

            stats_seen = []

            def aggregate(values):
                stats_seen.extend(values)
                return values[0]

            dependencies = LeRobotDatasetDependencies(
                metadata_cls=MetadataFactory,
                dataset_cls=DatasetFactory,
                resolve_delta_timestamps=resolve,
                aggregate_stats=aggregate,
                load_episode_with_stats=load_episode,
            )
            dataset = load_virtual_act_bc_dataset(
                (RootSelection(root, (1,)),),
                policy_config=SimpleNamespace(chunk_size=30),
                dependencies=dependencies,
            )
            self.assertEqual(dataset.num_episodes, 1)
            self.assertEqual(loaded_episodes, [1])
            self.assertEqual(len(stats_seen), 1)
            self.assertEqual(
                constructed[0]["delta_timestamps"]["action"],
                [index / 15 for index in range(30)],
            )
            self.assertEqual(constructed[0]["episodes"], [1])

    def test_loader_rejects_episode_that_is_not_actually_successful(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "epoch"
            (root / "meta" / "episodes").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "videos").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps({"codebase_version": "v3.0", "features": _features()}),
                encoding="utf-8",
            )
            (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
            metadata = _FakeMeta(episodes=(0,))
            dependencies = LeRobotDatasetDependencies(
                metadata_cls=lambda *_args, **_kwargs: metadata,
                dataset_cls=lambda *_args, **_kwargs: _FakeDataset("failed", metadata, 1),
                resolve_delta_timestamps=lambda *_args: {"action": [0.0]},
                aggregate_stats=lambda values: values[0],
                load_episode_with_stats=lambda *_args: _episode_row(successful=False),
            )
            with self.assertRaisesRegex(ValueError, "not labeled successful"):
                load_virtual_act_bc_dataset(
                    (RootSelection(root, (0,)),),
                    policy_config=SimpleNamespace(chunk_size=30),
                    dependencies=dependencies,
                )

    def test_loader_accepts_unlabeled_imitation_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "unlabeled"
            (root / "meta" / "episodes").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "videos").mkdir()
            features = _features(include_success=False)
            (root / "meta" / "info.json").write_text(
                json.dumps({"codebase_version": "v3.0", "features": features}),
                encoding="utf-8",
            )
            (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
            metadata = _FakeMeta(features=features, episodes=(0, 1))
            loaded_episodes = []

            dependencies = LeRobotDatasetDependencies(
                metadata_cls=lambda *_args, **_kwargs: metadata,
                dataset_cls=lambda *_args, **_kwargs: _FakeDataset("demo", metadata, 2),
                resolve_delta_timestamps=lambda *_args: {"action": [0.0]},
                aggregate_stats=lambda values: values[0],
                load_episode_with_stats=lambda _dataset, episode: (
                    loaded_episodes.append(episode)
                    or _episode_row(float(episode + 1), include_success=False)
                ),
            )

            dataset = load_virtual_act_bc_dataset(
                (RootSelection(root, (0, 1)),),
                policy_config=SimpleNamespace(chunk_size=30),
                dependencies=dependencies,
            )

            self.assertEqual(dataset.num_episodes, 2)
            self.assertEqual(loaded_episodes, [0, 1])
            self.assertNotIn("episode_success", dataset.meta.stats)


if __name__ == "__main__":
    unittest.main()
