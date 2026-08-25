"""Focused tests for the semantic ACT-TD3 training-data identity."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cyclo_brain.algorithm.rl.act_td3.training_identity import (
    build_act_td3_multi_root_training_data_identity,
    build_act_td3_training_data_identity,
)


def _write(path: Path, value: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode() if isinstance(value, str) else value)


class _FakeMetadata:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.total_episodes = 2
        self.video_keys = ["observation.images.camera"]
        self.robot_type = "ffw_sg2_rev1"

    def get_data_file_path(self, episode_index: int) -> Path:
        return Path(f"data/chunk-000/file-{episode_index:03d}.parquet")

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        assert video_key == "observation.images.camera"
        return Path(f"videos/{video_key}/chunk-000/file-{episode_index:03d}.mp4")


class _FakeDataset:
    def __init__(self, root: Path, episodes: list[int] | None = None) -> None:
        self.root = root
        self.meta = _FakeMetadata(root)
        self.episodes = episodes
        self._video_backend = "pyav"


def _build_fixture(root: Path) -> tuple[_FakeDataset, Path, SimpleNamespace]:
    dataset_root = root / "dataset"
    checkpoint_root = root / "checkpoint"
    robot_config = root / "robot" / "ffw.yaml"
    urdf = root / "robot" / "urdf" / "ffw.urdf"

    _write(dataset_root / "meta/info.json", '{"codebase_version":"v3.0"}')
    _write(dataset_root / "meta/tasks.parquet", b"tasks")
    _write(dataset_root / "meta/stats.json", "{}")
    _write(dataset_root / "meta/subtasks.parquet", b"subtasks")
    _write(dataset_root / "meta/episodes/chunk-000/file-000.parquet", b"episodes")
    _write(dataset_root / "meta/frame_reuse.parquet", b"ignored frame reuse")
    _write(dataset_root / "annotations/chunk-000/episode_000000.json", b"ignored annotation")
    for episode_index in range(2):
        _write(
            dataset_root / f"data/chunk-000/file-{episode_index:03d}.parquet",
            f"data {episode_index}",
        )
        _write(
            dataset_root
            / "videos/observation.images.camera/chunk-000"
            / f"file-{episode_index:03d}.mp4",
            f"video {episode_index}",
        )

    _write(checkpoint_root / "config.json", '{"type":"act"}')
    _write(checkpoint_root / "model.safetensors", b"model weights")
    _write(
        checkpoint_root / "policy_preprocessor.json",
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "state_file": "preprocessor.safetensors",
                    }
                ]
            }
        ),
    )
    _write(
        checkpoint_root / "policy_postprocessor.json",
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "state_file": "postprocessor.safetensors",
                    }
                ]
            }
        ),
    )
    _write(checkpoint_root / "preprocessor.safetensors", b"preprocessor state")
    _write(checkpoint_root / "postprocessor.safetensors", b"postprocessor state")
    _write(checkpoint_root / "train_config.json", b"ignored train config")

    _write(robot_config, b"robot config")
    _write(urdf, b"robot urdf")
    domain = SimpleNamespace(
        names=("joint", "linear_x"),
        action_groups=("arm", "mobile"),
        dataset_info_path=dataset_root / "meta/info.json",
        robot_config_path=robot_config,
        urdf_path=urdf,
        physical_low=[-1.0, 0.0],
        physical_high=[1.0, 0.0],
        passthrough_mask=[False, True],
    )
    return _FakeDataset(dataset_root), checkpoint_root, domain


def _identity(dataset: _FakeDataset, checkpoint: Path, domain: SimpleNamespace):
    return build_act_td3_training_data_identity(
        dataset,
        dataset.root,
        checkpoint,
        domain,
        robot_type="ffw_sg2_rev1",
        video_backend="pyav",
    )


class ACTTD3TrainingIdentityTest(unittest.TestCase):
    def test_deterministic_manifest_hashes_only_semantically_consumed_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint, domain = _build_fixture(Path(directory))
            first = _identity(dataset, checkpoint, domain)
            second = _identity(dataset, checkpoint, domain)

            self.assertEqual(first, second)
            self.assertTrue(first.identity.startswith("sha256:"))
            self.assertEqual(first.file_count, 17)
            self.assertEqual(
                set(first.component_sha256),
                {"dataset", "act_checkpoint", "robot", "virtual_contract"},
            )
            json.dumps(first.to_dict(), allow_nan=False)
            manifest_paths = {entry.path for entry in first.manifest}
            self.assertNotIn("meta/frame_reuse.parquet", manifest_paths)
            self.assertNotIn("train_config.json", manifest_paths)
            self.assertFalse(any(path.startswith("annotations/") for path in manifest_paths))

            _write(dataset.root / "meta/frame_reuse.parquet", b"changed but ignored")
            _write(dataset.root / "annotations/chunk-000/episode_000000.json", b"changed")
            _write(checkpoint / "train_config.json", b"changed")
            self.assertEqual(first.identity, _identity(dataset, checkpoint, domain).identity)

            _write(dataset.root / "data/chunk-000/file-000.parquet", b"selected changed")
            self.assertNotEqual(first.identity, _identity(dataset, checkpoint, domain).identity)

    def test_selected_episode_indices_bind_files_and_virtual_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint, domain = _build_fixture(Path(directory))
            all_episodes = _identity(dataset, checkpoint, domain)
            dataset.episodes = [1]
            selected = _identity(dataset, checkpoint, domain)

            self.assertNotEqual(all_episodes.identity, selected.identity)
            self.assertEqual(selected.virtual_contract["episode_indices"], [1])
            selected_paths = {entry.path for entry in selected.manifest}
            self.assertNotIn("data/chunk-000/file-000.parquet", selected_paths)
            self.assertIn("data/chunk-000/file-001.parquet", selected_paths)
            self.assertIn("meta/episodes/chunk-000/file-000.parquet", selected_paths)

    def test_sharded_safetensors_and_processor_state_references_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, checkpoint, domain = _build_fixture(Path(directory))
            (checkpoint / "model.safetensors").unlink()
            _write(checkpoint / "model-00001-of-00002.safetensors", b"shard one")
            _write(checkpoint / "model-00002-of-00002.safetensors", b"shard two")
            _write(
                checkpoint / "model.safetensors.index.json",
                json.dumps(
                    {
                        "weight_map": {
                            "actor.first": "model-00001-of-00002.safetensors",
                            "actor.second": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
            )

            first = _identity(dataset, checkpoint, domain)
            paths = {entry.path for entry in first.manifest}
            self.assertIn("model.safetensors.index.json", paths)
            self.assertIn("model-00001-of-00002.safetensors", paths)
            self.assertIn("model-00002-of-00002.safetensors", paths)
            _write(checkpoint / "model-00002-of-00002.safetensors", b"changed shard")
            self.assertNotEqual(first.identity, _identity(dataset, checkpoint, domain).identity)

    def test_path_escape_and_symbolic_links_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, checkpoint, domain = _build_fixture(root)
            dataset.meta.get_data_file_path = lambda _episode: Path("../outside.parquet")
            with self.assertRaisesRegex(ValueError, "outside the dataset root"):
                _identity(dataset, checkpoint, domain)

            dataset, checkpoint, domain = _build_fixture(root / "second")
            selected = dataset.root / "data/chunk-000/file-000.parquet"
            selected.unlink()
            selected.symlink_to(root / "outside.parquet")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                _identity(dataset, checkpoint, domain)

    def test_multi_root_identity_namespaces_files_and_records_ordered_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, checkpoint, first_domain = _build_fixture(root / "epoch_0000")
            second, _, second_domain = _build_fixture(root / "epoch_0001")
            second.episodes = [1]

            identity = build_act_td3_multi_root_training_data_identity(
                (first, second),
                (first.root, second.root),
                checkpoint,
                (first_domain, second_domain),
                robot_type="ffw_sg2_rev1",
                video_backend="pyav",
            )

            data_roots = identity.virtual_contract["data_roots"]
            self.assertEqual(identity.virtual_contract["episode_indices"], [0, 1, 2])
            self.assertEqual(len(data_roots), 2)
            self.assertEqual(data_roots[0]["root"], str(first.root.resolve()))
            self.assertEqual(data_roots[0]["episode_indices"], [0, 1])
            self.assertEqual(data_roots[0]["global_episode_indices"], [0, 1])
            self.assertEqual(data_roots[1]["episode_indices"], [1])
            self.assertEqual(data_roots[1]["global_episode_indices"], [2])
            self.assertEqual(
                data_roots[0]["identity"],
                _identity(first, checkpoint, first_domain).identity,
            )
            dataset_paths = {
                entry.path for entry in identity.manifest if entry.component == "dataset"
            }
            self.assertTrue(
                any(path.startswith("data_root_0000/") for path in dataset_paths)
            )
            self.assertTrue(
                any(path.startswith("data_root_0001/") for path in dataset_paths)
            )
            self.assertIn("data_root_0000", identity.component_sha256)
            self.assertIn("data_root_0001", identity.component_sha256)
            json.dumps(identity.to_dict(), allow_nan=False)

            reordered = build_act_td3_multi_root_training_data_identity(
                (second, first),
                (second.root, first.root),
                checkpoint,
                (second_domain, first_domain),
                robot_type="ffw_sg2_rev1",
                video_backend="pyav",
            )
            self.assertNotEqual(identity.identity, reordered.identity)

    def test_multi_root_identity_rejects_duplicate_and_incompatible_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, checkpoint, first_domain = _build_fixture(root / "epoch_0000")
            second, _, second_domain = _build_fixture(root / "epoch_0001")
            with self.assertRaisesRegex(ValueError, "must be unique"):
                build_act_td3_multi_root_training_data_identity(
                    (first, first),
                    (first.root, first.root),
                    checkpoint,
                    (first_domain, first_domain),
                    robot_type="ffw_sg2_rev1",
                    video_backend="pyav",
                )

            second_domain.names = ("different", "linear_x")
            with self.assertRaisesRegex(ValueError, "incompatible virtual contract"):
                build_act_td3_multi_root_training_data_identity(
                    (first, second),
                    (first.root, second.root),
                    checkpoint,
                    (first_domain, second_domain),
                    robot_type="ffw_sg2_rev1",
                    video_backend="pyav",
                )


if __name__ == "__main__":
    unittest.main()
