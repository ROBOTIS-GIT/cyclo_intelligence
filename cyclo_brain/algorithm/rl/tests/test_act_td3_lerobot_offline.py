"""Unit tests for the fixed-horizon LeRobot ACT-TD3 adapter."""

from __future__ import annotations

import unittest

import torch

from cyclo_brain.algorithm.rl.act_td3.batch import ACTTD3Batch
from cyclo_brain.algorithm.rl.act_td3.lerobot_offline import (
    ACTTD3LeRobotCollator,
    FixedHorizonLeRobotACTTD3Dataset,
    VirtualCumulativeLeRobotACTTD3Dataset,
)


class _FakeTable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: str) -> list[object]:
        return [row[key] for row in self._rows]


class _FakeLeRobotDataset:
    delta_timestamps = None
    fps = 2
    _return_uint8 = False

    def __init__(
        self,
        episodes: tuple[tuple[int, bool], ...] = ((5, True), (3, False)),
    ) -> None:
        rows: list[dict[str, object]] = []
        for episode_index, (length, successful) in enumerate(episodes):
            for frame_index in range(length):
                rows.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "episode_success": successful,
                        "action": torch.tensor(
                            [
                                100.0 * episode_index + 10.0 * frame_index + 1.0,
                                100.0 * episode_index + 10.0 * frame_index + 2.0,
                            ]
                        ),
                        "observation.state": torch.tensor(
                            [float(episode_index), float(frame_index)]
                        ),
                    }
                )
        self._rows = rows
        self.hf_dataset = _FakeTable(rows)
        self.features = {
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_success": {"dtype": "bool", "shape": [1]},
            "action": {"dtype": "float32", "shape": [2]},
            "observation.state": {"dtype": "float32", "shape": [2]},
        }

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return dict(self._rows[index])


class _OffsetPreprocessor:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        first = next(iter(batch.values()))
        self.batch_sizes.append(int(first.shape[0]))
        result = dict(batch)
        if "observation.state" in result:
            result["observation.state"] = result["observation.state"] + 5.0
        if "action" in result:
            # Deliberately makes raw zero padding non-zero.  The adapter must
            # restore the strict normalized-space zero padding afterwards.
            result["action"] = result["action"] + 10.0
        return result


class FixedHorizonLeRobotACTTD3DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _FakeLeRobotDataset()
        self.dataset = FixedHorizonLeRobotACTTD3Dataset(
            self.source,
            execution_horizon=3,
            observation_keys=("observation.state",),
        )

    def test_builds_non_overlapping_full_and_terminal_partial_blocks(self) -> None:
        self.assertEqual(len(self.dataset), 3)
        self.assertEqual(self.dataset.num_episodes, 2)
        self.assertEqual(self.dataset.num_successes, 1)
        self.assertEqual(self.dataset.num_failures, 1)
        self.assertEqual(
            self.dataset.episode_records,
            ((0, 5, True), (1, 3, False)),
        )
        self.assertEqual(self.dataset.action_dim, 2)
        self.assertEqual(self.dataset.fps, 2.0)

        first = self.dataset[0]
        self.assertEqual((first.episode_index, first.start_frame_index), (0, 0))
        self.assertFalse(first.terminated)
        self.assertTrue(first.episode_success)
        self.assertFalse(first.truncated)
        self.assertTrue(first.next_observation_valid)
        self.assertTrue(first.bootstrap_allowed)
        torch.testing.assert_close(
            first.behavior_action_chunk,
            torch.tensor([[1.0, 2.0], [11.0, 12.0], [21.0, 22.0]]),
        )
        torch.testing.assert_close(
            first.next_observations["observation.state"], torch.tensor([0.0, 3.0])
        )
        torch.testing.assert_close(first.rewards, torch.zeros(3))
        torch.testing.assert_close(first.step_durations_s, torch.full((3,), 0.5))

        final_success = self.dataset[1]
        self.assertEqual(
            (final_success.episode_index, final_success.start_frame_index), (0, 3)
        )
        self.assertTrue(final_success.terminated)
        self.assertTrue(final_success.episode_success)
        self.assertFalse(final_success.truncated)
        self.assertFalse(final_success.next_observation_valid)
        self.assertFalse(final_success.bootstrap_allowed)
        torch.testing.assert_close(
            final_success.behavior_action_chunk,
            torch.tensor([[31.0, 32.0], [41.0, 42.0], [0.0, 0.0]]),
        )
        self.assertTrue(
            torch.equal(final_success.executed_mask, torch.tensor([True, True, False]))
        )
        torch.testing.assert_close(final_success.rewards, torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(
            final_success.step_durations_s, torch.tensor([0.5, 0.5, 0.0])
        )
        torch.testing.assert_close(
            final_success.next_observations["observation.state"], torch.zeros(2)
        )

        full_failure = self.dataset[2]
        self.assertEqual((full_failure.episode_index, full_failure.start_frame_index), (1, 0))
        self.assertTrue(full_failure.terminated)
        self.assertFalse(full_failure.episode_success)
        self.assertFalse(full_failure.next_observation_valid)
        self.assertTrue(torch.equal(full_failure.executed_mask, torch.ones(3, dtype=torch.bool)))
        torch.testing.assert_close(full_failure.rewards, torch.zeros(3))

    def test_checkpoint_collator_builds_strict_actor_ready_batch(self) -> None:
        preprocessor = _OffsetPreprocessor()
        collator = ACTTD3LeRobotCollator(preprocessor)
        batch = collator([self.dataset[index] for index in range(len(self.dataset))])

        self.assertIsInstance(batch, ACTTD3Batch)
        self.assertEqual((batch.batch_size, batch.execution_horizon, batch.action_dim), (3, 3, 2))
        self.assertEqual(preprocessor.batch_sizes, [3, 1])
        torch.testing.assert_close(
            batch.observations["observation.state"],
            torch.tensor([[5.0, 5.0], [5.0, 8.0], [6.0, 5.0]]),
        )
        torch.testing.assert_close(
            batch.next_observations["observation.state"],
            torch.tensor([[5.0, 8.0], [0.0, 0.0], [0.0, 0.0]]),
        )
        torch.testing.assert_close(
            batch.behavior_action_chunks[1],
            torch.tensor([[41.0, 42.0], [51.0, 52.0], [0.0, 0.0]]),
        )
        self.assertTrue(
            torch.equal(batch.terminated, torch.tensor([False, True, True]))
        )
        self.assertTrue(
            torch.equal(batch.episode_success, torch.tensor([True, True, False]))
        )
        self.assertTrue(
            torch.equal(batch.next_observation_valid, torch.tensor([True, False, False]))
        )
        self.assertTrue(
            torch.equal(batch.bootstrap_allowed, torch.tensor([True, False, False]))
        )
        torch.testing.assert_close(
            batch.rewards,
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        )

    def test_h100_boundaries_have_no_off_by_one_drop_or_duplicate(self) -> None:
        for episode_length, expected_lengths in (
            (37, [37]),
            (100, [100]),
            (201, [100, 100, 1]),
        ):
            with self.subTest(episode_length=episode_length):
                dataset = FixedHorizonLeRobotACTTD3Dataset(
                    _FakeLeRobotDataset(((episode_length, True),)),
                    execution_horizon=100,
                    observation_keys=("observation.state",),
                )
                transitions = [dataset[index] for index in range(len(dataset))]
                self.assertEqual(
                    [int(transition.executed_mask.sum()) for transition in transitions],
                    expected_lengths,
                )
                self.assertEqual(
                    [transition.start_frame_index for transition in transitions],
                    list(range(0, episode_length, 100)),
                )
                self.assertEqual(
                    [transition.terminated for transition in transitions],
                    [False] * (len(transitions) - 1) + [True],
                )
                self.assertEqual(
                    [transition.next_observation_valid for transition in transitions],
                    [True] * (len(transitions) - 1) + [False],
                )
                flattened = torch.cat(
                    [
                        transition.behavior_action_chunk[
                            transition.executed_mask, 0
                        ]
                        for transition in transitions
                    ]
                )
                torch.testing.assert_close(
                    flattened,
                    torch.tensor(
                        [10.0 * frame_index + 1.0 for frame_index in range(episode_length)]
                    ),
                )
                flattened_rewards = torch.cat(
                    [transition.rewards[transition.executed_mask] for transition in transitions]
                )
                expected_rewards = torch.zeros(episode_length)
                expected_rewards[-1] = 1.0
                torch.testing.assert_close(flattened_rewards, expected_rewards)

    def test_rejects_windowed_or_incomplete_episode_data(self) -> None:
        windowed = _FakeLeRobotDataset()
        windowed.delta_timestamps = {"action": [0.0]}
        with self.assertRaisesRegex(ValueError, "delta_timestamps"):
            FixedHorizonLeRobotACTTD3Dataset(
                windowed,
                execution_horizon=3,
                observation_keys=("observation.state",),
            )

        uint8_images = _FakeLeRobotDataset()
        uint8_images._return_uint8 = True
        with self.assertRaisesRegex(ValueError, "return_uint8=False"):
            FixedHorizonLeRobotACTTD3Dataset(
                uint8_images,
                execution_horizon=3,
                observation_keys=("observation.state",),
            )

        incomplete = _FakeLeRobotDataset()
        incomplete._rows[0]["frame_index"] = 1
        with self.assertRaisesRegex(ValueError, "starting at frame 0"):
            FixedHorizonLeRobotACTTD3Dataset(
                incomplete,
                execution_horizon=3,
                observation_keys=("observation.state",),
            )


class VirtualCumulativeLeRobotACTTD3DatasetTest(unittest.TestCase):
    @staticmethod
    def _root(
        episodes: tuple[tuple[int, bool], ...],
    ) -> FixedHorizonLeRobotACTTD3Dataset:
        return FixedHorizonLeRobotACTTD3Dataset(
            _FakeLeRobotDataset(episodes),
            execution_horizon=3,
            observation_keys=("observation.state",),
        )

    def test_maps_ordered_roots_without_copying_and_remaps_episode_indices(self) -> None:
        first = self._root(((5, True), (3, False)))
        second = self._root(((4, False), (2, True)))
        replay = VirtualCumulativeLeRobotACTTD3Dataset((first, second))

        self.assertEqual(replay.num_roots, 2)
        self.assertEqual(replay.num_episodes, 4)
        self.assertEqual(replay.num_successes, 2)
        self.assertEqual(replay.num_failures, 2)
        self.assertEqual(replay.root_episode_ranges, ((0, 2), (2, 4)))
        self.assertEqual(
            replay.episode_records,
            ((0, 5, True), (1, 3, False), (2, 4, False), (3, 2, True)),
        )
        self.assertEqual(len(replay), len(first) + len(second))
        self.assertEqual(
            [replay[index].episode_index for index in range(len(replay))],
            [0, 0, 1, 2, 2, 3],
        )
        # Root-local reads remain intact: root 1 episode 0 still exposes its
        # root-local action values even though its logical identity is now 2.
        torch.testing.assert_close(
            replay[len(first)].behavior_action_chunk[0], torch.tensor([1.0, 2.0])
        )

    def test_rejects_incompatible_fps_action_camera_and_feature_schema(self) -> None:
        reference = self._root(((2, True),))

        different_fps_source = _FakeLeRobotDataset(((2, False),))
        different_fps_source.fps = 15
        different_fps = FixedHorizonLeRobotACTTD3Dataset(
            different_fps_source,
            execution_horizon=3,
            observation_keys=("observation.state",),
        )
        with self.assertRaisesRegex(ValueError, "fps"):
            VirtualCumulativeLeRobotACTTD3Dataset((reference, different_fps))

        different_action_source = _FakeLeRobotDataset(((2, False),))
        different_action_source.features["action"] = {
            "dtype": "float32",
            "shape": [3],
        }
        for row in different_action_source._rows:
            row["action"] = torch.tensor([1.0, 2.0, 3.0])
        different_action = FixedHorizonLeRobotACTTD3Dataset(
            different_action_source,
            execution_horizon=3,
            observation_keys=("observation.state",),
        )
        with self.assertRaisesRegex(ValueError, "action dimension"):
            VirtualCumulativeLeRobotACTTD3Dataset((reference, different_action))

        different_schema_source = _FakeLeRobotDataset(((2, False),))
        different_schema_source.features["observation.state"]["names"] = ["x", "y"]
        different_schema = FixedHorizonLeRobotACTTD3Dataset(
            different_schema_source,
            execution_horizon=3,
            observation_keys=("observation.state",),
        )
        with self.assertRaisesRegex(ValueError, "feature schema"):
            VirtualCumulativeLeRobotACTTD3Dataset((reference, different_schema))


if __name__ == "__main__":
    unittest.main()
