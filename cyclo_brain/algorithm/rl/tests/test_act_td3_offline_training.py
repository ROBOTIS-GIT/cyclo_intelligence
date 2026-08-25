"""Tests for versioned cumulative-replay ACT-TD3 training rounds."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from cyclo_brain.algorithm.rl.act_td3 import (
    ACTTD3Learner,
    ACTTD3LeRobotCollator,
    ACTTD3OfflineTrainingRunner,
    ACTTD3TrainingDataIdentity,
    ACTTD3UpdateResult,
    FixedHorizonLeRobotACTTD3Dataset,
    VirtualCumulativeLeRobotACTTD3Dataset,
)
from cyclo_brain.algorithm.rl.tests.test_act_td3_lerobot_offline import (
    _FakeLeRobotDataset,
    _OffsetPreprocessor,
)
from cyclo_brain.algorithm.rl.tests.test_act_td3_offline_warmup import (
    _assert_tree_equal,
    _learner as _warmup_learner,
)
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE


def _learner(
    actor_trainable_groups: tuple[str, ...] | None = None,
) -> ACTTD3Learner:
    source = _warmup_learner()
    config = replace(source.config, critic_warmup_updates=0)
    if actor_trainable_groups is not None:
        config = replace(
            config,
            actor_trainable_groups=actor_trainable_groups,
        )
    return ACTTD3Learner(
        source.actor,
        source.critic,
        config,
        random_seed=17,
    )


def _dataset(
    episodes: tuple[tuple[int, bool], ...] = ((5, True), (3, False)),
) -> FixedHorizonLeRobotACTTD3Dataset:
    source = _FakeLeRobotDataset(episodes)
    source.features[OBS_ENV_STATE] = {"dtype": "float32", "shape": [2]}
    for row in source._rows:
        row[OBS_ENV_STATE] = row[OBS_STATE].clone()
    return FixedHorizonLeRobotACTTD3Dataset(
        source,
        execution_horizon=3,
        observation_keys=(OBS_STATE, OBS_ENV_STATE),
    )


def _identity(dataset: FixedHorizonLeRobotACTTD3Dataset) -> ACTTD3TrainingDataIdentity:
    episode_indices = [record[0] for record in dataset.episode_records]
    suffix = f"episodes-{len(episode_indices)}"
    return ACTTD3TrainingDataIdentity(
        identity=f"sha256:{suffix}",
        file_count=3,
        byte_count=len(dataset),
        component_sha256={
            "dataset": f"sha256:dataset-{suffix}",
            "act_checkpoint": "sha256:fixed-actor",
            "robot": "sha256:fixed-robot",
            "virtual_contract": f"sha256:virtual-{suffix}",
        },
        manifest=(),
        virtual_contract={
            "episode_indices": episode_indices,
            "robot_type": "ffw_sg2_rev1",
            "video_backend": "pyav",
            "video_keys": ["observation.images.camera"],
        },
    )


def _virtual_dataset(
    roots: tuple[tuple[tuple[int, bool], ...], ...],
) -> VirtualCumulativeLeRobotACTTD3Dataset:
    return VirtualCumulativeLeRobotACTTD3Dataset(tuple(_dataset(root) for root in roots))


def _multi_identity(
    dataset: VirtualCumulativeLeRobotACTTD3Dataset,
    *,
    root_names: tuple[str, ...],
    first_legacy_identity: ACTTD3TrainingDataIdentity | None = None,
) -> ACTTD3TrainingDataIdentity:
    if len(root_names) != dataset.num_roots:
        raise AssertionError("root_names disagree")
    data_roots = []
    for ordinal, (name, episode_range) in enumerate(
        zip(root_names, dataset.root_episode_ranges, strict=True)
    ):
        start, stop = episode_range
        if ordinal == 0 and first_legacy_identity is not None:
            root_identity = first_legacy_identity.identity
            dataset_sha256 = first_legacy_identity.component_sha256["dataset"]
        else:
            root_identity = f"sha256:identity-{name}"
            dataset_sha256 = f"sha256:dataset-{name}"
        data_roots.append(
            {
                "ordinal": ordinal,
                "root": f"/dataset/{name}",
                "name": name,
                "identity": root_identity,
                "dataset_sha256": dataset_sha256,
                "episode_indices": list(range(stop - start)),
                "global_episode_indices": list(range(start, stop)),
                "file_count": 3,
                "byte_count": stop - start,
            }
        )
    suffix = "-".join(root_names)
    return ACTTD3TrainingDataIdentity(
        identity=f"sha256:multi-{suffix}",
        file_count=3 * len(data_roots),
        byte_count=len(dataset),
        component_sha256={
            "dataset": f"sha256:multi-dataset-{suffix}",
            "act_checkpoint": "sha256:fixed-actor",
            "robot": "sha256:fixed-robot",
            "virtual_contract": f"sha256:multi-virtual-{suffix}",
            **{
                f"data_root_{index:04d}": root["dataset_sha256"]
                for index, root in enumerate(data_roots)
            },
        },
        manifest=(),
        virtual_contract={
            "episode_indices": list(range(dataset.num_episodes)),
            "robot_type": "ffw_sg2_rev1",
            "video_backend": "pyav",
            "video_keys": ["observation.images.camera"],
            "data_roots": data_roots,
        },
    )


def _install_fast_update(learner: ACTTD3Learner) -> None:
    def update(_batch) -> ACTTD3UpdateResult:
        learner.completed_critic_updates += 1
        actor_updated = learner.completed_critic_updates % 2 == 0
        if actor_updated:
            learner.completed_actor_updates += 1
        return ACTTD3UpdateResult(
            critic_loss=float(learner.completed_critic_updates),
            target_mean=0.5,
            actor_updated=actor_updated,
            actor_loss=(float(learner.completed_actor_updates) if actor_updated else None),
            cvae_bc_loss=(1.0 if actor_updated else None),
            deterministic_bc_loss=(1.0 if actor_updated else None),
            actor_q_loss=(0.0 if actor_updated else None),
            actor_q_weight=(0.1 if actor_updated else None),
            actor_q_full_row_count=(1 if actor_updated else None),
            completed_critic_updates=learner.completed_critic_updates,
            completed_actor_updates=learner.completed_actor_updates,
            target_critic_updated=actor_updated,
        )

    learner.update = update  # type: ignore[method-assign]


def _runner(
    checkpoint: Path,
    *,
    dataset: (
        FixedHorizonLeRobotACTTD3Dataset
        | VirtualCumulativeLeRobotACTTD3Dataset
        | None
    ) = None,
    identity: ACTTD3TrainingDataIdentity | None = None,
    resume_from: Path | None = None,
    critic_epochs: int = 10,
    actor_equivalent_epochs: int = 5,
    actor_trainable_groups: tuple[str, ...] | None = None,
) -> ACTTD3OfflineTrainingRunner:
    replay = dataset or _dataset()
    learner = _learner(actor_trainable_groups)
    _install_fast_update(learner)
    return ACTTD3OfflineTrainingRunner(
        learner,
        replay,
        ACTTD3LeRobotCollator(_OffsetPreprocessor()),
        batch_size=2,
        sampling_seed=19,
        training_data_identity=identity or _identity(replay),
        checkpoint_path=checkpoint,
        resume_from=resume_from,
        critic_epochs=critic_epochs,
        actor_equivalent_epochs=actor_equivalent_epochs,
        checkpoint_interval=3,
        progress_interval=1,
    )


def _downgrade_round_checkpoint_to_legacy_v3(
    checkpoint: Path,
) -> dict[str, object]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    learner_state = state["learner"]
    learner_state["format"] = ACTTD3Learner.LEGACY_ALL_TRAINABLE_STATE_FORMAT
    del learner_state["contract"]["actor_trainable_groups"]
    del learner_state["config"]["actor_trainable_groups"]
    del state["base_contract"]["learner"]["config"][
        "actor_trainable_groups"
    ]
    torch.save(state, checkpoint)
    return state


class ACTTD3OfflineTrainingRunnerTest(unittest.TestCase):
    def test_ten_exact_replay_epochs_interleave_five_actor_equivalents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = _runner(Path(directory) / "round_001_ep0002.pt")
            sampled: list[tuple[int, ...]] = []
            original_update = runner.learner.update

            def recorded_update(batch):
                sampled.append(runner.last_sampled_indices)
                return original_update(batch)

            runner.learner.update = recorded_update  # type: ignore[method-assign]

            result = runner.run()

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.completed_epochs, 10)
            self.assertEqual(result.completed_critic_updates, 20)
            self.assertEqual(result.completed_actor_updates, 10)
            self.assertEqual(result.percentage, 100.0)
            self.assertEqual(result.durable_critic_updates, 20)
            self.assertTrue(Path(result.checkpoint_path).is_file())
            self.assertEqual([len(indices) for indices in sampled], [2, 1] * 10)
            for start in range(0, len(sampled), 2):
                self.assertEqual(
                    sorted((*sampled[start], *sampled[start + 1])),
                    [0, 1, 2],
                )
            state = torch.load(
                runner.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            learner_contract = state["base_contract"]["learner"]
            self.assertNotIn("passthrough_mask", learner_contract)
            self.assertEqual(
                learner_contract["action_domain"],
                "saved_act_preprocessor_mean_std_normalized",
            )
            self.assertIs(learner_contract["action_clamp"], False)

    def test_partial_epoch_resume_matches_continuous_state_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous = _runner(root / "continuous.pt")
            continuous.run()

            checkpoint = root / "split.pt"
            first = _runner(checkpoint)
            partial = first.run(max_round_critic_updates=7)
            self.assertEqual(partial.status, "segment_complete")
            self.assertEqual(partial.completed_epochs, 3)
            self.assertEqual(partial.completed_critic_updates, 7)
            self.assertEqual(partial.completed_actor_updates, 3)

            resumed = _runner(checkpoint, resume_from=checkpoint)
            resumed.run()

            _assert_tree_equal(
                self,
                continuous.learner.state_dict(),
                resumed.learner.state_dict(),
            )
            continuous_state = torch.load(
                root / "continuous.pt", map_location="cpu", weights_only=True
            )
            resumed_state = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            torch.testing.assert_close(
                continuous_state["sampler_state"],
                resumed_state["sampler_state"],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                continuous_state["last_sampled_indices"],
                resumed_state["last_sampled_indices"],
            )

    def test_legacy_v3_round_resumes_with_default_all_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            first = _runner(checkpoint)
            first.run(max_round_critic_updates=7)
            legacy = _downgrade_round_checkpoint_to_legacy_v3(checkpoint)

            resumed = _runner(checkpoint, resume_from=checkpoint)

            self.assertEqual(resumed.learner.completed_critic_updates, 7)
            self.assertEqual(resumed.learner.completed_actor_updates, 3)
            self.assertEqual(resumed._completed_epochs, 3)  # noqa: SLF001
            self.assertNotIn(
                "actor_trainable_groups",
                legacy["base_contract"]["learner"]["config"],
            )
            result = resumed.run(max_round_critic_updates=8)
            self.assertEqual(result.completed_critic_updates, 8)

    def test_legacy_v3_round_rejects_partial_freeze_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            _runner(checkpoint).run(max_round_critic_updates=1)
            _downgrade_round_checkpoint_to_legacy_v3(checkpoint)

            with self.assertRaisesRegex(
                ValueError,
                "only with all ACT actor trainable groups",
            ):
                _runner(
                    checkpoint,
                    resume_from=checkpoint,
                    actor_trainable_groups=("action_decoder",),
                )

    def test_legacy_v3_completed_parent_seeds_grown_replay_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "legacy_parent.pt"
            _runner(parent).run()
            _downgrade_round_checkpoint_to_legacy_v3(parent)
            grown = _dataset(((5, True), (3, False), (4, True)))

            child = _runner(
                root / "round_002.pt",
                dataset=grown,
                resume_from=parent,
            )

            self.assertEqual(child.round_index, 2)
            self.assertEqual(child.new_episode_count, 1)
            self.assertEqual(child.learner.completed_critic_updates, 20)
            self.assertEqual(child.learner.completed_actor_updates, 10)

    def test_v4_round_does_not_normalize_missing_group_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v4.pt"
            _runner(checkpoint).run(max_round_critic_updates=1)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            del state["base_contract"]["learner"]["config"][
                "actor_trainable_groups"
            ]
            torch.save(state, checkpoint)

            with self.assertRaisesRegex(ValueError, "base contract disagrees"):
                _runner(checkpoint, resume_from=checkpoint)

    def test_completed_checkpoint_starts_new_version_on_grown_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "round_001_ep0002.pt"
            first = _runner(first_path)
            first.run()

            grown = _dataset(((5, True), (3, False), (4, True)))
            second_path = root / "round_002_ep0003.pt"
            second = _runner(
                second_path,
                dataset=grown,
                resume_from=first_path,
            )

            self.assertEqual(second.round_index, 2)
            self.assertEqual(second.learner.completed_critic_updates, 20)
            self.assertEqual(second.learner.completed_actor_updates, 10)
            result = second.run()
            self.assertEqual(result.completed_critic_updates, 30)
            self.assertEqual(result.completed_actor_updates, 15)
            self.assertEqual(second.learner.completed_critic_updates, 50)
            self.assertEqual(second.learner.completed_actor_updates, 25)
            state = torch.load(second_path, map_location="cpu", weights_only=True)
            self.assertEqual(len(state["history"]), 1)
            self.assertEqual(state["current_round"]["round_index"], 2)
            self.assertEqual(state["current_round"]["new_episode_count"], 1)

    def test_custom_schedule_is_exact_and_may_change_after_completed_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "round_001.pt"
            first = _runner(
                checkpoint,
                critic_epochs=6,
                actor_equivalent_epochs=3,
            )
            partial = first.run(max_round_critic_updates=5)
            self.assertEqual(partial.completed_critic_updates, 5)
            self.assertEqual(partial.completed_actor_updates, 2)

            with self.assertRaisesRegex(ValueError, "same schedule"):
                _runner(checkpoint, resume_from=checkpoint)

            resumed = _runner(
                checkpoint,
                resume_from=checkpoint,
                critic_epochs=6,
                actor_equivalent_epochs=3,
            )
            result = resumed.run()
            self.assertEqual(result.completed_epochs, 6)
            self.assertEqual(result.completed_critic_updates, 12)
            self.assertEqual(result.completed_actor_updates, 6)

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(state["base_contract"]["critic_epochs"], 6)
            self.assertEqual(state["base_contract"]["actor_equivalent_epochs"], 3)
            grown = _dataset(((5, True), (3, False), (4, True)))
            second_path = root / "round_002.pt"
            second = _runner(
                second_path,
                dataset=grown,
                resume_from=checkpoint,
            )
            self.assertEqual(second.round_index, 2)
            second.run()
            second_state = torch.load(
                second_path, map_location="cpu", weights_only=True
            )
            self.assertEqual(
                second_state["history"][-1]["schedule"],
                {"critic_epochs": 6, "actor_equivalent_epochs": 3},
            )
            self.assertEqual(
                second_state["current_round"]["schedule"],
                {"critic_epochs": 10, "actor_equivalent_epochs": 5},
            )

    def test_schedule_requires_exact_fixed_two_to_one_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "critic_epochs must equal"):
                _runner(
                    Path(directory) / "bad.pt",
                    critic_epochs=6,
                    actor_equivalent_epochs=2,
                )

    def test_each_round_accepts_one_through_fifty_new_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "first.*1..50"):
                _runner(
                    root / "too_many_first.pt",
                    dataset=_dataset(tuple((1, False) for _ in range(51))),
                )

            first_path = root / "round_001.pt"
            _runner(first_path).run()
            prefix = ((5, True), (3, False))
            fifty_more = tuple((1, bool(index % 2)) for index in range(50))
            second = _runner(
                root / "round_002.pt",
                dataset=_dataset((*prefix, *fifty_more)),
                resume_from=first_path,
            )
            self.assertEqual(second.round_index, 2)
            self.assertEqual(second.new_episode_count, 50)

            fifty_one_more = tuple((1, bool(index % 2)) for index in range(51))
            with self.assertRaisesRegex(ValueError, "add 1..50"):
                _runner(
                    root / "round_002_too_many.pt",
                    dataset=_dataset((*prefix, *fifty_one_more)),
                    resume_from=first_path,
                )

    def test_growth_requires_completed_prior_round_and_new_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "round_001.pt"
            first = _runner(first_path)
            first.run(max_round_critic_updates=1)
            grown = _dataset(((5, True), (3, False), (4, True)))

            with self.assertRaisesRegex(ValueError, "before a round completes"):
                _runner(root / "round_002.pt", dataset=grown, resume_from=first_path)

    def test_multi_root_child_preserves_ordered_prefix_and_adds_new_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_replay = _virtual_dataset((((5, True), (3, False)),))
            first_identity = _multi_identity(first_replay, root_names=("epoch_0000",))
            first_path = root / "round_001.pt"
            _runner(
                first_path,
                dataset=first_replay,
                identity=first_identity,
            ).run()

            grown = _virtual_dataset(
                (((5, True), (3, False)), ((4, True), (2, False)))
            )
            grown_identity = _multi_identity(
                grown, root_names=("epoch_0000", "epoch_0001")
            )
            child = _runner(
                root / "round_002.pt",
                dataset=grown,
                identity=grown_identity,
                resume_from=first_path,
            )
            self.assertEqual(child.round_index, 2)
            self.assertEqual(child.new_episode_count, 2)
            state = child._checkpoint_state(0.0)  # noqa: SLF001
            recorded_roots = state["current_round"]["dataset"]["training_data"][
                "virtual_contract"
            ]["data_roots"]
            self.assertEqual(
                [entry["name"] for entry in recorded_roots],
                ["epoch_0000", "epoch_0001"],
            )

    def test_multi_root_growth_rejects_reordered_or_modified_prior_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_replay = _virtual_dataset((((5, True), (3, False)),))
            first_identity = _multi_identity(first_replay, root_names=("epoch_0000",))
            first_path = root / "round_001.pt"
            _runner(
                first_path,
                dataset=first_replay,
                identity=first_identity,
            ).run()

            grown = _virtual_dataset(
                (((5, True), (3, False)), ((4, True),))
            )
            changed_prefix = _multi_identity(
                grown, root_names=("epoch_0000_changed", "epoch_0001")
            )
            with self.assertRaisesRegex(ValueError, "ordered data-root prefix"):
                _runner(
                    root / "changed.pt",
                    dataset=grown,
                    identity=changed_prefix,
                    resume_from=first_path,
                )

    def test_legacy_single_root_checkpoint_resumes_with_multi_root_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_replay = _dataset()
            legacy_identity = _identity(legacy_replay)
            checkpoint = root / "legacy_interrupted.pt"
            _runner(
                checkpoint,
                dataset=legacy_replay,
                identity=legacy_identity,
            ).run(max_round_critic_updates=3)

            upgraded_replay = _virtual_dataset((((5, True), (3, False)),))
            upgraded_identity = _multi_identity(
                upgraded_replay,
                root_names=("epoch_0000",),
                first_legacy_identity=legacy_identity,
            )
            resumed = _runner(
                checkpoint,
                dataset=upgraded_replay,
                identity=upgraded_identity,
                resume_from=checkpoint,
            )
            self.assertEqual(resumed.round_index, 1)
            self.assertEqual(resumed._completed_critic_updates(), 3)  # noqa: SLF001

            resumed.run()
            grown = _virtual_dataset(
                (((5, True), (3, False)), ((4, True),))
            )
            grown_identity = _multi_identity(
                grown,
                root_names=("epoch_0000", "epoch_0001"),
                first_legacy_identity=legacy_identity,
            )
            child = _runner(
                root / "round_002.pt",
                dataset=grown,
                identity=grown_identity,
                resume_from=checkpoint,
            )
            self.assertEqual(child.new_episode_count, 1)

    def test_resume_rejects_corrupt_cursor_and_round_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "round_001.pt"
            _runner(checkpoint).run(max_round_critic_updates=7)
            original = torch.load(checkpoint, map_location="cpu", weights_only=True)

            corrupt_cursor = dict(original)
            corrupt_cursor["cursor"] = 1
            torch.save(corrupt_cursor, checkpoint)
            with self.assertRaisesRegex(ValueError, "replay permutation"):
                _runner(checkpoint, resume_from=checkpoint)

            corrupt_counter = dict(original)
            corrupt_counter["round_start_critic_updates"] = 1
            torch.save(corrupt_counter, checkpoint)
            with self.assertRaisesRegex(ValueError, "round counters"):
                _runner(checkpoint, resume_from=checkpoint)

    def test_rejects_warmup_configuration_and_more_than_200_episodes(self) -> None:
        replay = _dataset()
        learner = _warmup_learner()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "critic_warmup_updates=0"):
                ACTTD3OfflineTrainingRunner(
                    learner,
                    replay,
                    ACTTD3LeRobotCollator(_OffsetPreprocessor()),
                    batch_size=2,
                    sampling_seed=19,
                    training_data_identity=_identity(replay),
                    checkpoint_path=Path(directory) / "bad.pt",
                )

        too_many = _dataset(tuple((1, False) for _ in range(201)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "1..200 episodes"):
                _runner(Path(directory) / "too_many.pt", dataset=too_many)


if __name__ == "__main__":
    unittest.main()
