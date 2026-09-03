"""Tests for versioned cumulative-replay ACT-TD3 training rounds."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import torch

from cyclo_brain.algorithm.rl.act_td3 import (
    ACTTD3Learner,
    ACTTD3LeRobotCollator,
    ACTTD3OfflineTrainingProgress,
    ACTTD3OfflineTrainingRunner,
    ACTTD3TrainingDataIdentity,
    ACTTD3UpdateResult,
    FixedHorizonLeRobotACTTD3Dataset,
    VirtualCumulativeLeRobotACTTD3Dataset,
)
from cyclo_brain.algorithm.rl.act_td3.offline_training import (
    load_policy_local_warmup_critic,
    policy_update_period_for_epoch_schedule,
)
from cyclo_brain.algorithm.rl.act_td3.offline_warmup import _module_sha256
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
    *,
    policy_update_period: int = 2,
) -> ACTTD3Learner:
    source = _warmup_learner()
    config = replace(
        source.config,
        critic_warmup_updates=0,
        policy_update_period=policy_update_period,
    )
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
        actor_updated = (
            learner.completed_critic_updates
            % learner.config.policy_update_period
            == 0
        )
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


def _write_policy_warmup_critic(
    actor_root: Path,
    learner: ACTTD3Learner,
    replay: VirtualCumulativeLeRobotACTTD3Dataset,
    identity: ACTTD3TrainingDataIdentity,
) -> tuple[Path, Path]:
    if not learner.critic_optimizer.state:
        _seed_adam_state(learner)
    critic_dir = actor_root / "critic"
    critic_dir.mkdir(parents=True, exist_ok=True)
    latest = critic_dir / "latest.pt"
    actor_sha256 = _module_sha256(learner.actor)
    warm_config = asdict(replace(
        learner.config,
        critic_warmup_updates=5000,
    ))
    learner_contract = {
        "config": warm_config,
        "prediction_horizon": learner.prediction_horizon,
        "execution_horizon": learner.execution_horizon,
        "action_dim": learner.action_dim,
        "observation_keys": tuple(learner.critic.observation_keys),
        "action_domain": learner.ACTION_DOMAIN,
        "target_policy_smoothing": learner.TARGET_POLICY_SMOOTHING,
        "actor_q_gradient": learner.ACTOR_Q_GRADIENT,
        "action_clamp": False,
        "device": str(learner.device),
        "dtype": str(learner.dtype),
    }
    dataset_contract = {
        "transition_count": len(replay),
        "episode_count": replay.num_episodes,
        "success_count": replay.num_successes,
        "failure_count": replay.num_failures,
        "fps": float(replay.fps),
        "execution_horizon": replay.execution_horizon,
        "action_dim": replay.action_dim,
    }
    artifact = {
        "format": "cyclo_brain.act_td3_critic/v1",
        "status": "complete",
        "contract": {
            "training_data_identity": identity.identity,
            "sampling": "uniform_without_replacement_within_batch",
            "sampling_seed": 19,
            "batch_size": 2,
            "dataset": dataset_contract,
            "learner": learner_contract,
        },
        "actor_sha256": actor_sha256,
        "actor_target_sha256": actor_sha256,
        "critic": learner.critic.state_dict(),
        "critic_target": learner.critic_target.state_dict(),
        "critic_optimizer": learner.critic_optimizer.state_dict(),
        "completed_critic_updates": 5000,
        "completed_actor_updates": 0,
    }
    torch.save(artifact, latest)
    checkpoint_bytes = latest.read_bytes()
    roots = identity.virtual_contract["data_roots"]
    manifest = {
        "format": "cyclo_brain.act_td3_critic_manifest/v1",
        "status": "complete",
        "created_at": "2026-08-25T00:00:00+00:00",
        "base_policy": {
            "path": str(actor_root.resolve()),
            "actor_sha256": actor_sha256,
        },
        "artifact": {
            "format": "cyclo_brain.act_td3_critic/v1",
            "checkpoint_path": "latest.pt",
            "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "byte_count": len(checkpoint_bytes),
        },
        "training_data": {
            "identity": identity.identity,
            "dataset_roots": [root["root"] for root in roots],
            "file_count": identity.file_count,
            "byte_count": identity.byte_count,
            "component_sha256": identity.component_sha256,
            "virtual_contract": identity.virtual_contract,
        },
        "dataset": dataset_contract,
        "learner": json.loads(json.dumps(learner_contract)),
        "completed_critic_updates": 5000,
        "completed_actor_updates": 0,
        "actor_exactly_unchanged": True,
    }
    manifest_path = critic_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return latest, manifest_path


def _rewrite_policy_warmup_artifact(
    latest: Path,
    manifest_path: Path,
    artifact: dict[str, object],
    manifest: dict[str, object],
) -> None:
    torch.save(artifact, latest)
    checkpoint_bytes = latest.read_bytes()
    artifact_reference = manifest["artifact"]
    assert isinstance(artifact_reference, dict)
    artifact_reference["sha256"] = hashlib.sha256(checkpoint_bytes).hexdigest()
    artifact_reference["byte_count"] = len(checkpoint_bytes)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _seed_adam_state(learner: ACTTD3Learner) -> None:
    for index, parameter in enumerate(learner.critic.parameters(), start=1):
        learner.critic_optimizer.state[parameter] = {
            "step": torch.tensor(float(index)),
            "exp_avg": torch.full_like(parameter, 0.01 * index),
            "exp_avg_sq": torch.full_like(parameter, 0.001 * index),
        }


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
    batch_size: int = 2,
) -> ACTTD3OfflineTrainingRunner:
    replay = dataset or _dataset()
    policy_update_period = policy_update_period_for_epoch_schedule(
        critic_epochs,
        actor_equivalent_epochs,
    )
    learner = _learner(
        actor_trainable_groups,
        policy_update_period=policy_update_period,
    )
    _install_fast_update(learner)
    return ACTTD3OfflineTrainingRunner(
        learner,
        replay,
        ACTTD3LeRobotCollator(_OffsetPreprocessor()),
        batch_size=batch_size,
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

    def test_rl_metric_history_uses_exact_round_means_and_replay_return(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "round_001.pt"
            first = _runner(checkpoint)
            partial = first.run(max_round_critic_updates=3)

            self.assertEqual(len(partial.rl_metric_history), 1)
            point = partial.rl_metric_history[0]
            self.assertEqual(point.rl_epoch, 1)
            self.assertEqual(point.critic_loss_mean, 2.0)
            self.assertEqual(point.actor_loss_mean, 1.0)
            self.assertEqual(point.replay_average_reward, 0.5)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(
                state["current_round"]["telemetry"],
                {
                    "critic_loss_sum": 6.0,
                    "critic_loss_count": 3,
                    "actor_loss_sum": 1.0,
                    "actor_loss_count": 1,
                },
            )

            resumed = _runner(checkpoint, resume_from=checkpoint)
            complete = resumed.run()
            completed_point = complete.rl_metric_history[0]
            self.assertEqual(completed_point.critic_loss_mean, 10.5)
            self.assertEqual(completed_point.actor_loss_mean, 5.5)

            grown = _dataset(((5, True), (3, False), (4, True)))
            second = _runner(
                Path(directory) / "round_002.pt",
                dataset=grown,
                resume_from=checkpoint,
            )
            running: list[ACTTD3OfflineTrainingProgress] = []
            second.run(
                max_round_critic_updates=1,
                progress_callback=running.append,
            )
            self.assertEqual(len(running[0].rl_metric_history), 2)
            self.assertEqual(running[0].rl_metric_history[0], completed_point)
            self.assertEqual(running[0].rl_metric_history[1].rl_epoch, 2)
            self.assertIsNone(running[0].rl_metric_history[1].critic_loss_mean)
            self.assertIsNone(running[0].rl_metric_history[1].actor_loss_mean)
            self.assertAlmostEqual(
                running[0].rl_metric_history[1].replay_average_reward,
                2.0 / 3.0,
            )

    def test_legacy_checkpoint_without_telemetry_remains_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy_metrics.pt"
            _runner(checkpoint).run(max_round_critic_updates=3)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            del state["current_round"]["telemetry"]
            torch.save(state, checkpoint)

            resumed = _runner(checkpoint, resume_from=checkpoint)
            result = resumed.run(max_round_critic_updates=4)
            self.assertEqual(result.rl_metric_history, ())
            rewritten = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            self.assertNotIn("telemetry", rewritten["current_round"])

    def test_checkpoint_rejects_inexact_round_metric_accumulators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bad_metrics.pt"
            _runner(checkpoint).run(max_round_critic_updates=3)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state["current_round"]["telemetry"]["critic_loss_count"] = 2
            torch.save(state, checkpoint)

            with self.assertRaisesRegex(ValueError, "round telemetry disagrees"):
                _runner(checkpoint, resume_from=checkpoint)

    def test_legacy_v3_round_cannot_resume_changed_actor_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            first = _runner(checkpoint)
            first.run(max_round_critic_updates=7)
            _downgrade_round_checkpoint_to_legacy_v3(checkpoint)

            with self.assertRaisesRegex(ValueError, "objective contract changed"):
                _runner(checkpoint, resume_from=checkpoint)

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

    def test_legacy_v3_completed_parent_cannot_seed_new_objective_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "legacy_parent.pt"
            _runner(parent).run()
            _downgrade_round_checkpoint_to_legacy_v3(parent)
            grown = _dataset(((5, True), (3, False), (4, True)))

            with self.assertRaisesRegex(ValueError, "objective contract changed"):
                _runner(
                    root / "round_002.pt",
                    dataset=grown,
                    resume_from=parent,
                )

    def test_v5_round_does_not_normalize_missing_group_contract(self) -> None:
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

    def test_schedule_requires_an_exact_integer_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exact integer multiple"):
                _runner(
                    Path(directory) / "bad.pt",
                    critic_epochs=5,
                    actor_equivalent_epochs=3,
                )

    def test_equal_epoch_schedule_updates_actor_and_critic_one_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = _runner(
                Path(directory) / "one_to_one.pt",
                critic_epochs=1,
                actor_equivalent_epochs=1,
                batch_size=2,
            )
            self.assertEqual(runner.policy_update_period, 1)
            result = runner.run()
            self.assertEqual(
                result.completed_critic_updates,
                result.completed_actor_updates,
            )
            self.assertEqual(
                result.total_critic_updates,
                result.total_actor_updates,
            )

    def test_initial_replay_may_exceed_fifty_but_growth_stays_capped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial_episodes = tuple(
                (1, bool(index % 2)) for index in range(59)
            )
            first_path = root / "round_001.pt"
            first = _runner(
                first_path,
                dataset=_dataset(initial_episodes),
                critic_epochs=2,
                actor_equivalent_epochs=1,
                batch_size=59,
            )
            self.assertEqual(first.new_episode_count, 59)
            first.run()

            one_more = _runner(
                root / "round_002_one_more.pt",
                dataset=_dataset((*initial_episodes, (1, True))),
                resume_from=first_path,
                critic_epochs=2,
                actor_equivalent_epochs=1,
                batch_size=59,
            )
            self.assertEqual(one_more.round_index, 2)
            self.assertEqual(one_more.new_episode_count, 1)

    def test_later_round_accepts_one_through_fifty_new_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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

    def test_policy_warmup_loads_only_critic_and_accepts_appended_roots(self) -> None:
        warm_replay = _virtual_dataset((((5, True), (3, False)),))
        warm_identity = _multi_identity(
            warm_replay,
            root_names=("epoch_0000",),
        )
        grown_replay = _virtual_dataset(
            (((5, True), (3, False)), ((4, True),))
        )
        grown_identity = _multi_identity(
            grown_replay,
            root_names=("epoch_0000", "epoch_0001"),
        )
        source = _learner()
        _seed_adam_state(source)
        with torch.no_grad():
            for parameter in source.critic.parameters():
                parameter.fill_(0.125)
            for parameter in source.critic_target.parameters():
                parameter.fill_(-0.25)

        target = _learner(
            actor_trainable_groups=("action_decoder",),
            policy_update_period=1,
        )
        actor_before = {
            name: value.detach().clone()
            for name, value in target.actor.state_dict().items()
        }
        actor_target_before = {
            name: value.detach().clone()
            for name, value in target.actor_target.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            latest, manifest_path = _write_policy_warmup_critic(
                actor_root,
                source,
                warm_replay,
                warm_identity,
            )
            # The directory may be copied/moved, and a critic warmed on one
            # device may be consumed on another.  Path/device are provenance;
            # actor/content hashes and tensor contracts remain authoritative.
            artifact = torch.load(latest, map_location="cpu", weights_only=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact["contract"]["learner"]["device"] = "cuda:99"
            manifest["learner"]["device"] = "cuda:99"
            # Critic-only artifacts predate selectable actor objectives. Their
            # weights remain reusable because no actor objective ran in warm-up.
            del artifact["contract"]["learner"]["config"]["actor_objective"]
            del manifest["learner"]["config"]["actor_objective"]
            manifest["base_policy"]["path"] = "/original/copied-policy-location"
            _rewrite_policy_warmup_artifact(
                latest,
                manifest_path,
                artifact,
                manifest,
            )
            loaded = load_policy_local_warmup_critic(
                target,
                grown_replay,
                grown_identity,
                act_checkpoint=actor_root,
            )

        self.assertEqual(loaded, latest)
        _assert_tree_equal(self, source.critic.state_dict(), target.critic.state_dict())
        _assert_tree_equal(
            self,
            source.critic_target.state_dict(),
            target.critic_target.state_dict(),
        )
        _assert_tree_equal(
            self,
            source.critic_optimizer.state_dict(),
            target.critic_optimizer.state_dict(),
        )
        _assert_tree_equal(self, actor_before, target.actor.state_dict())
        _assert_tree_equal(self, actor_target_before, target.actor_target.state_dict())
        self.assertEqual(target.completed_critic_updates, 0)
        self.assertEqual(target.completed_actor_updates, 0)

    def test_policy_warmup_missing_pair_is_random_but_partial_pair_fails(self) -> None:
        replay = _virtual_dataset((((5, True), (3, False)),))
        identity = _multi_identity(replay, root_names=("epoch_0000",))
        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            runs = actor_root / "critic" / "runs"
            runs.mkdir(parents=True)
            (runs / "stopped.pt").write_bytes(b"uncommitted-run")
            self.assertIsNone(load_policy_local_warmup_critic(
                _learner(),
                replay,
                identity,
                act_checkpoint=actor_root,
            ))
            critic_dir = actor_root / "critic"
            (critic_dir / "latest.pt").write_bytes(b"unfinished")
            with self.assertRaisesRegex(ValueError, "pair is incomplete"):
                load_policy_local_warmup_critic(
                    _learner(),
                    replay,
                    identity,
                    act_checkpoint=actor_root,
                )

        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            blocked = actor_root / "critic" / "latest.pt"
            real_lstat = os.lstat

            def denied(path):
                if Path(path) == blocked:
                    raise PermissionError("denied")
                return real_lstat(path)

            with mock.patch.object(os, "lstat", side_effect=denied):
                with self.assertRaisesRegex(ValueError, "cannot be inspected safely"):
                    load_policy_local_warmup_critic(
                        _learner(),
                        replay,
                        identity,
                        act_checkpoint=actor_root,
                    )

        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            critic_dir = actor_root / "critic"
            critic_dir.mkdir(parents=True)
            (critic_dir / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pair is incomplete"):
                load_policy_local_warmup_critic(
                    _learner(),
                    replay,
                    identity,
                    act_checkpoint=actor_root,
                )

    def test_policy_warmup_rejects_sha_actor_and_replay_mismatches(self) -> None:
        replay = _virtual_dataset((((5, True), (3, False)),))
        identity = _multi_identity(replay, root_names=("epoch_0000",))

        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            _latest, manifest_path = _write_policy_warmup_critic(
                actor_root,
                _learner(),
                replay,
                identity,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_policy_local_warmup_critic(
                    _learner(), replay, identity, act_checkpoint=actor_root
                )

        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            latest, manifest_path = _write_policy_warmup_critic(
                actor_root,
                _learner(),
                replay,
                identity,
            )
            artifact = torch.load(latest, map_location="cpu", weights_only=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact["critic_optimizer"]["param_groups"][0]["lr"] = 1.0
            _rewrite_policy_warmup_artifact(
                latest,
                manifest_path,
                artifact,
                manifest,
            )
            with self.assertRaisesRegex(ValueError, "parameter groups disagree"):
                load_policy_local_warmup_critic(
                    _learner(), replay, identity, act_checkpoint=actor_root
                )

        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            latest, manifest_path = _write_policy_warmup_critic(
                actor_root,
                _learner(),
                replay,
                identity,
            )
            artifact = torch.load(latest, map_location="cpu", weights_only=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact["contract"]["dataset"]["transition_count"] -= 1
            manifest["dataset"]["transition_count"] -= 1
            _rewrite_policy_warmup_artifact(
                latest,
                manifest_path,
                artifact,
                manifest,
            )
            with self.assertRaisesRegex(ValueError, "replay tensor contract"):
                load_policy_local_warmup_critic(
                    _learner(), replay, identity, act_checkpoint=actor_root
                )

        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            _latest, manifest_path = _write_policy_warmup_critic(
                actor_root,
                _learner(),
                replay,
                identity,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["base_policy"]["actor_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "base actor identity"):
                load_policy_local_warmup_critic(
                    _learner(), replay, identity, act_checkpoint=actor_root
                )

        unrelated = _virtual_dataset((((5, True), (3, False)),))
        unrelated_identity = _multi_identity(
            unrelated,
            root_names=("different_epoch",),
        )
        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            _write_policy_warmup_critic(
                actor_root,
                _learner(),
                replay,
                identity,
            )
            with self.assertRaisesRegex(ValueError, "immutable prefix"):
                load_policy_local_warmup_critic(
                    _learner(),
                    unrelated,
                    unrelated_identity,
                    act_checkpoint=actor_root,
                )

        two_root_replay = _virtual_dataset(
            (((5, True), (3, False)), ((4, True), (2, False)))
        )
        ordered_identity = _multi_identity(
            two_root_replay,
            root_names=("epoch_0000", "epoch_0001"),
        )
        reordered_identity = _multi_identity(
            two_root_replay,
            root_names=("epoch_0001", "epoch_0000"),
        )
        with tempfile.TemporaryDirectory() as directory:
            actor_root = Path(directory) / "actor"
            actor_root.mkdir()
            _write_policy_warmup_critic(
                actor_root,
                _learner(),
                two_root_replay,
                ordered_identity,
            )
            with self.assertRaisesRegex(ValueError, "immutable prefix"):
                load_policy_local_warmup_critic(
                    _learner(),
                    two_root_replay,
                    reordered_identity,
                    act_checkpoint=actor_root,
                )


if __name__ == "__main__":
    unittest.main()
