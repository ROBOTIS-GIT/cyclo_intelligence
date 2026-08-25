"""Host-testable lifecycle contracts for the live Flow-SDE PPO adapter."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cyclo_brain.algorithm.rl.flow_sde_ppo.batch import FlowSDERollout
from cyclo_brain.algorithm.rl.flow_sde_ppo.live_cli import _configure_zenoh_cache
from cyclo_brain.algorithm.rl.flow_sde_ppo.live_source import (
    ACTION_DIM,
    ACTION_KEYS,
    ACTION_WIDTHS,
    EXECUTION_HORIZON,
    ActionStepReceipt,
    AtomicOutcomeFile,
    CycloFlowSDEEpisodeSource,
    CycloLeRobotObservationSource,
    EpisodeOutcome,
    FlowSDECollectionCancelled,
    SensorMarker,
)
from cyclo_brain.algorithm.rl.flow_sde_ppo.runner import FlowSDEActionDecision


def _marker(generation: int) -> SensorMarker:
    return SensorMarker((("generation", float(generation)),))


def _decision(value: float = 0.0) -> FlowSDEActionDecision:
    chains = torch.zeros(1, 3, EXECUTION_HORIZON, ACTION_DIM)
    chains[:, -1] = value
    rollout = FlowSDERollout(
        chains=chains,
        denoise_indices=torch.zeros(1, dtype=torch.long),
        old_log_probs=torch.zeros(1, EXECUTION_HORIZON, ACTION_DIM),
        action_mask=torch.ones(1, EXECUTION_HORIZON, ACTION_DIM, dtype=torch.bool),
    )
    return FlowSDEActionDecision(
        conditioning=torch.zeros(1, 3),
        rollout=rollout,
        executed_actions=torch.full((1, EXECUTION_HORIZON, ACTION_DIM), value),
        old_values=torch.tensor([0.25]),
    )


class _FakeRunner:
    def __init__(self):
        self.adapter = SimpleNamespace(
            encode_conditioning=lambda batch: batch["conditioning"],
        )
        self.decisions = 0

    def sample_preprocessed_batch(self, _batch):
        self.decisions += 1
        return _decision(float(self.decisions))

    @staticmethod
    def value(_conditioning):
        return torch.tensor([2.5])


class _FakeObservations:
    def __init__(self):
        self.generation = 0
        self.snapshot_calls = 0
        self.observe_baselines = []
        self.closed = False

    def snapshot(self, *, timeout=None):
        self.snapshot_calls += 1
        return _marker(self.generation)

    def observe(self, *, newer_than=None, timeout=None):
        self.observe_baselines.append(newer_than)
        self.generation += 1
        marker = _marker(self.generation)
        if newer_than is not None:
            assert marker.as_dict()["generation"] > newer_than.as_dict()["generation"]
        return {"conditioning": torch.ones(1, 3) * self.generation}, marker

    def close(self):
        self.closed = True


class _FakeActions:
    def __init__(self):
        self.reset_calls = 0
        self.begin_calls = 0
        self.cancel_calls = 0
        self.chunks = []
        self.closed = False

    def reset_environment(self):
        self.reset_calls += 1

    def begin_episode(self):
        self.begin_calls += 1
        return self.begin_calls

    def execute_chunk(self, actions, *, source_seq_id):
        chunk = np.asarray(actions)
        self.chunks.append((chunk.copy(), source_seq_id))
        return tuple(
            ActionStepReceipt(
                session_id=self.begin_calls,
                step_id=index + 1,
                environment_step=index + 1,
                executed_action=tuple(float(v) for v in action),
                simulator_timestamp=1.0,
                duration=0.01,
                received_at=1.0,
                command_max_abs_error=0.125,
            )
            for index, action in enumerate(chunk)
        )

    def cancel_current(self):
        self.cancel_calls += 1

    def close(self):
        self.closed = True


class _QueuedOutcomes:
    def __init__(self, values):
        self.values = list(values)

    def poll(self):
        return self.values.pop(0) if self.values else None


def _outcome(name: str, sequence: int = 1) -> EpisodeOutcome:
    return EpisodeOutcome("job", name, sequence, 1.0)


class LiveFlowSDEEpisodeSourceTest(unittest.TestCase):
    def _source(self, outcomes, *, max_decisions=2):
        observations = _FakeObservations()
        actions = _FakeActions()
        source = CycloFlowSDEEpisodeSource(
            observations=observations,
            actions=actions,
            outcomes=_QueuedOutcomes(outcomes),
            postprocessor=lambda action: action,
            max_chunk_decisions=max_decisions,
            sensor_timeout=1.0,
        )
        return source, observations, actions

    def test_success_resets_then_sends_physical_16_by_22_chunk(self):
        source, observations, actions = self._source([None, _outcome("success")])

        episode = source.collect_episode(_FakeRunner())

        self.assertEqual(ACTION_KEYS, ("arm_left", "arm_right", "head", "lift", "mobile"))
        self.assertEqual(ACTION_WIDTHS, (8, 8, 2, 1, 3))
        self.assertEqual(sum(ACTION_WIDTHS), ACTION_DIM)
        self.assertEqual(actions.reset_calls, 1)
        self.assertEqual(actions.begin_calls, 1)
        self.assertEqual(actions.chunks[0][0].shape, (16, 22))
        self.assertEqual(len(episode.transitions), 1)
        self.assertTrue(episode.transitions[0].terminated)
        self.assertEqual(episode.episode_return, 1.0)
        self.assertEqual(source.last_episode_diagnostics["primitive_steps"], 16)
        self.assertEqual(source.last_episode_diagnostics["command_max_abs_error"], 0.125)
        self.assertEqual(observations.snapshot_calls, 1)
        self.assertIsNotNone(observations.observe_baselines[0])

    def test_fresh_success_before_reset_is_not_silently_discarded(self):
        source, _observations, actions = self._source([_outcome("success")])

        episode = source.collect_episode(_FakeRunner())

        self.assertEqual(len(actions.chunks), 1)
        self.assertEqual(episode.episode_return, 1.0)
        self.assertTrue(episode.transitions[-1].terminated)

    def test_max_decisions_truncates_and_bootstraps_fresh_conditioning(self):
        source, observations, actions = self._source([None, None, None, None, None])

        episode = source.collect_episode(_FakeRunner())

        self.assertEqual(len(actions.chunks), 2)
        self.assertEqual(len(episode.transitions), 2)
        self.assertTrue(episode.transitions[-1].truncated)
        self.assertEqual(episode.bootstrap_value, 2.5)
        # reset barrier + one post-action barrier per chunk
        self.assertEqual(len(observations.observe_baselines), 3)

    def test_cancel_during_post_action_barrier_stops_without_episode(self):
        source, _observations, actions = self._source([None, None, _outcome("cancel")])

        with self.assertRaises(FlowSDECollectionCancelled):
            source.collect_episode(_FakeRunner())

        self.assertGreaterEqual(actions.cancel_calls, 2)


class AtomicOutcomeAndCacheTest(unittest.TestCase):
    def test_sensor_marker_requires_only_three_cameras_and_complete_state(self):
        status = {
            "cameras": {
                "cam_left_head": {"timestamp": 1.0},
                "cam_left_wrist": {"timestamp": 1.0},
                "cam_right_wrist": {"timestamp": 1.0},
                # cam_right_head is deliberately absent.
            },
            "joint_groups": {
                "follower_arm_left": {"timestamp": 1.0},
                "follower_arm_right": {"timestamp": 1.0},
                "follower_head": {"timestamp": 1.0},
                "follower_lift": {"timestamp": 1.0},
            },
            "sensors": {"odom": {"timestamp": 1.0}},
        }
        baseline = CycloLeRobotObservationSource._marker_from_status(status)
        self.assertIsNotNone(baseline)
        newer_status = {
            section: {
                name: {"timestamp": values["timestamp"] + 1.0}
                for name, values in entries.items()
            }
            for section, entries in status.items()
        }
        newer = CycloLeRobotObservationSource._marker_from_status(newer_status)
        self.assertTrue(CycloLeRobotObservationSource._is_newer(newer, baseline))
        newer_status["cameras"]["cam_left_wrist"]["timestamp"] = 1.0
        stale = CycloLeRobotObservationSource._marker_from_status(newer_status)
        self.assertFalse(CycloLeRobotObservationSource._is_newer(stale, baseline))

    def test_atomic_outcome_ignores_stale_sequence_and_reads_new_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "outcome.json"
            path.write_text(
                json.dumps({"job_id": "job", "outcome": "success", "sequence": 3}),
                encoding="utf-8",
            )
            reader = AtomicOutcomeFile(path, job_id="job")
            self.assertIsNone(reader.poll())
            path.write_text(
                json.dumps({"job_id": "job", "outcome": "fail", "sequence": 4}),
                encoding="utf-8",
            )
            self.assertEqual(reader.poll().outcome, "fail")

    def test_uid_readable_zenoh_cache_environment(self):
        original = os.environ.pop("ZENOH_ROS2_SDK_CACHE", None)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                self.assertEqual(_configure_zenoh_cache(temporary), temporary)
                self.assertEqual(os.environ["ZENOH_ROS2_SDK_CACHE"], temporary)
        finally:
            if original is None:
                os.environ.pop("ZENOH_ROS2_SDK_CACHE", None)
            else:
                os.environ["ZENOH_ROS2_SDK_CACHE"] = original


if __name__ == "__main__":
    unittest.main()
