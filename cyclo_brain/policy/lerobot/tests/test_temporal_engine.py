"""Real Worker/Engine/RobotClient callbacks; only weights and transport are fake."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from test_fastwam_runtime import engine_module

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sdk/robot_client/tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sdk/action_chunk_processing"))
from test_initial_pose_sync import RobotClient, robot_client_impl

from engine_process.protocol import CMD_GET_ACTION, CMD_LOAD_POLICY, CMD_UNLOAD_POLICY, CMD_UPDATE_CONTEXT, EngineCommandRequest
from engine_process.worker import EngineWorker
from inference_context.contract import LoadedExecution
from inference_context.execution import ActionRecord, ResetRecord
from lerobot_engine.input_pipeline import InputPipelineConfig


@pytest.mark.parametrize("pause_generation", [0, 1])
def test_temporal_model_registration_load_capture_predict_reset_and_cached_reload(pause_generation):
    definition_class = engine_module.resolve_adapter("act").__class__
    plan_calls = []

    def plan():
        plan_calls.append(config)
        # Explicit input recipe, independent of model config.
        return InputPipelineConfig({
            "sources": {
                "history": {"source": "joint:follower_arm", "offsets_s": [-.1, 0.], "max_age_s": .09},
                "image": {"source": "camera:eye", "max_age_s": .5},
            },
            "nodes": {
                "stacked": {"op": "stack", "inputs": ["history"], "options": {"sequence": True}},
                "batched": {"op": "unsqueeze", "inputs": ["stacked"], "options": {"axis": 0}},
                "temporal": {"op": "to_tensor", "inputs": ["batched"],
                             "options": {"dtype": "float32", "device": "cpu"}},
            },
            "outputs": {"before": {"temporal": "temporal", "image": "image"},
                        "after": {"*": "processed"}},
        }, {}, Path("temporal_test.yaml")).compile(engine)

    config = SimpleNamespace(type="temporal_test", n_obs_steps=2, sample_period_s=.1)
    calls, resets, robots = [], [], []

    def predict(batch):
        calls.append(batch)
        return batch["temporal"]

    policy = SimpleNamespace(config=config, predict_action_chunk=predict, reset=lambda: resets.append(True))
    engine = engine_module.LeRobotEngine()
    engine._resolve_model_dir = lambda path: path
    engine._load_policy_assets = mock.Mock(return_value=(policy, lambda x: x, lambda x: x))

    def init_robot(robot_type):
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            robot = RobotClient("ffw_sg2_rev1")
        robot._config = {"cameras": {"eye": {}, "unused": {}},
                         "joint_groups": {"follower_arm": {}}, "sensors": {}}
        robot._joint_children = {}
        robots.append(robot)
        engine._robot = robot
        engine._cameras = {"eye": "image"}
        engine._state_modalities = ["arm"]
        engine._action_keys = ["arm"]

    engine._init_robot = init_robot
    worker = EngineWorker(engine)
    load = EngineCommandRequest(command=CMD_LOAD_POLICY, model_path="/synthetic", robot_type="test", seq_id=1)
    with mock.patch.object(engine_module, "load_input_pipeline", return_value=SimpleNamespace(compile=lambda engine, **kwargs: plan())), \
            mock.patch.object(engine_module, "resolve_adapter", return_value=definition_class()), \
            mock.patch.object(torch.cuda, "is_available", return_value=False):
        try:
            response = worker.handle(load)
            assert response.success, response.message
            execution = LoadedExecution.from_json(response.capabilities_json)
            assert execution.contract.mode == "chunk" and execution.context is not None
            assert execution.contract.observation_warmup_timeout_s == pytest.approx(1.1)
            assert execution.contract.pending_command_count == 0
            robot = robots[-1]
            history = robot._observation_capture
            assert history.sources == {"joint:follower_arm"}
            assert history.bytes_used == 0

            def feed(start):
                for offset, values in ((0., [1., 2.]), (.1, [3., 4.])):
                    with mock.patch.object(robot_client_impl.time, "monotonic", return_value=start + offset):
                        robot._update_joint("follower_arm", SimpleNamespace(name=["a", "b"], position=values,
                                                                              velocity=[], effort=[]))
                image = np.full((2, 4, 3), [1, 2, 3], dtype=np.uint8)
                with mock.patch.object(robot_client_impl.time, "monotonic", return_value=start + .1), \
                        mock.patch.object(robot_client_impl.cv2, "imdecode", return_value=image):
                    robot._update_image("eye", SimpleNamespace(data=b"frame"))
                    robot._update_image("unused", SimpleNamespace(data=b"frame"))

            feed(10.)
            assert history.bytes_used == 16  # Two float32 states, no images retained.
            running = replace(execution.context, phase="running", revision=1)
            with mock.patch.object(robot_client_impl.time, "monotonic", return_value=10.15):
                action = worker.handle(EngineCommandRequest(command=CMD_GET_ACTION, seq_id=2,
                                        execution_context_json=running.to_json()))
            assert action.success, action.message
            assert (action.chunk_size, action.action_dim) == (2, 2)
            assert action.action_list == [1., 2., 3., 4.]
            np.testing.assert_array_equal(calls[0]["image"][0, 0], [3, 2, 1])
            assert len(plan_calls) == 1 and len(resets) == 1
            assert history.bytes_used == 16

            paused = replace(running, generation=pause_generation, phase="paused", revision=2)
            assert worker.handle(EngineCommandRequest(command=CMD_UPDATE_CONTEXT, seq_id=3,
                                 execution_context_json=paused.to_json())).success
            assert history.bytes_used == 0 and len(resets) == 2
            robot._input_received_monotonic["camera:eye"] = 20.
            with mock.patch.object(robot_client_impl.time, "monotonic", side_effect=[20., 20., 22.]), \
                    mock.patch("time.sleep"):
                missing = engine._build_observation("")
            assert not missing["success"] and "joint:follower_arm" in missing["message"]
            assert len(calls) == 1  # Empty history never repeats a cached state.

            cached = worker.handle(replace(load, seq_id=4))
            assert cached.success, cached.message
            assert robot._closed and robot._observation_capture is None
            assert history.bytes_used == 0
            assert robots[-1]._observation_capture is not history
            assert len(plan_calls) == 2
            engine._load_policy_assets.assert_called_once()
            assert worker.handle(EngineCommandRequest(command=CMD_UNLOAD_POLICY, seq_id=5)).success
            assert robots[-1]._closed and robots[-1]._observation_capture is None
        finally:
            engine.cleanup()


@pytest.mark.parametrize("whole_context", [False, True])
def test_contextual_chunk_uses_emitted_feedback_and_pending_prefix_without_robot_history(whole_context):
    definition_class = engine_module.resolve_adapter("act").__class__

    def plan(*args, **kwargs):
        sources = {
            "state": {"source": "joint:follower_arm"},
            "published": {"source": "execution:published:command", "count": 2, "min_count": 0},
            "pending": {"source": "execution:pending:command", "count": 1, "min_count": 0},
        }
        if whole_context:
            sources["context"] = {"source": "execution:context"}
        return InputPipelineConfig({
            "sources": sources,
            "nodes": {},
            "outputs": {"before": {key: key for key in sources}, "after": {"*": "processed"}},
        }, {}, Path("feedback_test.yaml")).compile(engine)

    batches = []

    def predict(batch):
        batches.append(batch)
        return torch.tensor([[1., 2.]])

    policy = SimpleNamespace(config=SimpleNamespace(type="feedback_test"), predict_action_chunk=predict,
                             reset=mock.Mock())
    engine = engine_module.LeRobotEngine()
    engine._resolve_model_dir = lambda path: path
    engine._load_policy_assets = mock.Mock(return_value=(policy, lambda x: x, lambda x: x))
    robots = []

    def init_robot(robot_type):
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            robot = RobotClient("ffw_sg2_rev1")
        robot._config = {"cameras": {}, "joint_groups": {"follower_arm": {}}, "sensors": {}}
        robot._joint_children = {}
        robot._update_joint("follower_arm", SimpleNamespace(name=["a", "b"], position=[3., 4.],
                                                           velocity=[], effort=[]))
        robots.append(robot)
        engine._robot = robot
        engine._cameras = {}
        engine._state_modalities = ["arm"]
        engine._action_keys = ["arm"]

    engine._init_robot = init_robot
    worker = EngineWorker(engine)
    load = EngineCommandRequest(command=CMD_LOAD_POLICY, model_path="/feedback", robot_type="test", seq_id=1)
    with mock.patch.object(engine_module, "load_input_pipeline", return_value=SimpleNamespace(compile=lambda engine, **kwargs: plan())), \
            mock.patch.object(engine_module, "resolve_adapter", return_value=definition_class()), \
            mock.patch.object(torch.cuda, "is_available", return_value=False):
        try:
            loaded = worker.handle(load)
            assert loaded.success, loaded.message
            execution = LoadedExecution.from_json(loaded.capabilities_json)
            assert execution.contract.mode == "chunk" and execution.context is not None
            assert execution.contract.pending_command_count == (None if whole_context else 1)
            session = next(iter(engine._observation_sessions.values()))
            assert session.history is None and robots[-1]._observation_capture is None
            assert session.required_observations == {"camera_names": [], "joint_groups": ["follower_arm"],
                                                     "sensor_names": []}

            def request(context, seq):
                result = worker.handle(EngineCommandRequest(command=CMD_GET_ACTION, seq_id=seq,
                                        execution_context_json=context.to_json()))
                assert result.success, result.message
                return result

            running = replace(execution.context, phase="running", revision=1)
            request(running, 2)
            assert batches[-1]["published"] == () and batches[-1]["pending"] == ()
            np.testing.assert_array_equal(batches[-1]["state"], [3., 4.])
            receipts = tuple(ActionRecord("previous", "published", "command", (float(i), 0.),
                planned_values=(float(i), 9.), event_id=i, command_id=i, recorded_s=100. + i)
                for i in (1, 2, 3))
            pending = tuple(ActionRecord("next", "planned", "command", (float(i), 5.), command_id=i)
                            for i in (4, 5))
            feedback = replace(running, revision=2, latest_event_id=3, actions=receipts + pending)
            first = request(feedback, 3)
            assert batches[-1]["published"] == receipts[-2:]
            assert batches[-1]["published"][-1].values != batches[-1]["published"][-1].planned_values
            assert batches[-1]["pending"] == pending[:1]
            assert session.execution.retained_record_count == 3
            retry = request(feedback, 3)
            assert retry.action_list == first.action_list and len(batches) == 2

            acknowledged = replace(feedback, revision=3, after_event_id=3, actions=())
            request(acknowledged, 4)
            assert batches[-1]["published"] == receipts[-2:] and batches[-1]["pending"] == ()
            paused = replace(acknowledged, phase="paused", generation=1, revision=4, latest_event_id=4,
                             resets=(ResetRecord(4, "stop", 104.),))
            stopped = worker.handle(EngineCommandRequest(command=CMD_UPDATE_CONTEXT, seq_id=5,
                                    execution_context_json=paused.to_json()))
            assert stopped.success, stopped.message
            assert session.execution.retained_record_count == 0
            request(replace(paused, phase="running", revision=5, after_event_id=4, resets=()), 6)
            assert batches[-1]["published"] == () and batches[-1]["pending"] == ()

            cached = worker.handle(replace(load, seq_id=7))
            assert cached.success, cached.message
            assert session.execution.retained_record_count == 0 and robots[-2]._closed
            engine._load_policy_assets.assert_called_once()
            from main_runtime.execution_feedback import ExecutionFeedback
            reloaded = LoadedExecution.from_json(cached.capabilities_json)
            ledger = ExecutionFeedback(reloaded.context, postprocess=False,
                                       pending_command_count=reloaded.contract.pending_command_count)
            ledger.phase = "running"
            ledger.buffer.enqueue(20, np.arange(10.).reshape(5, 2))
            emitted = ledger.buffer.take()
            ledger.buffer.finish(emitted.command_id, status="published", emitted_values=(0., 0.))
            context = ledger.project(ledger.capture())
            assert len([a for a in context.actions if a.status == "planned"]) == (4 if whole_context else 1)
            request(context, 8)
            assert len(ledger.buffer.snapshot().pending) == 4
            assert len(batches[-1]["pending"]) == 1
            assert batches[-1]["pending"][0].values == (2., 3.)
            assert batches[-1]["published"][0].values == (0., 0.)
            ledger.acknowledge(context)
            assert not ledger.capture().snapshot.events
            assert worker.handle(EngineCommandRequest(command=CMD_UNLOAD_POLICY, seq_id=9)).success
            assert robots[-1]._closed
        finally:
            engine.cleanup()


def test_two_second_history_can_warm_up_without_fabricating_samples():
    definition_class = engine_module.resolve_adapter("act").__class__

    def plan(*args, **kwargs):
        return InputPipelineConfig({
            "sources": {"samples": {"source": "joint:follower_arm", "offsets_s": [-2., 0.], "max_age_s": .1}},
            "nodes": {"history": {"op": "stack", "inputs": ["samples"], "options": {"sequence": True}}},
            "outputs": {"before": {"history": "history"}, "after": {"*": "processed"}},
        }, {}, Path("history_test.yaml")).compile(engine)

    engine = engine_module.LeRobotEngine()
    engine._policy = SimpleNamespace(config=SimpleNamespace(type="temporal_test"))
    engine._adapter_definition = definition_class()
    engine._input_pipeline_config = SimpleNamespace(compile=lambda engine, **kwargs: plan())
    with mock.patch.object(RobotClient, "_init_subscriptions"):
        robot = RobotClient("ffw_sg2_rev1")
    robot._config = {"cameras": {}, "joint_groups": {"follower_arm": {}}, "sensors": {}}
    robot._joint_children = {}
    engine._robot = robot
    engine._state_modalities = ["arm"]
    _, session = engine._observation_plan(False)
    clock = [10.]

    def feed():
        robot._update_joint("follower_arm", SimpleNamespace(name=["a"], position=[clock[0]],
                                                           velocity=[], effort=[]))

    def receive_during_wait(seconds):
        clock[0] += seconds
        feed()

    try:
        with mock.patch.object(robot_client_impl.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(robot_client_impl.time, "sleep", side_effect=receive_during_wait):
            feed()
            batch = engine._build_observation("")
        assert "history" in batch, batch
        assert batch["history"].shape == (2, 1)
        assert 1.9 <= batch["history"][1, 0] - batch["history"][0, 0] <= 2.1
        assert clock[0] >= 12.
        assert engine._observation_wait_s >= 2.
        assert session.read_timeout_s == 1.
        session.reset()
        assert session.read_timeout_s == 3.
    finally:
        session.close()
        robot.close()
