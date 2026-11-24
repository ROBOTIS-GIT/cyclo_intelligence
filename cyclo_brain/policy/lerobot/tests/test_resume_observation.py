"""Resume must reset loaded chunks and wait for real post-transition callbacks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from test_temporal_engine import RobotClient, robot_client_impl, engine_module
from lerobot_engine.input_pipeline import InputPipelineConfig
from engine_process.protocol import CMD_GET_ACTION, CMD_LOAD_POLICY, CMD_UPDATE_CONTEXT, EngineCommandRequest
from engine_process.worker import EngineWorker
from inference_context.contract import LoadedExecution
from inference_context.execution import ResetRecord


@pytest.mark.parametrize("stale_source", ["camera:eye", "joint:follower_arm"])
def test_plain_chunk_resume_retains_weights_clears_caches_and_waits_for_each_source(stale_source):
    calls, queue = [], []

    def predict(batch):
        calls.append(batch)
        queue.append(99)
        return torch.tensor([[3.]])

    policy = SimpleNamespace(config=SimpleNamespace(type="act"),
                             predict_action_chunk=predict, reset=queue.clear)
    pre, post = mock.Mock(side_effect=lambda x: x), mock.Mock(side_effect=lambda x: x)
    engine = engine_module.LeRobotEngine()
    engine._resolve_model_dir = lambda path: path
    engine._load_policy_assets = mock.Mock(return_value=(policy, pre, post))
    definition = engine_module.resolve_adapter("act").__class__()
    clock = [10.]

    def init_robot(robot_type):
        engine._channel_mapping = SimpleNamespace(action=lambda chunk: chunk)
        with mock.patch.object(RobotClient, "_init_subscriptions"):
            engine._robot = RobotClient("ffw_sg2_rev1")
        engine._robot._config = {"cameras": {"eye": {}}, "joint_groups": {"follower_arm": {}}, "sensors": {}}
        engine._robot._joint_children = {}
        engine._cameras, engine._state_modalities, engine._action_keys = {"eye": "image"}, ["arm"], ["arm"]

    recipe = InputPipelineConfig({
        "sources": {"state": {"source": "joint:follower_arm"}, "image": {"source": "camera:eye"}},
        "nodes": {}, "outputs": {"before": {"state": "state", "image": "image"},
                                  "after": {"*": "processed"}},
    }, {}, Path("resume_test.yaml"))
    engine._init_robot = init_robot
    worker = EngineWorker(engine)

    def feed(source):
        if source.startswith("joint:"):
            engine._robot._update_joint("follower_arm", SimpleNamespace(name=["j"], position=[clock[0]], velocity=[], effort=[]))
        else:
            with mock.patch.object(robot_client_impl.cv2, "imdecode", return_value=np.zeros((2, 2, 3), np.uint8)):
                engine._robot._update_image("eye", SimpleNamespace(data=b"frame"))

    with mock.patch.object(engine_module, "load_input_pipeline", return_value=recipe), \
         mock.patch.object(engine_module, "resolve_adapter", return_value=definition), \
         mock.patch.object(torch.cuda, "is_available", return_value=False), \
         mock.patch.object(robot_client_impl.time, "monotonic", side_effect=lambda: clock[0]), \
         mock.patch.object(robot_client_impl.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
        try:
            loaded = worker.handle(EngineCommandRequest(command=CMD_LOAD_POLICY, model_path="/test", robot_type="test"))
            assert loaded.success, loaded.message
            context = LoadedExecution.from_json(loaded.capabilities_json).context
            assert context is not None
            for source in ("camera:eye", "joint:follower_arm"):
                feed(source)
            running = replace(context, phase="running", revision=1)
            first = worker.handle(EngineCommandRequest(command=CMD_GET_ACTION, seq_id=1, execution_context_json=running.to_json()))
            assert first.success, first.message
            assert queue == [99]
            clock[0] = 20.
            resumed = replace(running, generation=1, revision=2, latest_event_id=1,
                              resets=(ResetRecord(1, "control source resumed", 9000000.),))
            updated = worker.handle(EngineCommandRequest(command=CMD_UPDATE_CONTEXT, seq_id=2, execution_context_json=resumed.to_json()))
            assert updated.success, updated.message
            assert queue == [] and pre.reset.call_count == post.reset.call_count == 2
            clock[0] = 20.1
            feed("joint:follower_arm" if stale_source.startswith("camera:") else "camera:eye")
            stale = engine._build_observation("task")
            assert stale.get("success") is False and stale_source in stale["message"]
            assert len(calls) == 1
            feed(stale_source)
            fresh = worker.handle(EngineCommandRequest(command=CMD_GET_ACTION, seq_id=3, execution_context_json=resumed.to_json()))
            assert fresh.success, fresh.message
            assert len(calls) == 2 and engine._policy is policy
            engine._load_policy_assets.assert_called_once()
        finally:
            engine.cleanup()
