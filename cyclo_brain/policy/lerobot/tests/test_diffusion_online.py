"""Opt-in pinned LeRobot tests. No network, GPU, robot or running Worker needed.

Run separately from suites that stub LeRobot, with
CYCLO_TEST_POLICY_DEPENDENCIES=1. Queue parity tests substitute neural generation;
the small-model regression runs real randomly initialized weights on CPU.
"""

from dataclasses import replace
import os
import time
from unittest.mock import Mock

import numpy as np
import pytest

if os.environ.get("CYCLO_TEST_POLICY_DEPENDENCIES") != "1":
    pytest.skip("requires pinned LeRobot dependencies", allow_module_level=True)

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from inference_context.execution import ActionRecord, ExecutionContext
from lerobot_engine.adapters import resolve_adapter
from channel_fixtures import make_channel_mapping


def config(n_obs=2, n_actions=3):
    return DiffusionConfig(
        device="cpu", n_obs_steps=n_obs, n_action_steps=n_actions, horizon=8,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        pretrained_backbone_weights=None, use_group_norm=True, down_dims=(32, 64),
        diffusion_step_embed_dim=32, spatial_softmax_num_keypoints=4,
        num_train_timesteps=4, num_inference_steps=2,
    )


def processors(cfg, path):
    stats = {
        key: {"min": torch.full((6,), -10.), "max": torch.full((6,), 10.)}
        for key in ("observation.state", "action")
    }
    stats.update({
        key: {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)}
        for key in cfg.image_features
    })
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    return make_pre_post_processors(cfg, pretrained_path=path)


def observation(value):
    return {
        "observation.state": torch.full((1, 6), float(value)),
        "observation.images.head": torch.full((1, 3, 32, 32), float(value) / 10),
        "observation.images.wrist": torch.full((1, 3, 32, 32), float(value) / 20),
        "task": ["test"],
    }


class GenerationProbe(torch.nn.Module):
    """Replace neural sampling only, preserving the real public queue logic."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.batches = []

    def generate_actions(self, batch, noise=None):
        self.batches.append({key: value.clone() for key, value in batch.items()})
        state = batch["observation.state"]
        assert state.shape == (1, self.config.n_obs_steps, 6)
        assert batch["observation.images"].shape == (1, self.config.n_obs_steps, 2, 3, 32, 32)
        return state.mean(dim=1, keepdim=True).repeat(1, self.config.n_action_steps, 1)


def queue_policy(cfg):
    policy = DiffusionPolicy.__new__(DiffusionPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = cfg
    policy.diffusion = GenerationProbe(cfg)
    policy.reset()
    return policy.eval()


def start(policy, pre, post):
    step = resolve_adapter("diffusion").create_execution_adapter(
        policy, pre, post, lambda tensor: tensor.detach().cpu().numpy(),
    )
    step.update_execution_context(ExecutionContext("test", 0, 0, "ready"))
    step.update_execution_context(ExecutionContext("test", 0, 1, "running"))
    return step


def acknowledge(step, prediction_id, action):
    context = step._context
    event_id = context.latest_event_id + 1
    record = ActionRecord(
        str(prediction_id), "published", "command", tuple(float(x) for x in action[0]),
        command_id=prediction_id, event_id=event_id, recorded_s=float(prediction_id),
    )
    step.update_execution_context(replace(
        context, revision=context.revision + 1, after_event_id=context.latest_event_id,
        latest_event_id=event_id, actions=(record,),
    ))


@pytest.mark.parametrize("n_obs", [1, 2, 4])
@pytest.mark.parametrize("n_actions", [1, 3])
def test_adapter_matches_official_online_sequence_and_reset(tmp_path, n_obs, n_actions):
    cfg = config(n_obs, n_actions)
    pre, post = processors(cfg, tmp_path / "adapter")
    ref_pre, ref_post = processors(cfg, tmp_path / "reference")
    policy, reference = queue_policy(cfg), queue_policy(cfg)
    step = start(policy, pre, post)
    for i in range(2 * n_actions + 1):
        actual = step.predict(observation(i), i + 1)
        expected = ref_post(reference.select_action(ref_pre(observation(i))))
        np.testing.assert_allclose(actual, expected.numpy())
        acknowledge(step, i + 1, actual)
    assert len(policy.diffusion.batches) == 3
    for actual, expected in zip(policy.diffusion.batches, reference.diffusion.batches):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])
    # Upstream bootstrap repeats the first frame. Cyclo supplies one real frame
    # per published step, not a manufactured temporal tensor or a second queue.
    assert torch.count_nonzero(policy.diffusion.batches[0]["observation.state"]) == 0
    last = policy.diffusion.batches[-1]["observation.state"][0, :, 0]
    expected_values = [max(0, i) / 10 for i in range(2 * n_actions - n_obs + 1, 2 * n_actions + 1)]
    torch.testing.assert_close(last, torch.tensor(expected_values))

    step.update_execution_context(replace(step._context, generation=1, phase="paused"))
    assert all(len(queue) == 0 for queue in policy._queues.values())
    step.update_execution_context(replace(step._context, revision=step._context.revision + 1, phase="running"))
    step.predict(observation(9), 2 * n_actions + 2)
    torch.testing.assert_close(policy.diffusion.batches[-1]["observation.state"], torch.full((1, n_obs, 6), .9))


def test_camera_error_is_explained_before_mutating_model_queues(tmp_path):
    cfg = config()
    pre, post = processors(cfg, tmp_path)
    policy = queue_policy(cfg)
    step = start(policy, pre, post)
    obs = observation(0)
    obs["observation.images.wrist"] = torch.zeros(1, 3, 48, 32)
    with pytest.raises(ValueError, match="equal sizes before stacking"):
        step.predict(obs, 1)
    assert all(len(queue) == 0 for queue in policy._queues.values())
    assert not policy.diffusion.batches


@pytest.mark.parametrize("enabled", [False, True])
def test_relative_action_processor_cannot_reanchor_cached_chunk_each_step(tmp_path, enabled):
    from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

    cfg = config()
    pre, post = processors(cfg, tmp_path)
    pre.steps.append(RelativeActionsProcessorStep(enabled=enabled))
    policy = queue_policy(cfg)
    if enabled:
        with pytest.raises(ValueError, match="reanchored"):
            start(policy, pre, post)
    else:
        start(policy, pre, post)
    assert not policy.diffusion.batches


def test_real_small_diffusion_reproduces_old_error_then_runs_online(tmp_path):
    torch.set_num_threads(2)
    cfg = config()
    pre, post = processors(cfg, tmp_path)
    policy = DiffusionPolicy(cfg).eval()
    # The actual neural implementation rejects the old latest-only chunk input.
    with pytest.raises(AssertionError):
        policy.predict_action_chunk(pre(observation(0)))
    step = start(policy, pre, post)
    for i in range(4):
        result = step.predict(observation(i), i + 1)
        assert result.shape == (1, 6) and np.isfinite(result).all()
        acknowledge(step, i + 1, result)


def test_worker_load_context_input_pipeline_retry_and_cached_reload(tmp_path, monkeypatch):
    from engine_process.protocol import (
        CMD_LOAD_POLICY, CMD_GET_ACTION, CMD_UNLOAD_POLICY, EngineCommandRequest,
    )
    from engine_process.worker import EngineWorker
    from inference_context.contract import LoadedExecution
    from lerobot_engine.engine import LeRobotEngine

    cfg = config()
    cfg.save_pretrained(tmp_path)
    pre, post = processors(cfg, tmp_path)
    engine = LeRobotEngine()
    assets = Mock(side_effect=lambda *_: (queue_policy(cfg), pre, post))
    monkeypatch.setattr(engine, "_load_policy_assets", assets)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    robots = []

    class Robot:
        _config = {"cameras": {}}

        def __init__(self):
            self.closed = False
            self.value = 0
            self.sources = []

        def close(self):
            self.closed = True

        def get_input_snapshot(self):
            now = time.monotonic()
            return {
                "images": {name: np.full((32, 32, 3), self.value, np.uint8) for name in ("head", "wrist")},
                "joint_positions": {"follower_cyclo_input_0": np.full(6, self.value, np.float32)},
                "sensors": {}, "captured_monotonic_s": now,
                "reception_monotonic_timestamps": dict.fromkeys(
                    ("camera:head", "camera:wrist", "joint:follower_cyclo_input_0"), now,
                ),
            }

        def get_required_input_snapshot(self, sources, **kwargs):
            self.sources.append(set(sources))
            return self.get_input_snapshot()

    def attach(_robot_type):
        robot = Robot()
        robots.append(robot)
        engine._robot = robot
        engine._cameras = {name: f"observation.images.{name}" for name in ("head", "wrist")}
        engine._state_modalities = ["arm"]
        engine._channel_mapping = make_channel_mapping(6)
        engine._step_adapter.set_action_mapping(engine._channel_mapping.action)
        engine._action_keys = ["arm"]

    monkeypatch.setattr(engine, "_init_robot", attach)
    worker = EngineWorker(engine)
    load = EngineCommandRequest(command=CMD_LOAD_POLICY, model_path=str(tmp_path),
                                robot_type="synthetic", policy_id="lerobot:diffusion", seq_id=1)
    try:
        loaded = worker.handle(load)
        assert loaded.success, loaded.message
        contract = LoadedExecution.from_json(loaded.capabilities_json)
        assert contract.contract.is_step and contract.context is not None
        session = engine._observation_sessions[True]
        assert session.history is None  # Only the policy allocates an observation queue.
        context = replace(contract.context, phase="running", revision=1)
        request = EngineCommandRequest(command=CMD_GET_ACTION, seq_id=2,
                                       execution_context_json=context.to_json())
        first = worker.handle(request)
        assert first.success, first.message
        assert (first.chunk_size, first.action_dim) == (1, 6)
        calls = len(robots[-1].sources)
        repeat = worker.handle(request)
        assert repeat.action_list == first.action_list
        assert len(robots[-1].sources) == calls
        assert len(engine._policy.diffusion.batches) == 1
        assert robots[-1].sources[-1] == {"camera:head", "camera:wrist", "joint:follower_cyclo_input_0"}
        assert engine._policy.diffusion.batches[0]["observation.state"].shape == (1, 2, 6)

        cached = worker.handle(replace(load, seq_id=3))
        assert cached.success, cached.message
        assets.assert_called_once()
        assert robots[-2].closed and session._closed
        assert all(len(queue) == 0 for queue in engine._policy._queues.values())
        assert worker.handle(EngineCommandRequest(command=CMD_UNLOAD_POLICY, seq_id=4)).success
        assert not engine.is_ready and robots[-1].closed
    finally:
        engine.cleanup()
