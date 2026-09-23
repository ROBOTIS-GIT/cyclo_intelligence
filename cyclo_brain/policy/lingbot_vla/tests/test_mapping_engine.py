from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from lingbot_vla_engine import engine
from lingbot_vla_engine.mapping import RobotMapping


def robot_fixture(metadata):
    config = {"joint_groups": {"follower_arm": {"role": "follower", "joint_names": ["j1", "j2"]}},
              "cameras": {"head": {}, "left": {"rotation_deg": 270}, "right": {}}}
    groups = {"arm": {"joint_names": ["j1", "j2"]}}
    cameras = dict(zip(metadata["cameras"], config["cameras"]))
    mapping = RobotMapping(metadata, config, groups, cameras)
    snapshot = {"joint_positions": {"follower_lingbot_input_0": [2., 1.]}, "sensors": {},
                "images": {"head": np.full((6, 8, 3), 11, np.uint8),
                           "left": np.full((4, 2, 3), 22, np.uint8),
                           "right": np.full((3, 5, 3), 33, np.uint8)}}
    return config, groups, mapping, snapshot


def test_raw_observations_and_named_output(assets):
    metadata = assets[-1]
    _, _, mapping, snapshot = robot_fixture(metadata)
    snapshot["images"]["left"] = np.arange(24, dtype=np.uint8).reshape(4, 2, 3)
    observation = mapping.observation(snapshot, "pick")
    np.testing.assert_array_equal(observation["observation.state"], [2., 1.])
    assert observation["task"] == "pick"
    assert observation[metadata["cameras"][1]].shape == (2, 4, 3)
    np.testing.assert_array_equal(observation[metadata["cameras"][1]], np.rot90(snapshot["images"]["left"]))
    assert observation[metadata["cameras"][0]].shape == (6, 8, 3)
    observation[metadata["cameras"][0]][:] = 0
    assert snapshot["images"]["head"].min() == 11
    assert mapping.joint_views["follower_lingbot_input_0"]["joint_names"] == ["j2", "j1"]
    chunk = mapping.action({"action": [[2., 1.], [4., 3.]]}, 2)
    np.testing.assert_array_equal(chunk, [[1., 2.], [3., 4.]])
    assert chunk.dtype == np.float64
    assert mapping.required["camera_names"] == ["head", "left", "right"]


@pytest.mark.parametrize("copy_images", [True, False])
def test_image_ownership_and_rotation_allocation(assets, monkeypatch, copy_images):
    from lingbot_vla_engine import mapping as mapping_module

    metadata = assets[-1]
    _, _, mapping, snapshot = robot_fixture(metadata)
    rotated = []
    rotate = mapping_module.cv2.rotate

    def track_rotation(*args):
        result = rotate(*args)
        rotated.append(result)
        return result

    monkeypatch.setattr(mapping_module.cv2, "rotate", track_rotation)
    observation = mapping.observation(snapshot, "pick", copy_images=copy_images)
    head, left, right = metadata["cameras"]
    for key, name in ((head, "head"), (right, "right")):
        assert np.shares_memory(observation[key], snapshot["images"][name]) == (not copy_images)
    assert observation[left] is rotated[0]
    assert not np.shares_memory(observation[left], snapshot["images"]["left"])
    for key in metadata["cameras"]:
        assert observation[key].flags.c_contiguous


def test_owned_noncontiguous_image_preserves_pixels(assets):
    _, _, mapping, snapshot = robot_fixture(assets[-1])
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)[:, ::2]
    snapshot["images"]["head"] = image
    observation = mapping.observation(snapshot, "pick", copy_images=False)
    result = observation[assets[-1]["cameras"][0]]
    np.testing.assert_array_equal(result, image)
    assert result.flags.c_contiguous


@pytest.mark.parametrize("bad", [np.zeros((1, 2, 2)), np.zeros((2, 1)), np.full((2, 2), np.nan)])
def test_invalid_action_fails(assets, bad):
    mapping = robot_fixture(assets[-1])[2]
    with pytest.raises(ValueError, match="Invalid LingBot action"):
        mapping.action({"action": bad}, 2)


def test_whole_group_subset_but_no_partial_group(assets):
    metadata = assets[-1]
    config, groups, _, _ = robot_fixture(metadata)
    groups["head"] = {"joint_names": ["neck"]}
    mapping = RobotMapping(metadata, config, groups, dict(zip(metadata["cameras"], config["cameras"])))
    assert mapping.action_keys == ["arm"]
    metadata = deepcopy(metadata)
    metadata["action_names"] = ["j1"]
    with pytest.raises(ValueError, match="Partial action group"):
        RobotMapping(metadata, config, groups, {})


def test_load_warmup_reset_errors_and_reload(monkeypatch, bundle, assets):
    config, groups, mapping, snapshot = robot_fixture(assets[-1])
    clients, policies = [], []

    class FakeRobot:
        def __init__(self, robot_type, defer_subscriptions):
            assert defer_subscriptions
            self._config, self._action_groups = config, groups
            self.camera_names = list(config["cameras"])
            self.closed = False
            self.stale = False
            clients.append(self)

        def configure_joint_views(self, views):
            assert views == mapping.joint_views

        def start_observation_subscriptions(self, **kwargs):
            assert kwargs == mapping.required

        def wait_for_ready(self, **kwargs):
            return True

        def get_required_input_snapshot(self, sources, max_age_s):
            assert sources == mapping.sources
            assert max_age_s == 1.
            if self.stale:
                raise ValueError("Missing or stale input")
            return snapshot

        def close(self):
            self.closed = True

    class FakePolicy:
        def infer(self, observation):
            assert observation[assets[-1]["cameras"][0]] is snapshot["images"]["head"]
            self.last_action_chunk = {"action": np.asarray([[2., 1.], [4., 3.]])}
            self.global_step += 1
            return self.last_action_chunk

    def load_native(bundle, precision):
        assert precision == "float32"
        policy = FakePolicy()
        policies.append(policy)
        return policy

    module = ModuleType("robot_client")
    module.RobotClient = FakeRobot
    camera_module = ModuleType("robot_client.camera_mapping")
    camera_module.resolve_camera_feature_sources = lambda keys, names: dict(zip(keys, names))
    monkeypatch.setitem(sys.modules, "robot_client", module)
    monkeypatch.setitem(sys.modules, "robot_client.camera_mapping", camera_module)
    monkeypatch.setattr(engine, "load_native", load_native)
    request = SimpleNamespace(model_path=str(bundle), robot_type="test_robot", task_instruction="pick")
    worker = engine.create_engine()
    assert worker.load_policy(request)["success"]
    assert worker.policy.last_action_chunk is None  # warmup must not survive LOAD
    for _ in range(3):
        response = worker.get_action_chunk(request)
        assert response["success"]
        np.testing.assert_array_equal(response["action_chunk"], [1., 2., 3., 4.])
    worker.update_execution_context(SimpleNamespace(session_id="new", generation=1))
    assert worker.policy.last_action_chunk is None
    clients[-1].stale = True
    assert "stale" in worker.get_action_chunk(request)["message"]
    worker.cleanup()
    assert clients[-1].closed and not worker.is_ready
    assert worker.load_policy(request)["success"]
    assert len(policies) == 2
    worker.cleanup()


def test_wrong_robot_fails_before_native_load(monkeypatch, bundle):
    def unexpected(*args):
        pytest.fail("No model should load for a different robot")
    monkeypatch.setattr(engine, "load_native", unexpected)
    request = SimpleNamespace(model_path=str(bundle), robot_type="different", task_instruction="")
    worker = engine.create_engine()
    result = worker.load_policy(request)
    assert not result["success"] and "robot_type" in result["message"]
    assert not worker.is_ready
