import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abot_engine.bundle import Bundle, Normalization
from abot_engine.mapping import RobotMapping
from abot_engine.engine import ABotEngine


@pytest.fixture
def bundle_path(tmp_path):
    meta = {
        "policy_id": "abot:m0", "robot_type": "test", "checkpoint": "checkpoints/model.pt",
        "cameras": ["head", "left", "right"], "state_names": ["j2", "j1", "linear_x"],
        "action_names": ["j2", "j1"], "state_indices": [2, 0, 3], "action_indices": [2, 0],
        "state_normalization": ["min_max", "identity", "identity"],
        "action_normalization": ["min_max", "identity"],
        "statistics_key": "test", "action_mode": "absolute", "observation_offsets": [0],
        "include_state": True,
    }
    config = {
        "framework": {"name": "ABot_M0", "use_vggt": False,
                      "action_model": {"action_dim": 4, "state_dim": 4, "future_action_window_size": 1,
                                       "past_action_window_size": 0, "action_horizon": 2}},
        "datasets": {"vla_data": {"action_mode": "abs", "include_state": True}},
    }
    (tmp_path / "cyclo_input_metadata.json").write_text(json.dumps(meta))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "dataset_statistics.json").write_text(json.dumps({"test": {
        "state": {"min": [0, 0, 0], "max": [4, 4, 4]},
        "action": {"min": [0, 0], "max": [10, 10]},
    }}))
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/model.pt").touch()
    return tmp_path


def robot_config():
    return {
        "cameras": {"head": {}, "left": {"rotation_deg": 270}, "right": {}, "unused": {}},
        "joint_groups": {"follower_arm": {"role": "follower", "joint_names": ["j1", "j2"]}},
        "sensors": {"odom": {}},
    }


def snapshot():
    return {"images": {"head": np.zeros((4, 6, 3), np.uint8),
                       "left": np.zeros((3, 8, 3), np.uint8), "right": np.zeros((5, 4, 3), np.uint8)},
            "joint_positions": {"follower_abot_input_0": [2., 1.]},
            "sensors": {"odom": {"linear_velocity": [3., 0., 0.]}}}


def mapping(bundle):
    return RobotMapping(bundle, robot_config(), {"arm": {"joint_names": ["j1", "j2"]}},
                        {c: c for c in bundle.cameras})


def test_named_mapping_padding_rotation_and_different_sizes(bundle_path):
    transform = mapping(Bundle(bundle_path, "test"))
    raw = snapshot()
    obs = transform.observation(raw, "pick")
    assert [im.shape for im in obs["image"]] == [(4, 6, 3), (8, 3, 3), (5, 4, 3)]
    np.testing.assert_array_equal(obs["state"], [[1, 0, 0, 3]])
    assert obs["lang"] == "pick"
    obs["image"][0][:] = 255
    assert not raw["images"]["head"].any()
    assert transform.required == {"camera_names": ["head", "left", "right"],
                                  "joint_groups": ["follower_abot_input_0"], "sensor_names": ["odom"]}
    value = np.array([[[.3, 0, .5, 0], [.6, 0, 1, 0]]], np.float32)
    result = transform.action({"normalized_actions": value})
    np.testing.assert_allclose(result, [[.3, 7.5], [.6, 10]])
    np.testing.assert_array_equal(value[0, :, 2], [.5, 1])


@pytest.mark.parametrize("field,value", [
    ("action_mode", "delta"), ("observation_offsets", [-1, 0]), ("policy_id", "abot:m05"),
    ("robot_type", "other"), ("action_names", ["j1", "j1"]), ("state_indices", [0, 0, 2]),
    ("action_indices", [0, 4]), ("action_indices", [False, 1]), ("statistics_key", "missing"),
    ("checkpoint", "../outside.pt"), ("include_state", False), ("state_names", ["unknown"]),
    ("action_normalization", ["binary", "identity"]), ("unexpected", True),
])
def test_invalid_bundle_rejected_before_model_load(bundle_path, field, value):
    path = bundle_path / "cyclo_input_metadata.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    result = ABotEngine().load_policy(SimpleNamespace(model_path=str(bundle_path), robot_type="test"))
    assert not result["success"]


def test_no_state_model_needs_only_images(bundle_path):
    path = bundle_path / "cyclo_input_metadata.json"
    meta = json.loads(path.read_text())
    meta.update(include_state=False, state_names=[], state_indices=[], state_normalization=[])
    path.write_text(json.dumps(meta))
    path = bundle_path / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["datasets"]["vla_data"]["include_state"] = False
    path.write_text(yaml.safe_dump(cfg))
    transform = mapping(Bundle(bundle_path, "test"))
    assert transform.required["joint_groups"] == transform.required["sensor_names"] == []
    assert "state" not in transform.observation({"images": snapshot()["images"]}, "pick")


def test_robot_channels_must_match(bundle_path):
    bundle = Bundle(bundle_path, "test")
    with pytest.raises(ValueError, match="action layout"):
        RobotMapping(bundle, robot_config(), {"arm": {"joint_names": ["j1"]}}, {})
    bundle.state_names = ["missing"]
    with pytest.raises(ValueError, match="cannot provide"):
        mapping(bundle)


@pytest.mark.parametrize("value", [np.zeros((2, 2, 4)), np.zeros((1, 3, 4)),
                                  np.full((1, 2, 4), np.nan), np.full((1, 2, 4), np.inf)])
def test_invalid_predictions(bundle_path, value):
    with pytest.raises(ValueError, match="Invalid ABot"):
        Bundle(bundle_path, "test").actions({"normalized_actions": value})


def test_normalization_constants_and_explicit_identity():
    transform = Normalization(["min_max", "q99", "mean_std", "identity"], {
        "min": [2]*4, "max": [2]*4, "q01": [2]*4, "q99": [2]*4,
        "mean": [2]*4, "std": [0]*4,
    }, 4)
    np.testing.assert_array_equal(transform.apply([3, 3, 3, 3]), [0, 1, 3, 3])
    np.testing.assert_array_equal(transform.apply([0, 1, 3, 3], inverse=True), [2, 2, 2, 3])
    with pytest.raises(ValueError, match="Invalid max"):
        Normalization(["min_max"], {"min": [0], "max": [float("nan")]}, 1)
    with pytest.raises(ValueError, match="Negative"):
        Normalization(["min_max"], {"min": [2], "max": [1]}, 1)


def test_checkpoint_symlink_escape(bundle_path, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.pt")
    outside.touch()
    path = bundle_path / "checkpoints/model.pt"
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="inside the bundle"):
        Bundle(bundle_path, "test")


def test_load_warmup_repeated_calls_and_clear(bundle_path, monkeypatch):
    import abot_engine.engine as module
    clients = []

    class Robot:
        def __init__(self, robot_type, defer_subscriptions):
            assert robot_type == "test" and defer_subscriptions
            self._config = robot_config()
            self._action_groups = {"arm": {"joint_names": ["j1", "j2"]}}
            self.camera_names = list(self._config["cameras"])
            self.closed = self.stale = False
            clients.append(self)

        def configure_joint_views(self, views):
            assert views["follower_abot_input_0"]["joint_names"] == ["j2", "j1"]

        def start_observation_subscriptions(self, **required):
            assert "unused" not in required["camera_names"]

        def wait_for_ready(self, **kwargs):
            return True

        def get_required_input_snapshot(self, sources, max_age_s):
            assert max_age_s == 1 and "camera:unused" not in sources
            if self.stale:
                raise ValueError("Missing or stale observation")
            return snapshot()

        def close(self):
            self.closed = True

    for name, attributes in {
        "robot_client": {"RobotClient": Robot},
        "robot_client.camera_mapping": {"resolve_camera_feature_sources": lambda keys, _: {k: k for k in keys}},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    calls = []

    def predict(examples):
        calls.append(examples)
        return {"normalized_actions": np.zeros((1, 2, 4), np.float32)}

    monkeypatch.setattr(module, "load_policy", lambda _: SimpleNamespace(predict_action=predict))
    request = SimpleNamespace(model_path=str(bundle_path), robot_type="test", task_instruction="pick")
    engine = ABotEngine()
    for _ in range(2):
        loaded = engine.load_policy(request)
        assert loaded["success"], loaded
        assert loaded["action_keys"] == ["arm"]
        for _ in range(3):
            action = engine.get_action_chunk(request)
            assert action["success"] and action["chunk_size"] == action["action_dim"] == 2
            np.testing.assert_array_equal(action["action_chunk"], [0, 5, 0, 5])
        clients[-1].stale = True
        assert not engine.get_action_chunk(request)["success"]
        engine.cleanup()
        engine.cleanup()
        assert clients[-1].closed and not engine.is_ready
    assert len(calls) == 8
    def fail(_):
        raise RuntimeError("bad weights")
    monkeypatch.setattr(module, "load_policy", fail)
    assert not engine.load_policy(request)["success"]
    assert clients[-1].closed and not engine.is_ready
    monkeypatch.setattr(module, "load_policy", lambda _: SimpleNamespace(predict_action=fail))
    result = engine.load_policy(request)
    assert "warmup failed" in result["message"]
    assert clients[-1].closed and not engine.is_ready


def test_failed_load_before_robot_allocation(bundle_path, monkeypatch):
    import abot_engine.engine as module
    monkeypatch.setattr(module, "load_policy", lambda _: (_ for _ in ()).throw(RuntimeError("bad weights")))
    engine = ABotEngine()
    result = engine.load_policy(SimpleNamespace(model_path=str(bundle_path), robot_type="wrong"))
    assert not result["success"] and not engine.is_ready


def test_export_preserves_original_and_refuses_overwrite(bundle_path, tmp_path):
    export = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/export_checkpoint.py"))["export"]
    source = bundle_path / "checkpoints/model.pt"
    source.write_bytes(b"test weights")
    output = tmp_path / "export"
    metadata = bundle_path / "cyclo_input_metadata.json"
    original = metadata.read_bytes()
    export(source, bundle_path, metadata, output)
    assert (output / "checkpoints/model.pt").read_bytes() == source.read_bytes()
    assert metadata.read_bytes() == original
    with pytest.raises(FileExistsError):
        export(source, bundle_path, metadata, output)


def test_stale_and_nonfinite_observations(bundle_path):
    transform = mapping(Bundle(bundle_path, "test"))
    raw = snapshot()
    raw["joint_positions"]["follower_abot_input_0"][0] = float("nan")
    with pytest.raises(ValueError, match="Invalid state"):
        transform.observation(raw, "pick")
    raw = snapshot()
    raw["images"]["head"] = raw["images"]["head"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8 RGB"):
        transform.observation(raw, "pick")
