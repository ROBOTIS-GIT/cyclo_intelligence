"""Synthetic input tests, not qualification of a physical robot profile."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lingbot_vla_training.inputs import (
    CANONICAL_DIM,
    ChannelLayout,
    InputContract,
    InputTransform,
    MeanStd,
    plan_frames,
)


@pytest.fixture
def input_profile():
    state, action = ["joint", "velocity"], ["target", "velocity"]
    info = {
        "robot_type": "synthetic",
        "features": {
            "observation.state": {"shape": [2], "names": state},
            "action": {"shape": [2], "names": action},
            "observation.images.first": {"dtype": "video"},
            "observation.images.second": {"dtype": "video"},
        },
    }
    profile = {
        "version": 2,
        "model_id": "lingbot-vla-v2",
        "policy_id": "lingbot_vla:v2",
        "robot_type": "synthetic",
        "state_names": state,
        "action_names": action,
        "action_representation": "absolute",
        "normalization": "mean_std",
        "cameras": {"side": "observation.images.second", "front": "observation.images.first"},
        "camera_order": ["front", "side"],
        "canonical_mapping": {
            "state": {"joint": 54, "velocity": 3},
            "action": {"target": 1, "velocity": 40},
        },
        "transforms": {},
        "teacher_training": {
            "depth": True, "future_depth": True, "future_video": True,
            "trainable": False, "episode_boundary": "pad_and_mask",
        },
    }
    for kind, names, origin in (("state", state, "observation.state"), ("action", action, "action")):
        profile["transforms"][kind] = [
            {
                "start": i, "end": i + 1, "origin_key": origin, "names": [name],
                "semantic_role": "synthetic_test_only",
                "subtract_state": False, "relative_type": None,
                "canonical_slots": [profile["canonical_mapping"][kind][name]],
            }
            for i, name in enumerate(names)
        ]
    return profile, info


@pytest.fixture
def input_transform(input_profile):
    profile, info = input_profile
    contract = InputContract.from_profile(profile, info)
    return InputTransform(contract, {
        "state": {"names": list(contract.state.names), "mean": [10, -3], "std": [2, 4]},
        "action": {"names": list(contract.action.names), "mean": [20, -2], "std": [5, 2]},
    })


def test_contract_preserves_camera_order_and_does_not_mutate_inputs(input_profile):
    profile, info = input_profile
    before = deepcopy((profile, info))
    contract = InputContract.from_profile(profile, info)
    assert contract.state.slots == (54, 3)
    assert contract.action.slots == (1, 40)
    assert contract.cameras == (
        ("front", "observation.images.first"), ("side", "observation.images.second"),
    )
    assert (profile, info) == before
    profile["canonical_mapping"]["action"]["target"] = 10
    assert contract.action.slots == (1, 40)


@pytest.mark.parametrize("names", [None, [], ["same", "same"], ["x", 1], ["x", " y"], ["", "x"]])
def test_invalid_names_rejected(input_profile, names):
    profile, info = input_profile
    profile["state_names"] = names
    with pytest.raises(ValueError):
        InputContract.from_profile(profile, info)


@pytest.mark.parametrize("index", [-1, 55, 1.5, True, "1", None, 3])
def test_bad_or_duplicate_slots_rejected(input_profile, index):
    profile, info = input_profile
    profile["canonical_mapping"]["state"]["joint"] = index
    with pytest.raises(ValueError):
        InputContract.from_profile(profile, info)


@pytest.mark.parametrize("field,value", [
    ("version", 1), ("version", 2.0), ("model_id", "lerobot-lingbot-va"),
    ("policy_id", "other"), ("action_representation", "relative"), ("robot_type", "other"),
    ("normalization", "quantile"), ("normalization", "min_max"), ("normalization", None),
    ("camera_order", ["side", "side"]), ("camera_order", ["front"]),
    ("camera_order", None), ("cameras", {}), ("canonical_mapping", {}), ("transforms", {}),
])
def test_invalid_profile_rejected(input_profile, field, value):
    profile, info = input_profile
    profile[field] = value
    with pytest.raises(ValueError):
        InputContract.from_profile(profile, info)


@pytest.mark.parametrize("failure", ["reordered", "partial", "dimension", "missing", "camera_dtype", "duplicate_camera"])
def test_metadata_mismatches_rejected(input_profile, failure):
    profile, info = input_profile
    if failure == "reordered":
        info["features"]["action"]["names"] = ["velocity", "target"]
    elif failure == "partial":
        info["features"]["action"] = {"shape": [3], "names": ["target", "velocity", "extra"]}
    elif failure == "dimension":
        info["features"]["action"]["shape"] = [3]
    elif failure == "missing":
        del info["features"]["observation.state"]
    elif failure == "camera_dtype":
        info["features"]["observation.images.first"]["dtype"] = "float32"
    else:
        profile["cameras"]["side"] = profile["cameras"]["front"]
    with pytest.raises(ValueError):
        InputContract.from_profile(profile, info)


def test_motor_name_dictionary_supported(input_profile):
    profile, info = input_profile
    for key in ("observation.state", "action"):
        info["features"][key]["names"] = {"motors": info["features"][key]["names"]}
    assert InputContract.from_profile(profile, info).state.names == ("joint", "velocity")


@pytest.mark.parametrize("kind", ["state", "action"])
@pytest.mark.parametrize("field,value", [
    ("start", True), ("start", 1), ("end", 3), ("origin_key", "unknown"),
    ("names", ["wrong"]), ("semantic_role", ""), ("subtract_state", True),
    ("subtract_state", 0), ("relative_type", "quaternion_local"), ("canonical_slots", [2]),
])
def test_bad_transform_rejected(input_profile, kind, field, value):
    profile, info = input_profile
    profile["transforms"][kind][0][field] = value
    with pytest.raises(ValueError):
        InputContract.from_profile(profile, info)


@pytest.mark.parametrize("failure", ["missing_relative", "overlap", "gap", "reorder"])
def test_transform_coverage_rejected(input_profile, failure):
    profile, info = input_profile
    transforms = profile["transforms"]["action"]
    if failure == "missing_relative":
        del transforms[0]["relative_type"]
    elif failure == "overlap":
        transforms.append(deepcopy(transforms[-1]))
    elif failure == "gap":
        transforms.pop()
    else:
        transforms.reverse()
    with pytest.raises(ValueError):
        InputContract.from_profile(profile, info)


@pytest.mark.parametrize("field,value", [
    ("depth", False), ("future_depth", False), ("future_video", False),
    ("trainable", True), ("episode_boundary", "repeat"),
])
def test_teachers_cannot_be_silently_disabled(input_profile, field, value):
    profile, info = input_profile
    profile["teacher_training"][field] = value
    with pytest.raises(ValueError):
        InputContract.from_profile(profile, info)


def test_absolute_roundtrip_and_padding(input_transform):
    state = np.array([100, -6], dtype=np.float32)
    actions = np.array([[25, -4], [30, 2], [30, 2]], dtype=np.float32)
    state_before, actions_before = state.copy(), actions.copy()
    out = input_transform.prepare(state, actions, np.array([False, False, True]))
    assert out["state"].shape == (55,)
    assert out["action"].shape == (3, 55)
    np.testing.assert_allclose(input_transform.restore_actions(out["action"]), actions)
    # No state subtraction, velocity integration, or clipping.
    np.testing.assert_allclose(out["action"][0, [1, 40]], [1, -1], rtol=1e-6)
    assert not out["action"][:, ~out["action_joint_mask"]].any()
    assert not out["state"][~out["state_joint_mask"]].any()
    assert out["action_loss_mask"].sum() == 4
    assert out["chunk_joint_mask"].sum() == 6
    assert not out["action_loss_mask"][-1].any()
    np.testing.assert_array_equal(state_before, state)
    np.testing.assert_array_equal(actions_before, actions)


def test_state_and_action_dimensions_can_differ():
    layout = ChannelLayout(("first", "second", "third"), (54, 0, 12))
    for shape in ((3,), (4, 3), (2, 4, 3)):
        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        packed = layout.pack(raw)
        assert packed.shape == (*shape[:-1], CANONICAL_DIM)
        np.testing.assert_array_equal(layout.unpack(packed), raw)
        assert not packed[..., ~layout.active_mask].any()


@pytest.mark.parametrize("value", [[np.nan, 1], [np.inf, 1], [1e100, 1], ["1", "2"], [True, False], [1]])
def test_invalid_tensor_rejected(value):
    with pytest.raises(ValueError):
        ChannelLayout(("a", "b"), (0, 1)).pack(value)


@pytest.mark.parametrize("pad", [[0, 0], [False], [True, True], [True, False]])
def test_invalid_chunk_padding_rejected(input_transform, pad):
    with pytest.raises(ValueError):
        input_transform.prepare([1, 2], [[1, 2], [3, 4]], pad)


@pytest.mark.parametrize("failure", ["order", "negative_std", "nonfinite", "horizon", "missing"])
def test_stats_fail_closed(failure):
    layout = ChannelLayout(("a", "b"), (0, 54))
    stats = {"names": ["a", "b"], "mean": [1, 2], "std": [1, 2]}
    if failure == "order":
        stats["names"].reverse()
    elif failure == "negative_std":
        stats["std"][0] = -1
    elif failure == "nonfinite":
        stats["mean"][0] = np.inf
    elif failure == "horizon":
        stats["std"] = [[1, 2], [1, 2]]
    else:
        del stats["mean"]
    with pytest.raises(ValueError):
        MeanStd(layout, stats)


def test_zero_variance_channels_and_constant_inputs():
    normalizer = MeanStd(ChannelLayout(("constant",), (4,)), {
        "names": ["constant"], "mean": [3], "std": [0],
    })
    np.testing.assert_array_equal(normalizer.normalize([3]), [0])
    np.testing.assert_allclose(normalizer.denormalize(normalizer.normalize([4])), [4])


def test_mean_std_matches_pinned_upstream(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    upstream = Path(__file__).resolve().parents[1] / "lingbot-vla-v2"
    if not (upstream / "lingbotvla").is_dir():
        pytest.skip("Upstream submodule is not initialized")
    monkeypatch.syspath_prepend(str(upstream))
    # Import the actual transform file without loading the unrelated data-loader package.
    spec = importlib.util.spec_from_file_location(
        "lingbot_upstream_transform", upstream / "lingbotvla/data/vla_data/transform.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw = np.array([[3, -2], [8, 1]], dtype=np.float32)
    stats = {"names": ["a", "b"], "mean": [2, 0], "std": [0.5, 2]}
    ours = MeanStd(ChannelLayout(("a", "b"), (54, 3)), stats)
    native_stats = {key: np.array(stats[key], dtype=np.float32) for key in ("mean", "std")}
    native = module.Normalizer({"action.synthetic": native_stats}, {"action.synthetic": "meanstd"})
    expected = native.normalize({"action.synthetic": raw})["action.synthetic"]
    np.testing.assert_array_equal(ours.normalize(raw), expected)
    np.testing.assert_array_equal(
        ours.denormalize(expected),
        native.unnormalize({"action.synthetic": expected})["action.synthetic"],
    )


@pytest.mark.parametrize("length", [1, 2, 5, 100])
@pytest.mark.parametrize("chunk", [2, 5, 50])
@pytest.mark.parametrize("warmup", [False, True])
def test_every_episode_position_stays_within_episode(length, chunk, warmup):
    for index in range(length):
        plan = plan_frames(
            frame_index=index, episode_length=length, chunk_size=chunk,
            fps=15, use_warmup_frame=warmup,
        )
        assert len(plan.action_indices) == chunk
        assert all(0 <= i < length for i in (*plan.action_indices, *plan.video_indices))
        assert plan.action_is_pad == tuple(index + offset >= length for offset in range(chunk))
        assert plan.future_valid == (index + chunk - 1 < length)
        assert plan.video_is_pad[-1] == (not plan.future_valid)
        assert not plan.video_is_pad[plan.current_index]
        assert plan.video_indices[plan.current_index] == index
        if warmup:
            assert plan.video_indices[:2] == (index, index)
        assert plan.future_offset_seconds == (chunk - 1) / 15
        assert plan.effective_fps == 15 / (chunk - 1)


@pytest.mark.parametrize("field,value", [
    ("frame_index", -1), ("frame_index", 5), ("frame_index", True),
    ("episode_length", 0), ("chunk_size", 1), ("chunk_size", 2.0),
    ("fps", 0), ("fps", -1), ("fps", float("nan")), ("fps", float("inf")),
    ("fps", True), ("use_warmup_frame", 1),
])
def test_bad_frame_plan_rejected(field, value):
    args = dict(frame_index=0, episode_length=5, chunk_size=3, fps=15, use_warmup_frame=True)
    args[field] = value
    with pytest.raises(ValueError):
        plan_frames(**args)


def test_frame_plan_and_channel_masks_agree(input_transform):
    plan = plan_frames(frame_index=3, episode_length=5, chunk_size=4, fps=15, use_warmup_frame=True)
    episode_actions = np.arange(10).reshape(5, 2)
    out = input_transform.prepare([1, 2], episode_actions[list(plan.action_indices)], plan.action_is_pad)
    assert out["action_loss_mask"].sum() == 4
    assert plan.video_indices == (3, 3, 4)
    assert not plan.future_valid
    np.testing.assert_allclose(
        input_transform.restore_actions(out["action"]),
        episode_actions[[3, 4, 4, 4]], rtol=1e-6, atol=1e-6,
    )
