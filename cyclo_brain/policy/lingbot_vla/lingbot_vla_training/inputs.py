"""Explicit, reversible raw-channel to canonical-tensor mapping.

Only the input section of the Bench version-2 profile is validated here.
Asset receipts and the training runtime require separate validation.
"""

from dataclasses import dataclass
import math

import numpy as np


CANONICAL_DIM = 55
NORM_EPS = 1e-6  # Matches upstream Normalizer's meanstd transform.


def _names(value, field):
    if (
        not isinstance(value, list) or not value
        or any(not isinstance(name, str) or not name or name != name.strip() for name in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field} must contain ordered, unique, nonempty names")
    return tuple(value)


def _dataset_names(info, key):
    feature = info.get("features", {}).get(key, {})
    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("motors")
    names = _names(names, f"meta/info.json {key}")
    if feature.get("shape") != [len(names)]:
        raise ValueError(f"{key} shape must match its named one-dimensional vector")
    return names


@dataclass(frozen=True)
class ChannelLayout:
    names: tuple[str, ...]
    slots: tuple[int, ...]

    def __post_init__(self):
        _names(list(self.names), "channel names")
        if (
            len(self.names) != len(self.slots)
            or any(type(slot) is not int or not 0 <= slot < CANONICAL_DIM for slot in self.slots)
            or len(set(self.slots)) != len(self.slots)
        ):
            raise ValueError("Every raw channel requires a distinct canonical slot in 0..54")

    @property
    def active_mask(self):
        mask = np.zeros(CANONICAL_DIM, dtype=bool)
        mask[list(self.slots)] = True
        return mask

    def pack(self, raw):
        raw = _vector(raw, len(self.names), "raw channels")
        canonical = np.zeros((*raw.shape[:-1], CANONICAL_DIM), dtype=np.float32)
        canonical[..., list(self.slots)] = raw
        return canonical

    def unpack(self, canonical):
        canonical = _vector(canonical, CANONICAL_DIM, "canonical channels")
        return canonical[..., list(self.slots)].copy()


def _vector(value, dimension, field):
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[-1] != dimension or array.dtype.kind not in "fiu":
        raise ValueError(f"{field} must be numeric with last dimension {dimension}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains nonfinite values")
    with np.errstate(over="ignore"):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{field} exceeds float32 range")
    return array


@dataclass(frozen=True)
class InputContract:
    state: ChannelLayout
    action: ChannelLayout
    cameras: tuple[tuple[str, str], ...]

    @classmethod
    def from_profile(cls, profile, info):
        if not isinstance(profile, dict) or not isinstance(info, dict):
            raise ValueError("Profile and dataset metadata must be objects")
        if (
            type(profile.get("version")) is not int or profile["version"] != 2
            or profile.get("model_id") != "lingbot-vla-v2"
            or profile.get("policy_id") != "lingbot_vla:v2"
            or profile.get("action_representation") != "absolute"
        ):
            raise ValueError("Expected a version-2 native LingBot profile with absolute actions")
        robot = info.get("robot_type")
        if not isinstance(robot, str) or not robot or profile.get("robot_type") != robot:
            raise ValueError("Profile robot_type differs from dataset metadata")
        mapping, transforms = profile.get("canonical_mapping"), profile.get("transforms")
        if (
            not isinstance(mapping, dict) or set(mapping) != {"state", "action"}
            or not isinstance(transforms, dict) or set(transforms) != {"state", "action"}
        ):
            raise ValueError("Explicit state/action canonical_mapping and transforms are required")
        layouts = {}
        for kind, origin in (("state", "observation.state"), ("action", "action")):
            names = _names(profile.get(f"{kind}_names"), f"{kind}_names")
            if names != _dataset_names(info, origin):
                raise ValueError(f"{kind} must preserve all dataset channels in dataset order")
            slots = mapping[kind]
            if not isinstance(slots, dict) or set(slots) != set(names):
                raise ValueError(f"{kind} mapping must cover every raw channel exactly once")
            layouts[kind] = ChannelLayout(names, tuple(slots[name] for name in names))
            slices = transforms[kind]
            if not isinstance(slices, list) or not slices:
                raise ValueError(f"{kind} requires explicit transform slices")
            cursor = 0
            for item in slices:
                if not isinstance(item, dict):
                    raise ValueError(f"{kind} transform must be an object")
                start, end = item.get("start"), item.get("end")
                role = item.get("semantic_role")
                if (
                    type(start) is not int or type(end) is not int
                    or start != cursor or not start < end <= len(names)
                    or item.get("origin_key") != origin
                    or item.get("names") != list(names[start:end])
                    or item.get("canonical_slots") != list(layouts[kind].slots[start:end])
                    or not isinstance(role, str) or not role.strip()
                    or item.get("subtract_state") is not False
                    or "relative_type" not in item or item["relative_type"] is not None
                ):
                    raise ValueError(f"{kind} transform must describe ordered absolute channel slices")
                cursor = end
            if cursor != len(names):
                raise ValueError(f"{kind} transforms do not cover all channels")
        # The initial tensor implementation deliberately supports one tested transform.
        if profile.get("normalization") != "mean_std":
            raise ValueError("Only mean_std normalization is currently implemented")
        cameras, order = profile.get("cameras"), profile.get("camera_order")
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError("Explicit camera roles are required")
        order = _names(order, "camera_order")
        if set(order) != set(cameras):
            raise ValueError("camera_order must list every role exactly once")
        keys = _names(list(cameras.values()), "camera feature keys")
        for key in keys:
            feature = info.get("features", {}).get(key, {})
            if feature.get("dtype") != "video":
                raise ValueError(f"Camera {key} is not a dataset video feature")
        teacher = profile.get("teacher_training")
        if (
            not isinstance(teacher, dict)
            or any(teacher.get(key) is not True for key in ("depth", "future_depth", "future_video"))
            or teacher.get("trainable") is not False
            or teacher.get("episode_boundary") != "pad_and_mask"
        ):
            raise ValueError("Frozen depth/future-depth/future-video teachers and boundary masks are required")
        return cls(layouts["state"], layouts["action"], tuple((role, cameras[role]) for role in order))


class MeanStd:
    """Raw-order statistics; padding is added only after normalization."""

    def __init__(self, layout, stats):
        if not isinstance(stats, dict) or stats.get("names") != list(layout.names):
            raise ValueError("Statistics must declare the exact raw channel order")
        self.mean = _vector(stats.get("mean"), len(layout.names), "statistics mean").copy()
        std = _vector(stats.get("std"), len(layout.names), "statistics std")
        if self.mean.ndim != 1 or std.ndim != 1 or np.any(std < 0):
            raise ValueError("Statistics must be one-dimensional with nonnegative std")
        self.scale = std + NORM_EPS

    def normalize(self, raw):
        raw = _vector(raw, len(self.mean), "normalization input")
        with np.errstate(over="ignore", invalid="ignore"):
            result = (raw - self.mean) / self.scale
        return _vector(result, len(self.mean), "normalized channels")

    def denormalize(self, values):
        values = _vector(values, len(self.mean), "denormalization input")
        with np.errstate(over="ignore", invalid="ignore"):
            result = values * self.scale + self.mean
        return _vector(result, len(self.mean), "denormalized channels")


class InputTransform:
    """Prepare a single raw sample without modifying its absolute action values."""

    def __init__(self, contract, stats):
        if not isinstance(stats, dict) or set(stats) != {"state", "action"}:
            raise ValueError("Separate raw-order state and action statistics are required")
        self.contract = contract
        self.state_norm = MeanStd(contract.state, stats["state"])
        self.action_norm = MeanStd(contract.action, stats["action"])

    def prepare(self, state, actions, action_is_pad):
        state = _vector(state, len(self.contract.state.names), "state")
        actions = _vector(actions, len(self.contract.action.names), "actions")
        padding = np.asarray(action_is_pad)
        if state.ndim != 1 or actions.ndim != 2 or not len(actions):
            raise ValueError("Expected one state vector and a nonempty action chunk")
        if padding.dtype != np.bool_ or padding.shape != (len(actions),):
            raise ValueError("action_is_pad must be a boolean mask for each chunk step")
        if padding.all() or np.any(padding[:-1] & ~padding[1:]):
            raise ValueError("Action padding must be a trailing suffix after a valid current step")
        action_mask = self.contract.action.active_mask
        return {
            "state": self.contract.state.pack(self.state_norm.normalize(state)),
            "action": self.contract.action.pack(self.action_norm.normalize(actions)),
            "state_joint_mask": self.contract.state.active_mask,
            "action_joint_mask": action_mask,
            "action_is_pad": padding.copy(),
            "chunk_joint_mask": np.broadcast_to(action_mask, (len(actions), CANONICAL_DIM)).copy(),
            "action_loss_mask": ~padding[:, None] & action_mask[None, :],
        }

    def restore_actions(self, canonical):
        return self.action_norm.denormalize(self.contract.action.unpack(canonical))


@dataclass(frozen=True)
class FramePlan:
    """Indices are episode-local. Warmup repeats current; it is not past history."""

    action_indices: tuple[int, ...]
    action_is_pad: tuple[bool, ...]
    video_indices: tuple[int, ...]
    video_is_pad: tuple[bool, ...]
    current_index: int
    future_valid: bool
    future_offset_seconds: float
    effective_fps: float


def plan_frames(*, frame_index, episode_length, chunk_size, fps, use_warmup_frame):
    for name, value in (("frame_index", frame_index), ("episode_length", episode_length), ("chunk_size", chunk_size)):
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
    if not 0 <= frame_index < episode_length:
        raise ValueError("Current frame must be inside a nonempty episode")
    if chunk_size < 2:
        raise ValueError("Future teachers require a chunk of at least two steps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if type(use_warmup_frame) is not bool:
        raise ValueError("use_warmup_frame must be boolean")
    requested = tuple(range(frame_index, frame_index + chunk_size))
    indices = tuple(min(index, episode_length - 1) for index in requested)
    padding = tuple(index >= episode_length for index in requested)
    offset = (chunk_size - 1) / fps
    # Matches upstream's one-future-frame recipe, including its duplicate warmup.
    video = (frame_index, indices[-1])
    video_pad = (False, padding[-1])
    if use_warmup_frame:
        video = (frame_index, *video)
        video_pad = (False, *video_pad)
    return FramePlan(
        indices, padding, video, video_pad, int(use_warmup_frame),
        not padding[-1], offset, fps / (chunk_size - 1),
    )
