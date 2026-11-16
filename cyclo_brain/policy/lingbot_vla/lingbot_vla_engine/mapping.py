"""Transport mapping for raw LingBot features, before its saved transforms."""

import cv2
import numpy as np


def channel_names(names, label):
    if (not isinstance(names, list) or not names
            or not all(isinstance(n, str) and n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"{label}: expected unique, ordered channel names")
    return names


class RobotMapping:
    def __init__(self, metadata, robot_config, action_groups, cameras):
        self.state_names = channel_names(metadata["state_names"], "state_names")
        self.action_names = channel_names(metadata["action_names"], "action_names")
        self.state_key = metadata["state_key"]
        self.action_key = metadata["action_key"]
        self.cameras = dict(cameras)
        self.rotations = {key: robot_config["cameras"][name].get("rotation_deg", 0)
                          for key, name in cameras.items()}
        if any(r not in (0, 90, 180, 270) for r in self.rotations.values()):
            raise ValueError("Camera rotation must be 0, 90, 180, or 270 degrees")
        available = {}
        for group, config in robot_config["joint_groups"].items():
            if config.get("role") != "follower" or config.get("parent"):
                continue
            for name in config["joint_names"]:
                if name in available:
                    raise ValueError(f"Ambiguous state channel: {name}")
                available[name] = (group, None)
        if "odom" in robot_config.get("sensors", {}):
            for field, prefix in (("linear_velocity", "linear"), ("angular_velocity", "angular")):
                for i, axis in enumerate("xyz"):
                    name = f"{prefix}_{axis}"
                    if name in available:
                        raise ValueError(f"Ambiguous odometry channel: {name}")
                    available[name] = (f"sensor:odom.{field}", i)
        missing = set(self.state_names) - available.keys()
        if missing:
            raise ValueError(f"Unavailable state channels: {sorted(missing)}")
        parents = {}
        for name in self.state_names:
            group, index = available[name]
            if index is None:
                parents.setdefault(group, []).append(name)
        self.joint_views = {
            f"follower_lingbot_input_{i}": {"parent": parent, "joint_names": names}
            for i, (parent, names) in enumerate(parents.items())}
        for group, view in self.joint_views.items():
            for index, name in enumerate(view["joint_names"]):
                available[name] = (f"joint:{group}", index)
        self.state_sources = [available[n] for n in self.state_names]
        # Never manufacture values for an untrained joint or partially control a group.
        self.action_keys = []
        target = []
        for key, group in sorted(action_groups.items()):
            names = group["joint_names"]
            overlap = set(names) & set(self.action_names)
            if overlap:
                if overlap != set(names):
                    raise ValueError(f"Partial action group is unsupported: {key}")
                self.action_keys.append(key)
                target.extend(names)
        channel_names(target, "robot actions")
        if set(target) != set(self.action_names):
            raise ValueError("Checkpoint action names do not match robot command channels")
        self.action_indices = [self.action_names.index(n) for n in target]
        self.sources = {s for s, _ in self.state_sources} | {f"camera:{n}" for n in cameras.values()}
        self.required = {
            "camera_names": sorted(set(cameras.values())),
            "joint_groups": sorted(s.split(":", 1)[1] for s in self.sources if s.startswith("joint:")),
            "sensor_names": sorted({s.split(":", 1)[1].split(".")[0] for s in self.sources if s.startswith("sensor:")}),
        }

    def observation(self, snapshot, instruction, *, copy_images=True):
        """Set copy_images=False only for a snapshot owned by this request."""
        state = []
        for source, index in self.state_sources:
            kind, name = source.split(":", 1)
            if kind == "joint":
                value = snapshot["joint_positions"][name]
            else:
                name, field = name.split(".", 1)
                value = snapshot["sensors"][name][field]
            state.append(value[index])
        state = np.asarray(state, dtype=np.float32)
        if not np.isfinite(state).all():
            raise ValueError("Non-finite state observation")
        observation = {self.state_key: state, "task": instruction}
        for key, name in self.cameras.items():
            image = snapshot["images"][name]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{name}: expected uint8 RGB HWC")
            rotation = self.rotations[key]
            if rotation:
                image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE,
                                          180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
            elif copy_images:
                image = np.array(image, copy=True, order="C")
            # Rotation already allocates; owned contiguous snapshots need no copy.
            observation[key] = np.ascontiguousarray(image)
        return observation

    def action(self, actions, horizon):
        value = np.asarray(actions[self.action_key])
        if value.shape != (horizon, len(self.action_names)) or not np.isfinite(value).all():
            raise ValueError(f"Invalid LingBot action chunk shape/values: {value.shape}")
        return np.ascontiguousarray(value[:, self.action_indices], dtype=np.float64)
