"""Name-checked model/robot mapping, independent of Torch and transport."""

import cv2
import numpy as np


def channel_names(value, label):
    if (not isinstance(value, list) or not value
            or not all(isinstance(n, str) and n for n in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} requires unique ordered channel names")
    return value


class RobotMapping:
    def __init__(self, metadata, robot_config, action_groups, camera_sources):
        self.state_names = channel_names(metadata["state_names"], "state")
        self.action_names = channel_names(metadata["action_names"], "action")
        self.cameras = dict(camera_sources)
        self.rotations = {key: robot_config["cameras"][source].get("rotation_deg", 0)
                          for key, source in self.cameras.items()}
        for value in self.rotations.values():
            if value not in (0, 90, 180, 270):
                raise ValueError(f"Unsupported camera rotation: {value}")
        available = {}
        for group, cfg in robot_config["joint_groups"].items():
            if cfg.get("role") != "follower" or cfg.get("parent"):
                continue
            for index, name in enumerate(cfg["joint_names"]):
                if name in available:
                    raise ValueError(f"Ambiguous state channel: {name}")
                available[name] = (f"joint:{group}", index)
        if "odom" in robot_config.get("sensors", {}):
            for field, prefix in (("linear_velocity", "linear"), ("angular_velocity", "angular")):
                for index, axis in enumerate("xyz"):
                    name = f"{prefix}_{axis}"
                    if name in available:
                        raise ValueError(f"Ambiguous odometry channel: {name}")
                    available[name] = (f"sensor:odom.{field}", index)
        missing = set(self.state_names) - available.keys()
        if missing:
            raise ValueError(f"Robot cannot provide checkpoint state channels: {sorted(missing)}")
        # Physical arrays follow message order. Named views select the model's
        # channels atomically from each message, preserving reception timestamps.
        parents = {}
        for name in self.state_names:
            source, _ = available[name]
            if source.startswith("joint:"):
                parents.setdefault(source.split(":", 1)[1], []).append(name)
        self.joint_views = {
            f"follower_rldx_input_{i}": {"parent": parent, "joint_names": names}
            for i, (parent, names) in enumerate(parents.items())
        }
        for group, view in self.joint_views.items():
            for index, name in enumerate(view["joint_names"]):
                available[name] = (f"joint:{group}", index)
        self.state_sources = [available[name] for name in self.state_names]
        self.action_keys = sorted(action_groups)
        target = [name for key in self.action_keys for name in action_groups[key]["joint_names"]]
        channel_names(target, "robot action")
        if set(target) != set(self.action_names):
            raise ValueError("Checkpoint action channels must match the robot action layout exactly")
        self.action_indices = [self.action_names.index(name) for name in target]
        self.sources = {source for source, _ in self.state_sources} | {
            f"camera:{source}" for source in self.cameras.values()}
        self.required = {
            "camera_names": sorted(set(self.cameras.values())),
            "joint_groups": sorted({s.split(":", 1)[1] for s in self.sources if s.startswith("joint:")}),
            "sensor_names": sorted({s.split(":", 1)[1].split(".")[0] for s in self.sources if s.startswith("sensor:")}),
        }

    def observation(self, snapshot, instruction, language_key):
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
        videos = {}
        for key, source in self.cameras.items():
            image = snapshot["images"][source]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{source}: expected uint8 RGB HWC image")
            rotation = self.rotations[key]
            if rotation:
                image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE,
                                          180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
            videos[key] = np.ascontiguousarray(image[None, None])
        return {"video": videos, "state": {"joint_position": state[None, None]},
                "language": {language_key: [[instruction]]}}

    def action(self, actions, horizon):
        value = np.asarray(actions["joint_position"])
        if value.shape != (1, horizon, len(self.action_names)) or not np.isfinite(value).all():
            raise ValueError(f"Invalid RLDX action chunk shape/values: {value.shape}")
        return np.ascontiguousarray(value[0][:, self.action_indices], dtype=np.float64)
