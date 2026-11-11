"""Robot observations to named ABot channels, without transport or model imports."""

import cv2
import numpy as np

from .bundle import names


class RobotMapping:
    def __init__(self, bundle, robot_config, action_groups, cameras):
        self.bundle = bundle
        self.cameras = cameras
        self.rotations = {key: robot_config["cameras"][source].get("rotation_deg", 0)
                          for key, source in cameras.items()}
        if any(r not in (0, 90, 180, 270) for r in self.rotations.values()):
            raise ValueError("Unsupported robot camera rotation")
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
                for index, axis in enumerate("xyz"):
                    name = f"{prefix}_{axis}"
                    if name in available:
                        raise ValueError(f"Ambiguous odometry channel: {name}")
                    available[name] = (f"sensor:odom.{field}", index)
        missing = set(bundle.state_names) - available.keys()
        if missing:
            raise ValueError(f"Robot cannot provide checkpoint state channels: {sorted(missing)}")
        parents = {}
        for name in bundle.state_names:
            parent, index = available[name]
            if index is None:
                parents.setdefault(parent, []).append(name)
        self.joint_views = {
            f"follower_abot_input_{i}": {"parent": parent, "joint_names": channels}
            for i, (parent, channels) in enumerate(parents.items())}
        for group, view in self.joint_views.items():
            for index, name in enumerate(view["joint_names"]):
                available[name] = (f"joint:{group}", index)
        self.state_sources = [available[name] for name in bundle.state_names]
        self.action_keys = sorted(action_groups)
        target = names([n for key in self.action_keys for n in action_groups[key]["joint_names"]], "robot action")
        if set(target) != set(bundle.action_names):
            raise ValueError("Checkpoint action channels must match the robot action layout exactly")
        self.action_indices = [bundle.action_names.index(n) for n in target]
        self.sources = {source for source, _ in self.state_sources} | {
            f"camera:{source}" for source in cameras.values()}
        self.required = {
            "camera_names": sorted(set(cameras.values())),
            "joint_groups": sorted(self.joint_views),
            "sensor_names": sorted({s.split(":", 1)[1].split(".")[0]
                                    for s in self.sources if s.startswith("sensor:")}),
        }

    def observation(self, snapshot, instruction):
        state = []
        for source, index in self.state_sources:
            kind, name = source.split(":", 1)
            if kind == "joint":
                value = snapshot["joint_positions"][name]
            else:
                name, field = name.split(".", 1)
                value = snapshot["sensors"][name][field]
            state.append(value[index])
        result = {"lang": instruction, "image": []}
        for key in self.bundle.cameras:
            source = self.cameras[key]
            image = snapshot["images"][source]
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or not image.size:
                raise ValueError(f"{source}: expected nonempty uint8 RGB HWC image")
            rotation = self.rotations[key]
            if rotation:
                image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE,
                                          180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
            # Isolate RobotClient's shared image buffers from upstream transforms.
            result["image"].append(image.copy())
        normalized = self.bundle.state(np.asarray(state, dtype=np.float32))
        if normalized is not None:
            result["state"] = normalized
        return result

    def action(self, prediction):
        value = self.bundle.actions(prediction)
        return np.ascontiguousarray(value[:, self.action_indices], dtype=np.float64)
