"""Compile named external policy channels to observation sources and commands."""

import logging

import numpy as np

logger = logging.getLogger("lerobot_engine")


class ChannelMapping:
    def __init__(self, robot, config, metadata, legacy_modalities):
        from cyclo_lerobot_io.mapping import feature_dim, validate_mapping

        self.state_dim = feature_dim(config.input_features, "observation.state")
        self.action_dim = feature_dim(config.output_features, "action")
        groups = robot._config.get("joint_groups", {})
        available = {}
        for group, spec in groups.items():
            if spec.get("parent") or spec.get("role") != "follower":
                continue
            if spec.get("msg_type") != "sensor_msgs/msg/JointState":
                continue
            for name in spec.get("joint_names", []):
                if name in available:
                    raise ValueError(f"Ambiguous state channel: {name}")
                available[name] = group
        velocities = {}
        if "odom" in robot._config.get("sensors", {}):
            velocities = {f"{kind}_{axis}": (f"sensor:odom.{kind}_velocity", index)
                          for kind in ("linear", "angular") for index, axis in enumerate("xyz")}
        if set(available) & set(velocities):
            raise ValueError("Joint and velocity channel names overlap")

        commands = {}
        for group, spec in sorted(robot._action_groups.items()):
            if spec["msg_type"] == "geometry_msgs/msg/Twist":
                names = ["linear_x", "linear_y", "angular_z"]
                if spec.get("joint_names", names) != names:
                    raise ValueError(f"Unsupported Twist layout: {group}")
            elif spec["msg_type"] == "trajectory_msgs/msg/JointTrajectory":
                names = list(spec["joint_names"])
            else:
                raise ValueError(f"Unsupported command type: {spec['msg_type']}")
            commands[group] = names
        all_actions = [name for names in commands.values() for name in names]
        if not all_actions or len(set(all_actions)) != len(all_actions):
            raise ValueError("Robot command channels must be nonempty and unambiguous")

        legacy = metadata is None
        if legacy:
            state_names = [name for modality in legacy_modalities
                           for name in (["linear_x", "linear_y", "angular_z"] if modality == "mobile"
                                        else robot.get_joint_names(f"follower_{modality}"))]
            metadata = {"version": 1, "state_names": state_names, "action_names": all_actions}
        try:
            validate_mapping(metadata, state_dim=self.state_dim, action_dim=self.action_dim)
        except ValueError as exc:
            if legacy:
                raise ValueError(f"{exc}. Add verified cyclo_io_mapping.json; legacy mode cannot select channels.") from exc
            raise
        self.state_names = tuple(metadata["state_names"])
        self.action_names = tuple(metadata["action_names"])
        missing = set(self.state_names) - available.keys() - velocities.keys()
        if missing:
            raise ValueError(f"Unknown state channels: {sorted(missing)}")
        missing = set(self.action_names) - set(all_actions)
        if missing:
            raise ValueError(f"Unknown action channels: {sorted(missing)}")
        self.action_keys = []
        target = []
        for group, names in commands.items():
            selected = set(names) & set(self.action_names)
            if selected and selected != set(names):
                raise ValueError(f"Partial action group {group}: missing {sorted(set(names) - selected)}")
            if selected:
                self.action_keys.append(group)
                target.extend(names)
        if not target:
            raise ValueError("No command groups selected")
        self.action_indices = np.array([self.action_names.index(name) for name in target], dtype=np.intp)

        # Named views belong to this deferred observation client only. Each view
        # is captured atomically from one message, including history timestamps.
        parents = {}
        for name in self.state_names:
            if name in available:
                parents.setdefault(available[name], []).append(name)
        self.joint_views = {f"follower_cyclo_input_{i}": {"parent": parent, "joint_names": names}
                            for i, (parent, names) in enumerate(parents.items())}
        locations = dict(velocities)
        for group, spec in self.joint_views.items():
            locations.update({name: (f"joint:{group}", index) for index, name in enumerate(spec["joint_names"])})
        self.state_sources = tuple(dict.fromkeys(locations[name][0] for name in self.state_names))
        self.state_indices = tuple((self.state_sources.index(locations[name][0]), locations[name][1])
                                   for name in self.state_names)
        if legacy:
            logger.warning("No cyclo_io_mapping.json: using legacy robot channel order; equal dimensions do not prove semantics")

    def state(self, values):
        if len(values) != len(self.state_sources):
            raise ValueError("State sources changed after LOAD")
        result = np.asarray([values[source][index] for source, index in self.state_indices], dtype=np.float32)
        if result.shape != (self.state_dim,) or not np.isfinite(result).all():
            raise ValueError("State must contain finite values for every selected channel")
        return result

    def action(self, chunk):
        chunk = np.asarray(chunk)
        if (chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != self.action_dim
                or chunk.dtype.kind not in "iuf" or not np.isfinite(chunk).all()):
            raise ValueError(f"Postprocessed action must be finite (T, {self.action_dim}), got {chunk.shape}")
        return np.ascontiguousarray(chunk[:, self.action_indices])

    def validate_processors(self, preprocessor, postprocessor=None):
        for step in getattr(preprocessor, "steps", ()):
            if type(step).__name__ != "RelativeActionsProcessorStep" or not step.enabled:
                continue
            options = step.get_config()
            if len(self.state_names) < len(self.action_names):
                raise ValueError("Relative action processor requires a state prefix covering every action")
            excludes = [str(name).lower() for name in options.get("exclude_joints", []) if name]
            labels = options.get("action_names")
            if labels is not None and tuple(labels) != self.action_names:
                raise ValueError("Relative processor action names differ from mapped channels")
            for index, action_name in enumerate(self.action_names):
                excluded = (labels is not None
                            and any(token in action_name.lower() for token in excludes))
                if not excluded and self.state_names[index] != action_name:
                    raise ValueError(f"Relative action processor state/action prefix mismatch at {index}: {action_name}")
        for step in getattr(postprocessor, "steps", ()):
            if type(step).__name__ == "GrootN17ActionDecodeStep" and step.use_relative_action:
                from .adapters.groot import validate_relative_channels

                validate_relative_channels(self, step)
