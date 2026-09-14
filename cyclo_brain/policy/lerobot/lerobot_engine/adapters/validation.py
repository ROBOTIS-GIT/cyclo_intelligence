"""Shared layout checks; model constraints are registered by adapters."""

def _feature_dim(features, key):
    feature = features.get(key)
    shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
    if not shape or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
        raise ValueError(f"Checkpoint requires an explicit positive vector feature: {key}")
    return shape[0]


def validate_step_layout(config, robot, modalities, action_keys):
    """Do not let legacy padding/truncation reinterpret feedback-sensitive actions."""
    state_dim = sum(3 if name == "mobile" else len(robot.get_joint_names(f"follower_{name}"))
                    for name in modalities)
    groups = robot._action_groups
    action_dim = 0
    for key in action_keys:
        if key not in groups:
            raise ValueError(f"step action group is missing: {key}")
        group = groups[key]
        if group["msg_type"] == "geometry_msgs/msg/Twist":
            action_dim += 3
        else:
            action_dim += len(group["joint_names"])
    for name, actual, expected in (
        ("state", state_dim, _feature_dim(config.input_features, "observation.state")),
        ("action", action_dim, _feature_dim(config.output_features, "action")),
    ):
        if actual != expected:
            raise ValueError(f"step {name} layout mismatch: robot={actual}, checkpoint={expected}; no padding/truncation")



def validate_single_observation(config, model_path):
    if config.get("n_obs_steps", 1) != 1:
        raise ValueError(f"{config.get('type')}: Cyclo currently supplies one observation, not history")


def validate_equal_camera_sizes(config, batch, camera_keys):
    shapes = {key: tuple(batch[key].shape[-2:]) for key in camera_keys}
    if len(set(shapes.values())) > 1:
        raise ValueError(
            f"{config.type} cameras must have equal sizes before stacking: {shapes}. "
            f"Check {config.type}.yaml against the training preprocessing."
        )
