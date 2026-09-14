"""Constraints of the pinned WALL-X implementation; never truncate robot joints."""

from .definition import AdapterDefinition
from .validation import _feature_dim, validate_single_observation


def validate_checkpoint(config, model_path):
    validate_single_observation(config, model_path)
    for features, key, maximum in (
        ("input_features", "observation.state", "max_state_dim"),
        ("output_features", "action", "max_action_dim"),
    ):
        dim = _feature_dim(config.get(features, {}), key)
        # The pinned WALL-X core pads to a literal 20, independently of config.
        if config.get(maximum, 20) != 20 or dim > 20:
            raise ValueError(
                f"WALL-X requires {maximum}=20 and {key} dimension <= 20; got {dim}"
            )


def validate_wall_x_robot(config, robot, modalities, action_keys):
    """Never let the legacy state padding/truncation hide a WALL-X layout mismatch."""
    state_dim = sum(
        3 if name == "mobile" else len(robot.get_joint_names(f"follower_{name}"))
        for name in modalities
    )
    groups = robot._action_groups
    if set(action_keys) != set(groups):
        raise ValueError("WALL-X action modalities do not match the robot command layout")
    action_dim = sum(
        3 if groups[key]["msg_type"] == "geometry_msgs/msg/Twist"
        else len(groups[key]["joint_names"])
        for key in action_keys
    )
    for name, actual, expected in (
        ("state", state_dim, _feature_dim(config.input_features, "observation.state")),
        ("action", action_dim, _feature_dim(config.output_features, "action")),
    ):
        if actual > 20 or actual != expected:
            raise ValueError(
                f"WALL-X {name} dimension: robot={actual}, checkpoint={expected}, maximum=20. "
                "Joint truncation/padding is not supported."
            )

ADAPTER = AdapterDefinition(
    checkpoint_validator=validate_checkpoint,
    layout_validator=validate_wall_x_robot,
)
