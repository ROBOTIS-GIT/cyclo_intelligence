"""Synthetic named layouts using the same compiler as a real model LOAD."""

from types import SimpleNamespace

from lerobot_engine.channel_mapping import ChannelMapping


def make_channel_mapping(state_dim, action_dim=None):
    if action_dim is None:
        action_dim = state_dim
    states = [f"joint_{i}" for i in range(state_dim)]
    actions = [f"joint_{i}" for i in range(action_dim)]
    robot = SimpleNamespace(
        _config={"joint_groups": {"follower_arm": {
            "role": "follower", "msg_type": "sensor_msgs/msg/JointState", "joint_names": states}}},
        _action_groups={"arm": {"msg_type": "trajectory_msgs/msg/JointTrajectory", "joint_names": actions}},
    )
    config = SimpleNamespace(input_features={"observation.state": {"shape": [state_dim]}},
                             output_features={"action": {"shape": [action_dim]}})
    return ChannelMapping(robot, config, {"version": 1, "state_names": states, "action_names": actions}, [])
