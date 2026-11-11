"""Opt-in two-process Zenoh/CDR smoke; run only in a network-none container.

Uses a synthetic configuration, actual SDK publisher/subscriber and RobotClient.
No model allocation or command publisher. Not a ROS2/rmw interoperability test.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np


def configure(publisher):
    os.environ["ROS_DOMAIN_ID"] = "174"
    os.environ["ZENOH_CONFIG_OVERRIDE"] = (
        'transport/shared_memory/enabled=false;scouting/multicast/enabled=false;'
        + ('mode="client";connect/endpoints=["tcp/127.0.0.1:7447"];listen/endpoints=[]'
           if publisher else 'mode="peer";connect/endpoints=[];listen/endpoints=["tcp/127.0.0.1:7447"]')
    )


def publish():
    from zenoh_ros2_sdk import ROS2Publisher, get_message_class

    header_type = get_message_class("std_msgs/msg/Header")
    time_type = get_message_class("builtin_interfaces/msg/Time")
    publisher = ROS2Publisher(topic="/cyclo_test/state", msg_type="sensor_msgs/msg/JointState")
    try:
        time.sleep(1.)
        for index in range(1, 101):
            publisher.publish(
                header=header_type(stamp=time_type(sec=0, nanosec=index), frame_id="test"),
                name=["a", "b"], position=np.array([index, index + 1.], dtype=np.float64),
                velocity=np.zeros(2), effort=np.zeros(2),
            )
            time.sleep(.02)
    finally:
        publisher.close()


def subscribe():
    from robot_client import RobotClient
    import robot_client.robot_client as implementation

    physical = {"topic": "/cyclo_test/state", "msg_type": "sensor_msgs/msg/JointState", "joint_names": ["a", "b"]}
    config = {"cameras": {}, "sensors": {}, "joint_groups": {
        "first": dict(physical), "second": dict(physical),
        "child": {"parent": "second", "joint_names": ["b"]},
    }}
    with patch.object(implementation, "_build_runtime_config", return_value=config):
        robot = RobotClient("ffw_sg2_rev1", defer_subscriptions=True)
    producer = None
    try:
        robot.start_observation_subscriptions(camera_names=[], sensor_names=[],
                                             joint_groups=["first", "second", "child"])
        assert len(robot._state_subscribers) == 1
        producer = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--publisher"])
        deadline = time.monotonic() + 30
        distinct = set()
        while time.monotonic() < deadline:
            values = robot.get_joint_positions()
            if values:
                assert set(values) == {"first", "second", "child"}
                np.testing.assert_array_equal(values["first"], values["second"])
                np.testing.assert_array_equal(values["child"], values["second"][1:])
                distinct.add(float(values["first"][0]))
                if values["first"][0] == 100:
                    break
            if producer.poll() not in (None, 0):
                raise RuntimeError(f"publisher exited: {producer.returncode}")
            time.sleep(.005)
        else:
            raise TimeoutError("final test message did not arrive over Zenoh")
        assert len(distinct) >= 50, distinct
        assert producer.wait(timeout=10) == 0
        assert not robot.get_missing_observations()
        print(json.dumps({"passed": True, "transport": "Zenoh TCP/CDR, two processes",
                          "physical_subscriptions": len(robot._state_subscribers),
                          "logical_joint_views": 3, "distinct_samples_observed": len(distinct),
                          "robot_commands": 0}), flush=True)
    finally:
        if producer is not None and producer.poll() is None:
            producer.terminate()
            try:
                producer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                producer.kill()
                producer.wait()
        robot.close()


if __name__ == "__main__":
    publisher_mode = sys.argv[1:] == ["--publisher"]
    configure(publisher_mode)
    publish() if publisher_mode else subscribe()
