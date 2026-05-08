#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

"""
Test script for verifying cross-container Zenoh communication.

Run inside lerobot_server or groot_server container to test:
1. JointState subscription from rosbag replay
2. CompressedImage subscription from rosbag replay
3. Odometry subscription (odom type hash fix verification)
4. Latency measurement
5. Data integrity checks
"""
import os
import sys
import time
import threading

# Add zenoh_ros2_sdk to path
ZENOH_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk")
if os.path.exists(ZENOH_SDK_PATH):
    sys.path.insert(0, ZENOH_SDK_PATH)

from zenoh_ros2_sdk import ROS2Subscriber, get_logger
import numpy as np

logger = get_logger("test_subscriber")

# Test results
results = {
    "joint_state": {"count": 0, "last_msg": None, "latencies": [], "errors": []},
    "image": {"count": 0, "last_msg": None, "sizes": [], "errors": []},
    "odom": {"count": 0, "last_msg": None, "latencies": [], "errors": []},
}
lock = threading.Lock()

DOMAIN_ID = int(os.environ.get("ROS_DOMAIN_ID", "30"))
ROUTER_IP = os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1")
ROUTER_PORT = int(os.environ.get("ZENOH_ROUTER_PORT", "7447"))


def on_joint_state(msg):
    """Callback for JointState messages."""
    recv_time = time.time()
    with lock:
        results["joint_state"]["count"] += 1
        try:
            names = list(msg.name)
            positions = list(msg.position)
            # Compute latency from header stamp if available
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                msg_time = stamp.sec + stamp.nanosec * 1e-9
                if msg_time > 1e9:  # Valid timestamp
                    latency_ms = (recv_time - msg_time) * 1000
                    results["joint_state"]["latencies"].append(latency_ms)
            results["joint_state"]["last_msg"] = {
                "names": names,
                "positions": positions[:3],  # First 3 for display
                "num_joints": len(positions),
            }
        except Exception as e:
            results["joint_state"]["errors"].append(str(e))


def on_image(msg):
    """Callback for CompressedImage messages."""
    recv_time = time.time()
    with lock:
        results["image"]["count"] += 1
        try:
            data_size = len(msg.data)
            results["image"]["sizes"].append(data_size)
            # Compute latency from header stamp
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                msg_time = stamp.sec + stamp.nanosec * 1e-9
                if msg_time > 1e9:
                    latency_ms = (recv_time - msg_time) * 1000
                    # Only record reasonable latencies (rosbag timestamps may be old)
                    if abs(latency_ms) < 60000:
                        pass  # rosbag timestamps are from recording time, not useful for latency
            results["image"]["last_msg"] = {
                "format": getattr(msg, 'format', 'unknown'),
                "data_size_bytes": data_size,
            }
        except Exception as e:
            results["image"]["errors"].append(str(e))


def on_odom(msg):
    """Callback for Odometry messages (tests odom type hash fix)."""
    recv_time = time.time()
    with lock:
        results["odom"]["count"] += 1
        try:
            pos = msg.pose.pose.position
            orient = msg.pose.pose.orientation
            results["odom"]["last_msg"] = {
                "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                "orientation": {"x": orient.x, "y": orient.y, "z": orient.z, "w": orient.w},
                "child_frame_id": getattr(msg, 'child_frame_id', ''),
            }
            # Check covariance fields (the ones that caused the hash mismatch)
            if hasattr(msg.pose, 'covariance'):
                cov = msg.pose.covariance
                results["odom"]["last_msg"]["pose_covariance_len"] = len(cov)
            if hasattr(msg.twist, 'covariance'):
                cov = msg.twist.covariance
                results["odom"]["last_msg"]["twist_covariance_len"] = len(cov)
        except Exception as e:
            results["odom"]["errors"].append(str(e))


def main():
    """Run subscriber test."""
    common_kwargs = {
        "domain_id": DOMAIN_ID,
        "router_ip": ROUTER_IP,
        "router_port": ROUTER_PORT,
    }

    print("=" * 70)
    print("Cross-Container Zenoh Communication Test")
    print("=" * 70)
    print(f"  Domain ID: {DOMAIN_ID}")
    print(f"  Router: {ROUTER_IP}:{ROUTER_PORT}")
    print(f"  ZENOH_CONFIG_OVERRIDE: {os.environ.get('ZENOH_CONFIG_OVERRIDE', 'not set')}")
    print(f"  ZENOH_SHM_ENABLED: {os.environ.get('ZENOH_SHM_ENABLED', 'not set')}")
    print()

    subscribers = []

    # 1. JointState subscriber
    print("[1/3] Subscribing to /robot/arm_left_follower/joint_states...")
    sub_joint = ROS2Subscriber(
        topic="/robot/arm_left_follower/joint_states",
        msg_type="sensor_msgs/msg/JointState",
        callback=on_joint_state,
        **common_kwargs,
    )
    subscribers.append(sub_joint)
    print("      OK")

    # 2. CompressedImage subscriber
    print("[2/3] Subscribing to /robot/camera/cam_left_head/image_raw/compressed...")
    sub_image = ROS2Subscriber(
        topic="/robot/camera/cam_left_head/image_raw/compressed",
        msg_type="sensor_msgs/msg/CompressedImage",
        callback=on_image,
        **common_kwargs,
    )
    subscribers.append(sub_image)
    print("      OK")

    # 3. Odometry subscriber (tests odom type hash fix)
    print("[3/3] Subscribing to /odom (Odometry - type hash fix test)...")
    try:
        sub_odom = ROS2Subscriber(
            topic="/odom",
            msg_type="nav_msgs/msg/Odometry",
            callback=on_odom,
            **common_kwargs,
        )
        subscribers.append(sub_odom)
        print("      OK")
    except Exception as e:
        print(f"      FAILED: {e}")
        results["odom"]["errors"].append(f"Subscribe failed: {e}")

    # Wait and collect data
    test_duration = 15  # seconds
    print(f"\nCollecting data for {test_duration} seconds...")
    print()

    for i in range(test_duration):
        time.sleep(1)
        with lock:
            joint_count = results["joint_state"]["count"]
            image_count = results["image"]["count"]
            odom_count = results["odom"]["count"]
        print(f"  [{i+1:2d}s] joints={joint_count}, images={image_count}, odom={odom_count}")

    # Print results
    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    # JointState results
    print()
    print("--- JointState (/robot/arm_left_follower/joint_states) ---")
    with lock:
        r = results["joint_state"]
    if r["count"] > 0:
        print(f"  Status: PASS")
        print(f"  Messages received: {r['count']}")
        print(f"  Rate: ~{r['count'] / test_duration:.1f} Hz")
        if r["last_msg"]:
            print(f"  Last msg: {r['last_msg']['num_joints']} joints, names={r['last_msg']['names'][:3]}...")
            print(f"  Positions (first 3): {r['last_msg']['positions']}")
        if r["latencies"]:
            lats = r["latencies"]
            print(f"  Latency: avg={np.mean(lats):.1f}ms, min={np.min(lats):.1f}ms, max={np.max(lats):.1f}ms")
    else:
        print(f"  Status: FAIL - No messages received")
    if r["errors"]:
        print(f"  Errors: {r['errors'][:3]}")

    # CompressedImage results
    print()
    print("--- CompressedImage (/robot/camera/cam_left_head/image_raw/compressed) ---")
    with lock:
        r = results["image"]
    if r["count"] > 0:
        print(f"  Status: PASS")
        print(f"  Messages received: {r['count']}")
        print(f"  Rate: ~{r['count'] / test_duration:.1f} Hz")
        if r["sizes"]:
            sizes = r["sizes"]
            print(f"  Image size: avg={np.mean(sizes)/1024:.1f}KB, min={np.min(sizes)/1024:.1f}KB, max={np.max(sizes)/1024:.1f}KB")
        if r["last_msg"]:
            print(f"  Format: {r['last_msg']['format']}")
    else:
        print(f"  Status: FAIL - No messages received")
    if r["errors"]:
        print(f"  Errors: {r['errors'][:3]}")

    # Odometry results
    print()
    print("--- Odometry (/odom - type hash fix verification) ---")
    with lock:
        r = results["odom"]
    if r["count"] > 0:
        print(f"  Status: PASS (odom type hash fix VERIFIED)")
        print(f"  Messages received: {r['count']}")
        print(f"  Rate: ~{r['count'] / test_duration:.1f} Hz")
        if r["last_msg"]:
            print(f"  Position: {r['last_msg']['position']}")
            print(f"  Orientation: {r['last_msg']['orientation']}")
            if "pose_covariance_len" in r["last_msg"]:
                print(f"  Pose covariance length: {r['last_msg']['pose_covariance_len']} (expected 36)")
            if "twist_covariance_len" in r["last_msg"]:
                print(f"  Twist covariance length: {r['last_msg']['twist_covariance_len']} (expected 36)")
    else:
        print(f"  Status: FAIL - No messages received")
        if r["errors"]:
            print(f"  Errors: {r['errors'][:3]}")

    print()
    print("=" * 70)
    total_pass = sum(1 for k in ["joint_state", "image", "odom"] if results[k]["count"] > 0)
    print(f"Overall: {total_pass}/3 tests passed")
    print("=" * 70)

    # Cleanup
    for sub in subscribers:
        try:
            sub.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
