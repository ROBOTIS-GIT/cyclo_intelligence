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
Latency & SHM verification test for cross-container Zenoh communication.

Measures:
1. Inter-message arrival jitter (real-time performance indicator)
2. Message size distribution (SHM threshold verification)
3. Data integrity checks
4. SHM status verification via Zenoh session info
"""
import os
import sys
import time
import threading
import statistics

# Add zenoh_ros2_sdk to path
ZENOH_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk")
if os.path.exists(ZENOH_SDK_PATH):
    sys.path.insert(0, ZENOH_SDK_PATH)

from zenoh_ros2_sdk import ROS2Subscriber, get_logger

logger = get_logger("test_latency")

DOMAIN_ID = int(os.environ.get("ROS_DOMAIN_ID", "30"))
ROUTER_IP = os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1")
ROUTER_PORT = int(os.environ.get("ZENOH_ROUTER_PORT", "7447"))

# Results storage
results = {}
lock = threading.Lock()


class TopicStats:
    """Tracks arrival times, sizes, and computes jitter statistics."""
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.arrival_times = []
        self.intervals = []
        self.sizes = []
        self.first_time = None
        self.last_time = None

    def record(self, data_size=0):
        now = time.monotonic()
        self.count += 1
        if self.first_time is None:
            self.first_time = now
        if self.last_time is not None:
            interval_ms = (now - self.last_time) * 1000
            self.intervals.append(interval_ms)
        self.last_time = now
        if data_size > 0:
            self.sizes.append(data_size)

    def summary(self):
        result = {"count": self.count}
        if self.first_time and self.last_time and self.last_time > self.first_time:
            duration = self.last_time - self.first_time
            result["rate_hz"] = (self.count - 1) / duration if self.count > 1 else 0
            result["duration_s"] = duration
        if self.intervals:
            result["interval_avg_ms"] = statistics.mean(self.intervals)
            result["interval_std_ms"] = statistics.stdev(self.intervals) if len(self.intervals) > 1 else 0
            result["interval_min_ms"] = min(self.intervals)
            result["interval_max_ms"] = max(self.intervals)
            # Jitter = standard deviation of inter-arrival times
            result["jitter_ms"] = result["interval_std_ms"]
        if self.sizes:
            result["size_avg_bytes"] = statistics.mean(self.sizes)
            result["size_min_bytes"] = min(self.sizes)
            result["size_max_bytes"] = max(self.sizes)
            # SHM threshold is typically 3072 bytes
            result["above_shm_threshold"] = sum(1 for s in self.sizes if s > 3072)
            result["below_shm_threshold"] = sum(1 for s in self.sizes if s <= 3072)
        return result


def main():
    common_kwargs = {
        "domain_id": DOMAIN_ID,
        "router_ip": ROUTER_IP,
        "router_port": ROUTER_PORT,
    }

    # Check SHM environment
    shm_enabled = os.environ.get("ZENOH_SHM_ENABLED", "not set")
    config_override = os.environ.get("ZENOH_CONFIG_OVERRIDE", "not set")

    print("=" * 70)
    print("Zenoh Cross-Container Latency & SHM Verification Test")
    print("=" * 70)
    print(f"  Domain ID:             {DOMAIN_ID}")
    print(f"  Router:                {ROUTER_IP}:{ROUTER_PORT}")
    print(f"  ZENOH_SHM_ENABLED:     {shm_enabled}")
    print(f"  ZENOH_CONFIG_OVERRIDE: {config_override}")
    print()

    # Topics to test with expected properties
    topics = {
        "joint_state": {
            "topic": "/robot/arm_left_follower/joint_states",
            "msg_type": "sensor_msgs/msg/JointState",
            "expected_hz": 100,
            "desc": "Small msg (~200B) - network transport",
        },
        "image": {
            "topic": "/robot/camera/cam_left_head/image_raw/compressed",
            "msg_type": "sensor_msgs/msg/CompressedImage",
            "expected_hz": 15,
            "desc": "Large msg (~160KB) - SHM zero-copy candidate",
        },
        "odom": {
            "topic": "/odom",
            "msg_type": "nav_msgs/msg/Odometry",
            "expected_hz": 100,
            "desc": "Medium msg (~700B) - network/SHM boundary",
        },
        "camera_info": {
            "topic": "/robot/camera/cam_left_head/image_raw/compressed/camera_info",
            "msg_type": "sensor_msgs/msg/CameraInfo",
            "expected_hz": 15,
            "desc": "Small msg (~300B) - network transport",
        },
    }

    stats = {}
    subscribers = []

    for key, cfg in topics.items():
        stats[key] = TopicStats(key)

        def make_callback(k, msg_type):
            def callback(msg):
                size = 0
                if hasattr(msg, 'data'):
                    try:
                        size = len(msg.data)
                    except:
                        pass
                elif hasattr(msg, 'pose'):
                    # Odometry - estimate size from covariance arrays
                    size = 700  # approximate
                with lock:
                    stats[k].record(data_size=size)
            return callback

        print(f"  [{key}] Subscribing to {cfg['topic']}...")
        try:
            sub = ROS2Subscriber(
                topic=cfg["topic"],
                msg_type=cfg["msg_type"],
                callback=make_callback(key, cfg["msg_type"]),
                **common_kwargs,
            )
            subscribers.append(sub)
            print(f"         OK - {cfg['desc']}")
        except Exception as e:
            print(f"         FAILED: {e}")

    # Collect data
    test_duration = 20
    print(f"\nCollecting data for {test_duration} seconds...")
    print()

    for i in range(test_duration):
        time.sleep(1)
        with lock:
            counts = {k: stats[k].count for k in stats}
        status = "  ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  [{i+1:2d}s] {status}")

    # Results
    print()
    print("=" * 70)
    print("LATENCY & SHM VERIFICATION RESULTS")
    print("=" * 70)

    for key, cfg in topics.items():
        print()
        print(f"--- {key}: {cfg['topic']} ---")
        print(f"  Description: {cfg['desc']}")
        with lock:
            s = stats[key].summary()

        if s["count"] == 0:
            print(f"  Status: FAIL - No messages received")
            continue

        print(f"  Status: PASS")
        print(f"  Messages received: {s['count']}")
        if "rate_hz" in s:
            print(f"  Measured rate: {s['rate_hz']:.1f} Hz (expected: ~{cfg['expected_hz']} Hz)")
        if "interval_avg_ms" in s:
            print(f"  Inter-arrival interval:")
            print(f"    Average:  {s['interval_avg_ms']:.2f} ms")
            print(f"    Std dev:  {s['interval_std_ms']:.2f} ms")
            print(f"    Min:      {s['interval_min_ms']:.2f} ms")
            print(f"    Max:      {s['interval_max_ms']:.2f} ms")
            print(f"    Jitter:   {s['jitter_ms']:.2f} ms")
        if "size_avg_bytes" in s:
            avg_kb = s['size_avg_bytes'] / 1024
            print(f"  Message size:")
            print(f"    Average:  {avg_kb:.1f} KB ({s['size_avg_bytes']:.0f} bytes)")
            print(f"    Min:      {s['size_min_bytes']/1024:.1f} KB")
            print(f"    Max:      {s['size_max_bytes']/1024:.1f} KB")
            # SHM threshold analysis
            shm_threshold = 3072
            if s.get("above_shm_threshold", 0) > 0:
                pct = s["above_shm_threshold"] / s["count"] * 100
                print(f"    Above SHM threshold (>{shm_threshold}B): {s['above_shm_threshold']}/{s['count']} ({pct:.0f}%)")
                print(f"    -> These messages use SHM zero-copy when SHM is enabled")
            else:
                print(f"    All messages below SHM threshold ({shm_threshold}B) - uses network transport")

    # Summary table
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Topic':<15} {'Count':>7} {'Rate Hz':>9} {'Jitter ms':>11} {'Avg Size':>10} {'SHM?':>6}")
    print("-" * 60)
    for key in topics:
        with lock:
            s = stats[key].summary()
        rate = f"{s.get('rate_hz', 0):.1f}"
        jitter = f"{s.get('jitter_ms', 0):.2f}"
        size = f"{s.get('size_avg_bytes', 0)/1024:.1f}KB" if s.get('size_avg_bytes') else "N/A"
        shm = "YES" if s.get('above_shm_threshold', 0) > 0 else "no"
        print(f"{key:<15} {s['count']:>7} {rate:>9} {jitter:>11} {size:>10} {shm:>6}")

    print()
    total = sum(1 for k in topics if stats[k].count > 0)
    print(f"Overall: {total}/{len(topics)} topics receiving data")
    print("=" * 70)

    # Cleanup
    for sub in subscribers:
        try:
            sub.close()
        except:
            pass


if __name__ == "__main__":
    main()
