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

"""Test RobotClient with rosbag replay.

Run this inside groot_server or lerobot_server container while
orchestrator replays a rosbag.

Usage:
    # In orchestrator:
    ros2 bag play /workspace/test_dataset --rate 1.0 --loop

    # In groot_server:
    python3 /workspace/test_rosbag_subscriber.py
"""
import sys
import os
import time
import logging
import json

# Setup paths
sys.path.insert(0, os.environ.get("ZENOH_SDK_PATH", "/zenoh_sdk"))
sys.path.insert(0, "/workspace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")

print("=" * 60)
print("RobotClient Rosbag Subscriber Test")
print("=" * 60)

# Import robot_client
from robot_client import RobotClient

# Use domain_id=0 to match orchestrator (which defaults to 0)
# groot container has ROS_DOMAIN_ID=30 but rosbag replay uses 0
DOMAIN_ID = 0

print(f"\n[1/6] Creating RobotClient (domain_id={DOMAIN_ID})...", flush=True)
t0 = time.time()
robot = RobotClient(
    "ffw_sg2_rev1",
    sync_check=False,
    router_ip="127.0.0.1",
    router_port=7447,
    domain_id=DOMAIN_ID,
)
print(f"  RobotClient created in {time.time()-t0:.2f}s", flush=True)

print(f"\n[2/6] Waiting for sensor data...", flush=True)
print(f"  Cameras: {robot.camera_names}", flush=True)
print(f"  Joint groups: {robot.joint_group_names}", flush=True)
print(f"  Total DOF: {robot.total_dof}", flush=True)

# Wait for data with detailed progress
ready = robot.wait_for_ready(timeout=30.0)
if ready:
    print("  All sensors ready!", flush=True)
else:
    print("  WARNING: Timeout. Checking partial data...", flush=True)

# Detailed status
print(f"\n[3/6] Data reception status:", flush=True)
status = robot.get_status()

cameras_ready = 0
cameras_total = len(status["cameras"])
print(f"\n  Cameras ({cameras_total}):", flush=True)
for name, info in status["cameras"].items():
    if info["ready"]:
        cameras_ready += 1
        print(f"    {name}: OK (shape={info['shape']})", flush=True)
    else:
        print(f"    {name}: MISSING", flush=True)

joints_ready = 0
joints_total = len(status["joint_groups"])
print(f"\n  Joint groups ({joints_total}):", flush=True)
for name, info in status["joint_groups"].items():
    if info["ready"]:
        joints_ready += 1
        print(f"    {name}: OK (dof={info['dof']})", flush=True)
    else:
        print(f"    {name}: MISSING", flush=True)

sensors_ready = 0
sensors_total = len(status["sensors"])
print(f"\n  Sensors ({sensors_total}):", flush=True)
for name, info in status["sensors"].items():
    if info["ready"]:
        sensors_ready += 1
        print(f"    {name}: OK", flush=True)
    else:
        print(f"    {name}: MISSING", flush=True)

print(f"\n  Summary: {cameras_ready}/{cameras_total} cameras, "
      f"{joints_ready}/{joints_total} joints, "
      f"{sensors_ready}/{sensors_total} sensors", flush=True)

# Check actual data values
print(f"\n[4/6] Data values:", flush=True)

images = robot.get_images()
for name, img in images.items():
    print(f"  Image {name}: shape={img.shape}, dtype={img.dtype}, "
          f"min={img.min()}, max={img.max()}", flush=True)

joints = robot.get_joint_positions()
for name, pos in joints.items():
    vals = [f"{v:.4f}" for v in pos[:4]]
    print(f"  Joint {name}: dof={len(pos)}, values=[{', '.join(vals)}...]", flush=True)

odom = robot.get_odom()
if odom:
    print(f"  Odom position: {odom['position']}", flush=True)
    print(f"  Odom orientation: {odom['orientation']}", flush=True)
    print(f"  Odom linear_velocity: {odom['linear_velocity']}", flush=True)

# Observation test
robot.set_task_instruction("pick up the red cup")
obs = robot.get_observation()
if obs:
    print(f"\n  Observation keys: {list(obs.keys())}", flush=True)
    print(f"  Images count: {len(obs['images'])}", flush=True)
    print(f"  Joint groups count: {len(obs['joint_positions'])}", flush=True)
    print(f"  Task instruction: '{obs['task_instruction']}'", flush=True)

    # Verify image format
    for name, img in obs['images'].items():
        print(f"    Image {name}: {img.shape} {img.dtype} (BGR)", flush=True)
        break

    # Verify with resize and rgb
    obs_rgb = robot.get_observation(resize=(256, 256), format="rgb")
    if obs_rgb:
        for name, img in obs_rgb['images'].items():
            print(f"    Image {name} (resized, rgb): {img.shape} {img.dtype}", flush=True)
            break

# Latency test
print(f"\n[5/6] Latency measurement:", flush=True)

import numpy as np

# get_observation latency
latencies = []
for i in range(100):
    t_start = time.time()
    obs = robot.get_observation()
    t_end = time.time()
    latencies.append((t_end - t_start) * 1000)

latencies = np.array(latencies)
print(f"  get_observation() latency (100 calls):", flush=True)
print(f"    Mean: {latencies.mean():.3f} ms", flush=True)
print(f"    Min:  {latencies.min():.3f} ms", flush=True)
print(f"    Max:  {latencies.max():.3f} ms", flush=True)
print(f"    P95:  {np.percentile(latencies, 95):.3f} ms", flush=True)

# get_images latency (with resize)
latencies_resize = []
for i in range(100):
    t_start = time.time()
    imgs = robot.get_images(resize=(256, 256), format="rgb")
    t_end = time.time()
    latencies_resize.append((t_end - t_start) * 1000)

latencies_resize = np.array(latencies_resize)
print(f"\n  get_images(resize=256, rgb) latency (100 calls):", flush=True)
print(f"    Mean: {latencies_resize.mean():.3f} ms", flush=True)
print(f"    Min:  {latencies_resize.min():.3f} ms", flush=True)
print(f"    Max:  {latencies_resize.max():.3f} ms", flush=True)
print(f"    P95:  {np.percentile(latencies_resize, 95):.3f} ms", flush=True)

# get_joint_positions latency
latencies_joints = []
for i in range(100):
    t_start = time.time()
    joints = robot.get_joint_positions()
    t_end = time.time()
    latencies_joints.append((t_end - t_start) * 1000)

latencies_joints = np.array(latencies_joints)
print(f"\n  get_joint_positions() latency (100 calls):", flush=True)
print(f"    Mean: {latencies_joints.mean():.3f} ms", flush=True)
print(f"    Min:  {latencies_joints.min():.3f} ms", flush=True)
print(f"    P95:  {np.percentile(latencies_joints, 95):.3f} ms", flush=True)

# Data freshness (how old is the latest data when we read it)
print(f"\n  Data freshness (age of latest sample):", flush=True)
time.sleep(0.1)  # let some data arrive
for cam in robot.camera_names[:2]:
    ts = robot.get_image_timestamp(cam)
    if ts:
        age_ms = (time.time() - ts) * 1000
        print(f"    Camera {cam}: {age_ms:.1f} ms old", flush=True)
for grp in list(robot.joint_group_names)[:2]:
    ts = robot.get_joint_timestamp(grp)
    if ts:
        age_ms = (time.time() - ts) * 1000
        print(f"    Joint {grp}: {age_ms:.1f} ms old", flush=True)

# Continuous reception test
print(f"\n[6/6] Continuous reception test (5s):", flush=True)
img_updates = {cam: 0 for cam in robot.camera_names}
jnt_updates = {grp: 0 for grp in robot.joint_group_names}
sensor_updates = {s: 0 for s in ["odom", "cmd_vel"]}

prev_img_ts = {cam: robot.get_image_timestamp(cam) or 0 for cam in robot.camera_names}
prev_jnt_ts = {grp: robot.get_joint_timestamp(grp) or 0 for grp in robot.joint_group_names}

t_start = time.time()
while time.time() - t_start < 5.0:
    for cam in robot.camera_names:
        ts = robot.get_image_timestamp(cam) or 0
        if ts > prev_img_ts[cam]:
            img_updates[cam] += 1
            prev_img_ts[cam] = ts
    for grp in robot.joint_group_names:
        ts = robot.get_joint_timestamp(grp) or 0
        if ts > prev_jnt_ts[grp]:
            jnt_updates[grp] += 1
            prev_jnt_ts[grp] = ts
    time.sleep(0.005)

print(f"  Camera update rates (5s window):", flush=True)
for cam, cnt in img_updates.items():
    hz = cnt / 5.0
    print(f"    {cam}: {cnt} updates ({hz:.1f} Hz)", flush=True)

print(f"  Joint update rates (5s window):", flush=True)
for grp, cnt in jnt_updates.items():
    hz = cnt / 5.0
    print(f"    {grp}: {cnt} updates ({hz:.1f} Hz)", flush=True)

# Summary
print(f"\n{'=' * 60}", flush=True)
print(f"TEST RESULTS SUMMARY", flush=True)
print(f"{'=' * 60}", flush=True)
all_ok = cameras_ready == cameras_total and joints_ready == joints_total and sensors_ready == sensors_total
print(f"  All sensors receiving: {'YES' if all_ok else 'NO'}", flush=True)
print(f"  Cameras: {cameras_ready}/{cameras_total}", flush=True)
print(f"  Joints:  {joints_ready}/{joints_total}", flush=True)
print(f"  Sensors: {sensors_ready}/{sensors_total}", flush=True)
print(f"  get_observation latency: {latencies.mean():.3f} ms avg", flush=True)
print(f"  get_images(256,rgb) latency: {latencies_resize.mean():.3f} ms avg", flush=True)
print(f"  get_joint_positions latency: {latencies_joints.mean():.3f} ms avg", flush=True)
print(f"{'=' * 60}", flush=True)

robot.close()
