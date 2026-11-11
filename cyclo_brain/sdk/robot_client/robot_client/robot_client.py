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
RobotClient - High-level abstraction for robot sensor data and control.

Provides simple Python API over zenoh_ros2_sdk, hiding all Zenoh/ROS2 details.
Users only need to specify robot type to get automatic topic subscription.
"""
import os
import sys
import time
import threading
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import cv2

# Add zenoh_ros2_sdk to path if not already available
_SDK_PATH = os.environ.get("ZENOH_SDK_PATH", "")
if _SDK_PATH and _SDK_PATH not in sys.path:
    sys.path.insert(0, _SDK_PATH)

from zenoh_ros2_sdk import ROS2Publisher, ROS2Subscriber, get_message_class  # noqa: E402


# -- robot config schema helper -----------------------------------------------
# shared/robot_configs/ is bind-mounted into the policy container at
# /orchestrator_config/, so schema.py lands beside the per-robot yamls.
# The module is intentionally self-contained (no `shared` package
# imports) so it can be picked up as a standalone file from that mount.
_SCHEMA_DIR = os.environ.get("ORCHESTRATOR_CONFIG_PATH", "/orchestrator_config")
if os.path.isdir(_SCHEMA_DIR) and _SCHEMA_DIR not in sys.path:
    sys.path.insert(0, _SCHEMA_DIR)
try:
    import schema as robot_schema  # type: ignore[import-not-found]
except ImportError:
    _src = Path(__file__).resolve()
    for _parent in _src.parents:
        _cand = _parent / "shared" / "robot_configs"
        if _cand.is_dir():
            sys.path.insert(0, str(_cand))
            break
    import schema as robot_schema  # type: ignore[import-not-found]


logger = logging.getLogger("robot_client")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _deadband(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def _build_runtime_config(section: dict) -> dict:
    """Translate the VLA-semantic schema into the cameras/joint_groups/
    sensors shape the RobotClient + inference engines consume.

    * ``observation.images``  → ``cameras``.
    * ``observation.state.<g>`` with JointState msg_type → physical
      follower joint group named ``follower_<g>``.
    * ``observation.state.<g>`` with Odometry msg_type   → ``sensors["odom"]``
      (treated as a sensor-backed state modality by GR00T).
    * Each ``action.<modality>`` (excluding mobile/Twist) gets a SYNTHETIC
      ``follower_<modality>`` joint_group with ``parent`` pointing at the
      first physical follower; ``_update_joint`` slices the parent's
      message by the action's ``joint_names`` to populate it.
    """
    cameras = robot_schema.get_image_topics(section)
    state_groups = robot_schema.get_state_groups(section)
    action_groups = robot_schema.get_action_groups(section)

    joint_groups: dict = {}
    sensors: dict = {}
    physical_follower_name: Optional[str] = None

    for name, cfg in state_groups.items():
        msg_type = cfg["msg_type"]
        if msg_type == "sensor_msgs/msg/JointState":
            group_name = f"follower_{name}"
            joint_groups[group_name] = {
                "topic": cfg["topic"],
                "msg_type": msg_type,
                "role": "follower",
                "joint_names": list(cfg["joint_names"]),
            }
            if physical_follower_name is None:
                physical_follower_name = group_name
        elif msg_type == "nav_msgs/msg/Odometry":
            sensors["odom"] = {
                "topic": cfg["topic"],
                "msg_type": msg_type,
            }
        else:
            # Unknown state msg_type — keep it in joint_groups for
            # diagnostics; subscribers will pick it up via the generic
            # JointState callback unless a more specific shape lands.
            joint_groups[f"follower_{name}"] = {
                "topic": cfg["topic"],
                "msg_type": msg_type,
                "role": "follower",
                "joint_names": list(cfg["joint_names"]),
            }

    if physical_follower_name is not None:
        for modality, cfg in action_groups.items():
            if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                # action.mobile is command-only; observation RobotClient
                # instances stay read-only.
                continue
            child_name = f"follower_{modality}"
            if child_name in joint_groups:
                # A physical follower already covers this modality.
                continue
            joint_groups[child_name] = {
                "parent": physical_follower_name,
                "role": "follower",
                "joint_names": list(cfg["joint_names"]),
            }

    return {
        "cameras": cameras,
        "joint_groups": joint_groups,
        "sensors": sensors,
    }


# Compatibility re-export for older engine code. Same VLA-semantic section in,
# same runtime-config dict out.
def derive_robot_config(section: dict) -> dict:
    return _build_runtime_config(section)


class RobotClient:
    """High-level robot interface over zenoh_ros2_sdk.

    Usage:
        robot = RobotClient("ffw_sg2_rev1")
        robot.wait_for_ready(timeout=10.0)
        images = robot.get_images()
        joints = robot.get_joint_positions()
    """

    def __init__(
        self,
        robot_type: str,
        sync_check: bool = False,
        sync_threshold_ms: float = 33.0,
        router_ip: str = "127.0.0.1",
        router_port: int = 7447,
        domain_id: Optional[int] = None,
        enable_command_publishers: bool = False,
        enable_preview_publisher: bool = False,
        subscribe_images: bool = True,
        subscribe_state: bool = True,
        subscribe_sensors: bool = True,
        defer_subscriptions: bool = False,
    ):
        section = robot_schema.load_robot_section(robot_type)
        # Phase 4: yaml is VLA-semantic (observation.images / state +
        # action.<modality>). _build_runtime_config translates that into
        # the cameras / joint_groups / sensors shape RobotClient and the
        # downstream inference engines have always consumed.
        self._config = _build_runtime_config(section)

        self._robot_type = robot_type
        self._sync_check = sync_check
        self._sync_threshold_ms = sync_threshold_ms
        self._router_ip = router_ip
        self._router_port = router_port
        self._domain_id = domain_id
        self._enable_command_publishers = bool(enable_command_publishers)
        self._enable_preview_publisher = bool(enable_preview_publisher)
        self._subscribe_images = bool(subscribe_images)
        self._subscribe_state = bool(subscribe_state)
        self._subscribe_sensors = bool(subscribe_sensors)
        self._defer_subscriptions = bool(defer_subscriptions)
        self._subscription_selection = None
        self._action_groups = robot_schema.get_action_groups(section)

        # Thread-safe data stores
        self._lock = threading.Lock()
        self._observation_capture = None
        self._observation_sequence = 0
        self._images: dict[str, np.ndarray] = {}
        self._image_timestamps: dict[str, float] = {}
        self._joint_positions: dict[str, np.ndarray] = {}
        self._joint_velocities: dict[str, np.ndarray] = {}
        self._joint_efforts: dict[str, np.ndarray] = {}
        self._joint_timestamps: dict[str, float] = {}
        self._joint_positions_by_name: dict[str, float] = {}
        self._joint_position_timestamps_by_name: dict[str, float] = {}
        self._sensors: dict[str, dict] = {}
        self._sensor_timestamps: dict[str, float] = {}
        self._task_instruction: str = ""

        self._subscribers: list = []
        self._state_subscribers: list = []
        self._command_publishers: dict[str, ROS2Publisher] = {}
        self._preview_publisher: Optional[ROS2Publisher] = None
        self._command_msg_types: dict[str, str] = {}
        self._command_joint_names: dict[str, list[str]] = {}
        self._action_keys = sorted(self._action_groups.keys())
        self._cmd_vel_linear_deadband = max(
            0.0,
            _float_env("CMD_VEL_LINEAR_DEADBAND", 0.0),
        )
        self._cmd_vel_angular_deadband = max(
            0.0,
            _float_env("CMD_VEL_ANGULAR_DEADBAND", 0.0),
        )
        self._initial_pose_sync_state_max_age_s = _float_env(
            "INITIAL_POSE_SYNC_STATE_MAX_AGE_S",
            1.0,
        )
        if (
            not math.isfinite(self._initial_pose_sync_state_max_age_s)
            or self._initial_pose_sync_state_max_age_s <= 0.0
        ):
            logger.warning(
                "INITIAL_POSE_SYNC_STATE_MAX_AGE_S must be positive and finite; "
                "using 1.0"
            )
            self._initial_pose_sync_state_max_age_s = 1.0
        self._closed = False

        self._init_subscriptions()
        if self._enable_command_publishers:
            self._init_command_publishers()
            if self._cmd_vel_linear_deadband or self._cmd_vel_angular_deadband:
                logger.info(
                    "cmd_vel deadband enabled: linear=%s angular=%s",
                    self._cmd_vel_linear_deadband,
                    self._cmd_vel_angular_deadband,
                )
        if self._enable_preview_publisher:
            self._init_preview_publisher()
        logger.info(f"RobotClient initialized: {robot_type} "
                     f"({len(self._config.get('cameras', {}))} cameras, "
                     f"{len(self._config.get('joint_groups', {}))} joint groups)")

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _selected_names(self, kind):
        selection = getattr(self, "_subscription_selection", None)
        return self._config.get(kind, {}) if selection is None else selection[kind]

    def start_observation_subscriptions(self, *, camera_names, joint_groups, sensor_names):
        """Start a deferred client's declared inputs once, after LOAD mapping.

        This is not live reconfiguration. A new LOAD creates a new RobotClient;
        command-publishing clients keep their existing default subscriptions.
        """
        if self._closed or not self._defer_subscriptions:
            raise RuntimeError("observation subscriptions require a fresh deferred client")
        selected = {"cameras": frozenset(camera_names), "joint_groups": frozenset(joint_groups),
                    "sensors": frozenset(sensor_names)}
        enabled = {"cameras": self._subscribe_images, "joint_groups": self._subscribe_state,
                   "sensors": self._subscribe_sensors}
        for kind, names in selected.items():
            missing = names - self._config.get(kind, {}).keys()
            if missing or (names and not enabled[kind]):
                raise ValueError(f"Invalid observation subscription selection {kind}: {sorted(names)}")
        capture = self._observation_capture
        if capture is not None:
            prefixes = {"camera": "cameras", "joint": "joint_groups", "sensor": "sensors"}
            for source in capture.sources:
                kind, name = source.split(":", 1)
                name = name.split(".", 1)[0] if kind == "sensor" else name
                if name not in selected[prefixes[kind]]:
                    raise ValueError(f"subscription selection omits captured source: {source}")
        self._subscription_selection = selected
        self._defer_subscriptions = False
        try:
            self._init_subscriptions()
        except Exception:
            self.close()
            raise

    def configure_joint_views(self, views: dict) -> None:
        """Register selected named slices before subscriptions/history are started."""
        if self._closed or not self._defer_subscriptions or self._observation_capture is not None:
            raise RuntimeError("Joint views require a fresh deferred observation client")
        groups = self._config["joint_groups"]
        additions = {}
        for name, view in views.items():
            parent = groups.get(view["parent"], {})
            names = list(view["joint_names"])
            if (name in groups or parent.get("parent") or parent.get("msg_type") != "sensor_msgs/msg/JointState"
                    or not names or len(set(names)) != len(names)
                    or not set(names) <= set(parent.get("joint_names", []))):
                raise ValueError(f"Invalid named observation view: {name}")
            additions[name] = {"parent": view["parent"], "role": "follower", "joint_names": names,
                               "strict_named": True}
        groups.update(additions)

    def _init_subscriptions(self):
        """Subscribe to configured topics, or the declared LOAD selection.

        Joint groups carrying a ``parent`` field have no physical topic of
        their own — they're synthetic per-modality views over a sibling
        group's data. ``_update_joint`` propagates from parent → children
        by name-based slicing inside the callback.
        """
        # Cameras
        if self._subscribe_images and not self._defer_subscriptions:
            for cam_name, cam_cfg in self._config.get("cameras", {}).items():
                if cam_name not in self._selected_names("cameras"):
                    continue
                sub = ROS2Subscriber(
                    topic=cam_cfg["topic"],
                    msg_type=cam_cfg["msg_type"],
                    callback=lambda msg, name=cam_name: self._update_image(name, msg),
                )
                self._subscribers.append(sub)
                logger.debug(f"Subscribed camera: {cam_name} -> {cam_cfg['topic']}")

        # Index parent → list of child group names so the upper-body
        # callback knows which slices to populate per message.
        self._joint_children: dict[str, list[str]] = {}
        for child_name, child_cfg in self._config.get("joint_groups", {}).items():
            parent = child_cfg.get("parent")
            if parent and child_name in self._selected_names("joint_groups"):
                self._joint_children.setdefault(parent, []).append(child_name)

        if self._subscribe_state and not self._defer_subscriptions:
            self._init_state_subscriptions()

        # Additional sensors. ``sensor_cfg`` may carry an optional
        # ``type_hash`` override — escape hatch for messages where
        # zenoh_ros2_sdk's hash computation needs to be pinned to a known
        # wire hash. Default is auto-compute via the SDK.
        if self._subscribe_sensors and not self._defer_subscriptions:
            for sensor_name, sensor_cfg in self._config.get("sensors", {}).items():
                if sensor_name not in self._selected_names("sensors"):
                    continue
                sub_kwargs = dict(
                    topic=sensor_cfg["topic"],
                    msg_type=sensor_cfg["msg_type"],
                    callback=lambda msg, name=sensor_name: self._update_sensor(name, msg),
                )
                if sensor_cfg.get("type_hash"):
                    sub_kwargs["type_hash"] = sensor_cfg["type_hash"]
                sub = ROS2Subscriber(**sub_kwargs)
                self._subscribers.append(sub)
                logger.debug(f"Subscribed sensor: {sensor_name} -> {sensor_cfg['topic']}")

    def _init_state_subscriptions(self) -> None:
        if self._state_subscribers:
            return
        groups = self._config.get("joint_groups", {})
        physical = {groups[name].get("parent") or name for name in self._selected_names("joint_groups")}
        topics: dict[tuple[str, str], list[str]] = {}
        for group_name, group_cfg in self._config.get("joint_groups", {}).items():
            if group_cfg.get("parent"):
                logger.debug(
                    f"Skipped joint subscription: {group_name} "
                    f"(synthetic view of {group_cfg['parent']})"
                )
                continue
            if group_name not in physical:
                continue
            topics.setdefault((group_cfg["topic"], group_cfg["msg_type"]), []).append(group_name)
        for (topic, msg_type), group_names in topics.items():
            subscriber = ROS2Subscriber(
                topic=topic,
                msg_type=msg_type,
                callback=lambda msg, names=tuple(group_names): self._update_joint_groups(names, msg),
            )
            self._state_subscribers.append(subscriber)
            self._subscribers.append(subscriber)
            logger.debug(f"Subscribed joints: {group_names} -> {topic}")

    def set_state_subscription(self, enabled: bool) -> None:
        """Enable or disable the joint-state subscription used for safe hold."""
        if self._defer_subscriptions:
            raise RuntimeError("start declared observation subscriptions before toggling state")
        enabled = bool(enabled)
        if enabled == self._subscribe_state:
            return
        self._subscribe_state = enabled
        if enabled:
            self._init_state_subscriptions()
            return

        subscribers = list(self._state_subscribers)
        self._state_subscribers.clear()
        self._subscribers = [
            subscriber
            for subscriber in self._subscribers
            if subscriber not in subscribers
        ]
        for subscriber in subscribers:
            try:
                subscriber.close()
            except Exception as exc:
                logger.debug(f"Error closing state subscriber: {exc}")
        with self._lock:
            self._joint_positions.clear()
            self._joint_velocities.clear()
            self._joint_efforts.clear()
            self._joint_timestamps.clear()
            self._joint_positions_by_name.clear()
            self._joint_position_timestamps_by_name.clear()

    def _init_command_publishers(self):
        """Create publishers for configured action topics."""
        common = {
            "router_ip": self._router_ip,
            "router_port": self._router_port,
        }
        if self._domain_id is not None:
            common["domain_id"] = self._domain_id

        for action_key in self._action_keys:
            cfg = self._action_groups[action_key]
            publisher_key = f"leader_{action_key}"
            self._command_msg_types[publisher_key] = cfg["msg_type"]
            self._command_joint_names[publisher_key] = list(cfg.get("joint_names", []))
            self._command_publishers[publisher_key] = ROS2Publisher(
                topic=cfg["topic"],
                msg_type=cfg["msg_type"],
                **common,
            )
            logger.debug(
                "Command publisher: %s -> %s (%s)",
                publisher_key,
                cfg["topic"],
                cfg["msg_type"],
            )

    def _init_preview_publisher(self):
        """Create a unified trajectory preview publisher for the 3D viewer."""
        common = {
            "router_ip": self._router_ip,
            "router_port": self._router_port,
        }
        if self._domain_id is not None:
            common["domain_id"] = self._domain_id
        self._preview_publisher = ROS2Publisher(
            topic="/inference/trajectory_preview",
            msg_type="trajectory_msgs/msg/JointTrajectory",
            **common,
        )
        logger.debug("Action preview publisher: /inference/trajectory_preview")

    # ------------------------------------------------------------------ #
    # Callback handlers
    # ------------------------------------------------------------------ #

    def attach_observation_capture(self, capture) -> None:
        """Attach one opt-in numeric reception sink, without replaying cached data.

        The sink exposes frozen ``sources`` and a non-throwing ``record`` method.
        record runs under the data lock and must only copy/store bounded samples:
        no inference, network I/O or calls back into this client. Sink failures
        must be latched and surfaced by its reader, not ignored.
        """
        available = set()
        if self._subscribe_images:
            available.update(f"camera:{key}" for key in self._selected_names("cameras"))
        if self._subscribe_state:
            available.update(f"joint:{key}" for key in self._selected_names("joint_groups"))
        if self._subscribe_sensors:
            fields = {
                "odom": ("position", "orientation", "linear_velocity", "angular_velocity"),
                "cmd_vel": ("linear", "angular"),
            }
            for sensor in self._selected_names("sensors"):
                available.update(f"sensor:{sensor}.{key}" for key in fields.get(sensor, ()))
        if not isinstance(capture.sources, frozenset) or not callable(capture.record):
            raise TypeError("capture requires frozen sources and record()")
        missing = capture.sources - available
        if missing:
            raise ValueError(f"Unavailable observation capture sources: {sorted(missing)}")
        with self._lock:
            if self._closed:
                raise RuntimeError("RobotClient is closed")
            if self._observation_capture is not None:
                raise RuntimeError("An observation capture is already attached")
            self._observation_capture = capture

    def detach_observation_capture(self, capture) -> None:
        """After return, no subscriber callback can append to this sink."""
        with self._lock:
            if self._observation_capture is capture:
                self._observation_capture = None

    def reset_observation_capture(self, capture) -> None:
        """Reset a session's history at a subscriber callback barrier."""
        with self._lock:
            if self._observation_capture is not capture:
                raise RuntimeError("The observation capture is not attached")
            capture.reset()

    def _capture_observation(self, source: str, value: np.ndarray) -> None:
        # Called under _lock. This time identifies decoded data becoming available,
        # not a sensor hardware timestamp or a later latest-value read.
        received_s = time.monotonic()
        if not hasattr(self, "_input_received_monotonic"):
            self._input_received_monotonic = {}
        self._input_received_monotonic[source] = received_s
        capture = getattr(self, "_observation_capture", None)
        if capture is None or source not in capture.sources:
            return
        self._observation_sequence += 1
        if source.startswith("camera:"):
            value = value[..., ::-1]  # Capture RGB; the existing latest store stays BGR.
        capture.record(source, self._observation_sequence, received_s, value)

    def _update_image(self, cam_name: str, msg):
        """CompressedImage -> BGR numpy array."""
        try:
            data = msg.data
            if isinstance(data, (list, tuple)):
                data = bytes(data)
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                with self._lock:
                    self._images[cam_name] = img
                    self._image_timestamps[cam_name] = time.time()
                    self._capture_observation(f"camera:{cam_name}", img)
        except Exception as e:
            logger.warning(f"Failed to decode image from {cam_name}: {e}")

    def _update_joint(self, group_name: str, msg):
        """Update one physical joint group and its synthetic views."""
        self._update_joint_groups((group_name,), msg)

    def _update_joint_groups(self, group_names: tuple[str, ...], msg):
        """Parse one physical message, then atomically update all logical views.

        Physical groups retain the full vector; only synthetic children slice
        by joint name. Stored arrays are replaced, never modified in place.
        """
        try:
            msg_names = list(msg.name) if hasattr(msg, 'name') else []
            position = list(msg.position) if hasattr(msg.position, '__iter__') else []
            velocity = list(msg.velocity) if hasattr(msg.velocity, '__iter__') else []
            effort = list(msg.effort) if hasattr(msg.effort, '__iter__') else []
            now = time.time()
            received_monotonic = time.monotonic()
            position_array = np.array(position, dtype=np.float32) if position else None
            velocity_array = np.array(velocity, dtype=np.float32) if velocity else None
            effort_array = np.array(effort, dtype=np.float32) if effort else None
            with self._lock:
                if position and msg_names:
                    self._joint_positions_by_name.update(
                        {name: float(value) for name, value in zip(msg_names, position)}
                    )
                    self._joint_position_timestamps_by_name.update(
                        {name: received_monotonic for name, _value in zip(msg_names, position)}
                    )
                for group_name in group_names:
                    if position_array is not None:
                        self._joint_positions[group_name] = position_array
                        self._capture_observation(f"joint:{group_name}", position_array)
                    if velocity_array is not None:
                        self._joint_velocities[group_name] = velocity_array
                    if effort_array is not None:
                        self._joint_efforts[group_name] = effort_array
                    self._joint_timestamps[group_name] = now

                # Propagate to synthetic child views.
                children = [child for group_name in group_names
                            for child in getattr(self, "_joint_children", {}).get(group_name, [])]
                if children:
                    name_to_idx = {n: i for i, n in enumerate(msg_names)}
                    for child in children:
                        child_cfg = self._config["joint_groups"].get(child, {})
                        wanted = child_cfg.get("joint_names", [])
                        strict = child_cfg.get("strict_named", False)
                        if not msg_names and not strict:
                            continue
                        if strict and (len(msg_names) != len(position) or len(set(msg_names)) != len(msg_names)):
                            self._joint_positions.pop(child, None)
                            self._joint_timestamps.pop(child, None)
                            continue
                        try:
                            indices = [name_to_idx[n] for n in wanted]
                        except KeyError as missing:
                            if strict:
                                self._joint_positions.pop(child, None)
                                self._joint_timestamps.pop(child, None)
                            # First few callbacks may race ahead of full
                            # name list — skip this child until the parent
                            # message carries every joint we expect.
                            logger.debug(
                                f"{child}: joint {missing} missing from "
                                f"{group_names} message"
                            )
                            continue
                        if strict and not np.isfinite(position_array[indices]).all():
                            self._joint_positions.pop(child, None)
                            self._joint_timestamps.pop(child, None)
                            continue
                        if position:
                            self._joint_positions[child] = np.array(
                                [position[i] for i in indices], dtype=np.float32
                            )
                        if velocity and len(velocity) == len(msg_names):
                            self._joint_velocities[child] = np.array(
                                [velocity[i] for i in indices], dtype=np.float32
                            )
                        if effort and len(effort) == len(msg_names):
                            self._joint_efforts[child] = np.array(
                                [effort[i] for i in indices], dtype=np.float32
                            )
                        self._joint_timestamps[child] = now
                        if position:
                            self._capture_observation(f"joint:{child}", self._joint_positions[child])
        except Exception as e:
            logger.warning(f"Failed to parse joint from {group_names}: {e}")

    def _update_sensor(self, sensor_name: str, msg):
        """Parse sensor messages (Odometry, Twist, etc.)."""
        try:
            data = {}
            if sensor_name == "odom":
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                lin = msg.twist.twist.linear
                ang = msg.twist.twist.angular
                data = {
                    "position": np.array([pos.x, pos.y, pos.z], dtype=np.float32),
                    "orientation": np.array([ori.x, ori.y, ori.z, ori.w], dtype=np.float32),
                    "linear_velocity": np.array([lin.x, lin.y, lin.z], dtype=np.float32),
                    "angular_velocity": np.array([ang.x, ang.y, ang.z], dtype=np.float32),
                }
            elif sensor_name == "cmd_vel":
                data = {
                    "linear": np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32),
                    "angular": np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float32),
                }
            else:
                data = {"raw": str(msg)}

            with self._lock:
                self._sensors[sensor_name] = data
                self._sensor_timestamps[sensor_name] = time.time()
                if not hasattr(self, "_input_received_monotonic"):
                    self._input_received_monotonic = {}
                self._input_received_monotonic[f"sensor:{sensor_name}"] = time.monotonic()
                for field, value in data.items():
                    if isinstance(value, np.ndarray):
                        self._capture_observation(f"sensor:{sensor_name}.{field}", value)
        except Exception as e:
            logger.warning(f"Failed to parse sensor {sensor_name}: {e}")

    # ------------------------------------------------------------------ #
    # Image API
    # ------------------------------------------------------------------ #

    @property
    def camera_names(self) -> list[str]:
        return list(self._config.get("cameras", {}).keys())

    def get_images(
        self,
        resize: Optional[tuple[int, int]] = None,
        format: str = "bgr",
    ) -> dict[str, np.ndarray]:
        """Get all camera images.

        Args:
            resize: Optional (width, height) tuple. None = original size.
            format: "bgr" (default) or "rgb".
        """
        with self._lock:
            result = {k: v.copy() for k, v in self._images.items()}
        if resize:
            result = {k: cv2.resize(v, resize) for k, v in result.items()}
        if format == "rgb":
            result = {k: cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for k, v in result.items()}
        return result

    def get_input_snapshot(self) -> dict:
        return self.get_required_input_snapshot()

    def get_required_input_snapshot(
        self, sources=None, *, max_age_s=None, after_s=None, max_age_by_source=None,
        readiness_check=None,
    ) -> dict:
        """Copy the latest inputs under one lock, without temporal alignment.

        Sensor timestamps retain their existing wall-clock reception semantics.
        captured_monotonic_s describes this read. reception_monotonic_timestamps
        describes actual callback updates, also when optional history is disabled.
        With explicit sources, validate readiness before copying/converting any
        pixels. Callbacks replace image arrays, so snapshot references survive
        subsequent updates; the RGB conversion makes the caller-owned copy.
        readiness_check may inspect bounded capture metadata at the same clock
        anchor. It runs under the callback lock and must not reenter RobotClient,
        perform model work, or do network I/O.
        """
        if max_age_s is not None and (not math.isfinite(max_age_s) or max_age_s <= 0):
            raise ValueError("max_age_s must be finite and positive")
        if after_s is not None and (not math.isfinite(after_s) or after_s < 0):
            raise ValueError("after_s must be a non-negative monotonic timestamp")
        requested = None if sources is None else frozenset(sources)
        ages = {} if max_age_by_source is None else dict(max_age_by_source)
        if any(age is not None and (type(age) not in (float, int)
                                   or not math.isfinite(age) or age <= 0) for age in ages.values()):
            raise ValueError("per-source maximum ages must be positive and finite or None")
        if requested is None and (max_age_s is not None or after_s is not None or ages):
            raise ValueError("readiness checks require explicit input sources")
        if requested is not None and not ages.keys() <= requested:
            raise ValueError("maximum age specified for an unrequested source")
        if readiness_check is not None and not callable(readiness_check):
            raise TypeError("readiness_check must be callable")
        with self._lock:
            captured = time.monotonic()
            received = getattr(self, "_input_received_monotonic", {})
            stores = {"camera": self._images, "joint": self._joint_positions, "sensor": self._sensors}
            if requested is not None:
                for source in sorted(requested):
                    kind, _, name = source.partition(":")
                    parent, dot, field = name.partition(".") if kind == "sensor" else (name, "", "")
                    if kind not in stores or parent not in stores[kind]:
                        raise ValueError(f"Missing input source: {source}")
                    if dot and field not in stores[kind][parent]:
                        raise ValueError(f"Missing input source: {source}")
                    age = ages.get(source, max_age_s)
                    if age is not None or after_s is not None or source in ages:
                        stamp = received.get(source)
                        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                                or stamp < 0 or stamp > captured):
                            raise ValueError(f"{source}: missing or invalid reception timestamp")
                        if after_s is not None and stamp <= after_s:
                            raise ValueError(f"{source}: observation predates publication barrier {after_s:g}")
                        if age is not None and captured - stamp > age:
                            raise ValueError(f"{source}: stale observation age={captured - stamp:.3f}s")
            if readiness_check is not None:
                readiness_check(captured)

            def selected(kind, values):
                return {key: value for key, value in values.items()
                        if requested is None or f"{kind}:{key}" in requested
                        or (kind == "sensor" and any(source.startswith(f"sensor:{key}.") for source in requested))}

            images = selected("camera", self._images)
            joints = {key: value.copy() for key, value in selected("joint", self._joint_positions).items()}
            sensors = deepcopy(selected("sensor", self._sensors))
            timestamps = {
                "images": selected("camera", self._image_timestamps),
                "joints": selected("joint", self._joint_timestamps),
                "sensors": selected("sensor", self._sensor_timestamps),
            }
            received = {key: value for key, value in received.items()
                        if requested is None or key in requested}
        return {
            "images": {key: cv2.cvtColor(value, cv2.COLOR_BGR2RGB) for key, value in images.items()},
            "joint_positions": joints,
            "sensors": sensors,
            "reception_wall_timestamps": timestamps,
            "captured_monotonic_s": captured,
            "reception_monotonic_timestamps": received,
        }

    def get_image(
        self,
        camera_name: str,
        resize: Optional[tuple[int, int]] = None,
        format: str = "bgr",
    ) -> Optional[np.ndarray]:
        """Get single camera image."""
        with self._lock:
            img = self._images.get(camera_name)
            if img is None:
                return None
            img = img.copy()
        if resize:
            img = cv2.resize(img, resize)
        if format == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def is_image_ready(self, camera_name: str) -> bool:
        with self._lock:
            return camera_name in self._images

    def get_image_timestamp(self, camera_name: str) -> Optional[float]:
        with self._lock:
            return self._image_timestamps.get(camera_name)

    # ------------------------------------------------------------------ #
    # Joint API
    # ------------------------------------------------------------------ #

    @property
    def joint_group_names(self) -> list[str]:
        return list(self._config.get("joint_groups", {}).keys())

    @property
    def total_dof(self) -> int:
        return self._config.get("total_dof", 0)

    def get_joint_names(self, group_name: str) -> list[str]:
        cfg = self._config.get("joint_groups", {}).get(group_name, {})
        return cfg.get("joint_names", [])

    def get_dof(self, group_name: str) -> int:
        cfg = self._config.get("joint_groups", {}).get(group_name, {})
        return cfg.get("dof", 0)

    def get_joint_positions(
        self, group: Optional[str] = None
    ) -> Union[dict[str, np.ndarray], np.ndarray]:
        """Get joint positions. Returns dict if no group, or np.ndarray for specific group."""
        with self._lock:
            if group:
                arr = self._joint_positions.get(group)
                return arr.copy() if arr is not None else np.array([], dtype=np.float32)
            return {k: v.copy() for k, v in self._joint_positions.items()}

    def get_named_joint_positions(self, joint_names: list[str], max_age_s: float = 1.0) -> dict[str, float]:
        """Read a complete, finite, fresh snapshot by actual joint name."""
        with self._lock:
            now = time.monotonic()
            missing = [name for name in joint_names
                       if name not in self._joint_positions_by_name
                       or name not in self._joint_position_timestamps_by_name
                       or not 0.0 <= now - self._joint_position_timestamps_by_name[name] <= max_age_s]
            if missing:
                raise RuntimeError("Waiting for fresh joint state: " + ", ".join(missing))
            positions = {name: float(self._joint_positions_by_name[name]) for name in joint_names}
        if not all(math.isfinite(value) for value in positions.values()):
            raise RuntimeError("Joint state contains non-finite positions.")
        return positions

    def get_joint_velocities(
        self, group: Optional[str] = None
    ) -> Union[dict[str, np.ndarray], np.ndarray]:
        with self._lock:
            if group:
                arr = self._joint_velocities.get(group)
                return arr.copy() if arr is not None else np.array([], dtype=np.float32)
            return {k: v.copy() for k, v in self._joint_velocities.items()}

    def get_joint_efforts(
        self, group: Optional[str] = None
    ) -> Union[dict[str, np.ndarray], np.ndarray]:
        with self._lock:
            if group:
                arr = self._joint_efforts.get(group)
                return arr.copy() if arr is not None else np.array([], dtype=np.float32)
            return {k: v.copy() for k, v in self._joint_efforts.items()}

    def is_joint_ready(self, group_name: str) -> bool:
        with self._lock:
            return group_name in self._joint_positions

    def get_joint_timestamp(self, group_name: str) -> Optional[float]:
        with self._lock:
            return self._joint_timestamps.get(group_name)

    # ------------------------------------------------------------------ #
    # Sensor API
    # ------------------------------------------------------------------ #

    def get_odom(self) -> Optional[dict]:
        with self._lock:
            return self._sensors.get("odom")

    def is_sensor_ready(self, sensor_name: str) -> bool:
        with self._lock:
            return sensor_name in self._sensors

    # ------------------------------------------------------------------ #
    # Command API
    # ------------------------------------------------------------------ #

    @property
    def action_keys(self) -> list[str]:
        return list(self._action_keys)

    def publish_action(self, action: np.ndarray, action_keys: Optional[list[str]] = None) -> None:
        """Publish one flat action vector to the robot command topics.

        Main process control loops use this method. Engine process instances keep
        ``enable_command_publishers=False`` and remain read-only.
        """
        if not self._command_publishers:
            raise RuntimeError("RobotClient command publishers are not enabled")

        keys = list(action_keys) if action_keys else self._action_keys
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        offset = 0
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None:
                continue
            publisher_key = f"leader_{publish_key}"
            msg_type = cfg["msg_type"]
            width = 3 if msg_type == "geometry_msgs/msg/Twist" else len(cfg["joint_names"])
            segment = values[offset:offset + width]
            offset += width

            publisher = self._command_publishers.get(publisher_key)
            if publisher is None:
                continue
            if msg_type == "geometry_msgs/msg/Twist":
                self._publish_twist(publisher, segment)
            else:
                self._publish_joint_trajectory(
                    publisher,
                    self._command_joint_names.get(publisher_key, []),
                    segment,
                )

    def publish_action_with_receipt(
        self, action: np.ndarray, action_keys: Optional[list[str]] = None,
        *, zero_twist: bool = False,
    ) -> np.ndarray:
        """Validate all groups and return the values sent by a successful publish.

        This is a transport receipt, not a controller execution acknowledgement.
        Any exception (including partial multi-topic publication) means callers
        must not record a successful whole-vector publication.
        zero_twist preserves position targets but stops velocity modalities when
        a model step expires. The receipt reports zeros, not the original action.
        """
        segments = self._build_action_segments(action, action_keys)
        emitted = []
        for publisher, joint_names, values, msg_type in segments:
            if msg_type == "geometry_msgs/msg/Twist":
                if zero_twist:
                    values = np.zeros(3, dtype=np.float64)
                values = self._publish_twist(publisher, values)
            else:
                self._publish_joint_trajectory(publisher, joint_names, values)
            emitted.extend(values)
        return np.asarray(emitted, dtype=np.float64)

    def publish_idle_action(self, action_keys: Optional[list[str]] = None) -> None:
        """Publish safe idle commands for velocity-like action topics.

        Position trajectory controllers hold their last target when the action
        buffer is empty. Twist command topics do not have that same semantics,
        so publish an explicit zero velocity for any commanded Twist modality.
        """
        if not self._command_publishers:
            raise RuntimeError("RobotClient command publishers are not enabled")

        keys = list(action_keys) if action_keys else self._action_keys
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None or cfg["msg_type"] != "geometry_msgs/msg/Twist":
                continue
            publisher = self._command_publishers.get(f"leader_{publish_key}")
            if publisher is None:
                continue
            self._publish_twist(publisher, np.zeros(3, dtype=np.float64))

    def publish_initial_pose_sync(
        self,
        action: np.ndarray,
        action_keys: Optional[list[str]] = None,
        duration_s: float = 5.0,
    ) -> None:
        """Publish one validated action as a slow position-only transition.

        Twist modalities are forced to zero. A complete current-position
        snapshot is required before any target is published so an interrupted
        transition can be replaced with a hold trajectory.
        """
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be a positive finite value")

        segments = self._build_action_segments(action, action_keys)
        hold_segments = self._build_current_position_segments(action_keys)
        if not hold_segments:
            raise ValueError("initial pose sync requires a position action group")

        try:
            for publisher, _joint_names, _values, msg_type in segments:
                if msg_type == "geometry_msgs/msg/Twist":
                    self._publish_twist(publisher, np.zeros(3, dtype=np.float64))
            for publisher, joint_names, values, msg_type in segments:
                if msg_type != "geometry_msgs/msg/Twist":
                    self._publish_joint_trajectory(
                        publisher,
                        joint_names,
                        values,
                        time_from_start_s=duration_s,
                    )
        except Exception:
            try:
                self.publish_current_pose_hold(action_keys, duration_s=0.1)
            except Exception as hold_error:
                logger.error(
                    "failed to hold current pose after initial sync publish error: %s",
                    hold_error,
                )
            raise

    def publish_named_pose(self, positions: dict[str, float], duration_s: float = 5.0) -> None:
        """Send one timed target per position controller; keep base velocity zero."""
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or not 0.1 <= duration_s <= 60.0:
            raise ValueError("duration_s must be between 0.1 and 60 seconds")
        keys = list(self._action_groups)
        segments = self._build_current_position_segments(keys)
        required = {name for _publisher, names, _values in segments for name in names}
        if not required or set(positions) != required:
            raise ValueError("Pose joints must match all configured position controllers.")
        values = {name: float(positions[name]) for name in required}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Pose contains non-finite positions.")
        self.get_named_joint_positions(list(required))
        self.publish_idle_action(keys)
        for publisher, names, _current in segments:
            self._publish_joint_trajectory(
                publisher, names, np.asarray([values[name] for name in names]),
                time_from_start_s=duration_s, zero_terminal_derivatives=True,
            )

    def publish_current_pose_hold(
        self,
        action_keys: Optional[list[str]] = None,
        duration_s: float = 0.1,
    ) -> None:
        """Attempt every zero/hold independently, then report any failed groups."""
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be a positive finite value")
        failures = []
        position_keys = []
        seen = set()
        for action_key in (action_keys or self._action_keys):
            try:
                key = self._resolve_action_key(action_key)
                if key in seen:
                    continue
                seen.add(key)
                cfg = self._action_groups.get(key)
                if cfg is None:
                    raise ValueError("unknown action key")
                if cfg["msg_type"] != "geometry_msgs/msg/Twist":
                    position_keys.append(key)
                    continue
                publisher = self._command_publishers.get(f"leader_{key}")
                if publisher is None:
                    raise RuntimeError("publisher unavailable")
                self._publish_twist(publisher, np.zeros(3, dtype=np.float64))
            except Exception as exc:
                failures.append(f"{action_key}: {exc}")

        # Freshness and transport failures affect only their own controller.
        for key in position_keys:
            try:
                segments = self._build_current_position_segments([key])
                if not segments or not segments[0][1]:
                    raise ValueError("position group has no joints")
                for publisher, names, values in segments:
                    self._publish_joint_trajectory(
                        publisher, names, values, time_from_start_s=duration_s,
                    )
            except Exception as exc:
                failures.append(f"{key}: {exc}")
        if not position_keys:
            failures.append("current pose hold requires a position action group")
        if failures:
            raise RuntimeError("current-pose hold failed: " + "; ".join(failures))

    def _build_action_segments(
        self,
        action: np.ndarray,
        action_keys: Optional[list[str]],
    ) -> list[tuple[ROS2Publisher, list[str], np.ndarray, str]]:
        if not self._command_publishers:
            raise RuntimeError("RobotClient command publishers are not enabled")

        keys = list(action_keys) if action_keys else self._action_keys
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("action contains non-finite values")

        segments = []
        offset = 0
        resolved_keys: set[str] = set()
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            if publish_key in resolved_keys:
                raise ValueError(f"duplicate action key: {action_key}")
            resolved_keys.add(publish_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None:
                raise ValueError(f"unknown action key: {action_key}")
            publisher_key = f"leader_{publish_key}"
            publisher = self._command_publishers.get(publisher_key)
            if publisher is None:
                raise RuntimeError(f"publisher unavailable for action key: {action_key}")
            msg_type = cfg["msg_type"]
            joint_names = list(self._command_joint_names.get(publisher_key, []))
            width = 3 if msg_type == "geometry_msgs/msg/Twist" else len(joint_names)
            if width <= 0:
                raise ValueError(f"action key has no configured dimensions: {action_key}")
            segment = values[offset:offset + width]
            if len(segment) != width:
                raise ValueError(
                    f"action dimension too small for {action_key}: "
                    f"expected {width}, got {len(segment)}"
                )
            offset += width
            segments.append((publisher, joint_names, segment.copy(), msg_type))

        if offset != len(values):
            raise ValueError(
                f"action dimension mismatch: expected {offset}, got {len(values)}"
            )
        return segments

    def _build_current_position_segments(
        self,
        action_keys: Optional[list[str]],
    ) -> list[tuple[ROS2Publisher, list[str], np.ndarray]]:
        keys = list(action_keys) if action_keys else self._action_keys
        planned = []
        missing = []
        stale = []
        now = time.monotonic()
        with self._lock:
            positions_by_name = dict(self._joint_positions_by_name)
            timestamps_by_name = dict(self._joint_position_timestamps_by_name)
            max_age_s = self._initial_pose_sync_state_max_age_s
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None:
                raise ValueError(f"unknown action key: {action_key}")
            if cfg["msg_type"] == "geometry_msgs/msg/Twist":
                continue
            publisher_key = f"leader_{publish_key}"
            publisher = self._command_publishers.get(publisher_key)
            if publisher is None:
                raise RuntimeError(f"publisher unavailable for action key: {action_key}")
            joint_names = list(self._command_joint_names.get(publisher_key, []))
            group_missing = [
                name
                for name in joint_names
                if name not in positions_by_name or name not in timestamps_by_name
            ]
            missing.extend(group_missing)
            if group_missing:
                continue
            group_stale = [
                (name, max(0.0, now - timestamps_by_name[name]))
                for name in joint_names
                if max(0.0, now - timestamps_by_name[name]) > max_age_s
            ]
            stale.extend(group_stale)
            if group_stale:
                continue
            planned.append(
                (
                    publisher,
                    joint_names,
                    np.asarray(
                        [positions_by_name[name] for name in joint_names],
                        dtype=np.float64,
                    ),
                )
            )
        if missing:
            missing_names = ", ".join(sorted(set(missing)))
            raise RuntimeError(f"current joint state unavailable: {missing_names}")
        if stale:
            stale_by_name = {name: age for name, age in stale}
            stale_details = ", ".join(
                f"{name}={age:.3f}s"
                for name, age in sorted(stale_by_name.items())
            )
            raise RuntimeError(
                "current joint state stale: "
                f"{stale_details} (max {max_age_s:.3f}s)"
            )
        if any(not np.all(np.isfinite(values)) for _publisher, _names, values in planned):
            raise RuntimeError("current joint state contains non-finite positions")
        return planned

    def build_action_preview(
        self,
        action: np.ndarray,
        action_keys: Optional[list[str]] = None,
    ) -> tuple[list[str], np.ndarray]:
        """Build a single joint trajectory point for preview-only consumers."""
        keys = list(action_keys) if action_keys else self._action_keys
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        joint_names: list[str] = []
        positions: list[float] = []
        offset = 0
        for action_key in keys:
            publish_key = self._resolve_action_key(action_key)
            cfg = self._action_groups.get(publish_key)
            if cfg is None:
                continue
            msg_type = cfg["msg_type"]
            width = 3 if msg_type == "geometry_msgs/msg/Twist" else len(cfg["joint_names"])
            segment = values[offset:offset + width]
            offset += width
            if msg_type == "geometry_msgs/msg/Twist":
                continue
            names = list(cfg.get("joint_names", []))
            joint_names.extend(names)
            positions.extend(float(v) for v in segment[:len(names)])
        return joint_names, np.asarray(positions, dtype=np.float64)

    def publish_action_preview(
        self,
        action: np.ndarray,
        action_keys: Optional[list[str]] = None,
    ) -> None:
        """Publish preview-only action data for the 3D viewer."""
        if self._preview_publisher is None:
            return
        joint_names, positions = self.build_action_preview(action, action_keys)
        if not joint_names or len(joint_names) != len(positions):
            return
        self._publish_joint_trajectory(self._preview_publisher, joint_names, positions)

    def _resolve_action_key(self, action_key: str) -> str:
        if action_key in self._action_groups:
            return action_key
        if action_key == "odometry" and "mobile" in self._action_groups:
            return "mobile"
        return action_key

    def _publish_twist(self, publisher: ROS2Publisher, values: np.ndarray) -> np.ndarray:
        Vector3 = get_message_class("geometry_msgs/msg/Vector3")
        linear_x = float(values[0]) if len(values) > 0 else 0.0
        linear_y = float(values[1]) if len(values) > 1 else 0.0
        angular_z = float(values[2]) if len(values) > 2 else 0.0
        filtered_linear_x = _deadband(linear_x, self._cmd_vel_linear_deadband)
        filtered_linear_y = _deadband(linear_y, self._cmd_vel_linear_deadband)
        filtered_angular_z = _deadband(angular_z, self._cmd_vel_angular_deadband)

        linear = Vector3(
            x=filtered_linear_x,
            y=filtered_linear_y,
            z=0.0,
        )
        angular = Vector3(
            x=0.0,
            y=0.0,
            z=filtered_angular_z,
        )
        publisher.publish(linear=linear, angular=angular)
        return np.array([filtered_linear_x, filtered_linear_y, filtered_angular_z], dtype=np.float64)

    def _publish_joint_trajectory(
        self,
        publisher: ROS2Publisher,
        joint_names: list[str],
        values: np.ndarray,
        time_from_start_s: float = 0.0,
        zero_terminal_derivatives: bool = False,
    ) -> None:
        Header = get_message_class("std_msgs/msg/Header")
        Time = get_message_class("builtin_interfaces/msg/Time")
        Duration = get_message_class("builtin_interfaces/msg/Duration")
        JointTrajectoryPoint = get_message_class(
            "trajectory_msgs/msg/JointTrajectoryPoint"
        )
        duration_s = float(time_from_start_s)
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("time_from_start_s must be finite and non-negative")
        duration_sec = int(math.floor(duration_s))
        duration_nanosec = int(round((duration_s - duration_sec) * 1_000_000_000))
        if duration_nanosec >= 1_000_000_000:
            duration_sec += 1
            duration_nanosec -= 1_000_000_000
        point = JointTrajectoryPoint(
            positions=np.asarray(values, dtype=np.float64),
            velocities=np.zeros(len(joint_names) if zero_terminal_derivatives else 0, dtype=np.float64),
            accelerations=np.zeros(len(joint_names) if zero_terminal_derivatives else 0, dtype=np.float64),
            effort=np.zeros(0, dtype=np.float64),
            time_from_start=Duration(sec=duration_sec, nanosec=duration_nanosec),
        )
        publisher.publish(
            header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
            joint_names=list(joint_names),
            points=[point],
        )

    # ------------------------------------------------------------------ #
    # Task instruction
    # ------------------------------------------------------------------ #

    def set_task_instruction(self, instruction: str):
        self._task_instruction = instruction

    @property
    def task_instruction(self) -> str:
        return self._task_instruction

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def get_observation(
        self,
        resize: Optional[tuple[int, int]] = None,
        format: str = "bgr",
    ) -> Optional[dict]:
        """Get full observation for inference.

        Returns:
            Dict with images, joint_positions, task_instruction.
            None if sync_check is enabled and data is out of sync.
        """
        if self._sync_check and not self._check_sync():
            return None
        return {
            "images": self.get_images(resize=resize, format=format),
            "joint_positions": self.get_joint_positions(),
            "task_instruction": self._task_instruction,
        }

    def _check_sync(self) -> bool:
        """Check if image and joint timestamps are within threshold."""
        threshold_s = self._sync_threshold_ms / 1000.0
        with self._lock:
            if not self._image_timestamps or not self._joint_timestamps:
                return False
            img_times = list(self._image_timestamps.values())
            jnt_times = list(self._joint_timestamps.values())

        latest_img = max(img_times) if img_times else 0
        latest_jnt = max(jnt_times) if jnt_times else 0
        return abs(latest_img - latest_jnt) < threshold_s

    # ------------------------------------------------------------------ #
    # Readiness / waiting
    # ------------------------------------------------------------------ #

    def wait_for_ready(
        self, timeout: float = 10.0, *, camera_names=None, joint_groups=None,
        sensor_names=(),
    ) -> bool:
        """Wait for selected inputs; omitted camera/joint lists keep legacy defaults."""
        deadline = time.monotonic() + timeout
        while True:
            missing = self.get_missing_observations(
                camera_names=camera_names, joint_groups=joint_groups,
                sensor_names=sensor_names,
            )
            if not missing:
                logger.info("Required observations ready")
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f"Timeout waiting for sensors. Missing: {missing}")
                return False
            time.sleep(min(0.1, remaining))

    def wait_for_image(self, camera_name: str, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_image_ready(camera_name):
                return True
            time.sleep(0.1)
        return False

    def wait_for_joint(self, group_name: str, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_joint_ready(group_name):
                return True
            time.sleep(0.1)
        return False

    def _all_ready(self) -> bool:
        return not self.get_missing_observations()

    def _get_missing(self) -> list[str]:
        return self.get_missing_observations()

    def get_missing_observations(
        self, *, camera_names=None, joint_groups=None, sensor_names=(),
    ) -> list[str]:
        """Report absent inputs without requiring unused robot capabilities."""
        if camera_names is None:
            camera_names = self._selected_names("cameras") if self._subscribe_images else ()
        if joint_groups is None:
            joint_groups = self._selected_names("joint_groups") if self._subscribe_state else ()
        missing = []
        with self._lock:
            for cam in camera_names:
                if cam not in self._images:
                    missing.append(f"camera:{cam}")
            for group in joint_groups:
                if group not in self._joint_positions:
                    missing.append(f"joint:{group}")
            for sensor in sensor_names:
                if sensor not in self._sensors:
                    missing.append(f"sensor:{sensor}")
        return missing

    # ------------------------------------------------------------------ #
    # Info / diagnostics
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """Get current status of all subscriptions."""
        with self._lock:
            return {
                "robot_type": self._robot_type,
                "cameras": {
                    name: {
                        "ready": name in self._images,
                        "shape": self._images[name].shape if name in self._images else None,
                        "timestamp": self._image_timestamps.get(name),
                    }
                    for name in self._config.get("cameras", {})
                },
                "joint_groups": {
                    name: {
                        "ready": name in self._joint_positions,
                        "dof": len(self._joint_positions[name]) if name in self._joint_positions else 0,
                        "timestamp": self._joint_timestamps.get(name),
                    }
                    for name in self._config.get("joint_groups", {})
                },
                "sensors": {
                    name: {
                        "ready": name in self._sensors,
                        "timestamp": self._sensor_timestamps.get(name),
                    }
                    for name in self._config.get("sensors", {})
                },
            }

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def close(self):
        """Close all subscriptions and command publishers."""
        if hasattr(self, '_closed') and self._closed:
            return
        self._closed = True
        with self._lock:
            self._observation_capture = None
        for sub in self._subscribers:
            try:
                sub.close()
            except Exception as e:
                logger.debug(f"Error closing subscriber: {e}")
        self._subscribers.clear()
        for pub in self._command_publishers.values():
            try:
                pub.close()
            except Exception as e:
                logger.debug(f"Error closing command publisher: {e}")
        self._command_publishers.clear()
        if self._preview_publisher is not None:
            try:
                self._preview_publisher.close()
            except Exception as e:
                logger.debug(f"Error closing preview publisher: {e}")
            self._preview_publisher = None
        logger.info("RobotClient closed")

    def __del__(self):
        self.close()
