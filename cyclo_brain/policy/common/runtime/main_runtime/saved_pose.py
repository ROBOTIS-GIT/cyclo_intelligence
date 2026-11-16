"""Session-scoped named poses, independent of the loaded policy."""
from __future__ import annotations

import math
import re
import time
import uuid
import xml.etree.ElementTree as ET


class SavedPoseManager:
    DEFAULT_DURATION_S = 5.0

    def __init__(self, robot_factory):
        self._factory = robot_factory
        self.device_id = uuid.uuid4().hex
        self._poses = {}
        self._durations = {}
        self._active_duration_s = self.DEFAULT_DURATION_S
        self._profiles = {}
        self._robot = None
        self.robot_type = ""
        self._target = None
        self._started = 0.0
        self.error = ""

    @property
    def returning(self):
        return self._target is not None

    def profile(self, robot_type):
        if not re.fullmatch(r"[a-zA-Z0-9_]+", robot_type):
            raise ValueError("Unknown robot type.")
        if robot_type not in self._profiles:
            from robot_client.robot_client import robot_schema

            section = robot_schema.load_robot_section(robot_type)
            groups = robot_schema.get_action_groups(section)
            names = [name for group in groups.values()
                     if group["msg_type"] == "trajectory_msgs/msg/JointTrajectory"
                     for name in group["joint_names"]]
            state_names = {name for group in robot_schema.get_state_groups(section).values()
                           if group["msg_type"] == "sensor_msgs/msg/JointState"
                           for name in group["joint_names"]}
            if not names or len(names) != len(set(names)) or not set(names).issubset(state_names):
                raise ValueError("Invalid position-joint configuration.")
            units = {}
            for joint in ET.parse(robot_schema.get_urdf_path(section)).getroot().findall("joint"):
                name = joint.get("name")
                if name not in names:
                    continue
                units[name] = "m" if joint.get("type") == "prismatic" else "rad"
            if set(units) != set(names):
                raise ValueError("Command joints are missing from the robot description.")
            self._profiles[robot_type] = {"names": names, "units": units}
        return self._profiles[robot_type]

    def _validate(self, robot_type, positions):
        profile = self.profile(robot_type)
        if not isinstance(positions, dict) or set(positions) != set(profile["names"]):
            raise ValueError("Saved joints do not match this robot.")
        result = {name: float(positions[name]) for name in profile["names"]}
        for name, value in result.items():
            if not math.isfinite(value):
                raise ValueError(f"Invalid joint position: {name}")
        return result

    def _connect(self, robot_type):
        self.profile(robot_type)
        if self._robot is not None and robot_type == self.robot_type:
            return
        if self.returning:
            raise RuntimeError("Stop the active pose return before changing robot type.")
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._robot = self._factory(robot_type)
        self.robot_type = robot_type
        self.error = ""

    def _snapshot(self):
        return self._robot.get_named_joint_positions(self.profile(self.robot_type)["names"])

    def _load(self, robot_type):
        try:
            pose = self._poses[robot_type]
        except KeyError:
            raise FileNotFoundError("Initial pose has not been saved in this session.") from None
        if (not isinstance(pose, dict) or pose.get("schema_version") != 1 or pose.get("robot_type") != robot_type
                or pose.get("device_id") != self.device_id
                or pose.get("units") != self.profile(robot_type)["units"]):
            raise ValueError("Saved pose identity or joint configuration has changed.")
        return self._validate(robot_type, pose.get("positions"))

    def save(self, robot_type):
        if self.returning:
            raise RuntimeError("Cannot save while returning.")
        self._connect(robot_type)
        positions = self._validate(robot_type, self._snapshot())
        pose = {"schema_version": 1, "robot_type": robot_type, "device_id": self.device_id,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "positions": positions,
                "units": self.profile(robot_type)["units"]}
        self._poses[robot_type] = pose
        self.error = ""

    def duration(self, robot_type):
        return self._durations.get(robot_type, self.DEFAULT_DURATION_S)

    def set_duration(self, robot_type, duration_s):
        if self.returning:
            raise RuntimeError("Cannot change duration while returning.")
        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or not 1.0 <= duration_s <= 60.0:
            raise ValueError("Return duration must be between 1 and 60 seconds.")
        self._connect(robot_type)
        self._durations[robot_type] = duration_s

    def check_duration(self, robot_type, expected_duration_s):
        # Reject a stale UI confirmation if another client changed the setting.
        if expected_duration_s != 0 and expected_duration_s != self.duration(robot_type):
            raise ValueError("Return duration changed; review the current value and retry.")

    def restore(self, robot_type):
        if self.returning:
            raise RuntimeError("Pose return is already running.")
        positions = self._load(robot_type)
        self._connect(robot_type)
        self._snapshot()
        self.error = ""
        self._target = positions
        self._active_duration_s = self.duration(robot_type)
        self._started = time.monotonic()
        try:
            self._robot.publish_named_pose(positions, duration_s=self._active_duration_s)
            self._started = time.monotonic()
        except Exception as exc:
            try:
                self.stop()
            except Exception as hold_error:
                self.error = f"Pose publish failed: {exc}; current-pose hold failed: {hold_error}"
            raise

    def stop(self):
        if self.returning:
            try:
                self._robot.publish_current_pose_hold()
            except Exception as exc:
                # Failed Stop must not expire with the normal trajectory timer.
                self.error = f"Pose stop failed; current-pose hold failed: {exc}"
                raise
            self._target = None
        self.error = ""

    def poll(self):
        if not self.returning:
            return
        try:
            if self.error:
                raise RuntimeError(self.error)
            self._snapshot()
            elapsed = time.monotonic() - self._started
            # Duration expiry is not confirmation that the robot reached its target.
            if elapsed >= self._active_duration_s:
                self._target = None
        except Exception as exc:
            reason = str(exc).split("; current-pose hold failed:", 1)[0]
            try:
                self.stop()
            except Exception as hold_error:
                reason = f"{reason}; current-pose hold failed: {hold_error}"
            self.error = reason

    def status(self, robot_type, *, connect=True):
        profile = self.profile(robot_type)
        positions, connected, error = {}, False, self.error
        try:
            positions = self._load(robot_type)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as exc:
            error = f"Cannot read saved pose: {exc}"
        if connect:
            self._connect(robot_type)
        if self._robot is not None and robot_type == self.robot_type:
            try:
                self._snapshot()
                connected = True
            except (ValueError, RuntimeError):
                pass
        names = profile["names"] if positions else []
        return {"robot_type": robot_type, "device_id": self.device_id, "connected": connected,
                "duration_s": self.duration(robot_type),
                "saved": bool(positions), "returning": self.returning, "error": error,
                "joint_names": names, "positions": [positions[n] for n in names],
                "units": [profile["units"][n] for n in names]}

    def close(self):
        self.stop()
        self._poses.clear()
        self._durations.clear()
        if self._robot is not None:
            self._robot.close()
            self._robot = None
