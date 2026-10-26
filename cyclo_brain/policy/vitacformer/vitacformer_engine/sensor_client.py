# Copyright 2026 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0.

"""Read-only SH5 subscriptions and causal histories for ViTacFormer.

This adapter extends the upstream RobotClient subscription hooks only inside
the ViTacFormer engine process. The common command publisher is unchanged.
"""

from collections import deque
import logging
import time

import numpy as np

from robot_client import RobotClient
from robot_client.robot_client import robot_schema

from .constants import CAMERA_NAME, JOINT_NAMES, TACTILE_BASELINE_SAMPLES

logger = logging.getLogger("vitacformer_engine")


def resample_history(samples, history_size, sample_hz, label, now=None):
    """Select the latest received frame at or before each target timestamp."""
    if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 1:
        raise ValueError("history_size must be a positive integer")
    if isinstance(sample_hz, bool) or not np.isfinite(sample_hz) or sample_hz <= 0:
        raise ValueError("sample_hz must be finite and positive")
    if not samples:
        raise RuntimeError(f"{label} history is not ready")
    times = np.asarray([stamp for stamp, _ in samples], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise RuntimeError(f"{label} history timestamps must be finite and monotonic")
    now = time.monotonic() if now is None else now
    period = 1.0 / sample_hz
    tolerance = period + max(1e-9, period * 1e-6)
    age = now - times[-1]
    if not np.isfinite(age) or age < 0 or age > tolerance:
        raise RuntimeError(f"{label} history is stale or has an invalid clock")
    targets = times[-1] - np.arange(history_size - 1, -1, -1) * period
    indices = np.searchsorted(times, targets, side="right") - 1
    if np.any(indices < 0):
        raise RuntimeError(f"{label} history warmup incomplete")
    if np.any(targets - times[indices] > tolerance):
        raise RuntimeError(f"{label} history has a stale resampling gap")
    frames = np.stack([samples[index][1] for index in indices]).astype(np.float32)
    if not np.isfinite(frames).all():
        raise RuntimeError(f"{label} history contains NaN or Inf")
    return frames


class ViTacFormerSensorClient(RobotClient):
    def __init__(self, robot_type):
        # RobotClient marks itself open after allocating its resource lists.
        # A schema failure before that point must also be safe to finalize.
        self._closed = True
        self._joint_history_samples = {}
        self._tactile_history_samples = {}
        self._tactile_calibration_samples = {}
        self._input_errors = {}
        try:
            super().__init__(robot_type)
        except Exception:
            self.close()
            raise

    def _init_subscriptions(self):
        cameras = self._config["cameras"]
        if CAMERA_NAME not in cameras:
            raise ValueError(f"ViTacFormer requires camera {CAMERA_NAME}")
        self._config["cameras"] = {CAMERA_NAME: cameras[CAMERA_NAME]}
        physical = {
            name: dict(cfg, joint_names=list(JOINT_NAMES))
            for name, cfg in self._config["joint_groups"].items()
            if not cfg.get("parent") and set(JOINT_NAMES).issubset(cfg["joint_names"])
        }
        if len(physical) != 1:
            raise ValueError("ViTacFormer requires one SH5 joint-state topic")
        self._config["joint_groups"] = physical
        section = robot_schema.load_robot_section(self._robot_type)
        sensors = robot_schema.get_tactile_topics(section)
        for name, cfg in sensors.items():
            if cfg["msg_type"] != "robotis_interfaces/msg/HandPressures":
                raise ValueError(f"Unsupported ViTacFormer tactile type: {cfg['msg_type']}")
            self._config["sensors"][name] = dict(cfg, kind="tactile")
        self._config["tactile_modalities"] = list(sensors)
        super()._init_subscriptions()

    def _update_joint(self, group_name, msg):
        # Store frames in YAML order even if /joint_states publishes a
        # differently ordered superset of the configured SH5 joints.
        super()._update_joint(group_name, msg)
        try:
            names = list(msg.name)
            positions = np.asarray(msg.position, dtype=np.float32)
            if len(names) != len(set(names)) or len(names) != len(positions):
                raise ValueError("joint names and positions must be unique and equally sized")
            wanted = self._config["joint_groups"][group_name]["joint_names"]
            lookup = dict(zip(names, positions))
            frame = np.asarray([lookup[name] for name in wanted], dtype=np.float32)
            if not np.isfinite(frame).all():
                raise ValueError("joint positions contain NaN or Inf")
            with self._lock:
                self._joint_history_samples.setdefault(group_name, deque(maxlen=512)).append(
                    (time.monotonic(), frame)
                )
                self._input_errors.pop(group_name, None)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            with self._lock:
                self._input_errors[group_name] = str(exc)
            logger.debug("Invalid joint frame from %s: %s", group_name, exc)

    @staticmethod
    def _hand_pressure_taxels(msg):
        sensors = list(msg.sensors)
        if len(sensors) != 5:
            raise ValueError("HandPressures must contain five finger sensors")
        frames = []
        for sensor in sensors:
            raw = sensor.pressure_values
            frame = (np.frombuffer(raw, dtype=np.uint8) if isinstance(
                raw, (bytes, bytearray, memoryview)
            ) else np.asarray(raw)).astype(np.float32)
            if frame.size != 9 or not np.isfinite(frame).all():
                raise ValueError("Each finger must contain nine finite pressure values")
            frames.append(frame.reshape(3, 3))
        return np.stack(frames)

    def _update_sensor(self, sensor_name, msg):
        if self._config["sensors"][sensor_name].get("kind") != "tactile":
            return super()._update_sensor(sensor_name, msg)
        try:
            frame = self._hand_pressure_taxels(msg)
            with self._lock:
                self._tactile_history_samples.setdefault(sensor_name, deque(maxlen=512)).append(
                    (time.monotonic(), frame)
                )
                calibration = self._tactile_calibration_samples.setdefault(sensor_name, [])
                if len(calibration) < TACTILE_BASELINE_SAMPLES:
                    calibration.append(frame.copy())
                self._sensors[sensor_name] = {"taxels": frame}
                self._sensor_timestamps[sensor_name] = time.time()
                self._input_errors.pop(sensor_name, None)
        except (AttributeError, TypeError, ValueError) as exc:
            with self._lock:
                self._input_errors[sensor_name] = str(exc)
            logger.debug("Invalid tactile frame from %s: %s", sensor_name, exc)

    def _history_samples(self, store, name):
        with self._lock:
            if name in self._input_errors:
                raise RuntimeError(f"Invalid {name} frame: {self._input_errors[name]}")
            return list(store.get(name, ()))

    def get_joint_position_history(self, joint_names, history_size, sample_hz):
        names = list(joint_names)
        matches = [
            (name, cfg["joint_names"])
            for name, cfg in self._config["joint_groups"].items()
            if not cfg.get("parent") and set(names).issubset(cfg["joint_names"])
        ]
        if not names or len(matches) != 1:
            raise RuntimeError("ViTacFormer requires one physical joint group covering its state")
        group_name, configured_names = matches[0]
        frames = resample_history(
            self._history_samples(self._joint_history_samples, group_name),
            history_size, sample_hz, group_name,
        )
        return frames[:, [configured_names.index(name) for name in names]]

    def get_tactile_taxel_history(self, sensor_name, history_size, sample_hz):
        return resample_history(
            self._history_samples(self._tactile_history_samples, sensor_name),
            history_size, sample_hz, sensor_name,
        )

    def _wait_for(self, read, timeout):
        if isinstance(timeout, bool) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        while True:
            try:
                return read()
            except RuntimeError as exc:
                if self._closed or time.monotonic() >= deadline:
                    raise RuntimeError(f"ViTacFormer input warmup timed out: {exc}") from exc
            time.sleep(0.01)

    def wait_for_joint_position_history(self, joint_names, history_size, sample_hz, timeout):
        return self._wait_for(
            lambda: self.get_joint_position_history(joint_names, history_size, sample_hz), timeout
        )

    def wait_for_tactile_taxel_history(self, sensor_name, history_size, sample_hz, timeout):
        return self._wait_for(
            lambda: self.get_tactile_taxel_history(sensor_name, history_size, sample_hz), timeout
        )

    def wait_for_tactile_samples(self, sensor_name, sample_count, timeout):
        if sample_count != TACTILE_BASELINE_SAMPLES:
            raise ValueError("ViTacFormer calibration requires exactly 20 frames")

        def read():
            with self._lock:
                samples = list(self._tactile_calibration_samples.get(sensor_name, ()))
            if len(samples) < sample_count:
                raise RuntimeError(f"Waiting for {sensor_name} baseline frames")
            return np.stack(samples)

        return self._wait_for(read, timeout)

    def action_dimension(self, action_keys):
        return sum(len(self._action_groups[key]["joint_names"]) for key in action_keys)
