# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CameraInfo cache for recording format v2.

Subscribes to every camera_info topic at construction time with
TRANSIENT_LOCAL durability to receive latched driver publications. Each
subscription is released after the first calibrated message for that camera.
Save writes available calibration under ``<episode>/camera_info/``; Discard
writes none. Robot reconfiguration starts a new cache.
"""

from __future__ import annotations

from itertools import chain
from math import isfinite
from pathlib import Path
import threading
from typing import Dict, Optional

import yaml
from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo


# camera_info publishers commonly use TRANSIENT_LOCAL so a late
# subscriber still receives the latched message. Match that for
# reliable one-shot capture; depth=1 because we only want the latest.
_SUB_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)


def _camera_info_to_dict(msg: CameraInfo) -> dict:
    return {
        "header": {
            "frame_id": msg.header.frame_id,
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
        },
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": msg.distortion_model,
        "d": [float(v) for v in msg.d],
        "k": [float(v) for v in msg.k],
        "r": [float(v) for v in msg.r],
        "p": [float(v) for v in msg.p],
        "binning_x": int(msg.binning_x),
        "binning_y": int(msg.binning_y),
        "roi": {
            "x_offset": int(msg.roi.x_offset),
            "y_offset": int(msg.roi.y_offset),
            "height": int(msg.roi.height),
            "width": int(msg.roi.width),
            "do_rectify": bool(msg.roi.do_rectify),
        },
    }


class CameraInfoSnapshot:
    """Cache calibration once per camera configuration, not once per episode."""

    # Preserve the grace period only for an interrupted episode on shutdown.
    _SHUTDOWN_WAIT_TIMEOUT_SEC = 1.0

    def __init__(
        self,
        node: Node,
        camera_info_topics: Dict[str, str],
        callback_group: Optional[CallbackGroup] = None,
    ) -> None:
        self._node = node
        self._spec: Dict[str, str]
        self._cb_group = callback_group or ReentrantCallbackGroup()

        self._cache_ready = threading.Condition()
        self._subs: Dict[str, object] = {}
        self._calibration: Dict[str, CameraInfo] = {}
        self._output_dir: Optional[Path] = None

        self.reconfigure(camera_info_topics)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, episode_dir: Path) -> None:
        if self._output_dir is not None:
            raise RuntimeError("CameraInfoSnapshot already running")
        output_dir = Path(episode_dir) / "camera_info"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir

    def stop_episode(self) -> Dict[str, Path]:
        """Flush cached CameraInfo messages to YAML, one per camera.

        Returns ``{cam_name: yaml_path}`` for cameras whose cache held a
        message. Cameras with no message are omitted with a warning;
        subscriptions for them stay alive for a later episode.
        """
        output_dir = self._output_dir
        if output_dir is None:
            return {}
        with self._cache_ready:
            snapshot = dict(self._calibration)
        written: Dict[str, Path] = {}
        for cam_name, topic in self._spec.items():
            msg = snapshot.get(cam_name)
            if msg is None:
                self._node.get_logger().warn(
                    f"CameraInfoSnapshot: no valid calibration from {cam_name} ({topic})"
                )
                continue
            yaml_path = output_dir / f"{cam_name}.yaml"
            with open(yaml_path, "w") as f:
                yaml.safe_dump(_camera_info_to_dict(msg), f, sort_keys=False)
            written[cam_name] = yaml_path

        self._output_dir = None
        return written

    def discard_episode(self) -> None:
        """Detach the episode without touching files or clearing the live cache."""
        self._output_dir = None

    def reconfigure(self, camera_info_topics: Dict[str, str]) -> None:
        """Swap the topic set — destroy current subs, build new ones."""
        if self._output_dir is not None:
            raise RuntimeError("Cannot reconfigure while recording")
        with self._cache_ready:
            self._teardown_subscriptions()
            self._spec = dict(camera_info_topics)
            self._calibration.clear()
            try:
                self._build_subscriptions(self._spec)
            except Exception:
                self._teardown_subscriptions()
                raise

    def close(self) -> None:
        """Tear down all subscriptions — called on node shutdown."""
        if self._output_dir is not None:
            try:
                with self._cache_ready:
                    self._cache_ready.wait_for(
                        lambda: self._spec.keys() <= self._calibration.keys(),
                        timeout=self._SHUTDOWN_WAIT_TIMEOUT_SEC,
                    )
                self.stop_episode()
            except Exception:  # pragma: no cover - defensive
                pass
        with self._cache_ready:
            self._teardown_subscriptions()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_msg(self, topics: Dict[str, str], cam_name: str, msg: CameraInfo) -> None:
        with self._cache_ready:
            if topics is not self._spec or cam_name not in self._subs:
                return  # Stale callback after capture, close or reconfiguration.
            # Drivers can publish an uncalibrated placeholder (K[0] == 0).
            if (not msg.width or not msg.height or msg.k[0] <= 0 or msg.k[4] <= 0
                    or not all(map(isfinite, chain(msg.k, msg.d, msg.r, msg.p)))):
                return
            if cam_name not in self._calibration:
                self._calibration[cam_name] = msg
                self._cache_ready.notify_all()
                self._node.get_logger().info(f"CameraInfoSnapshot: captured {cam_name}")
            try:
                self._node.destroy_subscription(self._subs[cam_name])
            except Exception as exc:
                self._node.get_logger().warning(
                    f"Could not release CameraInfo subscription for {cam_name}: {exc}",
                    throttle_duration_sec=5.0,
                )
                return  # Retry on the next calibrated message or during teardown.
            del self._subs[cam_name]

    def _build_subscriptions(self, topics: Dict[str, str]) -> None:
        for cam_name, topic in topics.items():
            sub = self._node.create_subscription(
                CameraInfo,
                topic,
                lambda msg, n=cam_name: self._on_msg(topics, n, msg),
                _SUB_QOS,
                callback_group=self._cb_group,
            )
            self._subs[cam_name] = sub
            self._node.get_logger().info(
                f"CameraInfoSnapshot: {cam_name} <- {topic} subscribed (idle)"
            )

    def _teardown_subscriptions(self) -> None:
        for cam_name, sub in list(self._subs.items()):
            try:
                self._node.destroy_subscription(sub)
            except Exception:  # pragma: no cover - destroy is best-effort
                pass
        self._subs.clear()
