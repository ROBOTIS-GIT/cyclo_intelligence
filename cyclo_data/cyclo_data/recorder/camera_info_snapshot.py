# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""One-shot CameraInfo capture for recording format v2.

Subscribes to every requested camera_info topic with TRANSIENT_LOCAL
durability so cached driver publications are delivered immediately,
records the first message per topic to a YAML file under
``<episode>/camera_info/<cam_name>.yaml``, and unsubscribes.
"""

from __future__ import annotations

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
    """Capture one CameraInfo message per camera, then unsubscribe."""

    def __init__(
        self,
        node: Node,
        camera_info_topics: Dict[str, str],
        callback_group: Optional[CallbackGroup] = None,
    ) -> None:
        self._node = node
        self._spec = dict(camera_info_topics)
        self._cb_group = callback_group or ReentrantCallbackGroup()

        self._lock = threading.Lock()
        self._subs: Dict[str, object] = {}
        self._captured: Dict[str, dict] = {}
        self._output_dir: Optional[Path] = None
        self._running = False

    def start(self, episode_dir: Path) -> None:
        if self._running:
            raise RuntimeError("CameraInfoSnapshot already running")
        self._output_dir = Path(episode_dir) / "camera_info"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for cam_name, topic in self._spec.items():
            sub = self._node.create_subscription(
                CameraInfo,
                topic,
                lambda msg, n=cam_name: self._on_msg(n, msg),
                _SUB_QOS,
                callback_group=self._cb_group,
            )
            self._subs[cam_name] = sub
        self._running = True

    def _on_msg(self, cam_name: str, msg: CameraInfo) -> None:
        with self._lock:
            if cam_name in self._captured:
                return
            self._captured[cam_name] = _camera_info_to_dict(msg)
            sub = self._subs.pop(cam_name, None)
        if sub is not None:
            try:
                self._node.destroy_subscription(sub)
            except Exception:  # pragma: no cover - destroy is best-effort
                pass
        self._node.get_logger().info(
            f"CameraInfoSnapshot: captured {cam_name}"
        )

    def stop(self) -> Dict[str, Path]:
        """Tear down outstanding subscriptions and write yaml snapshots.

        Returns ``{cam_name: yaml_path}`` for cameras that produced a
        message. Cameras without a captured message are omitted and a
        warning is logged.
        """
        if not self._running:
            return {}
        # Cancel any subscriptions that never produced a message.
        for cam_name, sub in list(self._subs.items()):
            try:
                self._node.destroy_subscription(sub)
            except Exception:  # pragma: no cover
                pass
        self._subs.clear()

        written: Dict[str, Path] = {}
        for cam_name, topic in self._spec.items():
            data = self._captured.get(cam_name)
            if data is None:
                self._node.get_logger().warn(
                    f"CameraInfoSnapshot: no message from {cam_name} ({topic})"
                )
                continue
            yaml_path = (self._output_dir / f"{cam_name}.yaml")  # type: ignore[union-attr]
            with open(yaml_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            written[cam_name] = yaml_path
        self._running = False
        return written
