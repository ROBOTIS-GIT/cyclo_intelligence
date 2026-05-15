# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CameraInfo cache for recording format v2.

Subscribes to every camera_info topic at construction time with
TRANSIENT_LOCAL durability so cached driver publications are delivered
immediately, then keeps the latest received message per camera in
``_latest``. ``start_episode`` / ``stop_episode`` flush the cache to a
YAML file under ``<episode>/camera_info/<cam_name>.yaml`` — the
subscription stays alive across episodes and is only torn down by
``reconfigure()`` (robot_type change) or ``close()`` (node shutdown).
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
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
    """Cache the latest CameraInfo per camera; flush to YAML per episode.

    Subscriptions live from ``__init__`` through ``close()`` / a
    ``reconfigure()``. Each incoming message overwrites the cached dict
    for that camera. ``start_episode`` opens the output dir and
    ``stop_episode`` writes one YAML per camera from the cache.
    """

    # If start/stop arrives before any TRANSIENT_LOCAL latched message
    # has been delivered, wait up to this long for the cache to fill
    # before giving up.
    _STOP_WAIT_TIMEOUT_SEC = 1.0
    _STOP_WAIT_POLL_SEC = 0.05

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
        self._latest: Dict[str, dict] = {}
        self._output_dir: Optional[Path] = None
        self._episode_active = False

        self._build_subscriptions(self._spec)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, episode_dir: Path) -> None:
        if self._episode_active:
            raise RuntimeError("CameraInfoSnapshot already running")
        self._output_dir = Path(episode_dir) / "camera_info"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._episode_active = True

    def stop_episode(self) -> Dict[str, Path]:
        """Flush cached CameraInfo dicts to YAML, one per camera.

        Returns ``{cam_name: yaml_path}`` for cameras whose cache held a
        message. Cameras with no message are omitted with a warning;
        subscriptions for them stay alive so a late publisher can
        backfill before the next episode.
        """
        if not self._episode_active:
            return {}

        self._wait_for_cache_fill()

        output_dir = self._output_dir
        written: Dict[str, Path] = {}
        if output_dir is not None:
            with self._lock:
                snapshot = dict(self._latest)
            for cam_name, topic in self._spec.items():
                data = snapshot.get(cam_name)
                if data is None:
                    self._node.get_logger().warn(
                        f"CameraInfoSnapshot: no message from {cam_name} ({topic})"
                    )
                    continue
                yaml_path = output_dir / f"{cam_name}.yaml"
                with open(yaml_path, "w") as f:
                    yaml.safe_dump(data, f, sort_keys=False)
                written[cam_name] = yaml_path

        self._output_dir = None
        self._episode_active = False
        return written

    def reconfigure(self, camera_info_topics: Dict[str, str]) -> None:
        """Swap the topic set — destroy current subs, build new ones."""
        if self._episode_active:
            raise RuntimeError("Cannot reconfigure while recording")
        self._teardown_subscriptions()
        self._spec = dict(camera_info_topics)
        with self._lock:
            self._latest.clear()
        self._build_subscriptions(self._spec)

    def close(self) -> None:
        """Tear down all subscriptions — called on node shutdown."""
        if self._episode_active:
            try:
                self.stop_episode()
            except Exception:  # pragma: no cover - defensive
                pass
        self._teardown_subscriptions()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_msg(self, cam_name: str, msg: CameraInfo) -> None:
        # No one-shot self-destroy — keep the subscription alive across
        # episodes and just overwrite the cache. CameraInfo rarely
        # changes mid-session, but if it does the freshest value wins.
        snapshot = _camera_info_to_dict(msg)
        with self._lock:
            already_cached = cam_name in self._latest
            self._latest[cam_name] = snapshot
        if not already_cached:
            self._node.get_logger().info(
                f"CameraInfoSnapshot: captured {cam_name}"
            )

    def _build_subscriptions(self, topics: Dict[str, str]) -> None:
        for cam_name, topic in topics.items():
            sub = self._node.create_subscription(
                CameraInfo,
                topic,
                lambda msg, n=cam_name: self._on_msg(n, msg),
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

    def _wait_for_cache_fill(self) -> None:
        """Brief poll for any camera with no cached message yet.

        TRANSIENT_LOCAL publishers normally deliver immediately when we
        subscribe at REFRESH_TOPICS time, so by stop_episode the cache
        is usually full. This handles the edge case of a publisher that
        came up after init and still hasn't sent.
        """
        with self._lock:
            missing = [cam for cam in self._spec if cam not in self._latest]
        if not missing:
            return
        deadline = time.monotonic() + self._STOP_WAIT_TIMEOUT_SEC
        while time.monotonic() < deadline:
            time.sleep(self._STOP_WAIT_POLL_SEC)
            with self._lock:
                missing = [cam for cam in self._spec if cam not in self._latest]
            if not missing:
                return
