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
# Author: Howon Kim, Seongwoo Kim

"""Cached full grids plus incremental Navigation costmap updates."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any
import zlib


logger = logging.getLogger("supervisor_api.navigation_topics")

GRID_TOPICS = frozenset({"/map", "/global_costmap/costmap"})
GRID_UPDATE_TOPICS = {
    "/global_costmap/costmap_updates": "/global_costmap/costmap",
}


def occupancy_grid_data_crc32(message: Any) -> int | None:
    """Return CRC32 of OccupancyGrid.data without constructing a Python list."""
    data = (
        message.get("data")
        if isinstance(message, dict)
        else getattr(message, "data", None)
    )
    if data is None:
        return None
    try:
        return zlib.crc32(data)
    except (BufferError, TypeError, ValueError):
        pass
    try:
        return zlib.crc32(memoryview(data))
    except (BufferError, TypeError, ValueError):
        pass
    if not isinstance(data, list):
        return None
    try:
        marker = 0
        chunk = bytearray()
        for value in data:
            chunk.append(int(value) & 0xFF)
            if len(chunk) == 65536:
                marker = zlib.crc32(chunk, marker)
                chunk.clear()
        return zlib.crc32(chunk, marker)
    except (TypeError, ValueError, OverflowError):
        return None


def _time_to_dict(value: Any) -> dict[str, int]:
    return {
        "sec": int(getattr(value, "sec", 0)),
        "nanosec": int(getattr(value, "nanosec", 0)),
    }


def _pose_to_dict(value: Any) -> dict[str, Any]:
    position = value.position
    orientation = value.orientation
    return {
        "position": {
            "x": float(position.x),
            "y": float(position.y),
            "z": float(position.z),
        },
        "orientation": {
            "x": float(orientation.x),
            "y": float(orientation.y),
            "z": float(orientation.z),
            "w": float(orientation.w),
        },
    }


def occupancy_grid_to_dict(message: Any) -> dict[str, Any]:
    """Convert only the OccupancyGrid fields consumed by the Navigation UI."""
    if isinstance(message, dict):
        return message
    return {
        "header": {
            "stamp": _time_to_dict(message.header.stamp),
            "frame_id": message.header.frame_id,
        },
        "info": {
            "map_load_time": _time_to_dict(message.info.map_load_time),
            "resolution": float(message.info.resolution),
            "width": int(message.info.width),
            "height": int(message.info.height),
            "origin": _pose_to_dict(message.info.origin),
        },
        "data": list(message.data),
    }


def occupancy_grid_update_to_dict(message: Any) -> dict[str, Any]:
    """Convert the OccupancyGridUpdate fields needed by the UI cache."""
    if isinstance(message, dict):
        return {
            "header": dict(message.get("header") or {}),
            "x": int(message.get("x", 0)),
            "y": int(message.get("y", 0)),
            "width": int(message.get("width", 0)),
            "height": int(message.get("height", 0)),
            "data": list(message.get("data") or []),
        }
    return {
        "header": {
            "stamp": _time_to_dict(message.header.stamp),
            "frame_id": message.header.frame_id,
        },
        "x": int(message.x),
        "y": int(message.y),
        "width": int(message.width),
        "height": int(message.height),
        "data": list(message.data),
    }


class OccupancyGridCache:
    """Keep one serialized grid and notify connected WebSocket clients."""

    def __init__(self, topic: str) -> None:
        if topic not in GRID_TOPICS:
            raise ValueError(f"Unsupported grid topic: {topic}")
        self.topic = topic
        self._lock = threading.Lock()
        self._marker: tuple[Any, ...] | None = None
        self._previous_marker: tuple[Any, ...] | None = None
        self._grid_signature: tuple[Any, ...] | None = None
        self._grid: dict[str, Any] | None = None
        self._latest_is_update = False
        self._payload: str | None = None
        self._full_payload: str | None = None
        self._listeners: dict[int, tuple[Any, Any]] = {}
        self._serial = 0

    @staticmethod
    def _metadata_marker(message: Any) -> tuple[Any, ...]:
        if isinstance(message, dict):
            header = message.get("header") or {}
            info = message.get("info") or {}
            origin = info.get("origin") or {}
            position = origin.get("position") or {}
            orientation = origin.get("orientation") or {}
            return (
                header.get("frame_id"), info.get("resolution"),
                info.get("width"), info.get("height"),
                position.get("x"), position.get("y"), position.get("z"),
                orientation.get("x"), orientation.get("y"),
                orientation.get("z"), orientation.get("w"),
            )
        info = message.info
        origin = info.origin
        return (
            message.header.frame_id, float(info.resolution),
            int(info.width), int(info.height),
            float(origin.position.x), float(origin.position.y),
            float(origin.position.z), float(origin.orientation.x),
            float(origin.orientation.y), float(origin.orientation.z),
            float(origin.orientation.w),
        )

    def cache_ros_message(self, message: Any) -> None:
        data_marker = occupancy_grid_data_crc32(message)
        if data_marker is None:
            return
        signature = (data_marker, *self._metadata_marker(message))
        grid = occupancy_grid_to_dict(message)
        grid = {
            **grid,
            "info": dict(grid.get("info") or {}),
            "data": list(grid.get("data") or []),
        }
        if "header" in grid:
            grid["header"] = dict(grid.get("header") or {})
        payload = json.dumps({
            "available": True,
            "data": grid,
        }, separators=(",", ":"))
        with self._lock:
            if signature == self._grid_signature:
                return
            self._serial += 1
            self._previous_marker = self._marker
            self._marker = ("grid", self._serial)
            self._grid_signature = signature
            self._grid = grid
            self._latest_is_update = False
            self._payload = payload
            self._full_payload = payload
            listeners = list(self._listeners.items())
        self._notify_listeners(listeners)

    def cache_ros_update(self, message: Any) -> None:
        """Merge a costmap dirty rectangle and notify clients with the delta."""
        if self.topic != "/global_costmap/costmap":
            return
        update = occupancy_grid_update_to_dict(message)
        x = update["x"]
        y = update["y"]
        width = update["width"]
        height = update["height"]
        update_data = update["data"]
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return
        if len(update_data) != width * height:
            return

        with self._lock:
            if self._grid is None:
                return
            info = self._grid.get("info") or {}
            grid_width = int(info.get("width") or 0)
            grid_height = int(info.get("height") or 0)
            grid_data = self._grid.get("data") or []
            if (
                x + width > grid_width
                or y + height > grid_height
                or len(grid_data) < grid_width * grid_height
            ):
                return

            min_changed_column = width
            min_changed_row = height
            max_changed_column = -1
            max_changed_row = -1
            for row in range(height):
                source_start = row * width
                target_start = (y + row) * grid_width + x
                row_data = update_data[source_start:source_start + width]
                existing_row = grid_data[target_start:target_start + width]
                if existing_row == row_data:
                    continue
                for column, value in enumerate(row_data):
                    if existing_row[column] == value:
                        continue
                    grid_data[target_start + column] = value
                    min_changed_column = min(min_changed_column, column)
                    max_changed_column = max(max_changed_column, column)
                    min_changed_row = min(min_changed_row, row)
                    max_changed_row = max(max_changed_row, row)
            if max_changed_column < 0 or max_changed_row < 0:
                return

            # Costmap2D reports the observation bounds, which can cover the
            # complete map when raytracing has a long range even if only a few
            # cells changed. Send the smallest rectangle containing the actual
            # value changes instead of forwarding those broad bounds.
            compact_width = max_changed_column - min_changed_column + 1
            compact_height = max_changed_row - min_changed_row + 1
            compact_data = []
            for row in range(min_changed_row, max_changed_row + 1):
                source_start = row * width + min_changed_column
                compact_data.extend(
                    update_data[source_start:source_start + compact_width]
                )
            compact_update = {
                **update,
                "x": x + min_changed_column,
                "y": y + min_changed_row,
                "width": compact_width,
                "height": compact_height,
                "data": compact_data,
            }

            if update["header"]:
                self._grid["header"] = update["header"]
            self._serial += 1
            self._previous_marker = self._marker
            self._marker = ("grid", self._serial)
            self._grid_signature = None
            self._latest_is_update = True
            self._payload = json.dumps({
                "available": True,
                "update": compact_update,
            }, separators=(",", ":"))
            # Build a full snapshot lazily only for a new or lagging client.
            self._full_payload = None
            listeners = list(self._listeners.items())
        self._notify_listeners(listeners)

    def clear(self) -> None:
        """Drop the cached grid and notify current clients to clear the map."""
        with self._lock:
            self._serial += 1
            self._previous_marker = self._marker
            self._marker = ("clear", self._serial)
            self._grid_signature = None
            self._grid = None
            self._latest_is_update = False
            self._payload = json.dumps(
                {"available": False},
                separators=(",", ":"),
            )
            self._full_payload = self._payload
            listeners = list(self._listeners.items())
        self._notify_listeners(listeners)

    def _notify_listeners(self, listeners: list[tuple[int, tuple[Any, Any]]]) -> None:
        stale_listeners = []
        for listener_id, (loop, event) in listeners:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                stale_listeners.append(listener_id)
        if stale_listeners:
            with self._lock:
                for listener_id in stale_listeners:
                    self._listeners.pop(listener_id, None)

    def serialized_if_changed(
        self, last_marker: tuple[Any, ...] | None
    ) -> tuple[tuple[Any, ...] | None, str | None]:
        """Return a WebSocket payload only when this client's marker changed."""
        with self._lock:
            marker = self._marker
            if marker is None or self._payload is None or marker == last_marker:
                return last_marker, None
            if self._latest_is_update and last_marker == self._previous_marker:
                return marker, self._payload
            if self._full_payload is None and self._grid is not None:
                self._full_payload = json.dumps({
                    "available": True,
                    "data": self._grid,
                }, separators=(",", ":"))
            return marker, self._full_payload or self._payload

    def add_listener(self, listener_id: int, loop: Any, event: Any) -> None:
        with self._lock:
            self._listeners[listener_id] = (loop, event)

    def remove_listener(self, listener_id: int) -> None:
        with self._lock:
            self._listeners.pop(listener_id, None)


GRID_CACHES = {topic: OccupancyGridCache(topic) for topic in GRID_TOPICS}
_ros_start_lock = threading.Lock()
_ros_thread: threading.Thread | None = None


def _ros_grid_spin() -> None:
    try:
        import rclpy
        from nav_msgs.msg import OccupancyGrid
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        try:
            from map_msgs.msg import OccupancyGridUpdate
        except ImportError:
            OccupancyGridUpdate = None
            logger.warning(
                "map_msgs is unavailable; Navigation grid cache will use "
                "full OccupancyGrid messages without costmap delta updates"
            )

        if not rclpy.ok():
            rclpy.init()
        node = Node("cyclo_navigation_grid_cache")
        fallback_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        ros_topics = (
            (*GRID_TOPICS, *GRID_UPDATE_TOPICS)
            if OccupancyGridUpdate is not None
            else tuple(GRID_TOPICS)
        )
        discovered_qos = {}
        for _ in range(20):
            for topic in ros_topics:
                publishers = node.get_publishers_info_by_topic(topic)
                if publishers:
                    discovered_qos[topic] = publishers[0].qos_profile
            if len(discovered_qos) == len(ros_topics):
                break
            executor.spin_once(timeout_sec=0.1)
        subscriptions = [
            node.create_subscription(
                OccupancyGrid,
                topic,
                GRID_CACHES[topic].cache_ros_message,
                discovered_qos.get(topic, fallback_qos),
            )
            for topic in GRID_TOPICS
        ]
        if OccupancyGridUpdate is not None:
            subscriptions.extend(
                node.create_subscription(
                    OccupancyGridUpdate,
                    update_topic,
                    GRID_CACHES[grid_topic].cache_ros_update,
                    discovered_qos.get(update_topic, fallback_qos),
                )
                for update_topic, grid_topic in GRID_UPDATE_TOPICS.items()
            )
        node._navigation_grid_subscriptions = subscriptions
        executor.spin()
    except Exception:
        logger.exception("Navigation ROS2 grid cache stopped")


def ensure_ros_grid_subscriber_started() -> None:
    """Start the single ROS subscriber shared by all WebSocket clients."""
    global _ros_thread
    with _ros_start_lock:
        if _ros_thread is not None and _ros_thread.is_alive():
            return
        _ros_thread = threading.Thread(
            target=_ros_grid_spin,
            daemon=True,
            name="navigation-grid-cache",
        )
        _ros_thread.start()
