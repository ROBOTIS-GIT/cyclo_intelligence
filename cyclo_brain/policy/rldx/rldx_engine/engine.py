#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""RLDX engine adapter for Cyclo's common two-process policy runtime.

The heavy RLDX model stays on an external GPU PC running RLDX's ZMQ
PolicyServer. This adapter only implements the backend-specific
``InferenceEngine`` contract: load endpoint metadata, build RobotClient
observations, call the remote server once, and return an action chunk.
The common runtime owns ``/rldx/inference_command``, buffering, preview
publishing, and robot command publishing.
"""

from __future__ import annotations

from collections import deque
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


_SCHEMA_DIR = os.environ.get("ORCHESTRATOR_CONFIG_PATH", "/orchestrator_config")
if os.path.isdir(_SCHEMA_DIR) and _SCHEMA_DIR not in sys.path:
    sys.path.insert(0, _SCHEMA_DIR)
try:
    import schema as robot_schema  # type: ignore[import-not-found]
except ImportError:
    _src_schema_dir = (
        Path(__file__).resolve().parents[4] / "shared" / "shared" / "robot_configs"
    )
    if _src_schema_dir.is_dir():
        sys.path.insert(0, str(_src_schema_dir))
    import schema as robot_schema  # type: ignore[import-not-found]


_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", "/robot_client_sdk")
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)


from engine import InferenceEngine  # noqa: E402
from .mapping import (  # noqa: E402
    aspect_area_resize_crop_sizes,
    build_state_observation,
    delta_indices,
    modality_keys,
    policy_action_chunk_from_rldx,
    robot_action_keys_from_policy,
    robot_action_chunk_from_rldx,
    stack_history_window,
)


logger = logging.getLogger("rldx_engine")

IMAGE_MAX_AREA = int(
    os.environ.get(
        "RLDX_IMAGE_MAX_AREA",
        str(int(os.environ.get("RLDX_IMAGE_SIZE", "256")) ** 2),
    )
)
IMAGE_RESIZE_M = int(os.environ.get("RLDX_IMAGE_RESIZE_M", "32"))
ROBOT_READY_TIMEOUT_S = float(os.environ.get("RLDX_ROBOT_READY_TIMEOUT_S", "10.0"))


def _camera_candidates(policy_key: str) -> list[str]:
    candidates = [policy_key]
    parts = policy_key.split("_")
    if len(parts) == 3 and parts[0] == "cam":
        candidates.append("_".join((parts[0], parts[2], parts[1])))
    aliases = {
        "cam_left_head": "cam_head_left",
        "cam_right_head": "cam_head_right",
        "cam_left_wrist": "cam_wrist_left",
        "cam_right_wrist": "cam_wrist_right",
    }
    alias = aliases.get(policy_key)
    if alias:
        candidates.append(alias)
    return list(dict.fromkeys(candidates))


class RLDXEngine(InferenceEngine):
    """Remote RLDX PolicyServer adapter used by common ``engine_process``."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._robot: Optional[Any] = None
        self._robot_type = ""
        self._task_instruction = ""
        self._session_id = ""
        self._remote_host = ""
        self._remote_port = 0
        self._remote_timeout_ms = 0

        self._video_keys: list[str] = []
        self._state_keys: list[str] = []
        self._language_keys: list[str] = []
        self._action_keys: list[str] = []
        self._robot_action_keys: list[str] = []
        self._video_t = 1
        self._video_delta_indices: list[int] = [0]
        self._video_history_size = 1
        self._video_history: dict[str, deque[np.ndarray]] = {}
        self._state_t = 1
        self._language_t = 1
        self._camera_rotations: dict[str, int] = {}
        self._reset_next = True
        self._last_policy_action_chunk: Optional[np.ndarray] = None
        self._send_action_prefix = self._bool_env("RLDX_SEND_ACTION_PREFIX", True)
        self._rtc_prefix_len = max(0, int(os.environ.get("RLDX_RTC_PREFIX_LEN", "4")))
        self._rtc_exec_horizon = max(0, int(os.environ.get("RLDX_RTC_EXEC_HORIZON", "12")))

    @property
    def is_ready(self) -> bool:
        return (
            self._client is not None
            and self._robot is not None
            and bool(self._video_keys)
            and bool(self._state_keys)
            and bool(self._language_keys)
            and bool(self._robot_action_keys)
        )

    def load_policy(self, request: Any) -> Dict[str, Any]:
        robot_type = str(getattr(request, "robot_type", "") or "").strip()
        if not robot_type:
            return self._fail("robot_type is required")

        try:
            self.cleanup()
            self._robot_type = robot_type
            self._task_instruction = str(
                getattr(request, "task_instruction", "") or ""
            )
            self._session_id = os.environ.get(
                "RLDX_SESSION_ID",
                f"cyclo-rldx-{self._robot_type}",
            )
            self._remote_host = self._remote_host_from_request(request)
            self._remote_port = self._remote_port_from_request(request)
            self._remote_timeout_ms = self._remote_timeout_from_request(request)

            self._client = self._make_remote_client()
            self._client.ping()
            self._load_modality_config()
            self._load_robot_config()

            self._robot = self._make_robot_client(
                self._robot_type,
                router_ip=os.environ.get("ZENOH_ROUTER_IP", "127.0.0.1"),
                router_port=int(os.environ.get("ZENOH_ROUTER_PORT", "7447")),
                domain_id=int(os.environ.get("ROS_DOMAIN_ID", "30")),
            )
            self._robot.wait_for_ready(timeout=ROBOT_READY_TIMEOUT_S)
            self._reset_next = True

            logger.info(
                "RLDX loaded: remote=%s:%s video=%s state=%s language=%s "
                "action=%s robot_action=%s",
                self._remote_host,
                self._remote_port,
                self._video_keys,
                self._state_keys,
                self._language_keys,
                self._action_keys,
                self._robot_action_keys,
            )
            return {
                "success": True,
                "message": f"connected to RLDX server {self._remote_host}:{self._remote_port}",
                "action_keys": list(self._robot_action_keys),
            }
        except Exception as exc:
            logger.error("load_policy failed: %s", exc, exc_info=True)
            self.cleanup()
            return self._fail(str(exc))

    def get_action_chunk(self, request: Any) -> Dict[str, Any]:
        if not self.is_ready:
            return self._fail("Not in inference mode")
        task_instruction = str(getattr(request, "task_instruction", "") or "")
        if task_instruction:
            self._task_instruction = task_instruction

        try:
            obs = self._build_observation()
            reset = self._reset_next
            options = {
                "session_ids": [self._session_id],
                "reset_memory": [bool(reset)],
            }
            action_prefix = self._build_action_prefix(reset=reset)
            if action_prefix is not None:
                options["action_prefix"] = action_prefix
                options["rtc_prefix_len"] = int(action_prefix.shape[0])
            started = time.monotonic()
            actions, _info = self._client.get_action(obs, options=options)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            self._reset_next = False

            self._last_policy_action_chunk = policy_action_chunk_from_rldx(
                actions,
                self._action_keys,
            )
            chunk = robot_action_chunk_from_rldx(actions, self._robot_action_keys)
            logger.info(
                "RLDX chunk T=%d D=%d remote_ms=%.1f reset=%s rtc_prefix=%d",
                chunk.shape[0],
                chunk.shape[1],
                elapsed_ms,
                reset,
                0 if action_prefix is None else int(action_prefix.shape[0]),
            )
            return {
                "success": True,
                "action_chunk": np.ascontiguousarray(chunk, dtype=np.float64),
                "chunk_size": int(chunk.shape[0]),
                "action_dim": int(chunk.shape[1]),
            }
        except Exception as exc:
            logger.error("get_action_chunk failed: %s", exc, exc_info=True)
            return self._fail(str(exc))

    def cleanup(self) -> None:
        if self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
            self._robot = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

        self._robot_type = ""
        self._task_instruction = ""
        self._session_id = ""
        self._remote_host = ""
        self._remote_port = 0
        self._remote_timeout_ms = 0
        self._video_keys = []
        self._state_keys = []
        self._language_keys = []
        self._action_keys = []
        self._robot_action_keys = []
        self._video_t = 1
        self._video_delta_indices = [0]
        self._video_history_size = 1
        self._video_history = {}
        self._state_t = 1
        self._language_t = 1
        self._camera_rotations = {}
        self._reset_next = True
        self._last_policy_action_chunk = None

    def _load_modality_config(self) -> None:
        if self._client is None:
            raise RuntimeError("RLDX client is not initialized")
        modality_config = self._client.get_modality_config()
        video_cfg = modality_config.get("video")
        state_cfg = modality_config.get("state")
        language_cfg = modality_config.get("language")
        action_cfg = modality_config.get("action")
        self._video_keys = modality_keys(video_cfg)
        self._state_keys = modality_keys(state_cfg)
        self._language_keys = modality_keys(language_cfg)
        self._action_keys = modality_keys(action_cfg)
        self._video_delta_indices = delta_indices(video_cfg) or [0]
        self._video_t = max(1, len(self._video_delta_indices))
        self._video_history_size = max(
            1,
            abs(min(self._video_delta_indices)) + 1,
        )
        self._video_history = {
            key: deque(maxlen=self._video_history_size)
            for key in self._video_keys
        }
        self._state_t = max(1, len(delta_indices(state_cfg)))
        self._language_t = max(1, len(delta_indices(language_cfg)))
        if (
            not self._video_keys
            or not self._state_keys
            or not self._language_keys
            or not self._action_keys
        ):
            raise RuntimeError(
                f"Invalid RLDX modality config: video={self._video_keys}, "
                f"state={self._state_keys}, language={self._language_keys}, "
                f"action={self._action_keys}"
            )
        if self._language_t != 1:
            raise RuntimeError(
                f"Unsupported RLDX language horizon: {self._language_t}; "
                "expected exactly one language timestep"
            )

    def _load_robot_config(self) -> None:
        section = robot_schema.load_robot_section(self._robot_type)
        robot_action_keys = sorted(robot_schema.get_action_groups(section).keys())
        self._robot_action_keys = robot_action_keys_from_policy(
            self._action_keys,
            robot_action_keys,
        )
        image_cfg = robot_schema.get_image_topics(section)
        self._camera_rotations = {
            name: int(cfg.get("rotation_deg", 0) or 0)
            for name, cfg in image_cfg.items()
        }

    def _build_observation(self) -> dict[str, Any]:
        if self._robot is None:
            raise RuntimeError("robot is not initialized")

        obs: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
        images = self._robot.get_images(format="rgb")
        for key in self._video_keys:
            image = self._select_image(images, key)
            history = self._video_history.setdefault(
                key,
                deque(maxlen=self._video_history_size),
            )
            history.append(image)
            frames = stack_history_window(history, self._video_delta_indices)
            obs["video"][key] = frames[None, :, :, :, :].astype(np.uint8)

        joints = self._robot.get_joint_positions()
        odom = self._robot.get_odom()
        obs["state"].update(
            build_state_observation(self._state_keys, joints, odom, self._state_t)
        )
        for key in self._language_keys:
            obs["language"][key] = [[self._task_instruction]]
        return obs

    def _select_image(self, images: dict[str, np.ndarray], policy_key: str) -> np.ndarray:
        import cv2

        source_key = None
        for candidate in _camera_candidates(policy_key):
            if candidate in images:
                source_key = candidate
                break
        if source_key is None:
            raise KeyError(
                f"Missing camera for RLDX key {policy_key!r}; "
                f"available={sorted(images)}"
            )

        image = images[source_key]
        rotation = self._camera_rotations.get(source_key, 0)
        rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        if rotation in rotate_map:
            image = cv2.rotate(image, rotate_map[rotation])

        (resize_h, resize_w), (crop_h, crop_w) = aspect_area_resize_crop_sizes(
            image.shape[0],
            image.shape[1],
            max_area=IMAGE_MAX_AREA,
            multiple=IMAGE_RESIZE_M,
        )
        if image.shape[0] != resize_h or image.shape[1] != resize_w:
            image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        if image.shape[0] != crop_h or image.shape[1] != crop_w:
            y0 = max(0, (image.shape[0] - crop_h) // 2)
            x0 = max(0, (image.shape[1] - crop_w) // 2)
            image = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _build_action_prefix(self, *, reset: bool) -> np.ndarray | None:
        if reset or not self._send_action_prefix or self._rtc_prefix_len <= 0:
            return None
        if self._last_policy_action_chunk is None:
            return None

        start = self._rtc_exec_horizon
        stop = start + self._rtc_prefix_len
        if self._last_policy_action_chunk.shape[0] < stop:
            return None
        return np.ascontiguousarray(
            self._last_policy_action_chunk[start:stop],
            dtype=np.float32,
        )

    def _make_remote_client(self) -> Any:
        from zmq_transport.client import RLDXRemoteClient

        return RLDXRemoteClient(
            host=self._remote_host,
            port=self._remote_port,
            timeout_ms=self._remote_timeout_ms,
            api_token=os.environ.get("RLDX_ZMQ_API_TOKEN") or None,
        )

    @staticmethod
    def _make_robot_client(robot_type: str, **kwargs: Any) -> Any:
        from robot_client import RobotClient

        return RobotClient(robot_type, **kwargs)

    @staticmethod
    def _remote_host_from_request(request: Any) -> str:
        return str(
            getattr(request, "remote_host", "")
            or os.environ.get("RLDX_ZMQ_HOST", "127.0.0.1")
        ).strip()

    @staticmethod
    def _remote_port_from_request(request: Any) -> int:
        return int(
            getattr(request, "remote_port", 0)
            or os.environ.get("RLDX_ZMQ_PORT", "5555")
        )

    @staticmethod
    def _remote_timeout_from_request(request: Any) -> int:
        return int(
            getattr(request, "remote_timeout_ms", 0)
            or os.environ.get("RLDX_ZMQ_TIMEOUT_MS", "300000")
        )

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _fail(message: str) -> Dict[str, Any]:
        return {"success": False, "message": str(message)}


def create_engine() -> RLDXEngine:
    return RLDXEngine()
