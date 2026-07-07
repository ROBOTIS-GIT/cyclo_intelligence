#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Small RLDX PolicyServer-compatible ZMQ client.

The remote GPU PC should run RLDX's own ``run_rldx_server.py``. This client
mirrors RLDX's wire codec closely enough for the robot-side bridge without
importing the full RLDX package into the Cyclo policy container.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import msgpack
import numpy as np


logger = logging.getLogger("rldx.zmq.client")


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return obj["as_json"]
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if isinstance(obj, np.generic):
            return obj.item()
        return obj


class RLDXRemoteClient:
    """REQ client for RLDX ``PolicyServer`` endpoints."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 120000,
        api_token: str | None = None,
    ) -> None:
        import zmq

        self._zmq = zmq
        self.host = str(host)
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.api_token = api_token
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def close(self) -> None:
        try:
            self.socket.close(0)
        finally:
            self.context.term()

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        *,
        requires_input: bool = True,
    ) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token:
            request["api_token"] = self.api_token

        summary = self._request_summary(endpoint, request)
        logger.info(
            "[RLDX-ZMQ] -> endpoint=%s remote=%s:%s%s",
            endpoint,
            self.host,
            self.port,
            f" {summary}" if summary else "",
        )
        started = time.monotonic()
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self.socket.recv())
        except self._zmq.Again as exc:
            self._init_socket()
            elapsed_ms = (time.monotonic() - started) * 1000.0
            logger.warning(
                "[RLDX-ZMQ] <- endpoint=%s timeout %.1fms remote=%s:%s",
                endpoint,
                elapsed_ms,
                self.host,
                self.port,
            )
            raise TimeoutError(
                f"RLDX endpoint {endpoint!r} timed out after {self.timeout_ms} ms"
            ) from exc

        if isinstance(response, dict) and "error" in response:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            logger.warning(
                "[RLDX-ZMQ] <- endpoint=%s error %.1fms message=%s",
                endpoint,
                elapsed_ms,
                response["error"],
            )
            raise RuntimeError(f"RLDX server error: {response['error']}")
        elapsed_ms = (time.monotonic() - started) * 1000.0
        response_summary = self._response_summary(endpoint, response)
        logger.info(
            "[RLDX-ZMQ] <- endpoint=%s ok %.1fms%s",
            endpoint,
            elapsed_ms,
            f" {response_summary}" if response_summary else "",
        )
        return response

    @staticmethod
    def _shape(value: Any) -> str:
        shape = getattr(value, "shape", None)
        if shape is None:
            return ""
        return "x".join(str(dim) for dim in tuple(shape))

    def _request_summary(self, endpoint: str, request: dict[str, Any]) -> str:
        data = request.get("data", {}) or {}
        if endpoint == "get_action":
            observation = data.get("observation", {}) or {}
            options = data.get("options", {}) or {}
            video_keys = [key for key in observation if str(key).startswith("video.")]
            state_keys = [key for key in observation if str(key).startswith("state.")]
            parts = [
                f"video={len(video_keys)}",
                f"state={len(state_keys)}",
            ]
            session_ids = options.get("session_ids") or []
            if session_ids:
                parts.append(f"session={session_ids[0]}")
            reset_memory = options.get("reset_memory")
            if reset_memory is not None:
                parts.append(f"reset={reset_memory}")
            action_prefix = options.get("action_prefix")
            prefix_shape = self._shape(action_prefix)
            if prefix_shape:
                parts.append(f"action_prefix={prefix_shape}")
            if "rtc_prefix_len" in options:
                parts.append(f"rtc_prefix_len={options['rtc_prefix_len']}")
            return " ".join(parts)
        if endpoint == "reset":
            options = data.get("options", {}) or {}
            session_ids = options.get("session_ids") or []
            return f"session={session_ids[0]}" if session_ids else ""
        return ""

    def _response_summary(self, endpoint: str, response: Any) -> str:
        if endpoint != "get_action":
            return ""
        actions = response[0] if isinstance(response, (tuple, list)) and response else response
        if not isinstance(actions, dict):
            return ""
        shapes = []
        for key, value in actions.items():
            shape = self._shape(value)
            shapes.append(f"{key}={shape}" if shape else str(key))
        return "actions=" + ",".join(shapes)

    def ping(self) -> bool:
        self.call_endpoint("ping", requires_input=False)
        return True

    def get_modality_config(self) -> dict:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        response = self.call_endpoint(
            "get_action",
            {"observation": observation, "options": options},
        )
        actions, info = tuple(response)
        return actions, info

    def reset(self, options: dict[str, Any] | None = None) -> dict:
        return self.call_endpoint("reset", {"options": options})
