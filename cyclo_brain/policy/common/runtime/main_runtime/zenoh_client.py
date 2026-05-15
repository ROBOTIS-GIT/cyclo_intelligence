#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Thin adapter around zenoh_ros2_sdk service clients.

The SDK is mounted into the policy container at runtime. Keeping this adapter
small lets unit tests exercise the Main<->Engine contract without importing the
SDK locally.
"""

from __future__ import annotations

from typing import Any

from engine_process.protocol import (
    ENGINE_COMMAND_REQUEST_DEF,
    ENGINE_COMMAND_RESPONSE_DEF,
    EngineCommandRequest,
    request_to_message_kwargs,
)


class ZenohEngineCommandClient:
    def __init__(
        self,
        service_name: str,
        router_ip: str,
        router_port: int,
        domain_id: int,
        node_name: str,
        namespace: str = "/",
    ) -> None:
        try:
            from zenoh_ros2_sdk import ROS2ServiceClient
        except Exception as e:  # pragma: no cover - depends on runtime mount.
            raise RuntimeError(
                "zenoh_ros2_sdk.ROS2ServiceClient is unavailable"
            ) from e

        self._client = ROS2ServiceClient(
            service_name=service_name,
            srv_type="interfaces/srv/EngineCommand",
            request_definition=ENGINE_COMMAND_REQUEST_DEF,
            response_definition=ENGINE_COMMAND_RESPONSE_DEF,
            router_ip=router_ip,
            router_port=router_port,
            domain_id=domain_id,
            node_name=node_name,
            namespace=namespace,
        )

    def call(self, request: EngineCommandRequest, timeout_s: float) -> Any:
        previous_timeout = getattr(self._client, "timeout", None)
        if previous_timeout is not None:
            self._client.timeout = timeout_s
        try:
            return self._client.call(**request_to_message_kwargs(request))
        finally:
            if previous_timeout is not None:
                self._client.timeout = previous_timeout

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()
