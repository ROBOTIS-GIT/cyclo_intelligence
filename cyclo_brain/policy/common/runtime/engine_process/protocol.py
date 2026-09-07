#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Internal Policy Runtime <-> Engine service contract.

External users call ``/policy/inference_command``. This protocol crosses the
Cyclo container boundary to one selected model Worker:

    Policy Runtime  -- EngineCommand srv -->  Engine process

``seq_id`` is intentionally part of both request and response. Timeouts mean
"Runtime stopped waiting", not necessarily "Engine stopped computing", so a
late Engine response can become stale and must be discarded by the requester.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List

import numpy as np


CMD_LOAD_POLICY = 0
CMD_GET_ACTION = 1
CMD_UNLOAD_POLICY = 2
CMD_DESCRIBE = 3
CMD_STATUS = 4
ENGINE_PROTOCOL_VERSION = "1.0"


ENGINE_COMMAND_REQUEST_DEF = """\
uint8 command
uint64 seq_id
string model_path
string embodiment_tag
string robot_type
string task_instruction
string acceleration_mode
string acceleration_engine_path
string policy_id
string policy_parameters_json
"""

ENGINE_COMMAND_RESPONSE_DEF = """\
uint64 seq_id
bool success
string message
string[] action_keys
int32 chunk_size
int32 action_dim
float64[] action_list
string protocol_version
string runtime_id
string worker_instance_id
string[] supported_policy_ids
string capabilities_json
string engine_state
"""


@dataclass
class EngineCommandRequest:
    command: int
    seq_id: int = 0
    model_path: str = ""
    embodiment_tag: str = ""
    robot_type: str = ""
    task_instruction: str = ""
    acceleration_mode: str = ""
    acceleration_engine_path: str = ""
    policy_id: str = ""
    policy_parameters_json: str = ""


@dataclass
class EngineCommandResponse:
    success: bool
    seq_id: int = 0
    message: str = ""
    action_keys: List[str] = field(default_factory=list)
    chunk_size: int = 0
    action_dim: int = 0
    action_list: List[float] = field(default_factory=list)
    protocol_version: str = ""
    runtime_id: str = ""
    worker_instance_id: str = ""
    supported_policy_ids: List[str] = field(default_factory=list)
    capabilities_json: str = "{}"
    engine_state: str = ""


def request_from_message(message: Any) -> EngineCommandRequest:
    """Normalize a ROS/Zenoh request object into a dataclass."""
    return EngineCommandRequest(
        command=int(getattr(message, "command", 0)),
        seq_id=int(getattr(message, "seq_id", 0)),
        model_path=str(getattr(message, "model_path", "") or ""),
        embodiment_tag=str(getattr(message, "embodiment_tag", "") or ""),
        robot_type=str(getattr(message, "robot_type", "") or ""),
        task_instruction=str(getattr(message, "task_instruction", "") or ""),
        acceleration_mode=str(getattr(message, "acceleration_mode", "") or ""),
        acceleration_engine_path=str(
            getattr(message, "acceleration_engine_path", "") or ""
        ),
        policy_id=str(getattr(message, "policy_id", "") or ""),
        policy_parameters_json=str(
            getattr(message, "policy_parameters_json", "") or ""
        ),
    )


def response_from_message(message: Any) -> EngineCommandResponse:
    """Normalize a ROS/Zenoh response object into a dataclass."""
    action_keys = getattr(message, "action_keys", None)
    action_list = getattr(message, "action_list", None)
    supported_policy_ids = getattr(message, "supported_policy_ids", None)
    return EngineCommandResponse(
        success=bool(getattr(message, "success", False)),
        seq_id=int(getattr(message, "seq_id", 0)),
        message=str(getattr(message, "message", "") or ""),
        action_keys=list(action_keys) if action_keys is not None else [],
        chunk_size=int(getattr(message, "chunk_size", 0)),
        action_dim=int(getattr(message, "action_dim", 0)),
        action_list=[float(v) for v in list(action_list)] if action_list is not None else [],
        protocol_version=str(getattr(message, "protocol_version", "") or ""),
        runtime_id=str(getattr(message, "runtime_id", "") or ""),
        worker_instance_id=str(getattr(message, "worker_instance_id", "") or ""),
        supported_policy_ids=(
            list(supported_policy_ids) if supported_policy_ids is not None else []
        ),
        capabilities_json=str(getattr(message, "capabilities_json", "{}") or "{}"),
        engine_state=str(getattr(message, "engine_state", "") or ""),
    )


def response_to_message_kwargs(response: EngineCommandResponse) -> dict:
    """Return kwargs for a generated EngineCommand response class."""
    return {
        "seq_id": int(response.seq_id),
        "success": bool(response.success),
        "message": str(response.message),
        "action_keys": list(response.action_keys),
        "chunk_size": int(response.chunk_size),
        "action_dim": int(response.action_dim),
        "action_list": np.asarray(response.action_list, dtype=np.float64),
        "protocol_version": str(response.protocol_version),
        "runtime_id": str(response.runtime_id),
        "worker_instance_id": str(response.worker_instance_id),
        "supported_policy_ids": list(response.supported_policy_ids),
        "capabilities_json": str(response.capabilities_json or "{}"),
        "engine_state": str(response.engine_state),
    }


def request_to_message_kwargs(request: EngineCommandRequest) -> dict:
    """Return kwargs for a generated EngineCommand request class."""
    return {
        "command": int(request.command),
        "seq_id": int(request.seq_id),
        "model_path": str(request.model_path),
        "embodiment_tag": str(request.embodiment_tag),
        "robot_type": str(request.robot_type),
        "task_instruction": str(request.task_instruction),
        "acceleration_mode": str(request.acceleration_mode),
        "acceleration_engine_path": str(request.acceleration_engine_path),
        "policy_id": str(request.policy_id),
        "policy_parameters_json": str(request.policy_parameters_json),
    }


def flatten_action_list(values: Any) -> List[float]:
    """Convert an ndarray/list action chunk into a plain flat list."""
    if hasattr(values, "reshape"):
        values = values.reshape(-1)
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, Iterable):
        return []
    return [float(v) for v in values]
