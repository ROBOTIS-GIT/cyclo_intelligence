#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Internal Main <-> Engine service contract.

External users still call ``/<backend>/inference_command``. This protocol is
only for the two Python processes inside a policy container:

    Main process  -- EngineCommand srv -->  Engine process

``seq_id`` is intentionally part of both request and response. Timeouts mean
"Main stopped waiting", not necessarily "Engine stopped computing", so a late
Engine response can become stale and must be discarded by the requester.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List

import numpy as np


CMD_LOAD_POLICY = 0
CMD_GET_ACTION = 1
CMD_UNLOAD_POLICY = 2


ENGINE_COMMAND_REQUEST_DEF = """\
uint8 command
uint64 seq_id
string model_path
string embodiment_tag
string robot_type
string task_instruction
"""

ENGINE_COMMAND_RESPONSE_DEF = """\
uint64 seq_id
bool success
string message
string[] action_keys
int32 chunk_size
int32 action_dim
float64[] action_list
"""


@dataclass
class EngineCommandRequest:
    command: int
    seq_id: int = 0
    model_path: str = ""
    embodiment_tag: str = ""
    robot_type: str = ""
    task_instruction: str = ""


@dataclass
class EngineCommandResponse:
    success: bool
    seq_id: int = 0
    message: str = ""
    action_keys: List[str] = field(default_factory=list)
    chunk_size: int = 0
    action_dim: int = 0
    action_list: List[float] = field(default_factory=list)


def request_from_message(message: Any) -> EngineCommandRequest:
    """Normalize a ROS/Zenoh request object into a dataclass."""
    return EngineCommandRequest(
        command=int(getattr(message, "command", 0)),
        seq_id=int(getattr(message, "seq_id", 0)),
        model_path=str(getattr(message, "model_path", "") or ""),
        embodiment_tag=str(getattr(message, "embodiment_tag", "") or ""),
        robot_type=str(getattr(message, "robot_type", "") or ""),
        task_instruction=str(getattr(message, "task_instruction", "") or ""),
    )


def response_from_message(message: Any) -> EngineCommandResponse:
    """Normalize a ROS/Zenoh response object into a dataclass."""
    action_keys = getattr(message, "action_keys", None)
    action_list = getattr(message, "action_list", None)
    return EngineCommandResponse(
        success=bool(getattr(message, "success", False)),
        seq_id=int(getattr(message, "seq_id", 0)),
        message=str(getattr(message, "message", "") or ""),
        action_keys=list(action_keys) if action_keys is not None else [],
        chunk_size=int(getattr(message, "chunk_size", 0)),
        action_dim=int(getattr(message, "action_dim", 0)),
        action_list=[float(v) for v in list(action_list)] if action_list is not None else [],
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
