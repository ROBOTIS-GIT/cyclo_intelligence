#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
MESSAGES_PATH = (
    REPO_ROOT
    / "cyclo_brain"
    / "sdk"
    / "robot_client"
    / "robot_client"
    / "messages"
    / "__init__.py"
)
SERVICE_PATH = REPO_ROOT / "interfaces" / "srv" / "InferenceCommand.srv"


def _load_request_definition() -> str:
    spec = importlib.util.spec_from_file_location(
        "robot_client_message_definitions", MESSAGES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.INFERENCE_COMMAND_REQUEST_DEF


def _field_lines(definition: str, *, stop_at_separator: bool = False) -> list[str]:
    fields = []
    for raw_line in definition.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if stop_at_separator and line == "---":
            break
        if not line or "=" in line:
            continue
        field_type, field_name, *_ = line.split()
        fields.append(f"{field_type} {field_name}")
    return fields


def test_dynamic_inference_command_definition_matches_ros_service() -> None:
    service_fields = _field_lines(
        SERVICE_PATH.read_text(encoding="utf-8"), stop_at_separator=True
    )
    dynamic_fields = _field_lines(_load_request_definition())

    assert dynamic_fields == service_fields
    assert dynamic_fields[-5:] == [
        "uint16 control_hz",
        "uint16 inference_hz",
        "float64 chunk_align_window_s",
        "bool initial_pose_sync",
        "float64 initial_pose_sync_duration_s",
    ]
