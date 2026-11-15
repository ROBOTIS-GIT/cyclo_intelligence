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
    assert "uint16 action_chunk_size" in dynamic_fields
    assert "string action_request_mode" in dynamic_fields
    assert "bool rlt_enabled" in dynamic_fields
    assert not any("ttrtc_" in field for field in dynamic_fields)



def test_dynamic_engine_command_definitions_match_service_and_runtime():
    import ast
    source = ast.parse((REPO_ROOT / 'cyclo_brain/policy/common/runtime/engine_process/protocol.py').read_text())
    runtime_definition = next(ast.literal_eval(node.value) for node in source.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == 'ENGINE_COMMAND_REQUEST_DEF' for target in node.targets))
    spec = importlib.util.spec_from_file_location('chunk_wire_definitions', MESSAGES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    service = (REPO_ROOT / 'interfaces/srv/EngineCommand.srv').read_text()
    fields = _field_lines(service, stop_at_separator=True)
    assert _field_lines(module.ENGINE_COMMAND_REQUEST_DEF) == fields
    assert _field_lines(runtime_definition) == fields
    assert fields[:2] == ["uint8 command", "uint16 action_chunk_size"]
    assert fields[-1] == "float64[] rtc_prefix_action_list"

    response = _field_lines(service.split("---", 1)[1])
    assert _field_lines(module.ENGINE_COMMAND_RESPONSE_DEF) == response
    runtime_response = next(ast.literal_eval(node.value) for node in source.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "ENGINE_COMMAND_RESPONSE_DEF" for target in node.targets))
    assert _field_lines(runtime_response) == response
    inference_response = _field_lines(SERVICE_PATH.read_text().split("---", 1)[1])
    assert _field_lines(module.INFERENCE_COMMAND_RESPONSE_DEF) == inference_response
