#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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
ENGINE_SERVICE_PATH = REPO_ROOT / "interfaces" / "srv" / "EngineCommand.srv"
ENGINE_PROTOCOL_PATH = (
    REPO_ROOT
    / "cyclo_brain"
    / "policy"
    / "common"
    / "runtime"
    / "engine_process"
    / "protocol.py"
)
CMAKE_PATH = REPO_ROOT / "interfaces" / "CMakeLists.txt"


def _load_definitions():
    spec = importlib.util.spec_from_file_location(
        "robot_client_message_definitions", MESSAGES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_engine_definitions():
    module_name = "policy_engine_protocol_definitions"
    spec = importlib.util.spec_from_file_location(module_name, ENGINE_PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def _service_response_fields(path: Path) -> list[str]:
    _request, separator, response = path.read_text(encoding="utf-8").partition("---")
    assert separator
    return _field_lines(response)


def test_dynamic_inference_command_definition_matches_ros_service() -> None:
    service_fields = _field_lines(
        SERVICE_PATH.read_text(encoding="utf-8"), stop_at_separator=True
    )
    dynamic_fields = _field_lines(_load_definitions().INFERENCE_COMMAND_REQUEST_DEF)

    assert dynamic_fields == service_fields
    assert "int32 action_steps" in dynamic_fields
    assert "int32 action_steps -1" in _load_definitions().INFERENCE_COMMAND_REQUEST_DEF
    assert dynamic_fields[-4:] == [
        "bool initial_pose_sync",
        "float64 initial_pose_sync_duration_s",
        "string policy_id",
        "string policy_parameters_json",
    ]
    dynamic_response_fields = _field_lines(
        _load_definitions().INFERENCE_COMMAND_RESPONSE_DEF
    )
    assert dynamic_response_fields == _service_response_fields(SERVICE_PATH)
    assert dynamic_response_fields[-16:] == [
        "string runtime_state",
        "string loaded_model_path",
        "string loaded_policy_id",
        "string loaded_policy_parameters_json",
        "bool publish_to_robot",
        "string loaded_action_request_mode",
        "string loaded_acceleration_mode",
        "string loaded_acceleration_engine_path",
        "uint16 loaded_control_hz",
        "uint16 loaded_inference_hz",
        "float64 loaded_chunk_align_window_s",
        "int32 loaded_action_steps",
        "int32 observed_chunk_size",
        "bool loaded_initial_pose_sync",
        "float64 loaded_initial_pose_sync_duration_s",
        "string runtime_error",
    ]


def test_action_steps_native_ros_and_worker_cdr_roundtrip():
    serialization = pytest.importorskip('rclpy.serialization')
    service = pytest.importorskip('interfaces.srv').InferenceCommand
    typesys = pytest.importorskip('rosbags.typesys')
    definitions = _load_definitions()
    store = typesys.get_typestore(typesys.Stores.EMPTY)
    for native_type, name, definition, field in (
        (service.Request, 'interfaces/srv/InferenceCommand_Request',
         definitions.INFERENCE_COMMAND_REQUEST_DEF, 'action_steps'),
        (service.Response, 'interfaces/srv/InferenceCommand_Response',
         definitions.INFERENCE_COMMAND_RESPONSE_DEF, 'loaded_action_steps'),
        (service.Response, 'interfaces/srv/InferenceCommand_Response',
         definitions.INFERENCE_COMMAND_RESPONSE_DEF, 'observed_chunk_size'),
    ):
        parsed = typesys.get_types_from_msg(definition, name)
        store.register(parsed)
        typename = next(iter(parsed))
        for steps in (0, 10, 2147483647):
            native = native_type(**{field: steps})
            worker = store.deserialize_cdr(serialization.serialize_message(native), typename)
            assert getattr(worker, field) == steps
            restored = serialization.deserialize_message(bytes(store.serialize_cdr(worker, typename)), native_type)
            assert getattr(restored, field) == steps


def test_dynamic_engine_command_definition_matches_ros_service() -> None:
    service_fields = _field_lines(
        ENGINE_SERVICE_PATH.read_text(encoding="utf-8"), stop_at_separator=True
    )
    definitions = _load_engine_definitions()
    dynamic_fields = _field_lines(definitions.ENGINE_COMMAND_REQUEST_DEF)

    assert dynamic_fields == service_fields
    assert dynamic_fields[-3:] == [
        "string policy_id",
        "string policy_parameters_json",
        "string execution_context_json",
    ]
    dynamic_response_fields = _field_lines(
        definitions.ENGINE_COMMAND_RESPONSE_DEF
    )
    assert dynamic_response_fields == _service_response_fields(ENGINE_SERVICE_PATH)
    assert dynamic_response_fields[-6:] == [
        "string protocol_version",
        "string runtime_id",
        "string worker_instance_id",
        "string[] supported_policy_ids",
        "string capabilities_json",
        "string engine_state",
    ]


def test_runtime_services_are_registered_for_rosidl_generation() -> None:
    cmake = CMAKE_PATH.read_text(encoding="utf-8")

    assert '"srv/InferenceCommand.srv"' in cmake
    assert '"srv/EngineCommand.srv"' in cmake
    assert '"srv/RobotPoseCommand.srv"' in cmake


def test_dynamic_pose_command_definition_matches_ros_service():
    path = REPO_ROOT / "interfaces/srv/RobotPoseCommand.srv"
    definitions = _load_definitions()
    assert _field_lines(definitions.ROBOT_POSE_COMMAND_REQUEST_DEF) == _field_lines(
        path.read_text(), stop_at_separator=True
    )
    assert _field_lines(definitions.ROBOT_POSE_COMMAND_RESPONSE_DEF) == _service_response_fields(path)
