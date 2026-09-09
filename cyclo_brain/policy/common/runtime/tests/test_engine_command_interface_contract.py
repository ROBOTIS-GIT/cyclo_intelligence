#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from engine_process.protocol import (  # noqa: E402
    ENGINE_COMMAND_REQUEST_DEF,
    ENGINE_COMMAND_RESPONSE_DEF,
    EngineCommandRequest,
    EngineCommandResponse,
    request_to_message_kwargs,
    response_to_message_kwargs,
)


def _definition_fields(definition: str) -> list[tuple[str, str]]:
    """Return ordered ``(type, name)`` fields, excluding ROS constants."""

    parsed: list[tuple[str, str]] = []
    for raw_line in definition.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" in line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise AssertionError(f"unexpected ROS interface line: {raw_line!r}")
        parsed.append((parts[0], parts[1]))
    return parsed


def _engine_command_service_path() -> Path | None:
    candidates = [
        REPOSITORY_ROOT / "interfaces" / "srv" / "EngineCommand.srv",
        Path(
            os.environ.get(
                "CYCLO_INTELLIGENCE_SOURCE",
                "/root/ros2_ws/src/cyclo_intelligence",
            )
        )
        / "interfaces"
        / "srv"
        / "EngineCommand.srv",
    ]
    return next((path for path in candidates if path.is_file()), None)


class EngineCommandInterfaceContractTests(unittest.TestCase):
    def test_ros_interface_matches_embedded_zenoh_definitions(self) -> None:
        service_path = _engine_command_service_path()
        if service_path is None:
            self.skipTest("canonical EngineCommand.srv is not mounted")
        request_source, response_source = service_path.read_text(
            encoding="utf-8"
        ).split("---", 1)

        self.assertEqual(
            _definition_fields(request_source),
            _definition_fields(ENGINE_COMMAND_REQUEST_DEF),
        )
        self.assertEqual(
            _definition_fields(response_source),
            _definition_fields(ENGINE_COMMAND_RESPONSE_DEF),
        )

    def test_request_dataclass_and_serializer_cover_every_wire_field(self) -> None:
        wire_field_names = [
            name for _field_type, name in _definition_fields(ENGINE_COMMAND_REQUEST_DEF)
        ]
        dataclass_field_names = [field.name for field in fields(EngineCommandRequest)]
        self.assertEqual(dataclass_field_names, wire_field_names)

        request = EngineCommandRequest(command=0)
        serialized = request_to_message_kwargs(request)
        self.assertEqual(list(serialized), wire_field_names)

        # Exercise construction against a generated-message-shaped object so
        # a missing TT-RTC field cannot hide behind permissive mocks.
        generated_shape = SimpleNamespace(**serialized)
        for field_name in (
            "action_request_mode",
            "rtc_delay_steps",
            "rtc_action_dim",
            "rtc_prefix_action_list",
        ):
            self.assertTrue(hasattr(generated_shape, field_name))

    def test_response_dataclass_and_serializer_cover_every_wire_field(self) -> None:
        wire_field_names = [
            name for _field_type, name in _definition_fields(ENGINE_COMMAND_RESPONSE_DEF)
        ]
        dataclass_field_names = [field.name for field in fields(EngineCommandResponse)]
        # The Python dataclass leads with success for ergonomic construction;
        # wire order is owned by the explicit serializer below.
        self.assertCountEqual(dataclass_field_names, wire_field_names)

        response = EngineCommandResponse(success=True)
        serialized = response_to_message_kwargs(response)
        self.assertEqual(list(serialized), wire_field_names)


if __name__ == "__main__":
    unittest.main()
