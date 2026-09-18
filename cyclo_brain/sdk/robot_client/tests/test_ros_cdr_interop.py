"""Real generated ROS types versus the Worker's rosbags CDR implementation.

Run after a clean interfaces build, with its install prefix on PYTHONPATH and
LD_LIBRARY_PATH. No ROS node, network connection or robot command is created.
"""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("rclpy")
interfaces = pytest.importorskip("interfaces")
from interfaces.srv import EngineCommand, InferenceCommand, RobotPoseCommand
from rclpy.serialization import deserialize_message, serialize_message
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "cyclo_brain/policy/common/runtime"))
from engine_process.protocol import ENGINE_COMMAND_REQUEST_DEF, ENGINE_COMMAND_RESPONSE_DEF

spec = importlib.util.spec_from_file_location(
    "cdr_robot_messages", ROOT / "cyclo_brain/sdk/robot_client/robot_client/messages/__init__.py"
)
messages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(messages)


def _sample_value(field_type, name, index):
    if field_type == "string":
        return f"{name}/\ud55c\uae00\nquoted:\"value\""
    if field_type == "sequence<string>":
        return [name, "\ud55c\uae00", ""]
    if field_type == "sequence<double>":
        return [0.25, -0.5, 1234.125]
    if field_type == "boolean":
        return index % 2 == 0
    if field_type == "double":
        return index + 0.375
    if field_type == "uint64":
        return 2**48 + index
    if field_type in {"uint8", "uint16", "int32"}:
        return index + 1
    raise AssertionError(f"Add explicit round-trip coverage for new field type: {field_type}")


@pytest.mark.parametrize("service,part,definition", [
    (EngineCommand, "Request", ENGINE_COMMAND_REQUEST_DEF),
    (EngineCommand, "Response", ENGINE_COMMAND_RESPONSE_DEF),
    (InferenceCommand, "Request", messages.INFERENCE_COMMAND_REQUEST_DEF),
    (InferenceCommand, "Response", messages.INFERENCE_COMMAND_RESPONSE_DEF),
    (RobotPoseCommand, "Request", messages.ROBOT_POSE_COMMAND_REQUEST_DEF),
    (RobotPoseCommand, "Response", messages.ROBOT_POSE_COMMAND_RESPONSE_DEF),
])
def test_generated_ros_and_worker_cdr_round_trip(service, part, definition):
    prefix = os.environ.get("CYCLO_TEST_INTERFACES_PREFIX")
    if prefix:
        assert Path(interfaces.__file__).resolve().is_relative_to(Path(prefix).resolve())

    native_type = getattr(service, part)
    fields = native_type.get_fields_and_field_types()
    values = {
        name: _sample_value(field_type, name, index)
        for index, (name, field_type) in enumerate(fields.items())
    }
    native = native_type(**values)
    typename = f"interfaces/srv/{service.__name__}_{part}"
    store = get_typestore(Stores.ROS2_JAZZY)
    definitions = get_types_from_msg(definition, typename)
    # rosbags normalizes a service part to <package>/srv/msg/<name>.
    # Use the parser's key, as the SDK does, rather than assuming a msg path.
    typename, = definitions
    store.register(definitions)
    dynamic = store.deserialize_cdr(serialize_message(native), typename)
    for name, expected in values.items():
        np.testing.assert_array_equal(getattr(dynamic, name), expected, err_msg=name)

    # Independently construct the SDK object, rather than merely reserializing
    # the result of its own decoder. Numeric sequences use the SDK's ndarray API.
    sdk_values = {
        name: np.asarray(value, dtype=np.float64)
        if fields[name] == "sequence<double>" else value
        for name, value in values.items()
    }
    sdk_message = store.types[typename](**sdk_values)
    decoded = deserialize_message(bytes(store.serialize_cdr(sdk_message, typename)), native_type)
    for name, expected in values.items():
        np.testing.assert_array_equal(getattr(decoded, name), expected, err_msg=name)
