from types import SimpleNamespace
from unittest.mock import Mock

from orchestrator.internal.communication.container_service_client import (
    ContainerServiceClient,
    ServiceResponse,
)


def test_set_action_policy_serializes_rlt_robot_override():
    client = ContainerServiceClient(node=None, service_prefix='/groot')
    client._inference_command_client = object()
    client._call_service = Mock(
        return_value=SimpleNamespace(success=True, message='switched')
    )

    result = client.inference_command(
        ContainerServiceClient.CMD_SET_ACTION_POLICY,
        action_policy_mode='rlt',
        rlt_robot_override=True,
    )

    request = client._call_service.call_args.args[1]
    assert result.success is True
    assert request.command == ContainerServiceClient.CMD_SET_ACTION_POLICY
    assert request.action_policy_mode == 'rlt'
    assert request.rlt_robot_override is True


def test_set_action_policy_defaults_rlt_robot_override_to_false():
    client = ContainerServiceClient(node=None, service_prefix='/groot')
    client._inference_command_client = object()
    client._call_service = Mock(
        return_value=SimpleNamespace(success=True, message='switched')
    )

    client.inference_command(
        ContainerServiceClient.CMD_SET_ACTION_POLICY,
        action_policy_mode='base',
    )

    request = client._call_service.call_args.args[1]
    assert request.rlt_robot_override is False


def test_tt_rtc_load_serializes_control_rate():
    client = ContainerServiceClient(node=None, service_prefix='/groot')
    client._inference_command_client = object()
    client._call_service = Mock(return_value=SimpleNamespace(success=True))
    client.inference_command(
        ContainerServiceClient.CMD_LOAD,
        action_request_mode='tt_rtc',
        control_hz=200.0,
    )
    request = client._call_service.call_args.args[1]
    assert request.control_hz == 200.0
    assert request.action_request_mode == 'tt_rtc'


def test_initial_pose_sync_serializes_deferred_activation_handshake():
    client = ContainerServiceClient(node=None, service_prefix='/groot')
    client._inference_command_client = object()
    client._call_service = Mock(return_value=ServiceResponse.from_service_response(
        SimpleNamespace(
            success=True,
            message='loaded',
            initial_pose_sync_duration_s=8.0,
        )
    ))

    result = client.inference_command(
        ContainerServiceClient.CMD_LOAD,
        initial_pose_sync=True,
        initial_pose_sync_duration_s=3.0,
        initial_pose_sync_defer_activation=True,
    )

    request = client._call_service.call_args.args[1]
    assert request.initial_pose_sync is True
    assert request.initial_pose_sync_duration_s == 3.0
    assert request.initial_pose_sync_defer_activation is True
    assert result.data['initial_pose_sync_duration_s'] == 8.0
