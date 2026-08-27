from types import SimpleNamespace
from unittest.mock import Mock

from orchestrator.internal.communication.container_service_client import (
    ContainerServiceClient,
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
