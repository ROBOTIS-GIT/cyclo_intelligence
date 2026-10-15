from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from orchestrator.bt.actions.home_tv import HomeTV
from orchestrator.bt.bt_core import NodeStatus


@pytest.mark.parametrize('operation', HomeTV.OPERATIONS)
@pytest.mark.parametrize('check_only', [False, True])
def test_tv_action_endpoint_and_repeatability(operation, check_only):
    node = Mock()
    client = node.create_client.return_value
    client.service_is_ready.return_value = True
    action = HomeTV(node, operation, check_only=check_only)
    assert node.create_client.call_args.args[1] == '/simulation/home_tv/' + ('check_' if check_only else '') + operation
    for cycle in range(2):
        future = Future(); client.call_async.return_value = future
        assert action.tick() == NodeStatus.RUNNING
        future.set_result(SimpleNamespace(success=True, message='applied'))
        assert action.tick() == NodeStatus.SUCCESS
        assert action.tick() == NodeStatus.SUCCESS
        assert client.call_async.call_count == cycle + 1
        action.reset()
    action.destroy()


def test_invalid_tv_operation():
    with pytest.raises(ValueError):
        HomeTV(Mock(), 'pickup_table')
