from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from orchestrator.bt.actions.home_laundry import HomeLaundry
from orchestrator.bt.bt_core import NodeStatus


@pytest.mark.parametrize('operation', HomeLaundry.OPERATIONS)
@pytest.mark.parametrize('check_only', [False, True])
def test_laundry_action_endpoint_and_repeatability(operation, check_only):
    node = Mock()
    client = node.create_client.return_value
    client.service_is_ready.return_value = True
    action = HomeLaundry(node, operation, check_only=check_only)
    assert node.create_client.call_args.args[1] == '/simulation/home_laundry/' + ('check_' if check_only else '') + operation
    for cycle in range(2):
        future = Future(); client.call_async.return_value = future
        assert action.tick() == NodeStatus.RUNNING
        future.set_result(SimpleNamespace(success=True, message='applied'))
        assert action.tick() == NodeStatus.SUCCESS
        assert action.tick() == NodeStatus.SUCCESS
        assert client.call_async.call_count == cycle + 1
        action.reset()
    action.destroy()


def test_invalid_laundry_operation():
    with pytest.raises(ValueError):
        HomeLaundry(Mock(), 'pickup_table')


def test_pause_waits_for_accepted_transition_and_resume_does_not_repeat_it():
    import threading
    import time
    node = Mock()
    client = node.create_client.return_value
    client.service_is_ready.return_value = True
    future = Future(); client.call_async.return_value = future
    action = HomeLaundry(node, 'pickup_basket')
    assert action.tick() == NodeStatus.RUNNING
    paused = threading.Event()
    def pause():
        action.pause(); paused.set()
    worker = threading.Thread(target=pause)
    worker.start()
    try:
        assert not paused.wait(.03)
        future.set_result(SimpleNamespace(success=True, message='applied'))
        assert paused.wait(1)
        action.resume()
        assert action.tick() == NodeStatus.SUCCESS
        assert client.call_async.call_count == 1
    finally:
        if not future.done(): future.set_result(SimpleNamespace(success=True, message='applied'))
        worker.join(1)
        action.destroy()
