from concurrent.futures import Future
from types import SimpleNamespace
import pytest
from orchestrator.bt.actions.home_meal import HomeMeal
from orchestrator.bt.bt_core import NodeStatus

class Client:
    def __init__(self):self.calls=0;self.ready=True;self.future=Future()
    def service_is_ready(self):return self.ready
    def call_async(self, request):self.calls+=1;return self.future

class Node:
    def __init__(self):self.client=Client();self.destroyed=False
    def create_client(self, *args):return self.client
    def destroy_client(self, client):self.destroyed=True
    def get_logger(self):return SimpleNamespace(error=lambda *_:None)

def test_success_acknowledges_only_applied_service_result():
    node=Node();action=HomeMeal(node,'put_sink')
    assert action.tick()==NodeStatus.RUNNING
    assert action.tick()==NodeStatus.RUNNING
    assert node.client.calls==1
    node.client.future.set_result(SimpleNamespace(success=True,message='sink_stacked'))
    assert action.tick()==NodeStatus.SUCCESS
    assert action.tick()==NodeStatus.SUCCESS
    assert node.client.calls==1
    action.destroy();assert node.destroyed

def test_action_can_run_again_after_tree_reset():
    node = Node()
    action = HomeMeal(node, 'pickup_table')
    for cycle in range(2):
        node.client.future = Future()
        assert action.tick() == NodeStatus.RUNNING
        assert action.tick() == NodeStatus.RUNNING
        node.client.future.set_result(SimpleNamespace(success=True, message='carrying_to_sink'))
        assert action.tick() == NodeStatus.SUCCESS
        assert node.client.calls == cycle + 1
        action.reset()
    action.destroy()
    assert node.destroyed


def test_rejected_scene_transition_fails_mission():
    node=Node();action=HomeMeal(node)
    action.tick();node.client.future.set_result(SimpleNamespace(success=False,message='wrong station'))
    assert action.tick()==NodeStatus.FAILURE

def test_unavailable_or_stalled_service_times_out():
    node=Node();node.client.ready=False;action=HomeMeal(node)
    assert action.tick()==NodeStatus.RUNNING
    action.started-=20
    assert action.tick()==NodeStatus.FAILURE
    action.reset();node.client.ready=True
    assert action.tick()==NodeStatus.RUNNING
    action.started-=20
    assert action.tick()==NodeStatus.FAILURE
    assert node.client.future.cancelled()

def test_reject_unknown_operation():
    with pytest.raises(ValueError):HomeMeal(Node(),'unknown')
