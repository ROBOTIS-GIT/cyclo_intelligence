from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import json,threading,time
import pytest

from orchestrator.bt.waypoint_targets import parse_waypoints,select_waypoint,resolve_waypoint
from orchestrator.bt import waypoint_navigation as nav

TARGET={'id':'start','label':'DockingStart','pose':{'frame_id':'map','x':1.5,'y':-1.6,'yaw':3.14}}

def test_resolves_id_or_unique_name_and_pins_pose():
    assert select_waypoint([TARGET],'start')==select_waypoint([TARGET],'DockingStart')
    result=select_waypoint([TARGET],'start');result['pose']['x']=99
    assert TARGET['pose']['x']==1.5
    with pytest.raises(ValueError):select_waypoint([TARGET],'missing')
    with pytest.raises(ValueError):select_waypoint([TARGET,{**TARGET,'id':'b'}],'DockingStart')
    assert len(parse_waypoints([TARGET,TARGET]))==1


def test_rejects_invalid_context_and_missing_mission():
    for value in ['{}',json.dumps([{**TARGET,'pose':{'x':float('nan'),'y':0,'yaw':0}}])]:
        with pytest.raises(ValueError):parse_waypoints(value)
    with pytest.raises(ValueError):resolve_waypoint([], 'start')
    with pytest.raises(ValueError):resolve_waypoint([], 'start','house','')


def test_explicit_saved_mission_lookup_uses_only_pose():
    response=Mock();response.read.return_value=json.dumps({'exists':True,'waypoints':[{**TARGET,'local_bt':'never-execute.xml'}]}).encode()
    context=Mock();context.__enter__=Mock(return_value=response);context.__exit__=Mock(return_value=False)
    with patch('orchestrator.bt.waypoint_targets.urlopen',return_value=context) as open_url:
        result=resolve_waypoint([], 'DockingStart','house','Test')
    assert result==TARGET and 'local_bt' not in result
    assert 'mission_name=Test' in open_url.call_args.args[0]


class Goal:
    def __init__(self,accepted=True):self.accepted=accepted;self.result=Future();self.cancels=0
    def get_result_async(self):return self.result
    def cancel_goal_async(self):
        self.cancels+=1
        if not self.result.done():self.result.set_result(NS(status=5,result=NS(error_code=0)))
        result=Future();result.set_result(NS(return_code=0));return result
    def finish(self,status=4,error=0):self.result.set_result(NS(status=status,result=NS(error_code=error)))


class Client:
    def __init__(self,*a,**kw):self.requests=[];self.futures=[];self.goals=[];self.destroyed=False
    def server_is_ready(self):return True
    def send_goal_async(self,message):
        self.requests.append(message);future=Future();self.futures.append(future);return future
    def accept(self,index=0,accepted=True):
        goal=Goal(accepted);self.goals.append(goal);self.futures[index].set_result(goal);return goal
    def destroy(self):self.destroyed=True


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(nav,'ActionClient',Client)
    node=NS(create_subscription=lambda *a,**kw:object(),destroy_subscription=lambda sub:None)
    rt=nav.WaypointNavigator(node,TARGET);rt.started-=1
    monkeypatch.setattr(rt,'slow',lambda after=0.:True)
    yield rt
    for index,future in enumerate(rt.client.futures):
        if not future.done():rt.client.accept(index)
    rt.close()


def test_goal_result_and_stationary_feedback(runtime):
    rt=runtime;assert rt.poll() is None
    goal=rt.client.accept();message=rt.client.requests[0]
    assert message.pose.header.frame_id=='map' and message.pose.pose.position.x==1.5
    goal.finish()
    for i in range(3):rt.odom_at=time.monotonic()+i;outcome=rt.poll()
    assert outcome is True


def test_rejected_goal_fails(runtime):
    rt=runtime;rt.poll();rt.client.accept(accepted=False)
    assert rt.poll() is False and 'rejected' in rt.message


def test_pause_cancels_own_goal_and_resume_resends_pose(runtime):
    rt=runtime;rt.poll();goal=rt.client.accept();rt.pause()
    assert goal.cancels==1 and rt.poll() is None
    rt.resume();rt.nav_active=False;rt.poll();rt.client.accept(1)
    assert len(rt.client.requests)==2
    assert rt.client.requests[0].pose==rt.client.requests[1].pose


def test_stop_drains_late_goal_acceptance(runtime):
    rt=runtime;rt.poll();errors=[]
    def close():
        try:rt.close()
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=close);thread.start();time.sleep(.05)
    assert thread.is_alive() and not rt.client.destroyed
    goal=rt.client.accept();thread.join(2)
    assert not errors and not thread.is_alive() and goal.cancels==1
    assert rt.client.destroyed


def test_external_navigation_is_not_preempted(runtime):
    rt=runtime;rt.nav_active=True
    assert rt.poll() is None and rt.client.requests==[]
    assert rt.poll() is False


def test_timeout_cancels_owned_goal(runtime):
    rt=runtime;rt.poll();goal=rt.client.accept();rt.started-=200
    assert rt.poll() is False and goal.cancels==1


def test_failed_initialization_releases_navigation_lease(monkeypatch):
    monkeypatch.setattr(nav,'ActionClient',Mock(side_effect=RuntimeError('unavailable')))
    with pytest.raises(RuntimeError):nav.WaypointNavigator(Mock(),TARGET)
    assert nav._NAV_LEASE.acquire(blocking=False)
    nav._NAV_LEASE.release()


def test_loader_context_is_per_tree_and_no_goal_is_sent_while_editing():
    import rclpy
    from orchestrator.bt.bt_nodes_loader import TreeLoader
    from orchestrator.bt.node_registry import build_registry,validate_registry
    rclpy.init();node=rclpy.create_node('return_waypoint_loader_test')
    try:
        loader=TreeLoader(node)
        import xml.etree.ElementTree as E
        root=E.fromstring('<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><ReturnToWaypoint waypoint="DockingStart"/></BehaviorTree></root>')
        root.set('mission_waypoints',json.dumps([TARGET]))
        action=loader.load_tree_from_string(E.tostring(root,encoding='unicode'))
        assert action.snapshot==[TARGET] and action.runtime is None
        standalone=loader.load_tree_from_string('<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><ReturnToWaypoint waypoint="DockingStart"/></BehaviorTree></root>')
        assert standalone.snapshot==[] and action.snapshot==[TARGET]
        assert validate_registry(build_registry())==[]
    finally:node.destroy_node();rclpy.shutdown()


def test_bt_preserves_goal_owner_until_stop_is_acknowledged():
    from orchestrator.bt.actions.return_to_waypoint import ReturnToWaypoint
    from orchestrator.bt.bt_core import NodeStatus
    action=ReturnToWaypoint(NS(get_logger=lambda:Mock()),waypoint='start')
    action.resolution=Future();action.resolution.set_result(TARGET)
    action.runtime=Mock(poll=Mock(return_value=False),message='timed out')
    action.runtime.close.side_effect=[RuntimeError('not yet stopped'),None]
    assert action.tick()==NodeStatus.RUNNING
    assert not action.failure
    assert action.tick()==NodeStatus.FAILURE
