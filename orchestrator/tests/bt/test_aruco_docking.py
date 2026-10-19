import math
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from orchestrator.bt.docking_core import DockConfig, dock_error, velocity, obstacle_in_motion
from orchestrator.bt.docking_vision import MarkerDetector, transform_pose


def test_front_and_side_offsets():
    assert dock_error([1.35, .2, 1.3],[-1,0,0],.85) == pytest.approx((.5,.2,0))
    assert velocity((0,0,0),DockConfig()) == ((0.,0.,0.),True)
    with pytest.raises(ValueError):dock_error([1,0,0],[1,0,0],.85)
    with pytest.raises(ValueError):dock_error([1,0,0],[0,0,1],.85)


@pytest.mark.parametrize('start',[(1.8,.2,.2),(1.4,-.3,-.3),(1.1,.1,.5)])
def test_closed_loop_converges(start):
    # World marker at origin, outward normal toward -X; robot faces +X.
    x,y,yaw=-start[0],start[1],start[2]; cfg=DockConfig(); settled=False
    for _ in range(2400):
        c,s=math.cos(yaw),math.sin(yaw)
        p=[-c*x-s*y,s*x-c*y,1.3]; n=[-c,s,0]
        command,settled=velocity(dock_error(p,n,cfg.distance),cfg)
        assert math.hypot(*command[:2]) <= cfg.max_speed+1e-9
        vx,vy,w=command
        x+=(c*vx-s*vy)*.05;y+=(s*vx+c*vy)*.05;yaw+=w*.05
        if settled:break
    assert settled
    assert math.hypot(x+.85,y) <= cfg.xy_tolerance


def test_obstacle_direction_and_rotation():
    assert obstacle_in_motion([(0.48,0)],(.1,0,0))
    assert not obstacle_in_motion([(-.5,0)],(.1,0,0))
    assert obstacle_in_motion([(0,.435)],(0,0,.1))


@pytest.mark.parametrize('marker_id',[10,11,12])
def test_detection_id_size_pose(marker_id):
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker=(cv2.aruco.generateImageMarker(dictionary,marker_id,180) if hasattr(cv2.aruco,'generateImageMarker')
            else cv2.aruco.drawMarker(dictionary,marker_id,180))
    image=np.full((480,640),255,np.uint8); image[150:330,230:410]=marker
    k=[600,0,320,0,600,240,0,0,1]
    result=MarkerDetector(marker_id,.3).detect(image,k,np.zeros(5))
    assert result is not None
    p,n,_,rms=result
    assert p[2] == pytest.approx(1.,abs=.015)
    assert n[2] < -.98 and rms < .1
    assert MarkerDetector(49,.3).detect(image,k,np.zeros(5)) is None
    # Optical forward (+Z) -> base forward (+X); optical right -> base right (-Y).
    tf=SimpleNamespace(rotation=SimpleNamespace(x=-.5,y=.5,z=-.5,w=.5),
                       translation=SimpleNamespace(x=.1,y=0,z=1.3))
    pb,nb=transform_pose(tf,p,n)
    err=dock_error(pb,nb,.85)
    assert err[0] == pytest.approx(.25,abs=.02)
    assert abs(err[1]) < .02 and abs(err[2]) < .02


@pytest.mark.parametrize('kwargs',[{'distance':.1},{'max_speed':float('nan')},{'timeout':0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):DockConfig(**kwargs)


def test_stop_pause_and_navigation_cannot_publish_motion():
    import threading
    from orchestrator.bt.docking_runtime import DockRuntime
    rt=DockRuntime.__new__(DockRuntime)
    rt.lock=threading.RLock();rt.halt=threading.Event();rt.paused=threading.Event()
    rt.leased=True;rt.nav_active=False
    emitted=[];rt.pub=SimpleNamespace(publish=emitted.append)
    rt.publish((.1,.02,.1));assert emitted[-1].linear.x == .1
    rt.paused.set();rt.publish((.1,0,0));assert emitted[-1].linear.x == 0
    rt.paused.clear();rt.halt.set();rt.publish((.1,0,0));assert emitted[-1].linear.x == 0
    rt.halt.clear();rt.nav_active=True;rt.publish((.1,0,0));assert emitted[-1].linear.x == 0
    rt.leased=False;count=len(emitted);rt.publish((.1,0,0));assert len(emitted)==count


def test_simulation_stale_and_future_frames():
    import time
    from orchestrator.bt.docking_runtime import DockRuntime
    rt=DockRuntime.__new__(DockRuntime);rt.clock=100.;rt.clock_at=time.monotonic()
    assert rt.source_fresh(SimpleNamespace(sec=99,nanosec=800000000))
    assert not rt.source_fresh(SimpleNamespace(sec=98,nanosec=0))
    assert not rt.source_fresh(SimpleNamespace(sec=101,nanosec=0))
    rt.clock_at-=1
    assert not rt.source_fresh(SimpleNamespace(sec=100,nanosec=0))


def test_xml_loader_and_catalog():
    import rclpy
    from orchestrator.bt.node_registry import build_registry,validate_registry,catalog_payload
    from orchestrator.bt.bt_nodes_loader import TreeLoader
    registry=build_registry();assert 'ArucoDock' in registry
    assert validate_registry(registry)==[]
    rclpy.init();node=rclpy.create_node('test_aruco_catalog')
    try:
        tree=TreeLoader(node).load_tree_from_string('<root BTCPP_format="4" main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><ArucoDock marker_id="12" marker_size="0.30"/></BehaviorTree></root>')
        assert tree.marker_id==12 and tree.config.distance==.85
        assert tree.runtime is None  # Building/editing a tree must not publish motion.
        tree.reset()
    finally:node.destroy_node();rclpy.shutdown()


def test_final_approach_exceeds_swerve_deadband():
    command,done=velocity((.04,0,0),DockConfig())
    assert not done and .03 < command[0] <= .1
    command,done=velocity((0,0,.07),DockConfig())
    assert not done and .03 < command[2] <= .18
    command,done=velocity((.02,.02,.02),DockConfig())
    assert done and command==(0.,0.,0.)


def test_xml_blank_optional_fields_use_defaults():
    from orchestrator.bt.actions.aruco_dock import ArucoDock
    action = ArucoDock.from_xml_params(SimpleNamespace(node=None), 'Dock10', {
        'marker_id': '10', 'lost_timeout': '', 'max_yaw_rate': ' ',
        'image_topic': '', 'odom_topic': '', 'target_distance': '.35'})
    assert action.config.distance == .35
    assert action.config.lost_timeout == 3.
    assert action.config.max_yaw_rate == .18
    assert action.transport[:2] == ('', '')
    assert action.camera_profile == 'VGA'
    assert action.transport[4] == '/odom'
    with pytest.raises(ValueError, match='ArucoDock Dock10: distance'):
        ArucoDock.from_xml_params(SimpleNamespace(node=None), 'Dock10', {'target_distance': '.349'})


def test_close_docking_clearance_keeps_collision_guard():
    assert not obstacle_in_motion([(.55, 0)], (.1, 0, 0))
    assert obstacle_in_motion([(.48, 0)], (.1, 0, 0))
    assert obstacle_in_motion([(.45, 0)], (.035, 0, 0))
    assert obstacle_in_motion([(.35, 0)], (.035, 0, 0))
    assert obstacle_in_motion([(0, .435)], (0, 0, .1))


def test_close_docking_margin_keeps_body_and_motion_guard():
    assert not obstacle_in_motion([(.48, 0)], (.035, 0, 0))
    assert obstacle_in_motion([(.45, 0)], (.035, 0, 0))
    assert obstacle_in_motion([(.42, 0)], (.035, 0, 0))


def test_high_resolution_profile_and_terminal_cleanup():
    from orchestrator.bt.actions.aruco_dock import ArucoDock
    from orchestrator.bt.bt_core import NodeStatus
    from unittest.mock import Mock
    action = ArucoDock(None, camera_profile='high_resolution')
    assert action.camera_profile == 'HD720'
    runtime = Mock(result=True)
    action.runtime = runtime
    assert action.tick() == NodeStatus.SUCCESS
    runtime.close.assert_called_once()
    assert action.runtime is None
    assert action.tick() == NodeStatus.SUCCESS
    action.reset()
    assert action.outcome is None
    with pytest.raises(ValueError, match='camera_profile'):
        ArucoDock(None, camera_profile='invalid')
