import threading
from types import SimpleNamespace as S
import pytest
from orchestrator.bt.actions.dock_retreat import DockRetreat
from orchestrator.bt.docking_retreat import DockRetreatRuntime,retreat_progress
from orchestrator.bt.docking_runtime import DockRuntime

def test_progress_tracks_initial_heading():
    assert retreat_progress((0,0,0),(-.2,0,0))==(.2,0,0)
    p,l,h=retreat_progress((0,0,1.57079632679),(0,-.2,1.57079632679))
    assert p==pytest.approx(.2);assert abs(l)<1e-8;assert h==0

def test_retreat_watchdog_does_not_require_images(monkeypatch):
    r=DockRetreatRuntime.__new__(DockRetreatRuntime);r.lock=threading.RLock();r.leased=True
    r.last_command=r.scan_at=r.odom_at=10;r.nav_active=False;sent=[];r.publish=sent.append
    monkeypatch.setattr('orchestrator.bt.docking_retreat.time.monotonic',lambda:10.1)
    r.watch();assert sent==[]
    r.scan_at=9;r.watch();assert sent==[(0.,0.,0.)]

def harness(monkeypatch,block=False,stale=False):
    now=[100.];monkeypatch.setattr('orchestrator.bt.docking_retreat.time.monotonic',lambda:now[0])
    r=DockRetreatRuntime.__new__(DockRetreatRuntime);r.lock=threading.RLock();r.leased=False;r.distance=.2;r.speed=.05;r.config=S(timeout=20)
    r.paused=threading.Event();r.nav_active=False;r.nav_received=True;r.message='';r.result=None
    r.node=S(count_publishers=lambda _:0,get_logger=lambda:S(error=lambda _:None));r.diagnostics=S(save_async=lambda *args:None)
    odom=S(header=S(frame_id='odom',stamp=None),twist=S(twist=S(linear=S(x=0,y=0),angular=S(z=0))))
    r.odom=odom;r.scan=S();r.source_fresh=lambda _:not stale;r.rear_visible=lambda _:None
    sent=[];r.publish=sent.append
    pose=[0.,0.,0.]
    def wait(_):
        now[0]+=.1;r.scan_at=r.odom_at=now[0]
        if sent and sent[-1][0]<0:pose[0]-=.005
        return False
    r.halt=S(wait=wait);r.pose=lambda _:tuple(pose)
    def check(*args):
        if block:raise ValueError('rear obstacle')
    r.check_obstacles=check
    return r,sent,pose

def test_retreat_finishes_only_after_distance_and_stationary(monkeypatch):
    r,sent,pose=harness(monkeypatch);r.run()
    assert r.result is True;assert -.21<=pose[0]<=-.195
    assert sent[-1]==(0.,0.,0.);assert any(c[0]<0 for c in sent)

@pytest.mark.parametrize('block,stale',[(True,False),(False,True)])
def test_retreat_holds_and_fails_without_valid_clear_feedback(monkeypatch,block,stale):
    r,sent,_=harness(monkeypatch,block,stale);r.run()
    assert r.result is False;assert all(c==(0.,0.,0.) for c in sent)

def test_rear_visibility_rejects_front_only_scan():
    r=DockRetreatRuntime.__new__(DockRetreatRuntime)
    r.tf=lambda *args:S(rotation=S(x=0,y=0,z=0,w=1))
    scan=S(header=S(frame_id='scan',stamp=None),angle_min=-.1,angle_increment=.01,ranges=[float('inf')]*20,range_min=.01,range_max=30)
    with pytest.raises(ValueError,match='Rear scan'):r.rear_visible(scan)
    scan.angle_min=-3.14159;scan.angle_increment=6.28318/360;scan.ranges=[float('inf')]*361
    r.rear_visible(scan)
