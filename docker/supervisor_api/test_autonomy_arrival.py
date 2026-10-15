from pathlib import Path
from types import SimpleNamespace as S
import math
import sys
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from supervisor_api.autonomy_arrival import check_arrival


def transform(stamp, x=1, y=2, yaw=0):
    return S(header=S(stamp=S(sec=int(stamp), nanosec=int((stamp-int(stamp))*1e9))),
             transform=S(translation=S(x=x,y=y), rotation=S(x=0,y=0,z=math.sin(yaw/2),w=math.cos(yaw/2))))

@pytest.mark.parametrize('now', [45., 1791779204.])
def test_sim_and_real_arrival(now):
    check_arrival({'x':1,'y':2}, transform(now-.1), now)

@pytest.mark.parametrize('stamp,now', [(45.,1791779204.), (1791779204.,45.), (40.,45.)])
def test_wrong_clock_or_stale(stamp, now):
    with pytest.raises(RuntimeError, match='clock mismatch or stale'):
        check_arrival({'x':1,'y':2}, transform(stamp), now)

@pytest.mark.parametrize('x,y,yaw', [(8,2,0), (1,2,1), (float('nan'),2,0)])
def test_false_success_rejected(x,y,yaw):
    with pytest.raises(RuntimeError):
        check_arrival({'x':1,'y':2}, transform(45,x,y,yaw),45)


def test_yaw_wrap_and_amcl_future_stamp():
    check_arrival({'x':1,'y':2,'yaw':-math.pi+.01}, transform(46,yaw=math.pi-.01),45)


@pytest.mark.parametrize('now', [45., 1791779204.])
def test_stateful_heading_alignment_drift_is_accepted(now):
    check_arrival({'x':1,'y':2}, transform(now-.1,x=1.273,yaw=.107),now)

@pytest.mark.parametrize('distance,yaw', [(.301,0),(.273,.151)])
def test_heading_drift_allowance_remains_bounded(distance,yaw):
    with pytest.raises(RuntimeError,match='reported success before arrival'):
        check_arrival({'x':1,'y':2},transform(45,x=1+distance,yaw=yaw),45)
