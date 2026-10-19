import json,re
from pathlib import Path
import numpy as np
import pytest
from orchestrator.bt.docking_footprint import SG2_BASE_OUTLINE, SG2_BASE_FOOTPRINT, swept_obstacle

def test_exact_studio_geometry_and_wheel_coverage():
    s=(Path(__file__).resolve().parents[2] / 'ui/src/components/navigation/sg2TopDownGeometry.js').read_text()
    parts=[json.loads(line.strip().rstrip(',')) for line in s.splitlines() if line.strip().startswith('{"name":')]
    assert next(p['points'] for p in parts if p['name']=='base')==SG2_BASE_OUTLINE
    for p in parts:
        if 'wheel' in p['name']:
            for point in p['points']: assert swept_obstacle([point],(.001,0,0))

def test_convex_ccw():
    p=np.array(SG2_BASE_FOOTPRINT);e=np.roll(p,-1,axis=0)-p
    assert np.all(np.cross(e,np.roll(e,-1,axis=0))>=-1e-10)

def test_front_45cm_is_free_but_base_and_swept_space_blocked():
    assert swept_obstacle([(.45,0)],(.1,0,0)) is None
    assert swept_obstacle([(.22,0)],(.1,0,0))
    assert swept_obstacle([(.27,0)],(.1,0,0))

def test_rear_and_side():
    assert swept_obstacle([(-.44,0)],(-.08,0,0))
    assert swept_obstacle([(-.5,0)],(-.08,0,0)) is None
    assert swept_obstacle([(0,.33)],(0,.08,0))

def test_rotation_sweeps_corners():
    p=np.array(SG2_BASE_FOOTPRINT)[8];a=.2
    point=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@p
    assert swept_obstacle([point],(0,0,.4))

def test_nonfinite_rejected_and_zero_no_motion():
    with pytest.raises(ValueError):swept_obstacle([(float('nan'),0)],(.1,0,0))
    assert swept_obstacle([(0,0)],(0,0,0)) is None
