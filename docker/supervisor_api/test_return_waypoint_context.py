"""Operate must resolve return targets from the same immutable mission snapshot."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET
from . import autonomy_config as config


def test_operate_injects_all_waypoints_without_modifying_saved_bt(tmp_path):
    original='<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><ReturnToWaypoint waypoint="start"/></BehaviorTree></root>'
    bt=tmp_path/'main.xml';bt.write_text(original)
    def waypoint(wid,local=''):
        row={'id':wid,'label':wid,'pose':{'frame_id':'map','x':1.,'y':2.,'yaw':0.},'local_bt':local}
        return SimpleNamespace(id=wid,model_dump=lambda:row.copy())
    manifest=SimpleNamespace(exists=True,revision=1,metadata={'reasoning_mission_id':'m',
        'mission_flow':{'edges':[{'source':'work','target':'end'}]}},
        waypoints=[waypoint('start'),waypoint('work','main.xml'),waypoint('end')])
    with patch.object(config.missions,'_read_manifest',return_value=manifest),patch.object(config.missions,'_resolve_bt_path',return_value=bt):
        steps=config.AutonomyRepository().mission('house',{'mission_name':'Test','mission_id':'m','revision':1})
    assert [s['id'] for s in steps]==['work','end']
    snapshot=json.loads(ET.fromstring(steps[0]['xml']).get('mission_waypoints'))
    assert [s['id'] for s in snapshot]==['start','work','end']
    assert bt.read_text()==original
