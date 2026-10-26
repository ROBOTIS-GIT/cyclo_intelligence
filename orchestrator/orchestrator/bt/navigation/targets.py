"""Resolve a named mission waypoint without replaying its local behavior tree."""
from copy import deepcopy
import json
import math
import os
from urllib.parse import urlencode, quote
from urllib.request import urlopen


def parse_waypoints(value):
    rows = json.loads(value) if isinstance(value, str) and value else value or []
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError('Invalid mission waypoint context')
    output = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('A mission waypoint requires an ID')
        pose = row.get('pose') or {}
        if pose.get('frame_id', 'map') != 'map':
            raise ValueError('ReturnToWaypoint requires map-frame poses')
        values = {k: float(pose[k]) for k in ('x', 'y', 'yaw')}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Non-finite waypoint pose')
        label = row.get('label') or row['id']
        if not isinstance(label, str):
            raise ValueError('Invalid waypoint label')
        normalized = {'id': row['id'], 'label': label, 'pose': {'frame_id': 'map', **values}}
        if row['id'] in seen:
            previous = next(w for w in output if w['id'] == row['id'])
            if previous['pose'] != normalized['pose']:
                raise ValueError('Conflicting poses for a mission waypoint')
            continue
        output.append(normalized)
        seen.add(row['id'])
    return output


def select_waypoint(rows, target):
    rows = parse_waypoints(rows)
    matches = [w for w in rows if w['id'] == target]
    if not matches:
        matches = [w for w in rows if w['label'] == target]
    if len(matches) != 1:
        raise ValueError('Select one existing waypoint by ID or unique name: ' + target)
    return deepcopy(matches[0])


def resolve_waypoint(snapshot, target, map_name='', mission_name=''):
    if bool(map_name) != bool(mission_name):
        raise ValueError('Supply both map_name and mission_name, or leave both empty for this mission')
    if map_name:
        port = int(os.environ.get('CYCLO_SUPERVISOR_API_PORT', '7100'))
        if not 1 <= port <= 65535: raise ValueError('Invalid supervisor port')
        url = f'http://127.0.0.1:{port}/navigation/missions/' + quote(map_name, safe='')
        url += '?' + urlencode({'mission_name': mission_name})
        with urlopen(url, timeout=5) as response:
            data = json.load(response)
        if not data.get('exists'):
            raise ValueError('Selected mission does not exist')
        snapshot = data.get('waypoints')
    elif not snapshot:
        raise ValueError('Mission context unavailable. Supply map_name and mission_name in Action Canvas')
    return select_waypoint(snapshot, target)
