"""One-time, explicit migration from an observation mission to map settings.

Existing mission IDs and dataset labels are retained. No runtime code relies on
ObservationMode; --observation-mission is simply the caller's migration input.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def migrate(root, map_name, observation_mission, tasks, apply=False):
    root = Path(root)
    if any('/' in name or name in ('.', '..') for name in [map_name, observation_mission, *tasks]):
        raise ValueError('Use map and mission names, not paths')
    map_dir = root / 'maps' / map_name
    plan_path = map_dir / 'autonomy.json'
    if plan_path.exists():
        return {'changed': False, 'reason': 'An observation plan already exists'}
    mission_dir = root / 'missions' / map_name
    legacy_dir = mission_dir / observation_mission
    legacy = json.loads((legacy_dir / 'mission.json').read_text())
    if len(legacy['waypoints']) != 1:
        raise ValueError('Migrate one observation waypoint at a time')
    point = legacy['waypoints'][0]
    old_id = legacy['metadata']['reasoning_mission_id']
    store = json.loads((map_dir / 'spots.json').read_text()) if (map_dir / 'spots.json').exists() else {'schema_version': 1, 'map_name': map_name, 'spots': []}
    def same_pose(a, b): return all(abs(a.get(k, 0) - b.get(k, 0)) < .0001 for k in ('x', 'y', 'yaw'))
    spot = next((s for s in store['spots'] if same_pose(s['pose'], point['pose'])), None)
    if spot is None:
        spot = {'id': point['id'], 'map_name': map_name, 'label': point['label'],
                'pose': point['pose'], 'linked_bt_tree': '', 'metadata': {'coordinate_space': 'map', 'role': 'observation'}}
        store['spots'].append(spot)
    arrival = (legacy_dir / point['local_bt']).read_text() if point.get('local_bt') else ''
    ET.fromstring(arrival) if arrival else None
    plan = {'revision': 1, 'mode': 'fixed', 'settle_seconds': 1, 'interval_seconds': 1,
            'excluded_mission_ids': [old_id], 'observations': [
                {'spot_id': spot['id'], 'dwell_seconds': 15, 'mission_ids': [], 'arrival_bt': arrival}]}
    updates = []
    for name in tasks:
        directory = mission_dir / name
        manifest = json.loads((directory / 'mission.json').read_text())
        tail = manifest['waypoints'][-1]
        if not same_pose(tail['pose'], point['pose']) or 'observation' not in tail['label'].lower():
            continue
        removed_id = tail['id']
        manifest['waypoints'] = manifest['waypoints'][:-1]
        flow = manifest['metadata'].get('mission_flow', {})
        flow['nodes'] = [n for n in flow.get('nodes', []) if n['id'] != removed_id]
        flow['edges'] = [e for e in flow.get('edges', []) if removed_id not in (e['source'], e['target'])]
        manifest['revision'] += 1
        completion = manifest['metadata'].get('completion_condition', '')
        manifest['metadata']['completion_condition'] = completion.replace(' and the robot has returned to observation', '')
        global_path = directory / manifest['global_bt']
        tree = ET.parse(global_path)
        for parent in tree.iter():
            for node in list(parent):
                if node.get('waypoint_id') == removed_id: parent.remove(node)
        updates.append((directory, manifest, global_path, tree))
    if apply:
        map_dir.mkdir(parents=True, exist_ok=True)
        write_json(map_dir / 'spots.json', store)
        for directory, manifest, global_path, tree in updates:
            write_json(directory / 'mission.json', manifest)
            tree.write(global_path, encoding='utf-8', xml_declaration=True)
        write_json(plan_path, plan)
    return {'changed': True, 'applied': apply, 'map': map_name, 'spot': spot['label'],
            'tasks': [manifest['mission_name'] for _, manifest, _, _ in updates]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/navigation'))
    parser.add_argument('--map', required=True)
    parser.add_argument('--observation-mission', required=True)
    parser.add_argument('--tasks', nargs='*', default=[])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(migrate(args.root, args.map, args.observation_mission, args.tasks, args.apply), indent=2))
