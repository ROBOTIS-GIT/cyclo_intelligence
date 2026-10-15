"""Create a linked Patrol recording for a completed mission's return movement.

This module owns only dataset transitions. The Record runner owns navigation.
"""
from copy import deepcopy
import shutil
from uuid import uuid4

from fastapi import HTTPException
if __package__:
    from .collection import atomic, now, read_json, write_json
    from .collection_episode_contract import CAPTURE_CONTRACT, episode_type_for
else:
    from collection import atomic, now, read_json, write_json
    from collection_episode_contract import CAPTURE_CONTRACT, episode_type_for


def _finish_source(store, source, child):
    source['ended_at'] = source.get('ended_at') or child['started_at']
    source['return_episode_id'] = child['id']
    source['next_episode_id'] = child['id']
    write_json(store.path('raw', source['id'], 'episode.json'), source)


def _existing_child(store, source):
    linked = source.get('return_episode_id') or source.get('next_episode_id')
    if linked:
        try:
            child = store.episode(linked)
        except HTTPException as exc:
            if exc.status_code == 404:
                # A deleted return must not be resurrected by an old retry.
                raise HTTPException(410, 'The return episode was deleted. Start a new recording instead.') from exc
            raise
        if child.get('source_episode_id') != source['id'] or child.get('capture_segment') != 'mission_return':
            raise HTTPException(409, 'The linked return episode is inconsistent. Reload the episode.')
        return child
    # Recover if publication succeeded but the source link could not be saved.
    matches = []
    for path in store.episode_paths(source['map_name']):
        safe = store.path(*path.relative_to(store.root).parts)
        child = read_json(safe)
        if child.get('previous_episode_id', child.get('source_episode_id')) == source['id']:
            matches.append(child)
    if len(matches) > 1:
        raise HTTPException(409, 'More than one return episode references this mission. Reload the dataset.')
    if matches and (matches[0].get('source_episode_id') != source['id'] or matches[0].get('capture_segment') != 'mission_return'):
        raise HTTPException(409, 'This mission already continues in a different recording segment. Reload the episode.')
    return matches[0] if matches else None


def open_return_episode(store, eid):
    """Caller holds the collection lock. Retries never start another episode."""
    source = store.episode(eid)
    if episode_type_for(source) != 'mission':
        raise HTTPException(422, 'Only a Mission episode can create a return recording.')
    existing = _existing_child(store, source)
    if existing:
        if source.get('return_episode_id') != existing['id'] or not source.get('ended_at'):
            _finish_source(store, source, existing)
        return store.get(existing['id'])
    spot_id = source.get('return_observation_spot_id')
    if not spot_id:
        raise HTTPException(422, 'Select a return observation spot before starting the mission episode.')
    view = store.observation_view(source['map_name'])
    ids = {point['spot_id'] for point in view['plan']['observations']}
    spot = next((point for point in view['spots'] if point['id'] == spot_id
                 and (point.get('metadata', {}).get('role') == 'observation' or spot_id in ids)), None)
    if spot is None:
        raise HTTPException(409, 'The return observation spot was deleted. Save this episode and select a new return spot.')
    pinned_spot = source.get('return_observation_spot')
    if not isinstance(pinned_spot, dict) or pinned_spot.get('id') != spot_id:
        raise HTTPException(422, 'The return observation snapshot is missing. Start a new recording.')
    recorded_plan = deepcopy(source.get('recording_plan') or {})
    recorded_point = next((p for p in recorded_plan.get('observations', []) if p['spot_id'] == spot_id), {})
    pinned_spot = {**deepcopy(pinned_spot), 'arrival_bt': pinned_spot.get('arrival_bt', recorded_point.get('arrival_bt', ''))}
    if not source.get('recording_mission_id'):
        raise HTTPException(422, 'The mission episode has no stable mission identity. Start a new recording.')
    catalog = deepcopy(source['catalog']) if source.get('observation_binding') else store.current_catalog(source['map_name'])
    directory = store.next_episode_directory(source['map_name'], 'patrol')
    child = {
        'schema_version': 'reasoning_episode_v4', 'id': str(uuid4()),
        'episode_type': 'patrol', 'capture_contract': CAPTURE_CONTRACT,
        'capture_segment': 'mission_return', 'source_episode_id': source['id'], 'previous_episode_id': source['id'],
        'return_mission_id': source['recording_mission_id'],
        'return_mission_name': source.get('recording_mission_name'),
        'return_observation_spot_id': spot_id, 'return_observation_spot': deepcopy(pinned_spot),
        'patrol_spot_ids': [spot_id], 'recording_plan': recorded_plan,
        **{key: source[key] for key in ('map_name', 'group_id', 'robot_type', 'environment')},
        'started_at': now(), 'ended_at': None, 'storage_path': str(directory),
        'catalog': catalog, 'catalog_snapshot_id': catalog['snapshot_id'],
    }
    child.update({key: source[key] for key in ('capture_mode', 'capture_interval_ms') if key in source})
    if source.get('observation_binding'):
        child.update({key: source[key] for key in ('observation_binding', 'plan_revision', 'source_observation_spot_id')})
    directory.mkdir(parents=True)
    try:
        atomic(directory / 'samples.jsonl', b'')
        write_json(directory / 'episode.json', child)
    except Exception:
        shutil.rmtree(directory)
        raise
    # Publish child before source closure so a failed link write can be recovered
    # without losing images or creating a duplicate on retry.
    _finish_source(store, source, child)
    return store.get(child['id'])
