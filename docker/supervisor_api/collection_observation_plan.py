"""Bind new Record episodes to a saved observation plan, without rewriting old data."""
from copy import deepcopy
from fastapi import HTTPException

BINDING = 'observation_plan_v1'

def departure_context(plan, spot_id):
    """Record follows its saved route, independently of an Operate fixed mode."""
    points = plan.get('observations', [])
    index = next((i for i, p in enumerate(points) if p['spot_id'] == spot_id), None)
    if index is None:
        return {'can_advance': False, 'next_spot_id': None}
    can_advance = len(points) > 1 and (plan.get('route_closed', True) or index + 1 < len(points))
    return {'can_advance': can_advance,
            'next_spot_id': points[(index + 1) % len(points)]['spot_id'] if can_advance else None,
            'departure_policy': points[index].get('departure_policy', 'reasoning')}



def episode_binding(request, kind, view, missions):
    if request.plan_revision is None:
        return {}  # Existing clients and historical episodes retain their contract.
    if request.capture_contract != 'task_aligned_v1':
        raise HTTPException(422, 'Observation plans require typed recording.')
    plan = view['plan']
    if request.plan_revision != plan['revision']:
        raise HTTPException(409, 'The observation plan changed. Refresh it in Record before starting an episode.')
    points = {p['spot_id']: p for p in plan['observations']}
    if kind == 'patrol':
        if any(sid not in points for sid in request.patrol_spot_ids):
            raise HTTPException(422, 'Every patrol stop must belong to the saved observation plan.')
        return {'observation_binding': BINDING, 'plan_revision': plan['revision']}
    point = points.get(request.observation_spot_id)
    if point is None:
        raise HTTPException(422, 'Select an observation in the saved Record plan.')
    excluded = set(plan.get('excluded_mission_ids', []))
    available = {m['mission_id'] for m in missions}
    candidates = [mid for mid in point.get('mission_ids', []) if mid not in excluded and mid in available]
    fields = {'observation_binding': BINDING, 'plan_revision': plan['revision'],
              'candidate_mission_ids': candidates}
    if kind == 'mission':
        selected = next((m for m in missions if m['mission_name'] == request.recording_mission_name), None)
        if not selected or selected['mission_id'] not in candidates:
            raise HTTPException(422, 'Select a mission linked to the source observation in Record.')
        if request.recording_mission_id and request.recording_mission_id != selected['mission_id']:
            raise HTTPException(409, 'The selected mission changed. Refresh it in Record.')
        if request.return_observation_spot_id and request.return_observation_spot_id != request.observation_spot_id:
            raise HTTPException(422, 'A mission must return to its source observation.')
        fields.update(source_observation_spot_id=request.observation_spot_id,
                      return_observation_spot_id=request.observation_spot_id)
    return fields


def capture_binding(episode, context):
    """Use the episode snapshot for both labels and training context."""
    if episode.get('observation_binding') != BINDING:
        return context
    context = deepcopy(context)
    context['plan_snapshot'] = deepcopy(episode['recording_plan'])
    context['plan_revision'] = episode['plan_revision']
    context['source_observation_spot_id'] = episode.get('source_observation_spot_id') or episode.get('observation_spot_id')
    if episode['episode_type'] == 'observation':
        context['candidate_mission_ids'] = list(episode['candidate_mission_ids'])
        context.pop('allowed_mission_ids', None)
    if context.get('reasoning_task', 'mission_selection') == 'mission_selection':
        observation = context.get('observation') or {}
        sid = observation.get('at_spot_id') or episode.get('observation_spot_id')
        context['observation'] = {**observation, **departure_context(episode['recording_plan'], sid)}
    context['mission_catalog_snapshot'] = deepcopy(episode['catalog'])
    return context


def live_record_scope(store, context):
    """Live Record preview uses the saved spot's missions, just like Operate."""
    if context.get('runtime_source') != 'record_runner_reported' or context.get('reasoning_task') != 'mission_selection':
        return context
    view = store.observation_view(context['map_name'])
    plan = view['plan']
    spot_id = context['observation']['at_spot_id']
    point = next((p for p in plan['observations'] if p['spot_id'] == spot_id), None)
    if point is None or not any(s['id'] == spot_id for s in view['spots']):
        raise HTTPException(409, 'Save this observation in the Record plan before previewing mission selection.')
    excluded = set(plan.get('excluded_mission_ids', []))
    return {**context, 'observation': {**context['observation'], **departure_context(plan, spot_id)},
            'candidate_mission_ids': [mid for mid in point.get('mission_ids', []) if mid not in excluded],
            'allowed_mission_ids': None, 'plan_revision': plan.get('revision', 0)}
