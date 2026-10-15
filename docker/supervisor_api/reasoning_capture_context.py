"""Match frames to server-owned execution history, never infer missing history."""
from copy import deepcopy
import math
import time
from fastapi import HTTPException
if __package__:
    from .reasoning_contract import input_context
else:
    from reasoning_contract import input_context

SERVER_FIELDS = ('reasoning_task', 'active_mission', 'observation', 'candidate_mission_ids',
                 'recent_missions', 'operator_instruction', 'map_name', 'localized',
                 'plan_revision', 'plan_snapshot', 'mission_catalog_snapshot', 'context_version',
                 'context_valid_from', 'context_valid_until')


def _identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 128


def _matched_report(report, frame):
    try:
        since, sampled = (float(report[key]) for key in ('context_valid_from_ms', 'sampled_at_ms'))
        captured = float(frame['captured_at']) * 1000
        matched = (all(math.isfinite(v) for v in (since, sampled, captured))
                   and since <= sampled and since <= captured <= sampled + 1500
                   and -1000 <= time.time() * 1000 - sampled <= 5000)
        return matched, captured - sampled
    except (KeyError, TypeError, ValueError):
        return False, 0


def reported_context(context, frame):
    """Validate Record's capture-time reports; retain their client-reported provenance."""
    report = context.pop('recording_execution', None)
    if report is None:
        return context
    if not isinstance(report, dict):
        raise HTTPException(422, 'Invalid recording execution report')
    # Existing clients only reported missions. New clients must name their role.
    task = report.get('reasoning_task', 'mission_supervision')
    version, status = report.get('context_version'), report.get('status')
    if (task not in ('mission_selection', 'patrol_supervision', 'mission_supervision')
            or report.get('map_name') != context.get('map_name')
            or type(version) is not int or version < 0):
        raise HTTPException(422, 'Invalid recording task, map or context version')
    mid, step, action = report.get('mission_id'), report.get('step'), report.get('action', '')
    if mid is not None and not _identifier(mid):
        raise HTTPException(422, 'Invalid recording mission ID')
    if task == 'mission_supervision':
        if (status not in ('running', 'paused') or type(step) is not int or step < 1
                or not isinstance(action, str) or len(action) > 120):
            raise HTTPException(422, 'Invalid recording mission status, step or action')
    elif task == 'mission_selection':
        if (status != 'arrived' or mid is not None or report.get('localized') is not True
                or not _identifier(report.get('observation_spot_id'))):
            raise HTTPException(422, 'Observation capture requires a localized, arrived observation spot')
    else:
        phase = report.get('phase', 'moving')
        from_spot, to_spot = report.get('from_spot_id'), report.get('to_spot_id')
        valid_source = (_identifier(from_spot) or from_spot is None) if phase == 'settling' else (from_spot is None if phase in ('returning', 'approaching') else _identifier(from_spot))
        if (status not in ('running', 'paused') or mid is not None
                or phase not in ('moving', 'returning', 'approaching', 'settling') or not valid_source or not _identifier(to_spot)
                or from_spot == to_spot):
            raise HTTPException(422, 'Patrol capture requires a running or paused route between observation spots')
    matched, delta = _matched_report(report, frame)
    if task == 'mission_supervision' and mid is None:
        matched = False
    context.update(recording_execution_report=deepcopy(report), context_confidence='client_reported',
                   reasoning_task=task)
    if not matched:
        context.update(runtime_source='unverified', source='record_context_unavailable', active_mission=None,
                       observation={'phase': 'unknown'}, candidate_mission_ids=[])
        return context
    context.update(runtime_source='record_runner_reported', source='record_runner_reported',
                   client_context_version=version, capture_report_delta_ms=round(delta, 3), active_mission=None)
    if task == 'mission_supervision':
        context.update(active_mission={'mission_id': mid, 'status': status, 'step': step, 'action': action},
                       candidate_mission_ids=[mid], observation={'phase': 'executing'})
    elif task == 'patrol_supervision':
        context.update(candidate_mission_ids=[], observation={'phase': report.get('phase', 'moving'), 'status': status,
                       'from_spot_id': report.get('from_spot_id'), 'target_spot_id': report['to_spot_id'],
                       'at_spot_id': report['to_spot_id'] if report.get('phase') == 'settling' else None})
    else:
        spot = report['observation_spot_id']
        context.update(localized=True, observation={'phase': 'observing', 'status': 'arrived',
                       'spot_id': spot, 'at_spot_id': spot, 'target_spot_id': None})
    return context


def context_for_frame(raw, frame, provider=None, *, normalized=False):
    context = deepcopy(raw) if normalized else input_context(raw)
    context.setdefault('reasoning_task', 'mission_selection')
    if provider is None:
        return reported_context(context, frame)
    captured_at = frame.get('captured_at')
    try:
        snapshot = provider(snapshot_time=captured_at)
    except Exception:
        snapshot = {}  # Saving images is still possible; unknown runtime stays explicit.
    if snapshot is None:
        return reported_context(context, frame)  # No autonomous runtime owns this capture.
    context.pop('recording_execution', None)
    valid = isinstance(snapshot, dict) and snapshot.get('context_version') is not None
    try:
        start, end, stamp = [float(snapshot.get(k)) for k in ('context_valid_from', 'context_valid_until')] + [float(captured_at)]
    except (ValueError, TypeError, AttributeError):
        valid = False
    else:
        valid = valid and all(math.isfinite(v) for v in (start, end, stamp)) and start <= stamp <= end
    if valid:
        context.update({key: deepcopy(snapshot[key]) for key in SERVER_FIELDS if key in snapshot})
        context['runtime_source'] = snapshot.get('runtime_source', 'autonomy')
        context['source'] = 'server_at_capture'
    else:
        requested_task = context.get('reasoning_task')
        for key in SERVER_FIELDS:
            if key not in ('map_name', 'localized'):
                context.pop(key, None)
        context.update(runtime_source='unverified', source='server_context_unavailable',
                       reasoning_task=requested_task or 'mission_selection', active_mission=None,
                       observation={'phase': 'unknown'}, candidate_mission_ids=[])
    return context
