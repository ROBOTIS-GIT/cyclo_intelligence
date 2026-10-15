"""Capture-time context history. Runtime checkpoints never become model targets."""
from collections import deque
from copy import deepcopy
import time


class ContextHistory:
    def __init__(self):
        self.entries = deque(maxlen=2048)
        self.version = 0

    def record(self, context):
        self.version += 1
        value = deepcopy(context)
        value['context_version'] = self.version
        self.entries.append((time.time(), value))
        return self.version

    def at(self, captured_at=None):
        now = time.time()
        stamp = now if captured_at is None else float(captured_at)
        if stamp > now or not self.entries:
            return {'runtime_source': 'unverified'}
        until = now
        for since, value in reversed(self.entries):
            if since <= stamp:
                return {**deepcopy(value), 'context_valid_from': since,
                        'context_valid_until': until}
            until = since
        return {'runtime_source': 'unverified'}


def context_for(state, allowed):
    mission = state.get('mission')
    runner = state.get('runner') or {}
    active = None
    if state.get('mode') == 'mission' and mission:
        active = {'mission_id': mission['mission_id'],
                  'status': 'paused' if state.get('paused') else
                            'returning' if state.get('phase') == 'returning' else 'running',
                  'step': runner.get('currentIndex', 0) + 1,
                  'action': runner.get('phase', state.get('phase', ''))}
    patrol = not active and (state.get('observation_context') or {}).get('phase') in ('moving', 'settling', 'returning')
    return {'map_name': state.get('map_name'), 'runtime_source': 'autonomy',
            'reasoning_task': 'mission_supervision' if active else 'patrol_supervision' if patrol else 'mission_selection',
            'active_mission': active, 'operator_instruction': state.get('operator_instruction') or None,
            'observation': deepcopy(state.get('observation_context') or {}),
            'candidate_mission_ids': [active['mission_id']] if active else [] if patrol else list(allowed),
            'recent_missions': deepcopy(state.get('recent_missions', [])),
            'plan_revision': state.get('plan_revision')}
