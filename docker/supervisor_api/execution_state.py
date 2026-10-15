"""Pure, versioned execution view over the compatibility autonomy state.

Snapshot revisions describe all published changes. Capture context versions are
separate: logs and inference results must not revoke an in-flight decision.
"""
from copy import deepcopy
from typing import Literal, TypedDict

Lifecycle = Literal['idle', 'starting', 'active', 'stopping', 'stopped', 'interrupted', 'failed']
Activity = Literal['idle', 'patrol_navigation', 'observation_posture', 'observing',
                   'mission_navigation', 'mission_action', 'return_navigation', 'return_posture']
Control = Literal['idle', 'running', 'pausing', 'paused', 'resuming', 'cancelling']
CONTROLS = {'idle', 'running', 'pausing', 'paused', 'resuming', 'cancelling'}


class ExecutionSnapshot(TypedDict):
    schema_version: int
    source: Literal['operate']
    authority: Literal['server']
    epoch: str
    session_id: str | None
    revision: int
    lifecycle: Lifecycle
    activity: Activity
    control: Control
    owns_execution: bool
    map_name: str | None
    plan_revision: int | None
    observation_spot_id: str | None
    target_spot_id: str | None
    mission_id: str | None
    step_index: int | None
    pause_causes: dict[str, str]
    return_checkpoint: dict | None
    error: str | None


def transition_state(previous, changes, *, replace=False):
    """Copy a state patch and advance only its publication revision."""
    if 'revision' in changes:
        raise ValueError('Execution revisions are assigned by the controller')
    if 'control_status' in changes and changes['control_status'] not in CONTROLS:
        raise ValueError('Unknown execution control status')
    state = {} if replace else dict(previous)
    state.update(deepcopy(changes))
    state['revision'] = previous.get('revision', 0) + 1
    return state


def _activity(state):
    phase = state.get('phase', 'idle')
    runner = state.get('runner') or {}
    action = runner.get('phase')
    if phase in ('idle', 'stopped', 'interrupted', 'starting', 'loading'):
        return 'idle'
    if action == 'return-posture': return 'return_posture'
    if action == 'return-navigation': return 'return_navigation'
    if phase == 'returning':
        return 'return_posture' if (state.get('observation_context') or {}).get('phase') == 'settling' else 'return_navigation'
    if state.get('mode') == 'mission' and state.get('mission'):
        if action == 'navigation': return 'mission_navigation'
        if action in ('waypoint-task', 'task-completed'): return 'mission_action'
        return 'idle'  # Admission has not dispatched a physical mission step yet.
    if action == 'observation-posture' or phase == 'settling': return 'observation_posture'
    if action == 'patrol-navigation' or phase == 'positioning': return 'patrol_navigation'
    if phase in ('observing', 'watching', 'waiting'): return 'observing'
    return 'idle'


def execution_snapshot(state, owns_execution, epoch) -> ExecutionSnapshot:
    """Normalize compatibility fields without inventing actuator evidence."""
    phase = state.get('phase', 'idle')
    lifecycle = {'idle': 'idle', 'starting': 'starting', 'loading': 'starting',
                 'stopping': 'stopping', 'stopped': 'stopped', 'interrupted': 'interrupted',
                 'error': 'failed'}.get(phase, 'active' if state.get('active') else 'idle')
    control = state.get('control_status') or ('paused' if state.get('paused') else
              'running' if state.get('active') else 'idle')
    if phase in ('stopping', 'cancelling') or (phase == 'error' and owns_execution):
        control = 'cancelling'
    if not owns_execution or phase == 'interrupted': control = 'idle'
    if control not in CONTROLS: raise ValueError('Unknown execution control status')
    observation = state.get('observation_context') or {}
    runner = state.get('runner') or {}
    step = runner.get('currentIndex')
    if type(step) is not int or step < 0: step = None
    revision = state.get('revision', 0)
    if type(revision) is not int or revision < 0: raise ValueError('Invalid execution revision')
    snapshot = ExecutionSnapshot(
        schema_version=1, source='operate', authority='server', epoch=epoch,
        session_id=state.get('session_id'), revision=revision, lifecycle=lifecycle,
        activity=_activity(state), control=control, owns_execution=bool(owns_execution),
        map_name=state.get('map_name'), plan_revision=state.get('plan_revision'),
        observation_spot_id=observation.get('at_spot_id'),
        target_spot_id=runner.get('activeSpotId') if state.get('mode') == 'mission'
                       else observation.get('target_spot_id'),
        mission_id=(state.get('mission') or {}).get('mission_id'), step_index=step,
        pause_causes=deepcopy(state.get('pause_causes') or {}),
        return_checkpoint=deepcopy(state.get('resume_checkpoint')), error=state.get('error'))
    return snapshot
