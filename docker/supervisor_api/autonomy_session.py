"""Patrol -> arrival confirmation -> supervised task -> same-spot return."""
import time
from .reasoning import StartRequest
from .autonomy_runtime import OperationCancelled
from .autonomy_direction import PatrolRedirect, next_route_index
from .autonomy_confirmation import MissionStartConfirmation, ObservationDepartureConfirmation
from .autonomy_reasoner import SessionReasoner
from .autonomy_actions import SessionActions
from .autonomy_mission_cancel import ChildMissionCancelled, cancel_child_mission


class PatrolSession:
    def __init__(self, controller, request, plan, points):
        self.c, self.request, self.plan, self.points = controller, request, plan, points
        self.index = 0
        self.previous = None
        self.last_arrived_index = None
        self.union = sorted({mid for p in points for mid in p['mission_ids']}
                            - set(plan.excluded_mission_ids))

    def run(self):
        c = self.c
        c.event('Autonomy started', f'{self.request.map_name} · {self.plan.mode}')
        with c.control_lock:
            c._check(self.request.map_name)
            c.runtime = c.runtime_factory()
            c.runtime.open()
        for point in self.points:
            self.validate_tree(point['arrival_bt'])
        with c.control_lock:
            c._check(self.request.map_name)
            c.reasoning.start(StartRequest(model_id=self.request.model_id, map_name=self.request.map_name))
        deadline = time.monotonic() + 660
        while True:
            status = c.reasoning.status()
            if status['phase'] == 'ready': break
            if status['phase'] in ('error', 'stopped'): raise RuntimeError(status.get('error') or 'Qwen stopped')
            if time.monotonic() > deadline: raise TimeoutError('Qwen model loading timed out')
            c.update(phase=status['phase']); c._wait(.25)
        self.reasoner = SessionReasoner(c, self.request, self.plan, status['session_id'])
        self.actions = SessionActions(c, self.reasoner, self.plan)
        while True:
            c._check(self.request.map_name)
            point = self.points[self.index]
            try:
                self.move_to(point)
            except PatrolRedirect:
                with c.control_lock:
                    c.runtime.patrol_redirect.clear()
                    if self.last_arrived_index is not None: self.index = self.last_arrived_index
                continue
            self.last_arrived_index = self.index
            while True:
                selected = self.observe(point)
                if selected is None: break
                try:
                    self.execute(selected, point)
                except ChildMissionCancelled as exc:
                    cancel_child_mission(c, selected[0], str(exc))
                self.move_to(point, returning=True)
                # Observe this same spot again; never skip it after a task.
            self.previous = point['spot_id']
            self.index = self.next_index()

    def next_index(self):
        return next_route_index(self.index, len(self.points),
                                self.c.state.get('patrol_direction', 1), self.plan.route_closed)

    def can_advance(self):
        return self.plan.mode == 'patrol' and len(self.points) > 1 and self.next_index() != self.index

    def validate_tree(self, xml):
        validate = getattr(self.c.runtime, 'validate_tree', None)
        if validate is not None: validate(xml)

    def move_to(self, point, returning=False):
        c = self.c
        c.update(patrol_index=self.index)
        phase = 'returning' if returning else 'positioning'
        context = {'phase': 'returning' if returning else 'moving',
                   'at_spot_id': None, 'from_spot_id': None if returning else self.previous,
                   'target_spot_id': point['spot_id']}
        c.transition(mode='observation', mission=None, mission_route=[], mission_dispatched=False,
                     phase=phase, observation=point, observation_context=context,
                     allowed=self.union, result=None,
                     runner={'phase': 'return-navigation' if returning else 'patrol-navigation'})
        c.event('Returning to observation' if returning else 'Moving to observation', point['label'])
        self.actions.run(c.runtime.navigate, point['pose'], supervised=True)
        with c.control_lock:
            if not returning and getattr(c.runtime, 'patrol_redirect', None) and c.runtime.patrol_redirect.is_set():
                raise PatrolRedirect()
            c.transition(phase='returning' if returning else 'settling',
                         observation_context={'phase': 'settling', 'spot_id': point['spot_id'],
                                              'label': point['label'], 'at_spot_id': point['spot_id'],
                                              'from_spot_id': None, 'target_spot_id': None},
                         runner={'phase': 'return-posture' if returning else 'observation-posture'})
        if point['arrival_bt']:
            self.actions.run(c.runtime.tree, point['arrival_bt'], supervised=True)
        deadline = time.monotonic() + self.plan.settle_seconds
        while time.monotonic() < deadline or c.state['paused']:
            self.reasoner.supervise()
            if time.monotonic() < deadline or c.state['paused']:
                c._wait(max(.05, min(self.plan.interval_seconds, self.plan.settle_seconds)))
        self.previous = point['spot_id']
        c.transition(mode='observation', phase='observing', mission=None, mission_route=[], runner={},
                     result=None, resume_checkpoint=None, mission_dispatched=False,
                     allowed=[m for m in point['mission_ids'] if m not in self.plan.excluded_mission_ids],
                     observation_context={'phase': 'observing', 'spot_id': point['spot_id'],
                                          'label': point['label'], 'at_spot_id': point['spot_id'],
                                          'from_spot_id': None, 'target_spot_id': None,
                                          'can_advance': self.can_advance(),
                                          'next_spot_id': self.points[self.next_index()]['spot_id'] if self.can_advance() else None,
                                          'departure_policy': point.get('departure_policy', 'reasoning')})
        c.event('Observation started', point['label'])

    def observe(self, point):
        c = self.c
        departure_confirmation = ObservationDepartureConfirmation()
        confirmation = MissionStartConfirmation()
        confirmation.required = self.plan.start_confirmations
        remaining = point['dwell_seconds']
        while True:
            c._check(self.request.map_name)
            started = time.monotonic()
            result = self.reasoner.infer()
            confirmed = False
            if result is None or c.state['paused']:
                confirmation.reset()
                departure_confirmation.reset()
            else:
                confirmed = confirmation.accept(result, c.instruction_version)
                if confirmation.stamps:
                    selected = result['selected_mission']
                    c.event('Mission confirmation',
                            f"{selected['mission_name']} · {len(confirmation.stamps)}/{confirmation.required}")
            if result and not c.state['paused'] and self.can_advance():
                decision = result.get('decision') or {}
                departure_confirmed = departure_confirmation.accept(
                    result, c.instruction_version, c.state['context_version'])
                if departure_confirmation.stamps:
                    c.event('Observation departure confirmation',
                            f"{point['label']} · {len(departure_confirmation.stamps)}/{departure_confirmation.required}")
                if departure_confirmed:
                    with c.control_lock, c.lock:
                        c._check(self.request.map_name)
                        if not c.state['paused'] and result.get('context_version') == c.state['context_version']:
                            c.event('Observation complete', f"{point['label']} · {decision.get('reason', '')}")
                            return None
            if confirmed:
                selected = result['selected_mission']
                if selected['mission_id'] in c.allowed:
                    try:
                        steps = c.repository.mission(self.request.map_name, selected)
                        for wp in steps: self.validate_tree(wp['xml'])
                    except (ValueError, RuntimeError) as exc:
                        confirmation.reset()
                        with c.lock:
                            c.update(result={**result, 'action_allowed': False, 'block_reason': str(exc)})
                        c.event('Decision rejected', str(exc))
                    else:
                        # Instructions or Stop arriving during preflight revoke this decision.
                        with c.control_lock, c.lock:
                            c._check(self.request.map_name)
                            if (not c.state['paused'] and result['context_version'] == c.state['context_version']):
                                checkpoint = {'return_spot_id': point['spot_id'],
                                    'next_spot_id': self.points[self.next_index()]['spot_id'],
                                    'plan_revision': self.plan.revision}
                                c.transition(mode='mission', phase='executing', mission=selected,
                                    result=None, resume_checkpoint=checkpoint, mission_dispatched=False,
                                    mission_route=[{k: wp[k] for k in ('id', 'label', 'pose')} for wp in steps],
                                    runner={'phase': 'mission-admission'},
                                    observation_context={**c.state['observation_context'], 'phase': 'executing'})
                                return selected, steps
            if not c.state['paused']: remaining -= time.monotonic() - started
            if (point.get('departure_policy', 'reasoning') == 'timed'
                    and self.can_advance() and remaining <= 0 and not c.state['paused']): return None
            c._wait(self.plan.interval_seconds)
            if not c.state['paused']: remaining -= self.plan.interval_seconds

    def execute(self, selection, point):
        selected, steps = selection
        c = self.c
        c.event('Mission accepted', selected['mission_name'])
        for index, wp in enumerate(steps):
            progress = {'currentIndex': index, 'total': len(steps), 'activeSpotId': wp['id'], 'label': wp['label']}
            try:
                c.transition(runner={**progress, 'phase': 'navigation'},
                    observation_context={'phase': 'executing', 'at_spot_id': None,
                                         'from_spot_id': None, 'target_spot_id': None})
                self.actions.run(c.runtime.navigate, wp['pose'], supervised=True)
                c.transition(runner={**progress, 'phase': 'waypoint-task', 'bt_xml': wp['xml']})
                if wp['xml']: self.actions.run(c.runtime.tree, wp['xml'], supervised=True)
                c.transition(runner={**progress, 'phase': 'task-completed'})
            except (OperationCancelled, ChildMissionCancelled): raise
            except Exception as exc:
                raise RuntimeError(f"{selected['mission_name']} · {wp['label']}: {exc}") from exc
        c.event('Mission completed', selected['mission_name'])
        with c.lock:
            c.update(recent_missions=(c.state['recent_missions'] + [
                {'mission_id': selected['mission_id'], 'status': 'completed'}])[-20:])
