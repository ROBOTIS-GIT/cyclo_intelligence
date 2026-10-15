"""Server-owned autonomy session, admission and exclusive execution control.

Patrol/mission flow, inference and ROS operations live in separate modules.
"""
from copy import deepcopy
from pathlib import Path
import json
import os
import threading
import time
from uuid import uuid4
from fastapi import HTTPException
from .autonomy_config import AutonomyRepository
from .autonomy_context import ContextHistory, context_for
from .autonomy_confirmation import MissionStartConfirmation
from .autonomy_routes import AutonomyStart, create_router
from .autonomy_runtime import OperationCancelled
from .execution_state import execution_snapshot, transition_state


SESSION_FIELDS = ('session_id', 'map_name', 'model_id', 'mode', 'phase', 'mission', 'runner',
                  'recent_missions', 'resume_checkpoint', 'paused', 'pause_causes',
                  'revision', 'control_status', 'plan_revision')


class AutonomyController:
    def __init__(self, reasoning, runtime_factory, check_navigation, repository=None,
                 session_path=None, recovery_stop=None):
        self.reasoning, self.runtime_factory = reasoning, runtime_factory
        self.repository = repository or AutonomyRepository()
        self.check_navigation, self.recovery_stop = check_navigation, recovery_stop
        self.lock, self.control_lock = threading.RLock(), threading.RLock()
        self.cancelled = threading.Event()
        self.thread = self.stop_thread = self.action_thread = None
        self.runtime = None
        self.route_points = []
        self.owned = False
        self.epoch = str(uuid4())
        self.session_path = Path(session_path) if session_path else None
        from .autonomy_conversation import AutonomyConversation
        self.conversation = AutonomyConversation(self)
        self.instruction_version = 0
        self.history = ContextHistory()
        self.context_closed_at = None
        self.allowed = []
        self.plan_snapshot, self.catalog = {}, []
        self.state = {'phase': 'idle', 'active': False, 'map_name': None, 'events': [],
                      'mission': None, 'runner': {}, 'error': None, 'result': None,
                      'paused': False, 'pause_causes': {}, 'control_status': 'idle', 'revision': 0}
        if self.session_path and self.session_path.exists():
            try: restored = json.loads(self.session_path.read_text())
            except (OSError, ValueError): restored = {}
            if not isinstance(restored, dict): restored = {}
            restored = {key: value for key, value in restored.items() if key in SESSION_FIELDS}
            for key in ('mission', 'runner', 'resume_checkpoint', 'pause_causes'):
                if key in restored and not isinstance(restored[key], dict):
                    restored[key] = {} if key in ('runner', 'pause_causes') else None
            if not isinstance(restored.get('recent_missions', []), list): restored['recent_missions'] = []
            # The new boot epoch makes even a persisted older revision unambiguous.
            restored.pop('revision', None)
            restored.update(phase='interrupted', active=False, control_status='idle',
                            error='The server restarted during autonomy. Stop before restarting.')
            self._commit(owned=True, **restored)

    def _persist(self):
        if not self.session_path: return
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.session_path.with_suffix('.tmp')
        temp.write_text(json.dumps({k: self.state.get(k) for k in SESSION_FIELDS}))
        os.replace(temp, self.session_path)

    def status(self):
        with self.lock:
            return {**deepcopy(self.state), 'needsStop': self.owned,
                    'execution': execution_snapshot(self.state, self.owned, self.epoch),
                    'loadedModel': self.reasoning.status()}

    def _commit(self, *, owned=None, replace=False, record_context=False,
                persist=False, allowed=None, **values):
        """The single locked publication path for compatibility and normalized state."""
        with self.lock:
            next_state = transition_state(self.state, values, replace=replace)
            if allowed is not None: self.allowed = list(allowed)
            if record_context:
                context = context_for(next_state, self.allowed)
                next_state['reasoning_task'] = context['reasoning_task']
                next_state['context_version'] = self.history.record(context)
                next_state['context_since'] = self.history.entries[-1][0]
            if owned is not None: self.owned = owned
            self.state = next_state
            if persist: self._persist()

    def update(self, **values):
        self._commit(**values)

    def transition(self, *, allowed=None, **values):
        with self.lock:
            if self.cancelled.is_set(): raise OperationCancelled()
            self._commit(record_context=True, persist=True, allowed=allowed, **values)

    def capture_context(self, snapshot_time=None):
        with self.lock:
            closed = self.context_closed_at
            if not self.owned or not self.state.get('active'):
                if snapshot_time is None or closed is None or float(snapshot_time) >= closed:
                    # A failed/stopping owner has not proved that actuators are idle.
                    return {'runtime_source': 'unverified'} if self.owned else None
            context = self.history.at(snapshot_time)
            if closed is not None and 'context_valid_until' in context:
                context['context_valid_until'] = min(context['context_valid_until'], closed)
            if context.get('runtime_source') == 'autonomy':
                context.update(plan_snapshot=deepcopy(self.plan_snapshot),
                               mission_catalog_snapshot={'missions': deepcopy(self.catalog)})
            return context

    def event(self, title, detail='', persist=True):
        row = {'title': title, 'detail': detail, 'time_ms': int(time.time() * 1000)}
        with self.lock:
            self.update(events=[row, *self.state['events']][:100])
            map_name = self.state['map_name']
        try:
            if persist: self.reasoning.activity.append(map_name, {'kind': 'operation', **row})
        except (OSError, ValueError): pass

    def start(self, request):
        with self.lock:
            if self.owned or (self.stop_thread and self.stop_thread.is_alive()):
                raise HTTPException(409, 'Stop the current autonomy session first')
            if not request.localized: raise HTTPException(409, 'Localize before starting autonomy')
            plan = self.repository.read(request.map_name)
            if plan.revision != request.plan_revision:
                raise HTTPException(409, 'Observation settings changed. Reload before starting.')
            try: points, catalog = self.repository.resolve(request.map_name, plan)
            except ValueError as exc: raise HTTPException(422, str(exc)) from exc
            if not points: raise HTTPException(409, 'Add an observation spot first')
            if plan.mode == 'fixed': points = points[:1]
            self.check_navigation(request.map_name)
            self.reasoning.models.resolve(request.model_id, request.map_name)
            self.cancelled = threading.Event()
            self.instruction_version = 0
            self.history = ContextHistory()
            self.context_closed_at = None
            self.plan_snapshot, self.catalog = plan.model_dump(), deepcopy(catalog)
            self.route_points = deepcopy(points)
            self.allowed = sorted({mid for point in points for mid in point['mission_ids']}
                                  - set(plan.excluded_mission_ids))
            initial = {'phase': 'starting', 'mode': 'observation', 'active': True,
                          'map_name': request.map_name, 'session_id': str(uuid4()),
                          'model_id': request.model_id, 'plan_revision': plan.revision,
                          'patrol_direction': 1, 'patrol_index': 0,
                          'observation_mode': plan.mode, 'observation': None, 'mission': None,
                          'runner': {}, 'result': None, 'events': [], 'error': None,
                          'operator_instruction': '', 'recent_missions': [],
                          'paused': False, 'pause_causes': {}, 'pause_reason': None,
                          'resume_checkpoint': None, 'mission_dispatched': False,
                          'control_status': 'running'}
            try: self._commit(owned=True, replace=True, record_context=True, persist=True, **initial)
            except OSError:
                self._commit(owned=False, active=False, phase='error', control_status='idle')
                raise
            self.thread = threading.Thread(target=self._run, args=(request, plan, points), daemon=True)
            self.thread.start()
        return self.status()

    def _run(self, request, plan, points):
        from .autonomy_session import PatrolSession
        try:
            PatrolSession(self, request, plan, points).run()
        except OperationCancelled:
            pass
        except Exception as exc:
            message = str(getattr(exc, 'detail', exc))
            with self.lock:
                if self.cancelled.is_set(): return
                self.cancelled.set()
                self.context_closed_at = time.time()
                self.update(phase='error', active=False, error=message, control_status='cancelling')
            try:
                if self.runtime: self.runtime.cancel()
            except Exception as failure: message += f'; Stop failed: {failure}'
            with self.lock:
                if self.state['phase'] == 'error': self.update(error=message)
            self.event('Autonomy paused after failure', message)

    def _check(self, map_name):
        if self.cancelled.is_set(): raise OperationCancelled()
        self.check_navigation(map_name)

    def _wait(self, seconds):
        if self.cancelled.wait(seconds): raise OperationCancelled()

    def pause(self, reason='Operator pause', source='manual'):
        with self.control_lock:
            with self.lock:
                if not self.state.get('active') or self.cancelled.is_set(): return self.status()
                causes = dict(self.state['pause_causes'])
                already = bool(causes)
                if causes.get(source) == reason: return self.status()
            if not already:
                if self.runtime is None: raise HTTPException(409, 'Runtime is still starting')
                with self.lock:
                    if self.cancelled.is_set(): return self.status()
                    self.update(control_status='pausing')
                try: self.runtime.pause()
                except Exception as exc: self._control_failure('Pause', exc)
            with self.lock:
                if self.cancelled.is_set(): return self.status()
                causes[source] = reason
                self.transition(paused=True, control_status='paused', pause_causes=causes, pause_reason='; '.join(causes.values()))
            self.event('Mission paused' if self.state['mode'] == 'mission' else 'Patrol paused', reason)
        return self.status()

    def resume(self, source='manual'):
        with self.control_lock:
            with self.lock:
                if not self.state.get('active') or self.cancelled.is_set(): return self.status()
                causes = dict(self.state['pause_causes'])
                if source not in causes: return self.status()
                del causes[source]
            if not causes:
                with self.lock:
                    if self.cancelled.is_set(): return self.status()
                    self.update(control_status='resuming')
                try: self.runtime.resume()
                except Exception as exc: self._control_failure('Resume', exc)
            with self.lock:
                if self.cancelled.is_set(): return self.status()
                self.transition(paused=bool(causes), control_status='paused' if causes else 'running', pause_causes=causes,
                                pause_reason='; '.join(causes.values()) or None)
            if not causes: self.event('Mission resumed' if self.state['mode'] == 'mission' else 'Patrol resumed')
        return self.status()

    def _control_failure(self, operation, exc):
        message = f'{operation} could not be confirmed: {exc}'
        with self.lock:
            # A delayed acknowledgement failure cannot replace an accepted Stop.
            if self.cancelled.is_set(): return
            self.cancelled.set()
            self.context_closed_at = time.time()
            self.update(active=False, phase='error', error=message, control_status='cancelling')
        try: self.runtime.cancel()
        except Exception as failure: message += f'; Stop failed: {failure}'
        with self.lock:
            if self.state['phase'] == 'error': self.update(error=message)
        self.event('Autonomy control failed', message)
        raise HTTPException(503, message) from exc

    def instruction(self, text):
        with self.lock:
            if not self.state['active']: raise HTTPException(409, 'Autonomy is not active')
            self.instruction_version += 1
            self.transition(operator_instruction=text)
        self.event('Operator instruction', text)
        return self.status()

    def stop(self):
        with self.lock:
            if not self.owned and self.reasoning.status().get('phase') in ('idle', 'stopped'):
                self.update(result=None)
                return self.status()
            self.cancelled.set()
            if self.context_closed_at is None: self.context_closed_at = time.time()
            self._commit(owned=True, active=False, phase='stopping', result=None, control_status='cancelling')
            if self.stop_thread and self.stop_thread.is_alive(): return self.status()
            self.stop_thread = threading.Thread(target=self._stop, daemon=True)
            self.stop_thread.start()
        return self.status()

    def _stop(self):
        errors = []
        with self.control_lock:
            try:
                if self.runtime: self.runtime.cancel()
                elif self.recovery_stop: self.recovery_stop()
            except Exception as exc: errors.append(str(exc))
        try: self.reasoning.stop()
        except Exception as exc: errors.append(str(getattr(exc, 'detail', exc)))
        for worker in (self.thread, self.action_thread):
            if worker: worker.join(timeout=110)
            if worker and worker.is_alive(): errors.append('Execution worker has not stopped yet')
        if not errors:
            try:
                if self.runtime: self.runtime.close()
            except Exception as exc: errors.append(str(exc))
        with self.lock:
            if not errors:
                try:
                    if self.session_path: self.session_path.unlink(missing_ok=True)
                except OSError as exc: errors.append(str(exc))
            if errors:
                self.update(phase='error', error='; '.join(errors), control_status='cancelling')
            else:
                self.runtime = None
                self.route_points = []
                self._commit(owned=False, phase='stopped', control_status='idle', mission=None, mission_route=[], runner={},
                                  result=None, error=None, paused=False, pause_causes={}, pause_reason=None,
                                  mission_dispatched=False)
        self.event('Autonomy stop failed' if errors else 'Autonomy stopped', '; '.join(errors))
