"""Latest-frame reasoning and mode-scoped decisions; no navigation ownership."""
from copy import deepcopy
from fastapi import HTTPException
from .reasoning import MissionDecisionRequest
from .autonomy_confirmation import ResumeConfirmation
from .autonomy_runtime import OperationCancelled
from .autonomy_mission_cancel import ChildMissionCancelled


class SessionReasoner:
    def __init__(self, controller, request, plan, session_id):
        self.c, self.request, self.plan = controller, request, plan
        self.session_id = session_id
        self.previous_stamp = None
        self.resume_scope = None
        self.resume_gate = ResumeConfirmation(plan.resume_confirmations)

    def infer(self, completed=None):
        c = self.c
        c._check(self.request.map_name)
        if getattr(c, 'conversation', None) and c.conversation.pending:
            c._wait(.1)
            return None
        if c.reasoning.status().get('session_id') != self.session_id:
            raise RuntimeError('Reasoning session changed. Stop and restart autonomy.')
        with c.lock:
            version = c.state['context_version']
            context = c.capture_context()
            captured_after = c.state['context_since']
        try:
            result = c.reasoning.decide(MissionDecisionRequest(
                map_name=self.request.map_name, localized=True,
                expected_session_id=self.session_id, previous_frame_stamp=self.previous_stamp,
                reasoning_task=context['reasoning_task'], active_mission=context['active_mission'],
                operator_instruction=context['operator_instruction'],
                recent_missions=context['recent_missions'],
                observation=context['observation'],
                allowed_mission_ids=context['candidate_mission_ids'],
                frame_captured_after=captured_after))
        except HTTPException as exc:
            if exc.status_code == 409 and exc.detail == 'A Qwen request is already in progress':
                # A conversation may claim the model between the pending check and decide().
                c._wait(.1)
                return None
            if exc.status_code not in (409, 422, 500, 502, 503, 504): raise
            result = {'error': str(exc.detail), 'action_allowed': False, 'decision': None}
        if c.cancelled.is_set(): raise OperationCancelled()
        with c.lock:
            if version != c.state['context_version'] or (completed and completed.is_set()):
                return None
            if result.get('frame_stamp'): self.previous_stamp = result['frame_stamp']
            result = {**result, 'reasoning_task': context['reasoning_task'], 'context_version': version}
            c.update(result=result)
        decision = result.get('decision') or {}
        label = decision.get('decision', 'unavailable')
        if label == 'start': label = 'start proposal'
        c.event(f"AI · {label}",
                decision.get('reason') or result.get('error', ''), persist=False)
        return result

    def supervise(self, completed=None):
        result = self.infer(completed)
        if result is None:
            self.resume_gate.reset(); self.resume_scope = None
            return
        c = self.c
        with c.control_lock:
            if c.cancelled.is_set(): raise OperationCancelled()
            if result['context_version'] != c.state['context_version']:
                self.resume_gate.reset(); self.resume_scope = None
                return
            patrol = result.get('reasoning_task') == 'patrol_supervision'
            mission = c.state.get('mission') or {}
            expected_id = None if patrol else mission.get('mission_id')
            scope = (result['context_version'], result.get('reasoning_task'), expected_id)
            if scope != self.resume_scope:
                self.resume_gate.reset()
                self.resume_scope = scope
            decision = result.get('decision') or {}
            valid = (not result.get('error') and result.get('action_allowed')
                     and (patrol or bool(expected_id))
                     and decision.get('mission_id') == expected_id)
            if (valid and not patrol and c.state['phase'] == 'executing'
                    and decision.get('decision') == 'cancel'):
                self.resume_gate.reset()
                raise ChildMissionCancelled(decision.get('reason') or 'AI requested mission cancellation')
            if not valid or decision.get('decision') not in ('pause', 'continue'):
                self.resume_gate.reset()
                c.pause(result.get('error') or 'Waiting for a valid supervision decision', source='ai')
            elif decision['decision'] == 'pause':
                self.resume_gate.reset()
                c.pause(decision.get('reason') or 'AI requested a pause', source='ai')
            elif 'ai' in c.state['pause_causes']:
                confirmed = self.resume_gate.accept(result)
                c.event('Resume confirmation',
                        f"{'Patrol' if patrol else mission.get('mission_name', expected_id)} · {len(self.resume_gate.stamps)}/{self.resume_gate.required}")
                if confirmed:
                    c.resume(source='ai')
                    self.resume_gate.reset()
            else:
                self.resume_gate.reset()

    def admission(self):
        # New tasks get a valid supervision decision before physical execution.
        while True:
            if (self.c.state.get('phase') == 'positioning' and getattr(self.c.runtime, 'patrol_redirect', None)
                    and self.c.runtime.patrol_redirect.is_set()):
                from .autonomy_direction import PatrolRedirect
                raise PatrolRedirect()
            self.supervise()
            if not self.c.state['paused']:
                result = self.c.state.get('result') or {}
                if ((result.get('decision') or {}).get('decision') == 'continue'
                        and result.get('context_version') == self.c.state['context_version']): return
            self.c._wait(self.plan.interval_seconds)
