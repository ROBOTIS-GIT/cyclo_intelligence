"""Bounded dialogue and whitelisted controller commands, separate from scene decisions."""
import threading
from copy import deepcopy
from fastapi import HTTPException

from .autonomy_dialogue import direct_action, answer_message


class AutonomyConversation:
    def __init__(self, controller):
        self.c = controller
        self.lock = threading.RLock()
        self.pending = False
        self.messages = []
        self.requests = []
        self.error = None

    def status(self):
        with self.lock:
            return {'messages': deepcopy(self.messages), 'pending': self.pending, 'error': self.error}

    def submit(self, text, request_id):
        text = text.strip()
        if not text: raise HTTPException(422, 'Enter a message.')
        with self.lock:
            if request_id in self.requests: return self.status()
            if self.pending: raise HTTPException(409, 'Wait for the current reply.')
            self.requests = (self.requests + [request_id])[-100:]
            self.messages = (self.messages + [{'id': request_id, 'role': 'user', 'content': text}])[-24:]
            self.pending = True; self.error = None
        with self.c.lock:
            session = self.c.state.get('session_id')
        threading.Thread(target=self._reply, args=(text, request_id, session), daemon=True).start()
        return self.status()

    def _reply(self, text, request_id, session):
        c = self.c
        try:
            # Exact emergency commands bypass model scheduling, including when the model is unavailable.
            direct = direct_action(text)
            with c.lock:
                snapshot = deepcopy(c.state)
            if direct:
                answer = {'action': direct, 'reply': ''}
            else:
                if c.reasoning.status().get('phase') != 'ready':
                    raise ValueError('Start reasoning to chat with Qwen. Stop and pause controls remain available.')
                from .reasoning import qwen_request
                if not c.reasoning.observe_lock.acquire(timeout=15):
                    raise ValueError('Qwen is finishing an inference. Please try again shortly.')
                try:
                    with c.lock: snapshot = deepcopy(c.state)
                    with self.lock: history = [{'role': m['role'], 'content': m['content'][:500]} for m in self.messages[:-1][-6:]]
                    robot = {k: snapshot.get(k) for k in ('active','phase','mode','paused','pause_reason','map_name','patrol_direction')}
                    for key, fields in {'observation':('spot_id','label'), 'mission':('mission_id','mission_name'), 'runner':('phase','label','currentIndex','total')}.items():
                        robot[key] = {k: (snapshot.get(key) or {}).get(k) for k in fields}
                    robot['saved_patrol_order'] = [p['label'] for p in c.route_points]
                    robot['last_scene_assessment'] = (snapshot.get('result') or {}).get('decision')
                    answer = answer_message(qwen_request, text, robot, history)
                finally: c.reasoning.observe_lock.release()
            action = answer['action']
            reply = answer['reply']
            if action != 'none':
                with c.control_lock:
                    with c.lock:
                        if c.state.get('session_id') != session or (not direct and c.state.get('context_version') != snapshot.get('context_version')):
                            raise ValueError('Robot state changed while answering. Please send the command again.')
                        if action != 'stop' and (not c.state.get('active') or c.cancelled.is_set()):
                            raise ValueError('Autonomy is not active. Start it before sending movement commands.')
                    if action in ('reverse_patrol','forward_patrol'):
                        from .autonomy_direction import change_direction
                        reply = change_direction(c, action)
                    elif action == 'instruction':
                        c.instruction(text)
                        reply = 'Instruction added to the robot’s live reasoning. It will be evaluated with the next camera observation; no mission has been started by this message.'
                    elif action == 'clear_instruction':
                        c.instruction('')
                        reply = 'The task instruction has been cleared.'
                    elif action == 'pause':
                        c.pause('Operator chat request', source='manual'); reply = 'Autonomy is paused.'
                    elif action == 'resume':
                        state = c.resume(source='manual')
                        reply = 'Manual pause is cleared. '+('Still waiting: '+str(state.get('pause_reason')) if state.get('paused') else 'Autonomy is running.')
                    else:
                        c.stop(); reply = 'Stop requested. Waiting for the robot and reasoning to stop.'
                c.event('Conversation command', action)
            with self.lock:
                self.messages = (self.messages + [{'id': request_id+'-reply', 'role': 'assistant', 'content': reply, 'action': action}])[-24:]
        except Exception as exc:
            with self.lock: self.error = str(getattr(exc, 'detail', exc))
        finally:
            with self.lock: self.pending = False
