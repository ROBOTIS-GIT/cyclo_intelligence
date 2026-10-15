"""One ROS action worker, while the controller keeps reasoning on latest frames."""
import threading
import time
from .autonomy_runtime import OperationCancelled


class SessionActions:
    def __init__(self, controller, reasoner, plan):
        self.c, self.reasoner, self.plan = controller, reasoner, plan

    def run(self, operation, *args, supervised=False):
        c = self.c
        if supervised:
            self.reasoner.admission()
        else:
            while c.state['paused']:
                self.reasoner.infer()
                c._wait(self.plan.interval_seconds)
        done, errors = threading.Event(), []
        def work():
            try: operation(*args, c.cancelled)
            except Exception as exc: errors.append(exc)
            finally: done.set()
        with c.control_lock:
            c._check(c.state['map_name'])
            if supervised and c.state['phase'] == 'executing' and not c.state.get('mission_dispatched'):
                c.update(mission_dispatched=True)
                c.event('Mission started', c.state['mission']['mission_name'])
            c.action_thread = threading.Thread(target=work, daemon=True)
            c.action_thread.start()
        try:
            while not done.is_set() or c.state['paused']:
                if errors: raise errors[0]
                c._check(c.state['map_name'])
                if supervised: self.reasoner.supervise(None if done.is_set() else done)
                else: self.reasoner.infer(done)
                if done.is_set() and not c.state['paused']: break
                # Completion wakes this cadence without waiting for an extra inference interval.
                if c.state['paused']: c._wait(self.plan.interval_seconds)
                else:
                    deadline = time.monotonic() + self.plan.interval_seconds
                    while not done.is_set() and time.monotonic() < deadline:
                        c._wait(min(.05, max(0, deadline - time.monotonic())))
            if errors: raise errors[0]
        finally:
            try:
                if not done.is_set():
                    c.runtime.cancel()
            finally:
                c.action_thread.join(timeout=30)
            if c.action_thread.is_alive():
                raise RuntimeError('ROS action worker did not stop')
        if c.cancelled.is_set(): raise OperationCancelled()
