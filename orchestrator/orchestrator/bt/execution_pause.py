"""ROS service adapter for BT pause; progress lives in existing tree objects."""
from functools import wraps

from .pause_control import pause_tree, resume_tree


def serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._execution_lock:
            return method(self, *args, **kwargs)
    return wrapped


def nonblocking_tick(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        # A pause may await actuator feedback. Do not occupy the node's default
        # callback group waiting for its lock, or service futures can deadlock.
        if not self._execution_lock.acquire(blocking=False):
            return
        try:
            return method(self, *args, **kwargs)
        finally:
            self._execution_lock.release()
    return wrapped


class ExecutionPause:
    @serialized
    def _set_paused_callback(self, request, response):
        try:
            mode = self.tree_execution_mode
            if mode in ('completed', 'failed'):
                # Completion can race the pause request; never start a new tree.
                response.success = True
                response.message = 'Task already terminal'
                return response
            if request.data:
                if mode == 'paused':
                    response.success = True
                    response.message = 'BT already paused'
                    return response
                if mode != 'running' or self.root is None:
                    raise RuntimeError('No running tree to pause')
                self.tree_execution_mode = 'pausing'
                supervisor = getattr(self, 'inference_supervisor', None)
                if supervisor is not None:
                    supervisor.pause()
                pause_tree(self.root)
                self.tree_execution_mode = 'paused'
                response.message = 'BT paused; tree progress preserved'
            else:
                if mode == 'running':
                    response.success = True
                    response.message = 'BT already running'
                    return response
                if mode != 'paused' or self.root is None:
                    raise RuntimeError('Only an acknowledged paused tree may resume')
                supervisor = getattr(self, 'inference_supervisor', None)
                if supervisor is not None:
                    supervisor.resume()
                try:
                    resume_tree(self.root)
                except Exception:
                    if supervisor is not None:
                        supervisor.pause()
                    raise
                self.tree_execution_mode = 'running'
                response.message = 'BT resumed at the preserved step'
            response.success = True
        except Exception as exc:
            # Never allow timer ticks after an unconfirmed actuator hold.
            if self.tree_execution_mode in ('pausing', 'paused'):
                self.tree_execution_mode = 'pause_failed'
            response.success = False
            response.message = str(exc)
        self._publish_status()
        return response
