"""Single owner for the runtime socket and robot command publishers."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path


@contextmanager
def runtime_process_lock():
    socket_path = os.environ.get('POLICY_RUNTIME_CONTROL_SOCKET', '/run/cyclo/policy-runtime.sock')
    path = Path(socket_path + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not unlink this inode: concurrent launches must contend on the same lock.
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Policy Runtime is already running; stop the existing launch first') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
