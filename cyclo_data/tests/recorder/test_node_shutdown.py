"""The real entrypoint must handle signals even inside a threading wait."""

import ast
import faulthandler
import importlib.util
from pathlib import Path
import os
import signal
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


def _signal_during_wait(signum):
    calls = []
    delivered = False
    main_thread = threading.current_thread()
    original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    class InterruptedEvent(threading.Event):
        def wait(self, timeout=None):
            nonlocal delivered
            if threading.current_thread() is main_thread and not delivered:
                delivered = True
                # Event.wait holds this non-reentrant lock before releasing it
                # in Condition.wait. Deliver an actual signal at that boundary.
                with self._cond:
                    os.kill(os.getpid(), signum)
            return super().wait(timeout)

    class ExternalShutdownException(Exception):
        pass

    def stop_recording():
        assert 'context' not in calls and 'executor' not in calls
        calls.append('recording')

    node = SimpleNamespace(
        _recording=SimpleNamespace(shutdown=stop_recording),
        _hub=SimpleNamespace(shutdown=lambda: calls.append('hub')),
        _conversion=SimpleNamespace(shutdown=lambda: calls.append('conversion')),
        destroy_node=lambda: calls.append('node'),
    )
    executor = SimpleNamespace(
        add_node=lambda node: None,
        spin_once=lambda timeout_sec: threading.Event().wait(0.01),
        shutdown=lambda: calls.append('executor'),
    )
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda **kwargs: executor
    executors.ExternalShutdownException = ExternalShutdownException
    sys.modules['rclpy.executors'] = executors
    path = Path(__file__).resolve().parents[2] / 'cyclo_data/cyclo_data_node.py'
    if not path.is_file():
        spec = importlib.util.find_spec('cyclo_data.cyclo_data_node')
        assert spec is not None and spec.origin is not None, 'cyclo_data node source not found'
        path = Path(spec.origin)
    tree = ast.parse(path.read_text())
    body = [item for item in tree.body if (
        isinstance(item, ast.FunctionDef) and item.name == 'main'
    ) or (
        isinstance(item, ast.ImportFrom) and item.module == 'rclpy.executors'
    )]
    namespace = {
        'rclpy': SimpleNamespace(init=lambda **kwargs: None, ok=lambda: 'context' not in calls,
                                 try_shutdown=lambda: calls.append('context')),
        'CycloDataNode': lambda: node,
        'signal': signal,
        'threading': SimpleNamespace(Event=InterruptedEvent, Thread=threading.Thread),
        'SignalHandlerOptions': SimpleNamespace(NO=0),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['main']()
    assert delivered
    assert calls == ['recording', 'hub', 'conversion', 'executor', 'node', 'context']
    assert {sig: signal.getsignal(sig) for sig in original_handlers} == original_handlers


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_signal_during_wait_does_not_deadlock_teardown(signum):
    try:
        result = subprocess.run(
            [sys.executable, __file__, str(signum)], capture_output=True, text=True, timeout=3)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'Signal handler blocked teardown:\n{exc.stderr}')
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == '__main__':
    faulthandler.dump_traceback_later(1)
    _signal_during_wait(int(sys.argv[1]))
    faulthandler.cancel_dump_traceback_later()
