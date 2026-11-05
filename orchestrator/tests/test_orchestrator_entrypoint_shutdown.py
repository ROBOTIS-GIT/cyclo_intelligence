"""Exercise both entrypoint cleanup paths without importing or starting ROS."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


class ExternalShutdownException(Exception):
    pass


class RCLError(RuntimeError):
    pass


@pytest.mark.parametrize('path,node_class,cleanup', [
    ('orchestrator/orchestrator/orchestrator_node.py', 'OrchestratorNode',
     ['monitor', 'executor', 'node', 'context']),
    ('cyclo_data/cyclo_data/cyclo_data_node.py', 'CycloDataNode',
     ['hub', 'conversion', 'recording', 'executor', 'node', 'context']),
])
@pytest.mark.parametrize('context_closed', [False, True])
@pytest.mark.parametrize('spin_error', [
    None, KeyboardInterrupt, ExternalShutdownException, RuntimeError,
])
@pytest.mark.parametrize('shutdown_error,error_context_closed', [
    (None, True),
    (RCLError('failed to shutdown: rcl_shutdown already called'), True),
    (RCLError('unexpected shutdown failure'), True),
    (RCLError('failed to shutdown: rcl_shutdown already called'), False),
    (ValueError('rcl_shutdown already called'), True),
], ids=['normal', 'shutdown-race', 'unrelated-error', 'still-open', 'wrong-type'])
def test_entrypoint_shutdown(monkeypatch, path, node_class, cleanup,
                             context_closed, spin_error,
                             shutdown_error, error_context_closed):
    calls = []
    context = {'closed': False}

    def spin():
        context['closed'] = context_closed
        if spin_error is not None:
            raise spin_error('executor exited')

    def shutdown():
        if context['closed']:
            raise RuntimeError('rcl_shutdown already called')
        try_shutdown()

    def try_shutdown():
        calls.append('context')
        if shutdown_error is not None:
            # Model the native signal handler closing the context during shutdown.
            context['closed'] = error_context_closed
            raise shutdown_error
        context['closed'] = True

    node = SimpleNamespace(
        _stop_inference_status_monitor=lambda: calls.append('monitor'),
        _hub=SimpleNamespace(shutdown=lambda: calls.append('hub')),
        _conversion=SimpleNamespace(shutdown=lambda: calls.append('conversion')),
        _recording=SimpleNamespace(shutdown=lambda: calls.append('recording')),
        destroy_node=lambda: calls.append('node'),
    )
    executor = SimpleNamespace(
        add_node=Mock(), spin=spin, shutdown=lambda: calls.append('executor'))
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda: executor
    executors.ExternalShutdownException = ExternalShutdownException
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    ros = SimpleNamespace(init=Mock(), shutdown=shutdown, try_shutdown=try_shutdown,
                          ok=lambda: not context['closed'])
    namespace = {'rclpy': ros, node_class: lambda: node}

    # Compile the real entrypoint and executor import, excluding ROS node setup.
    source = ast.parse((ROOT / path).read_text(), filename=path)
    body = [item for item in source.body if (
        isinstance(item, ast.FunctionDef) and item.name == 'main'
    ) or (
        isinstance(item, ast.ImportFrom) and item.module == 'rclpy.executors'
    )]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), namespace)
    args = ['--ros-args']
    suppress_shutdown_error = (
        isinstance(shutdown_error, RuntimeError)
        and 'rcl_shutdown already called' in str(shutdown_error)
        and error_context_closed
    )
    if shutdown_error is not None and not suppress_shutdown_error:
        with pytest.raises(type(shutdown_error)) as caught:
            namespace['main'](args)
        assert caught.value is shutdown_error
    elif spin_error is RuntimeError:
        with pytest.raises(RuntimeError, match='executor exited'):
            namespace['main'](args)
    else:
        namespace['main'](args)

    ros.init.assert_called_once_with(args=args)
    executor.add_node.assert_called_once_with(node)
    assert calls == cleanup
    assert context['closed'] is (True if shutdown_error is None else error_context_closed)
