"""Exercise both entrypoint cleanup paths without importing or starting ROS."""

import ast
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]


class ExternalShutdownException(Exception):
    pass


class RCLError(RuntimeError):
    pass


def _compile_main(path, namespace):
    source = ast.parse((ROOT / path).read_text(), filename=path)
    body = [item for item in source.body if (
        isinstance(item, ast.FunctionDef) and item.name == 'main'
    ) or (
        isinstance(item, ast.ImportFrom) and item.module == 'rclpy.executors'
    )]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), namespace)


def _signal_namespace():
    original = {2: object(), 15: object()}
    handlers = dict(original)

    def install(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        return previous

    return SimpleNamespace(SIGINT=2, SIGTERM=15, signal=install), handlers, original


@pytest.mark.parametrize('path,node_class,cleanup', [
    ('orchestrator/orchestrator/orchestrator_node.py', 'OrchestratorNode',
     ['monitor', 'executor', 'node', 'context']),
    ('cyclo_data/cyclo_data/cyclo_data_node.py', 'CycloDataNode',
     ['recording', 'hub', 'conversion', 'executor', 'node', 'context']),
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
    spin_called = False

    def spin():
        context['closed'] = context_closed
        if spin_error is not None:
            raise spin_error('executor exited')

    def spin_once(timeout_sec=0.1):
        nonlocal spin_called
        if not spin_called:
            spin_called = True
            spin()
            handlers[2](2, None)
        else:
            time.sleep(0.001)

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
        create_timer=Mock(return_value=SimpleNamespace(cancel=Mock())),
        state_callback_group=object(),
    )
    executor = SimpleNamespace(
        add_node=Mock(), spin=spin, spin_once=spin_once,
        shutdown=lambda: calls.append('executor'))
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda **kwargs: executor
    executors.ExternalShutdownException = ExternalShutdownException
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    ros = SimpleNamespace(init=Mock(), shutdown=shutdown, try_shutdown=try_shutdown,
                          ok=lambda: not context['closed'])
    signals, handlers, original_handlers = _signal_namespace()
    signal_options = SimpleNamespace(NO=object())
    namespace = {'rclpy': ros, node_class: lambda: node, 'signal': signals,
                 'threading': threading, 'SignalHandlerOptions': signal_options}

    # Compile the real entrypoint and executor import, excluding ROS node setup.
    _compile_main(path, namespace)
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

    if node_class == 'CycloDataNode':
        ros.init.assert_called_once_with(args=args, signal_handler_options=signal_options.NO)
        assert handlers == original_handlers
    else:
        ros.init.assert_called_once_with(args=args)
    executor.add_node.assert_called_once_with(node)
    assert calls == cleanup
    assert context['closed'] is (True if shutdown_error is None else error_context_closed)


@pytest.mark.parametrize('signum', [2, 15], ids=['SIGINT', 'SIGTERM'])
def test_data_signal_teardown_runs_with_live_context_before_executor_shutdown(monkeypatch, signum):
    calls = []
    context = {'closed': False}
    signals, handlers, originals = _signal_namespace()
    timer = SimpleNamespace(cancel=lambda: calls.append('timer cancelled'))
    callbacks = []
    state_group = object()
    main_thread = threading.current_thread()

    def create_timer(period, callback, *, callback_group):
        assert callback_group is state_group
        callbacks.append(callback)
        return timer

    def stop_recording():
        assert threading.current_thread() is main_thread
        assert not context['closed'], 'STOP must precede ROS context shutdown'
        assert 'executor' not in calls, 'executor must still deliver STOP response'
        calls.append('recording')

    def try_shutdown():
        if not context['closed']:
            calls.append('context')
        context['closed'] = True

    def spin():
        callbacks[0]()
        assert calls == []
        handlers[signum](signum, None)
        assert not context['closed']
        assert calls == [], 'signal handler must only request serialized teardown'
        callbacks[0]()
        assert context['closed']

    def spin_once(timeout_sec=0.1):
        handlers[signum](signum, None)
        time.sleep(0.01)

    node = SimpleNamespace(
        create_timer=create_timer, state_callback_group=state_group,
        _recording=SimpleNamespace(shutdown=stop_recording),
        _hub=SimpleNamespace(shutdown=lambda: calls.append('hub')),
        _conversion=SimpleNamespace(shutdown=lambda: calls.append('conversion')),
        destroy_node=lambda: calls.append('node'),
    )
    executor = SimpleNamespace(add_node=Mock(), spin=spin, spin_once=spin_once,
                               shutdown=lambda: calls.append('executor'))
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda **kwargs: executor
    executors.ExternalShutdownException = ExternalShutdownException
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    namespace = {
        'rclpy': SimpleNamespace(init=Mock(), try_shutdown=try_shutdown, ok=lambda: not context['closed']),
        'CycloDataNode': lambda: node, 'signal': signals, 'threading': threading,
        'SignalHandlerOptions': SimpleNamespace(NO=object()),
    }
    _compile_main('cyclo_data/cyclo_data/cyclo_data_node.py', namespace)
    namespace['main']()
    assert calls == ['recording', 'hub', 'conversion', 'executor', 'node', 'context']
    assert handlers == originals


@pytest.mark.parametrize('close_context_during_stop', [False, True])
def test_data_shutdown_keeps_executor_dispatching_delayed_stop(monkeypatch, close_context_during_stop):
    signals, handlers, originals = _signal_namespace()
    requested = threading.Event()
    acknowledged = threading.Event()
    context = {'closed': False}
    calls = []
    first_spin = True

    def spin_once(timeout_sec=0.1):
        nonlocal first_spin
        if first_spin:
            first_spin = False
            handlers[2](2, None)
        if requested.wait(0.01):
            if close_context_during_stop:
                context['closed'] = True
            else:
                acknowledged.set()

    def stop_recording():
        calls.append('recording')
        assert 'executor' not in calls
        requested.set()
        if close_context_during_stop:
            deadline = time.monotonic() + 1
            while not context['closed'] and time.monotonic() < deadline:
                time.sleep(0.01)
            assert context['closed']
            raise RuntimeError('context closed during STOP')
        assert acknowledged.wait(1), 'executor must dispatch delayed STOP acknowledgement'

    executor = SimpleNamespace(add_node=Mock(), spin=lambda: None, spin_once=spin_once,
                               shutdown=lambda: calls.append('executor'))
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda **kwargs: executor
    executors.ExternalShutdownException = ExternalShutdownException
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    node = SimpleNamespace(
        create_timer=Mock(return_value=SimpleNamespace(cancel=Mock())), state_callback_group=object(),
        _recording=SimpleNamespace(shutdown=stop_recording),
        _hub=SimpleNamespace(shutdown=lambda: calls.append('hub')),
        _conversion=SimpleNamespace(shutdown=lambda: calls.append('conversion')),
        destroy_node=lambda: calls.append('node'), get_logger=lambda: Mock(),
    )
    namespace = {'rclpy': SimpleNamespace(init=Mock(), ok=lambda: not context['closed'],
                 try_shutdown=lambda: context.update(closed=True)), 'CycloDataNode': lambda: node,
                 'signal': signals, 'threading': threading, 'SignalHandlerOptions': SimpleNamespace(NO=0)}
    _compile_main('cyclo_data/cyclo_data/cyclo_data_node.py', namespace)
    if close_context_during_stop:
        with pytest.raises(RuntimeError, match='context closed during STOP'):
            namespace['main']()
    else:
        namespace['main']()
    assert calls == ['recording', 'hub', 'conversion', 'executor', 'node']
    assert handlers == originals


def test_data_context_shutdown_exception_is_not_lost_in_executor_task(monkeypatch):
    signals, handlers, originals = _signal_namespace()
    context = {'closed': False}
    callbacks = []
    expected = RuntimeError('unrelated error after context closed')

    def try_shutdown():
        if not context['closed']:
            context['closed'] = True
            raise expected

    def spin():
        handlers[2](2, None)
        try:
            callbacks[0]()
        except RuntimeError:
            # Jazzy can exit when context.ok becomes false before observing a
            # worker Task's exception. The shutdown owner must not rely on it.
            pass

    def spin_once(timeout_sec=0.1):
        handlers[2](2, None)
        time.sleep(0.01)

    executor = SimpleNamespace(add_node=Mock(), spin=spin, spin_once=spin_once, shutdown=Mock())
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda **kwargs: executor
    executors.ExternalShutdownException = ExternalShutdownException
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    node = SimpleNamespace(
        create_timer=lambda period, cb, **kw: (callbacks.append(cb) or SimpleNamespace(cancel=Mock())),
        state_callback_group=object(), _recording=SimpleNamespace(shutdown=Mock()),
        _hub=SimpleNamespace(shutdown=Mock()), _conversion=SimpleNamespace(shutdown=Mock()),
        destroy_node=Mock(), get_logger=lambda: Mock(),
    )
    namespace = {'rclpy': SimpleNamespace(init=Mock(), ok=lambda: not context['closed'],
                 try_shutdown=try_shutdown), 'CycloDataNode': lambda: node, 'signal': signals,
                 'threading': threading, 'SignalHandlerOptions': SimpleNamespace(NO=0)}
    _compile_main('cyclo_data/cyclo_data/cyclo_data_node.py', namespace)
    with pytest.raises(RuntimeError) as caught:
        namespace['main']()
    assert caught.value is expected
    assert handlers == originals


@pytest.mark.parametrize('failure_at', ['node', 'executor'])
def test_data_setup_failure_restores_handlers_and_closes_context(monkeypatch, failure_at):
    signals, handlers, originals = _signal_namespace()
    expected = ValueError('setup failure')
    closed = []

    def fail():
        raise expected

    node = SimpleNamespace(
        _recording=SimpleNamespace(shutdown=Mock()), _hub=SimpleNamespace(shutdown=Mock()),
        _conversion=SimpleNamespace(shutdown=Mock()), destroy_node=Mock())
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda **kwargs: fail()
    executors.ExternalShutdownException = ExternalShutdownException
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    namespace = {'rclpy': SimpleNamespace(init=Mock(), ok=lambda: not closed,
                 try_shutdown=lambda: closed.append(True)),
                 'CycloDataNode': fail if failure_at == 'node' else lambda: node,
                 'signal': signals, 'threading': threading, 'SignalHandlerOptions': SimpleNamespace(NO=0)}
    _compile_main('cyclo_data/cyclo_data/cyclo_data_node.py', namespace)
    with pytest.raises(ValueError) as caught:
        namespace['main']()
    assert caught.value is expected
    assert handlers == originals
    assert closed


@pytest.mark.parametrize('signum', [2, 15], ids=['SIGINT', 'SIGTERM'])
def test_data_joins_spinner_before_destroying_executor_guards(monkeypatch, signum):
    signals, handlers, originals = _signal_namespace()
    threads = []
    closed = []

    def make_thread(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    def spin_once(timeout_sec):
        handlers[signum](signum, None)
        time.sleep(timeout_sec)

    def shutdown_executor():
        assert not threads[0].is_alive(), 'spin_once must exit before executor guards are cleared'

    executor = SimpleNamespace(add_node=Mock(), spin_once=spin_once, shutdown=shutdown_executor)
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda **kwargs: executor
    executors.ExternalShutdownException = ExternalShutdownException
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    node = SimpleNamespace(
        _recording=SimpleNamespace(shutdown=Mock()), _hub=SimpleNamespace(shutdown=Mock()),
        _conversion=SimpleNamespace(shutdown=Mock()), destroy_node=Mock())
    namespace = {'rclpy': SimpleNamespace(init=Mock(), ok=lambda: not closed,
                 try_shutdown=lambda: closed.append(True)), 'CycloDataNode': lambda: node,
                 'signal': signals, 'SignalHandlerOptions': SimpleNamespace(NO=0),
                 'threading': SimpleNamespace(Event=threading.Event, Thread=make_thread)}
    _compile_main('cyclo_data/cyclo_data/cyclo_data_node.py', namespace)
    try:
        namespace['main']()
    finally:
        for thread in threads:
            thread.join(timeout=1)
    node.destroy_node.assert_called_once()
    assert handlers == originals
    assert closed


def test_data_thread_start_failure_keeps_original_error_and_destroys_node(monkeypatch):
    signals, handlers, originals = _signal_namespace()
    expected = RuntimeError('cannot start executor thread')
    closed = []
    thread = threading.Thread(target=lambda: None)
    thread.start = Mock(side_effect=expected)
    executor = SimpleNamespace(add_node=Mock(), shutdown=Mock())
    executors = ModuleType('rclpy.executors')
    executors.MultiThreadedExecutor = lambda **kwargs: executor
    executors.ExternalShutdownException = ExternalShutdownException
    monkeypatch.setitem(sys.modules, 'rclpy.executors', executors)
    node = SimpleNamespace(
        _recording=SimpleNamespace(shutdown=Mock()), _hub=SimpleNamespace(shutdown=Mock()),
        _conversion=SimpleNamespace(shutdown=Mock()), destroy_node=Mock())
    namespace = {'rclpy': SimpleNamespace(init=Mock(), ok=lambda: not closed,
                 try_shutdown=lambda: closed.append(True)), 'CycloDataNode': lambda: node,
                 'signal': signals, 'SignalHandlerOptions': SimpleNamespace(NO=0),
                 'threading': SimpleNamespace(Event=threading.Event, Thread=lambda **kw: thread)}
    _compile_main('cyclo_data/cyclo_data/cyclo_data_node.py', namespace)
    with pytest.raises(RuntimeError) as caught:
        namespace['main']()
    assert caught.value is expected
    executor.shutdown.assert_called_once()
    node.destroy_node.assert_called_once()
    assert handlers == originals
    assert closed
