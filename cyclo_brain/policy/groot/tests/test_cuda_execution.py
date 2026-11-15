"""Verify stream dependencies and exception cleanup without requiring CUDA."""
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def make_execution():
    events = []

    class Stream:
        def __init__(self, device=0):
            self.device = device

        def wait_stream(self, source):
            events.append((self, source))

    caller = Stream()
    current = [caller]

    @contextmanager
    def activate(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    cuda = SimpleNamespace(Stream=Stream, stream=activate,
                           current_stream=lambda device: current[0])
    spec = importlib.util.spec_from_file_location(
        'cuda_execution_test', Path(__file__).resolve().parents[1] / 'runtime/cuda_execution.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'torch': SimpleNamespace(cuda=cuda)}):
        spec.loader.exec_module(module)
    return module.InferenceStream(), caller, current, events


def test_inference_orders_input_and_output_without_device_sync():
    execution, caller, current, events = make_execution()
    for _ in range(2):
        with execution.activate():
            assert current[0] is execution.stream
            assert events[-1] == (execution.stream, caller)
        assert current[0] is caller
        assert events[-1] == (caller, execution.stream)
    assert len(events) == 4


def test_exception_restores_caller_and_orders_pending_work():
    execution, caller, current, events = make_execution()
    with pytest.raises(ValueError, match='inference failed'):
        with execution.activate():
            raise ValueError('inference failed')
    assert current[0] is caller
    assert events == [(execution.stream, caller), (caller, execution.stream)]
