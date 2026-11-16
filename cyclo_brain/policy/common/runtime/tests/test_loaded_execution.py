"""The LOAD response selects scheduling; caller/model names never do."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from engine_process.protocol import CMD_GET_ACTION, CMD_LOAD_POLICY, EngineCommandRequest
from engine_process.worker import EngineWorker
from inference_context.contract import ExecutionContract, LoadedExecution
from inference_context.execution import ExecutionContext
from main_runtime.service_handler import CMD_LOAD, ServiceHandler
from main_runtime.session_state import SessionState
from .test_service_handler import FakeControlLoop, FakeRequester, make_response


def test_legacy_load_stays_chunk_without_feedback():
    execution = LoadedExecution.from_json("")
    assert execution == LoadedExecution()
    assert execution.to_json() == ""
    assert LoadedExecution.from_json("{}") == execution


def test_step_load_roundtrip():
    execution = LoadedExecution(ExecutionContract("step"), ExecutionContext("session", 0, 0, "ready"))
    assert LoadedExecution.from_json(execution.to_json()) == execution


@pytest.mark.parametrize("count", [None, 0, 3, 4096])
def test_pending_plan_budget_roundtrip(count):
    execution = LoadedExecution(ExecutionContract(pending_command_count=count),
                                ExecutionContext("session", 0, 0, "ready"))
    assert LoadedExecution.from_json(execution.to_json()) == execution


@pytest.mark.parametrize("count", [-1, 4097, True, 1.0, "3"])
def test_pending_plan_budget_rejects_invalid_values(count):
    with pytest.raises(ValueError, match="pending command count"):
        ExecutionContract(pending_command_count=count)


def test_pending_plan_budget_requires_context():
    with pytest.raises(ValueError, match="execution session"):
        LoadedExecution(ExecutionContract(pending_command_count=0))


@pytest.mark.parametrize("payload", [
    None, False, 0, "null", "[]", "x", " " * 4097,
    '{"execution_contract":{"mode":"step"},"execution_context":null}',
    '{"execution_contract":{"mode":"unknown"},"execution_context":null}',
    '{"execution_contract":{"mode":"chunk","extra":1},"execution_context":null}',
])
def test_malformed_load_fails_closed(payload):
    with pytest.raises(ValueError):
        LoadedExecution.from_json(payload)


def test_load_cannot_restore_running_context():
    context = ExecutionContext("session", 0, 0, "running")
    with pytest.raises(ValueError, match="empty and ready"):
        LoadedExecution(ExecutionContract("step"), context)


def test_worker_negotiates_step_context_and_refuses_legacy_get_action():
    contexts = []
    engine = SimpleNamespace(
        load_policy=lambda req: {"success": True, "execution_contract": ExecutionContract("step")},
        update_execution_context=contexts.append,
        get_action_chunk=lambda req: {"success": True, "chunk_size": 1, "action_dim": 1, "action_chunk": [1.]},
    )
    worker = EngineWorker(engine)
    response = worker.handle(EngineCommandRequest(command=CMD_LOAD_POLICY, seq_id=1))
    assert response.success
    loaded = LoadedExecution.from_json(response.capabilities_json)
    assert loaded.contract.is_step
    assert contexts == [loaded.context]
    unsafe = worker.handle(EngineCommandRequest(command=CMD_GET_ACTION, seq_id=2))
    assert not unsafe.success and "requires execution context" in unsafe.message
    running = replace(loaded.context, phase="running", revision=1)
    action = worker.handle(EngineCommandRequest(command=CMD_GET_ACTION, seq_id=3,
                                               execution_context_json=running.to_json()))
    assert action.success
    assert contexts[-1] == running


def test_step_engine_without_context_hook_cannot_load_successfully():
    engine = SimpleNamespace(load_policy=lambda req: {
        "success": True, "execution_contract": ExecutionContract("step"),
    })
    response = EngineWorker(engine).handle(EngineCommandRequest(command=CMD_LOAD_POLICY, seq_id=1))
    assert not response.success
    assert "execution context" in response.message


def test_temporal_chunk_load_negotiates_context_without_step_scheduling():
    contexts = []
    engine = SimpleNamespace(load_policy=lambda req: {
        "success": True, "execution_contract": ExecutionContract(),
        "requires_execution_context": True,
    }, update_execution_context=contexts.append)
    response = EngineWorker(engine).handle(EngineCommandRequest(command=CMD_LOAD_POLICY, seq_id=1))
    assert response.success
    loaded = LoadedExecution.from_json(response.capabilities_json)
    assert not loaded.contract.is_step and loaded.context is not None
    assert contexts == [loaded.context]


@pytest.mark.parametrize("valid", [True, False])
def test_service_configures_negotiated_step_or_rolls_back(valid):
    execution = LoadedExecution(ExecutionContract("step"), ExecutionContext("negotiated", 0, 0, "ready"))
    requester, loop, session = FakeRequester(), FakeControlLoop(), SessionState()
    requester.load_policy = lambda req: SimpleNamespace(
        success=True, message="loaded", action_keys=["arm"],
        capabilities_json=execution.to_json() if valid else '{"execution_contract":{}}',
    )
    handler = ServiceHandler(session, requester, loop, make_response, backend="lerobot")
    response = handler.handle(SimpleNamespace(command=CMD_LOAD, model_path="/models/test",
                                              robot_type="test", task_instruction="pick", publish_to_robot=True))
    assert response.success == valid
    assert session.loaded == valid
    if valid:
        assert loop.configures[-1]["execution_contract"] == execution.contract
        assert loop.configures[-1]["execution_context"] == execution.context
        assert requester.unload_count == 0
    else:
        assert not loop.configures
        assert requester.unload_count == 1
