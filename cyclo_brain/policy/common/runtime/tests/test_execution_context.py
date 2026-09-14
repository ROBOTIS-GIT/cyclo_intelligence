"""Context wire validation and opt-in Worker lifecycle behavior."""

from dataclasses import asdict, replace
import base64
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_process.protocol import (
    CMD_GET_ACTION, CMD_LOAD_POLICY, CMD_UPDATE_CONTEXT, EngineCommandRequest,
    request_from_message, request_to_message_kwargs,
)
from engine_process.worker import EngineWorker
import engine_process.worker as worker_module
from inference_context.execution import ActionRecord, ExecutionContext, PlanningRecord, ResetRecord
from main_runtime.inference_requester import InferenceRequester


class ContextEngine:
    def __init__(self):
        self.contexts = []
        self.calls = 0

    def load_policy(self, request):
        return {"success": True}

    def update_execution_context(self, context):
        self.contexts.append(context)

    def get_action_chunk(self, request):
        assert request.execution_context == self.contexts[-1]
        self.calls += 1
        return {"success": True, "chunk_size": 1, "action_dim": 2,
                "action_chunk": np.array([1., 2.])}


def context(**kwargs):
    return ExecutionContext(**{"session_id": "episode", "generation": 0,
                               "revision": 0, "phase": "running", **kwargs})


def request(command, ctx, seq=1, **kwargs):
    return EngineCommandRequest(command=command, seq_id=seq,
                                execution_context_json=ctx.to_json(), **kwargs)


def test_wire_roundtrip_and_explicit_spaces():
    ctx = context(actions=(ActionRecord("p1", "planned", "waypoint", (1., 2.), 0),))
    assert ExecutionContext.from_json(ctx.to_json()) == ctx
    req = request(CMD_GET_ACTION, ctx)
    from types import SimpleNamespace
    assert request_from_message(SimpleNamespace(**request_to_message_kwargs(req))) == req


def test_long_hold_feedback_keeps_every_receipt_and_timestamp():
    ctx = context(actions=tuple(
        ActionRecord("1", "published", "command", tuple(float(v) for v in range(22)),
                     command_id=i, event_id=i, recorded_s=100 + i / 100)
        for i in range(1, 2001)
    ), latest_event_id=2000)
    raw = ctx.to_json()
    assert json.loads(raw)["encoding"] == "zlib+base64"
    assert len(raw.encode()) < 65536
    assert ExecutionContext.from_json(raw) == ctx


@pytest.mark.parametrize("data", [b"not-zlib", zlib.compress(b"[]"),
    zlib.compress(b" " * (8 * 1024 * 1024 + 1)), zlib.compress(b"{}") + b"trailing"])
def test_invalid_compressed_context_is_bounded(data):
    raw = json.dumps({"encoding": "zlib+base64", "payload": base64.b64encode(data).decode()})
    with pytest.raises(ValueError):
        ExecutionContext.from_json(raw)


def test_large_incompressible_receipts_roundtrip_and_retry_once():
    engine = ContextEngine()
    worker = EngineWorker(engine)
    ready = context(phase="ready")
    assert worker.handle(request(CMD_LOAD_POLICY, ready)).success
    values = np.random.default_rng(42).normal(size=(3000, 22))
    records = tuple(ActionRecord("1", "published", "command", tuple(row),
                                 command_id=i, event_id=i, recorded_s=100 + i / 100)
                    for i, row in enumerate(values.tolist(), start=1))
    running = replace(ready, phase="running", revision=1, actions=records, latest_event_id=len(records))
    raw = running.to_json()
    assert 65536 < len(raw.encode()) < 8 * 1024 * 1024
    assert ExecutionContext.from_json(raw) == running
    req = request(CMD_GET_ACTION, running, seq=2)
    response = worker.handle(req)
    assert response.success
    assert worker.handle(req) == response and engine.calls == 1
    assert engine.contexts[-1].actions == records


def test_wire_and_decoded_context_budgets_are_bounded():
    with pytest.raises(ValueError, match="8 MiB"):
        ExecutionContext.from_json(" " * (8 * 1024 * 1024 + 1))
    values = tuple(.12345678912345678 for _ in range(256))
    ctx = context(actions=tuple(ActionRecord("1", "planned", "command", values, command_id=i)
                                for i in range(2048)))
    with pytest.raises(ValueError, match="expanded execution context exceeds 8 MiB"):
        ctx.to_json()


def test_compression_never_enlarges_the_wire_message(monkeypatch):
    ctx = context(actions=tuple(ActionRecord("1", "planned", "command", (1., 2.), command_id=i)
                                for i in range(256)))
    monkeypatch.setattr(zlib, "compress", lambda data: data * 2)
    raw = ctx.to_json()
    assert len(raw.encode()) > 65536
    assert "encoding" not in json.loads(raw)
    assert ExecutionContext.from_json(raw) == ctx


@pytest.mark.parametrize("raw", ["[]", "null", "{}", '{"extra":1}', '"' + "x" * 65536 + '"'])
def test_malformed_context_rejected(raw):
    with pytest.raises(ValueError):
        ExecutionContext.from_json(raw)


def test_invalid_action_facts_cannot_claim_execution():
    for kwargs in ({"status": "executed"}, {"space": "unknown"},
                   {"values": (float("nan"),)}, {"waypoint_index": -1}):
        with pytest.raises(ValueError):
            ActionRecord(**{"prediction_id": "p1", "status": "published",
                            "space": "command", "values": (1.,), **kwargs})


def test_requester_delivers_context_and_duplicate_request_does_not_advance_model():
    engine = ContextEngine()
    worker = EngineWorker(engine)
    ctx = context()
    assert worker.handle(request(CMD_LOAD_POLICY, ctx)).success

    class Client:
        def call(self, req, timeout_s):
            response = worker.handle(req)
            assert worker.handle(req) == response
            return response

    response = InferenceRequester(Client()).get_action("move", context=ctx)
    assert response.success
    assert engine.calls == 1
    assert response.action_list == [1., 2.]


def test_each_load_or_prediction_parses_context_once_even_on_retry(monkeypatch):
    engine = ContextEngine()
    worker = EngineWorker(engine)
    original = worker_module._parse_context
    parsed = []

    def parse(raw):
        parsed.append(raw)
        return original(raw)

    monkeypatch.setattr(worker_module, "_parse_context", parse)
    ctx = context()
    assert worker.handle(request(CMD_LOAD_POLICY, ctx)).success
    assert len(parsed) == 1
    req = request(CMD_GET_ACTION, ctx, seq=2)
    response = worker.handle(req)
    assert response.success and len(parsed) == 2
    assert worker.handle(req) == response and len(parsed) == 3
    assert engine.calls == 1


@pytest.mark.parametrize("count", [0, 20, 2000])
def test_shallow_record_encoding_keeps_exact_wire_bytes(count):
    ctx = context(actions=tuple(
        ActionRecord("1", "published", "command", tuple(float(v) for v in range(22)),
                     command_id=i, event_id=i, recorded_s=100 + i / 100,
                     planned_values=tuple(float(v + 1) for v in range(22)),
                     blend_weight=0.5, anchor_command_id=0,
                     anchor_values=tuple(0. for _ in range(22)), reason="publication")
        for i in range(1, count + 1)
    ), planning=(PlanningRecord(1, 1, 0, 1, count + 1, 101.),),
       resets=(ResetRecord(count + 2, "stop", 102.),), latest_event_id=count + 2)
    previous = json.dumps(asdict(ctx), allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(previous.encode()) > 65536:
        previous = json.dumps({"encoding": "zlib+base64", "payload": base64.b64encode(
            zlib.compress(previous.encode())).decode("ascii")}, separators=(",", ":"))
    assert ctx.to_json() == previous
    assert ExecutionContext.from_json(previous) == ctx


def test_stop_context_blocks_get_and_old_generations_cannot_return():
    engine = ContextEngine()
    worker = EngineWorker(engine)
    ctx = context()
    assert worker.handle(request(CMD_LOAD_POLICY, ctx)).success
    stopped = replace(ctx, generation=1, phase="stopped")
    assert worker.handle(request(CMD_UPDATE_CONTEXT, stopped, seq=2)).success
    assert not worker.handle(request(CMD_GET_ACTION, ctx, seq=3)).success
    assert not worker.handle(request(CMD_GET_ACTION, stopped, seq=4)).success
    assert engine.calls == 0
    resumed = replace(stopped, revision=1, phase="running")
    assert worker.handle(request(CMD_GET_ACTION, resumed, seq=5)).success
    assert engine.calls == 1


def test_context_revision_and_session_identity_are_not_ambiguous():
    worker = EngineWorker(ContextEngine())
    ctx = context()
    assert worker.handle(request(CMD_LOAD_POLICY, ctx)).success
    assert not worker.handle(request(CMD_UPDATE_CONTEXT, replace(ctx, phase="paused"))).success
    assert not worker.handle(request(CMD_GET_ACTION, replace(ctx, session_id="different"))).success
    assert not worker.handle(EngineCommandRequest(command=CMD_GET_ACTION)).success
    assert worker._engine_state == "loaded"


def test_unsupported_engine_rejects_context_before_model_load():
    class Legacy:
        def load_policy(self, request):
            pytest.fail("must validate contract before allocating weights")
    worker = EngineWorker(Legacy())
    response = worker.handle(request(CMD_LOAD_POLICY, context()))
    assert not response.success
    assert "does not implement" in response.message


def test_duplicate_id_with_new_context_cannot_mutate_engine():
    engine = ContextEngine()
    worker = EngineWorker(engine)
    ctx = context()
    worker.handle(request(CMD_LOAD_POLICY, ctx))
    assert worker.handle(request(CMD_GET_ACTION, ctx, seq=2)).success
    assert not worker.handle(request(CMD_GET_ACTION, replace(ctx, revision=1), seq=2)).success
    assert len(engine.contexts) == 1
    assert engine.calls == 1


def test_inference_failure_requires_new_generation():
    engine = ContextEngine()
    worker = EngineWorker(engine)
    ctx = context()
    worker.handle(request(CMD_LOAD_POLICY, ctx))
    original = engine.get_action_chunk
    engine.get_action_chunk = lambda request: {"success": False, "message": "failed after state mutation"}
    assert not worker.handle(request(CMD_GET_ACTION, ctx, seq=2)).success
    engine.get_action_chunk = original
    assert not worker.handle(request(CMD_GET_ACTION, ctx, seq=3)).success
    assert engine.calls == 0
    assert worker.handle(request(CMD_GET_ACTION, replace(ctx, generation=1), seq=4)).success
    assert engine.calls == 1


def test_partially_failed_context_hook_requires_new_generation():
    engine = ContextEngine()
    worker = EngineWorker(engine)
    ctx = context()
    assert worker.handle(request(CMD_LOAD_POLICY, ctx)).success
    original = engine.update_execution_context
    calls = []

    def fail_after_mutation(value):
        calls.append(value)
        raise RuntimeError("context reset partially failed")

    engine.update_execution_context = fail_after_mutation
    updated = replace(ctx, revision=1)
    assert not worker.handle(request(CMD_UPDATE_CONTEXT, updated, seq=2)).success
    engine.update_execution_context = original
    assert not worker.handle(request(CMD_GET_ACTION, updated, seq=3)).success
    assert len(calls) == 1 and engine.calls == 0
    assert worker.handle(request(CMD_GET_ACTION, replace(updated, generation=1), seq=4)).success


def test_contextual_timeout_requires_reset_even_if_worker_may_have_finished():
    from engine_process.protocol import EngineCommandResponse
    class Client:
        calls = 0

        def call(self, req, timeout_s):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("response lost after model advanced")
            return EngineCommandResponse(success=True, seq_id=req.seq_id)

    client = Client()
    requester = InferenceRequester(client)
    assert not requester.get_action("move", context=context()).success
    assert not requester.get_action("move", context=context()).success
    assert client.calls == 1
    assert requester.get_action("move", context=context(generation=1)).success


def test_nonfinite_model_result_does_not_advance_context_successfully():
    engine = ContextEngine()
    worker = EngineWorker(engine)
    ctx = context()
    worker.handle(request(CMD_LOAD_POLICY, ctx))
    engine.get_action_chunk = lambda req: {
        "success": True, "chunk_size": 1, "action_dim": 1,
        "action_chunk": [float("nan")],
    }
    assert not worker.handle(request(CMD_GET_ACTION, ctx, seq=2)).success
    response = worker.handle(request(CMD_GET_ACTION, ctx, seq=3))
    assert not response.success
    assert "reset" in response.message
