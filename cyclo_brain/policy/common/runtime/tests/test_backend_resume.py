"""Actual backend reset/prediction paths with synthetic weights and receptions."""

from dataclasses import replace
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from inference_context.execution import ExecutionContext, ResetRecord
from inference_context.inputs import ReceivedValues, SampleQuery
from inference_context import observation


@pytest.mark.parametrize("backend", ["abot", "lingbot_vla", "rldx"])
@pytest.mark.parametrize("stale_source", ["camera:eye", "joint:arm"])
@pytest.mark.parametrize("remote_time", [1., 9000000.])
def test_resume_resets_loaded_backend_and_rejects_each_old_source(monkeypatch, backend, stale_source, remote_time):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / backend))
    module = importlib.import_module(f"{backend}_engine.engine")
    engine = module.create_engine()
    clock = [20.]
    monkeypatch.setattr(observation.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(observation.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    values = {"camera:eye": np.zeros((2, 2, 3), np.uint8), "joint:arm": np.array([3.])}
    stamps = {key: 19. for key in values}
    calls, resets = [], []

    class Robot:
        def get_required_input_snapshot(self, sources, **options):
            received = ReceivedValues(values, stamps, after_s=options.get("after_s"))
            for source in sources:
                received.resolve(SampleQuery(source), clock[0])
            return {"images": {"eye": values["camera:eye"]},
                    "joint_positions": {"arm": values["joint:arm"]}, "sensors": {},
                    "captured_monotonic_s": clock[0], "reception_monotonic_timestamps": stamps.copy()}

    def predict(inputs, **kwargs):
        calls.append(inputs)
        return {"action": [[3.]]}

    policy = SimpleNamespace(predict_action=predict, infer=predict,
        get_action=lambda inputs, **kwargs: (predict(inputs, **kwargs), {}),
        reset=lambda: resets.append(True), global_step=9,
        last_action_chunk=[9], last_normalized_action_chunk=[9])
    engine.policy, engine.robot = policy, Robot()
    engine.mapping = SimpleNamespace(sources=frozenset(values),
        observation=lambda snapshot, *args, **kwargs: snapshot,
        action=lambda actions, *args: np.array([[3.]]))
    engine.bundle = SimpleNamespace(horizon=1)
    engine.horizon, engine.language_key, engine.max_age_s = 1, "task", 1.
    initial = ExecutionContext("session", 0, 0, "ready")
    engine.update_execution_context(initial)
    running = replace(initial, phase="running", revision=1)
    engine.update_execution_context(running)
    # Simulate caches left by the prediction that completed before handoff.
    policy.last_action_chunk = policy.last_normalized_action_chunk = [9]
    resume = replace(running, generation=1, revision=2, latest_event_id=1,
                     resets=(ResetRecord(1, "control source resumed", remote_time),))
    engine.update_execution_context(resume)
    assert engine.policy is policy
    if backend == "lingbot_vla":
        assert policy.last_action_chunk is None and policy.last_normalized_action_chunk is None
    if backend == "rldx":
        assert len(resets) == 2
    clock[0] = 20.1
    stamps.update({key: 20.1 for key in values if key != stale_source})
    result = engine.get_action_chunk(SimpleNamespace(task_instruction="task"))
    assert not result["success"], result
    assert stale_source in result["message"]
    assert calls == []
    stamps.update({key: clock[0] for key in values})
    result = engine.get_action_chunk(SimpleNamespace(task_instruction="task"))
    assert result["success"], result
    assert len(calls) == 1 and engine.policy is policy
