"""Public step API contract tests; policy weights and transport are not loaded."""

from dataclasses import replace
import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from inference_context.execution import ActionRecord, ExecutionContext


_spec = importlib.util.spec_from_file_location(
    "step_adapter_under_test", Path(__file__).resolve().parents[1] / "lerobot_engine/adapters/public_step.py"
)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def adapter(*, strict=False):
    policy = mock.Mock()
    policy.select_action.return_value = torch.tensor([[1., 2.]])
    pre = mock.Mock(side_effect=lambda obs: obs)
    post = mock.Mock(side_effect=lambda action: action * 10)
    result = module.PublicStepAdapter(
        policy, pre, post, lambda a: a.detach().cpu().numpy(), require_exact_publication=strict,
    )
    result.update_execution_context(ExecutionContext("episode", 0, 0, "ready"))
    result.update_execution_context(ExecutionContext("episode", 0, 1, "running"))
    return result


def receipt(step, **overrides):
    fact = ActionRecord(**{
        "prediction_id": "1", "status": "published", "space": "command",
        "values": (10., 20.), "command_id": 0, "event_id": 1, "recorded_s": 10., **overrides,
    })
    return replace(step._context, revision=step._context.revision + 1,
                   actions=(fact,), after_event_id=0, latest_event_id=1)


def test_public_api_keeps_batch_rank_and_does_not_prefill_observation_history():
    step = adapter()
    obs = {"observation.state": torch.tensor([[3., 4.]])}
    np.testing.assert_array_equal(step.predict(obs, 1), [[10., 20.]])
    step.policy.select_action.assert_called_once_with(obs)
    assert step.postprocessor.call_args.args[0].shape == (1, 2)
    step.policy.predict_action_chunk.assert_not_called()
    with pytest.raises(RuntimeError, match="no publisher receipt"):
        step.predict(obs, 2)
    step.update_execution_context(receipt(step))
    step.predict(obs, 2)
    assert step.policy.select_action.call_count == 2


@pytest.mark.parametrize("mode", ["paused", "stopped", "error", "syncing"])
def test_generation_reset_clears_policy_and_processor_state_without_reloading(mode):
    step = adapter()
    step.predict({}, 1)
    ctx = replace(step._context, generation=1, phase=mode)
    step.update_execution_context(ctx)
    assert step.policy.reset.call_count == 2
    assert step.preprocessor.reset.call_count == 2
    assert step.postprocessor.reset.call_count == 2
    step.update_execution_context(replace(ctx, revision=2, phase="running"))
    step.predict({}, 2)
    step.policy.from_pretrained.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "discarded"])
def test_no_feedback_ack_for_failed_or_preview_only_publication(status):
    step = adapter()
    step.predict({}, 1)
    with pytest.raises(RuntimeError, match="not published"):
        step.update_execution_context(receipt(step, status=status))
    with pytest.raises(RuntimeError, match="reset"):
        step.predict({}, 2)


def test_pending_plan_is_not_an_ack():
    step = adapter()
    step.predict({}, 1)
    planned = ActionRecord("1", "planned", "command", (10., 20.))
    step.update_execution_context(replace(step._context, revision=2, actions=(planned,)))
    with pytest.raises(RuntimeError, match="no publisher receipt"):
        step.predict({}, 2)


def test_reordered_command_receipt_preserves_model_space_result_memory():
    step = adapter(strict=True)
    results = []
    step.set_result_observer(lambda raw, processed: results.append((raw.clone(), processed.clone())))
    step.set_action_mapping(lambda chunk: chunk[:, [1, 0]])
    np.testing.assert_array_equal(step.predict({}, 1), [[20., 10.]])
    np.testing.assert_array_equal(results[0][0], [[1., 2.]])
    np.testing.assert_array_equal(results[0][1], [[10., 20.]])
    with pytest.raises(RuntimeError, match="cannot change"):
        step.set_action_mapping(lambda chunk: chunk)
    step.update_execution_context(receipt(step, values=(20., 10.)))
    np.testing.assert_array_equal(step.predict({}, 2), [[20., 10.]])


@pytest.mark.parametrize("strict", [False, True])
def test_model_cached_actions_require_exact_published_values(strict):
    step = adapter(strict=strict)
    step.predict({}, 1)
    feedback = receipt(step, values=(0., 20.), planned_values=(10., 20.))
    if strict:
        with pytest.raises(RuntimeError, match="differs"):
            step.update_execution_context(feedback)
    else:
        step.update_execution_context(feedback)
        step.predict({}, 2)


def test_interpolated_or_wrong_action_cannot_ack_public_step():
    step = adapter()
    step.predict({}, 1)
    with pytest.raises(RuntimeError, match="differs"):
        step.update_execution_context(receipt(step, values=(5., 10.)))


@pytest.mark.parametrize("action", [torch.ones(2, 2), torch.ones(1, 2, 2),
                                   torch.tensor([[float("nan"), 1.]])])
def test_bad_output_latches_adapter_until_reset(action):
    step = adapter()
    step.policy.select_action.return_value = action
    with pytest.raises(ValueError, match="one finite"):
        step.predict({}, 1)
    with pytest.raises(RuntimeError, match="reset"):
        step.predict({}, 2)
    step.update_execution_context(replace(step._context, generation=1))
    step.policy.select_action.return_value = torch.ones(1, 2)
    step.predict({}, 2)


def test_exception_during_model_step_does_not_reuse_partially_updated_cache():
    step = adapter()
    step.policy.select_action.side_effect = RuntimeError("half advanced")
    with pytest.raises(RuntimeError, match="half advanced"):
        step.predict({}, 1)
    step.policy.select_action.side_effect = None
    with pytest.raises(RuntimeError, match="reset"):
        step.predict({}, 2)


def test_context_and_prediction_identity_guards():
    step = adapter()
    step.predict({}, 1)
    step.update_execution_context(receipt(step))
    with pytest.raises(ValueError, match="increase"):
        step.predict({}, 1)
    with pytest.raises(ValueError, match="mismatch"):
        step.update_execution_context(replace(step._context, session_id="other"))


def test_observation_barrier_uses_first_publication_not_latest_hold():
    step = adapter()
    assert step.observation_after_s is None
    step.predict({}, 1)
    step.update_execution_context(receipt(step, recorded_s=10.))
    step.update_execution_context(receipt(step, recorded_s=10.5))
    assert step.observation_after_s == 10.
    step.predict({}, 2)
    step.update_execution_context(receipt(step, prediction_id="2", recorded_s=11.))
    assert step.observation_after_s == 11.
    step.update_execution_context(replace(step._context, generation=1))
    assert step.observation_after_s is None


def test_publication_without_timestamp_cannot_advance_step():
    step = adapter()
    step.predict({}, 1)
    with pytest.raises(RuntimeError, match="timestamp"):
        step.update_execution_context(receipt(step, recorded_s=None))
    with pytest.raises(RuntimeError, match="reset"):
        step.predict({}, 2)


@pytest.mark.parametrize("name,expected", [("act", "chunk"), ("groot", "chunk"),
                                         ("diffusion", "step"),
                                         ("multi_task_dit", "step"), ("lingbot_va", "step")])
def test_candidate_contract_is_resolved_at_adapter_boundary(name, expected):
    from lerobot_engine.adapters import resolve_adapter
    assert resolve_adapter(name).contract.mode == expected


@pytest.mark.parametrize("name,strict", [("diffusion", False), ("multi_task_dit", False), ("lingbot_va", True)])
def test_factory_sets_required_feedback_contract(name, strict):
    from lerobot_engine.adapters import resolve_adapter
    step = adapter()
    step.policy.config.type = name
    step.preprocessor.steps = ()
    candidate = resolve_adapter(name).create_execution_adapter(
        step.policy, step.preprocessor, step.postprocessor, step.to_numpy,
    )
    assert candidate.require_exact_publication is strict
    step.policy.config.type = "act"
    assert resolve_adapter("act").create_execution_adapter(
        step.policy, step.preprocessor, step.postprocessor, step.to_numpy) is None


def test_batch_validation_runs_after_processor_and_before_policy_queue_update():
    step = adapter()
    step.preprocessor.side_effect = lambda obs: {"aligned": True}
    step.batch_validator = mock.Mock(side_effect=ValueError("invalid processed input"))
    with pytest.raises(ValueError, match="invalid processed input"):
        step.predict({"raw": True}, 1)
    step.batch_validator.assert_called_once_with({"aligned": True})
    step.policy.select_action.assert_not_called()
    step.postprocessor.assert_not_called()
    with pytest.raises(RuntimeError, match="reset"):
        step.predict({}, 2)
