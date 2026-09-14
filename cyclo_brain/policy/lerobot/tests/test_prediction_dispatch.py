"""Select the public API without interpreting internal model faults as missing APIs."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from lerobot_engine.prediction import PredictionMixin


def runtime(policy, predictor=None):
    result = PredictionMixin()
    result._policy = policy
    result._chunk_predictor = predictor
    return result


@pytest.mark.parametrize("error", [
    AttributeError("missing encoder attribute"),
    ValueError("invalid state"),
    NotImplementedError("prediction mode is unsupported"),
])
def test_internal_model_error_does_not_advance_a_second_prediction(error):
    policy = SimpleNamespace(predict_action_chunk=mock.Mock(side_effect=error), select_action=mock.Mock())
    with pytest.raises(type(error), match=str(error)):
        runtime(policy)._predict_chunk({})
    policy.select_action.assert_not_called()


def test_internal_failure_after_state_change_is_not_retried_through_select_action():
    class StatefulPolicy:
        def __init__(self):
            self.predictions = 0

        def predict_action_chunk(self, batch):
            self.predictions += 1
            raise NotImplementedError("unsupported decoder")

        def select_action(self, batch):
            return self.predict_action_chunk(batch)

    policy = StatefulPolicy()
    with pytest.raises(NotImplementedError, match="unsupported decoder"):
        runtime(policy)._predict_chunk({})
    assert policy.predictions == 1


def test_missing_chunk_method_preserves_public_select_action_compatibility():
    policy = SimpleNamespace(select_action=mock.Mock(return_value=torch.ones(2)))
    assert runtime(policy)._predict_chunk({}).shape == (1, 1, 2)
    policy.select_action.assert_called_once_with({})


def test_adapter_predictor_is_used_without_mutating_or_falling_back_to_policy_methods():
    policy = SimpleNamespace(predict_action_chunk=mock.Mock(), select_action=mock.Mock())
    predictor = mock.Mock(return_value=torch.ones(1, 2))
    engine = runtime(policy, predictor)
    assert engine._predict_chunk({"state": 1}).shape == (1, 1, 2)
    predictor.assert_called_once_with({"state": 1})
    policy.predict_action_chunk.assert_not_called()
    predictor.side_effect = NotImplementedError("adapter rejected batch")
    with pytest.raises(NotImplementedError, match="adapter rejected"):
        engine._predict_chunk({})
    policy.select_action.assert_not_called()


@pytest.mark.parametrize("batch_size", [0, 2, 4])
def test_single_robot_path_rejects_extra_or_missing_batches_before_cpu_copy(batch_size):
    action = torch.zeros(batch_size, 8, 22)
    with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected transfer")):
        with pytest.raises(ValueError, match="single batch"):
            PredictionMixin._to_numpy_chunk(action)


@pytest.mark.parametrize("shape,expected", [
    ((1, 8, 22), (8, 22)),
    ((8, 22), (8, 22)),
    ((22,), (1, 22)),
])
def test_supported_action_shapes_keep_all_values(shape, expected):
    import numpy as np

    action = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    result = PredictionMixin._to_numpy_chunk(action)
    assert result.shape == expected and result.dtype == np.float64
    np.testing.assert_array_equal(result.reshape(-1), action.numpy().reshape(-1))
