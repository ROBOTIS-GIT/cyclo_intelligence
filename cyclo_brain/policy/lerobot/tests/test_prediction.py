#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest

import torch


ENGINE_DIR = Path(__file__).resolve().parents[1] / "lerobot_engine"
package = types.ModuleType("lerobot_engine")
package.__path__ = [str(ENGINE_DIR)]
sys.modules.setdefault("lerobot_engine", package)

spec = importlib.util.spec_from_file_location(
    "lerobot_engine.prediction",
    ENGINE_DIR / "prediction.py",
)
prediction = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prediction
spec.loader.exec_module(prediction)
PredictionMixin = prediction.PredictionMixin


class _Policy:
    def __init__(
        self,
        actions: torch.Tensor,
        *,
        policy_type: str,
        n_action_steps: int | None = None,
        temporal_ensemble_coeff: float | None = None,
    ) -> None:
        self.actions = actions
        self.config = SimpleNamespace(
            type=policy_type,
            n_action_steps=n_action_steps,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )

    def predict_action_chunk(self, _batch):
        return self.actions


class _FallbackPolicy:
    config = SimpleNamespace(type="tdmpc")

    def predict_action_chunk(self, _batch):
        raise NotImplementedError

    def select_action(self, _batch):
        return torch.tensor([[1.0, 2.0]])


class _NoChunkPolicy:
    config = SimpleNamespace(type="tdmpc")

    def select_action(self, _batch):
        return torch.tensor([[3.0, 4.0]])


class _BrokenChunkPolicy:
    config = SimpleNamespace(type="diffusion")

    def predict_action_chunk(self, _batch):
        raise AttributeError("internal policy defect")

    def select_action(self, _batch):
        raise AssertionError("internal AttributeError must not fall back")


class _Predictor(PredictionMixin):
    def __init__(self, policy) -> None:
        self._policy = policy


class PredictionTest(unittest.TestCase):
    def test_act_returns_only_official_execution_prefix(self) -> None:
        actions = torch.arange(60 * 19, dtype=torch.float32).reshape(1, 60, 19)
        predictor = _Predictor(
            _Policy(actions, policy_type="act", n_action_steps=16)
        )

        actual = predictor._predict_chunk({})

        self.assertEqual(actual.shape, (1, 16, 19))
        torch.testing.assert_close(actual, actions[:, :16], rtol=0.0, atol=0.0)

    def test_non_act_chunk_horizon_is_unchanged(self) -> None:
        actions = torch.zeros(1, 7, 3)
        predictor = _Predictor(_Policy(actions, policy_type="diffusion"))

        self.assertIs(predictor._predict_chunk({}), actions)

    def test_invalid_act_execution_horizon_fails_fast(self) -> None:
        for horizon in (0, 5):
            predictor = _Predictor(
                _Policy(
                    torch.zeros(1, 4, 2),
                    policy_type="act",
                    n_action_steps=horizon,
                )
            )
            with self.subTest(horizon=horizon):
                with self.assertRaisesRegex(ValueError, "execution horizon"):
                    predictor._predict_chunk({})

    def test_temporal_ensemble_act_fails_fast(self) -> None:
        predictor = _Predictor(
            _Policy(
                torch.zeros(1, 4, 2),
                policy_type="act",
                n_action_steps=1,
                temporal_ensemble_coeff=0.01,
            )
        )

        with self.assertRaisesRegex(ValueError, "temporal ensembling"):
            predictor._predict_chunk({})

    def test_select_action_fallback_remains_one_step_chunk(self) -> None:
        actual = _Predictor(_FallbackPolicy())._predict_chunk({})

        self.assertEqual(actual.shape, (1, 1, 2))
        torch.testing.assert_close(
            actual,
            torch.tensor([[[1.0, 2.0]]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_missing_chunk_api_uses_select_action(self) -> None:
        actual = _Predictor(_NoChunkPolicy())._predict_chunk({})

        torch.testing.assert_close(
            actual,
            torch.tensor([[[3.0, 4.0]]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_internal_attribute_error_is_not_hidden_by_fallback(self) -> None:
        with self.assertRaisesRegex(AttributeError, "internal policy defect"):
            _Predictor(_BrokenChunkPolicy())._predict_chunk({})


if __name__ == "__main__":
    unittest.main()
