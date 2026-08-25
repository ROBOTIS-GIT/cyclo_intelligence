#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
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


class _MultiTaskDiTPolicy:
    camera_keys = (
        "observation.images.rgb.cam_left_wrist",
        "observation.images.rgb.cam_left_head",
        "observation.images.rgb.cam_right_wrist",
    )

    def __init__(self, *, n_obs_steps: int = 1) -> None:
        self.config = SimpleNamespace(
            type="multi_task_dit",
            n_obs_steps=n_obs_steps,
            image_features={key: object() for key in self.camera_keys},
        )
        self.prepared_batch = None

    def predict_action_chunk(self, _batch):
        raise AssertionError("the queue-backed public API must not be called")

    def _prepare_batch(self, batch):
        prepared = dict(batch)
        prepared["observation.images"] = torch.stack(
            [prepared[key] for key in self.camera_keys], dim=-4
        )
        return prepared

    def _generate_actions(self, batch):
        self.prepared_batch = batch
        return torch.zeros(batch["observation.state"].shape[0], 16, 22)


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

    def test_multi_task_dit_adds_history_and_stacks_three_cameras(self) -> None:
        policy = _MultiTaskDiTPolicy()
        batch = {
            "observation.state": torch.zeros(2, 22),
            **{
                key: torch.zeros(2, 3, 32, 48)
                for key in policy.camera_keys
            },
            "observation.language.tokens": torch.zeros(2, 77, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(
                2, 77, dtype=torch.long
            ),
        }

        actual = _Predictor(policy)._predict_chunk(batch)

        self.assertEqual(actual.shape, (2, 16, 22))
        self.assertEqual(
            policy.prepared_batch["observation.state"].shape,
            (2, 1, 22),
        )
        self.assertEqual(
            policy.prepared_batch["observation.images"].shape,
            (2, 1, 3, 3, 32, 48),
        )
        for key in policy.camera_keys:
            self.assertEqual(policy.prepared_batch[key].shape, (2, 1, 3, 32, 48))
            self.assertEqual(batch[key].shape, (2, 3, 32, 48))
        self.assertEqual(batch["observation.state"].shape, (2, 22))

    def test_multi_task_dit_rejects_history_checkpoint_not_supported_by_engine(self) -> None:
        policy = _MultiTaskDiTPolicy(n_obs_steps=2)

        with self.assertRaisesRegex(ValueError, "requires n_obs_steps=1"):
            _Predictor(policy)._predict_chunk({"observation.state": torch.zeros(1, 22)})

    def test_multi_task_dit_rejects_missing_camera(self) -> None:
        policy = _MultiTaskDiTPolicy()
        batch = {
            "observation.state": torch.zeros(1, 22),
            **{
                key: torch.zeros(1, 3, 32, 48)
                for key in policy.camera_keys[:-1]
            },
        }

        with self.assertRaisesRegex(ValueError, "must have shape"):
            _Predictor(policy)._predict_chunk(batch)

    @unittest.skipUnless(
        os.environ.get("CYCLO_TEST_MULTI_TASK_DIT_CHECKPOINT"),
        "set CYCLO_TEST_MULTI_TASK_DIT_CHECKPOINT for the CUDA integration test",
    )
    def test_real_multi_task_dit_checkpoint_returns_16_by_22_chunk(self) -> None:
        from lerobot_engine.loading import LoadingMixin

        checkpoint = os.environ["CYCLO_TEST_MULTI_TASK_DIT_CHECKPOINT"]
        device = torch.device("cuda")
        policy, preprocessor, postprocessor = LoadingMixin._load_policy_assets(
            checkpoint,
            device,
        )
        raw_batch = {
            "observation.state": torch.zeros(1, 22),
            **{
                key: torch.full((1, 3, 256, 256), 0.5)
                for key in policy.config.image_features
            },
            "task": ["pick up the jelly bag"],
        }

        with torch.inference_mode():
            normalized = _Predictor(policy)._predict_chunk(
                preprocessor(raw_batch)
            )
            action = postprocessor(normalized)

        self.assertEqual(policy.config.type, "multi_task_dit")
        self.assertEqual(policy.config.objective, "flow_matching")
        self.assertEqual(action.shape, (1, 16, 22))
        self.assertTrue(bool(torch.isfinite(action).all()))


if __name__ == "__main__":
    unittest.main()
