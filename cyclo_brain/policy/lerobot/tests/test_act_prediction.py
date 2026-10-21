"""ACT execution horizon must match the saved n_action_steps, not chunk_size."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "act_prediction_helpers", Path(__file__).resolve().parents[1] / "lerobot_engine/prediction.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ActPredictionTest(unittest.TestCase):
    def engine(self, steps=3, policy_type="act"):
        engine = module.PredictionMixin()
        engine._policy = SimpleNamespace(
            config=SimpleNamespace(type=policy_type, n_action_steps=steps),
            predict_action_chunk=lambda batch: torch.arange(40).reshape(1, 5, 8),
        )
        return engine

    def test_uses_only_requested_action_steps(self):
        action = self.engine()._predict_chunk({})
        self.assertEqual(tuple(action.shape), (1, 3, 8))
        torch.testing.assert_close(action, torch.arange(24).reshape(1, 3, 8))

    def test_rejects_invalid_horizon(self):
        for steps in (0, 6):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.engine(steps)._predict_chunk({})

    def test_other_policy_keeps_full_chunk(self):
        action = self.engine(policy_type="diffusion")._predict_chunk({})
        self.assertEqual(tuple(action.shape), (1, 5, 8))

    def test_runtime_selection_overrides_saved_execution_steps(self):
        engine = self.engine(steps=1)
        engine._runtime_chunk_size = 4
        self.assertEqual(tuple(engine._predict_chunk({}).shape), (1, 4, 8))
        self.assertEqual(engine._policy.config.n_action_steps, 1)

    def test_temporal_ensemble_uses_selected_prediction_horizon(self):
        engine = self.engine(steps=1)
        engine._policy.config.chunk_size = 5
        engine._policy.config.temporal_ensemble_coeff = 0.01
        engine._runtime_chunk_size = 2
        seen = []
        def update(actions):
            seen.append(tuple(actions.shape))
            return actions[:, 0]
        engine._policy.temporal_ensembler = SimpleNamespace(update=update)
        self.assertEqual(tuple(engine._predict_chunk({}).shape), (1, 1, 8))
        self.assertEqual(seen, [(1, 2, 8)])

    def test_temporal_ensemble_uses_official_select_action(self):
        engine = self.engine(steps=1)
        engine._policy.config.temporal_ensemble_coeff = 0.01
        engine._policy.select_action = lambda batch: torch.full((1, 8), 42.)
        torch.testing.assert_close(engine._predict_chunk({}), torch.full((1, 1, 8), 42.))


if __name__ == "__main__":
    unittest.main()
