"""Focused host-side tests for deterministic value warm-up evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup import ChunkBoundaryRecord
from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup_eval import (
    CURRENT_VALUE_HEAD_HIDDEN_DIMS,
    ValueWarmupEvaluationSample,
    assert_exact_value_head_reload,
    evaluate_value_predictions,
    samples_from_records,
    validate_current_value_head_state_dict,
)
from cyclo_brain.algorithm.rl.flow_sde_ppo.value_warmup_eval_cli import (
    _atomic_json,
    _validate_bundle_manifest,
    _validate_checkpoint,
    _validate_output_path,
    build_parser,
)
from cyclo_brain.model.multi_task_dit.value_head import MultiTaskDiTValueHead


def _sample(
    episode: int,
    chunk: int,
    count: int,
    successful: bool,
    target: float,
    prediction: float,
    *,
    dataset: int = 0,
) -> ValueWarmupEvaluationSample:
    return ValueWarmupEvaluationSample(
        dataset_index=dataset,
        episode_index=episode,
        chunk_index=chunk,
        chunk_count=count,
        successful=successful,
        target_return=target,
        prediction=prediction,
    )


class FlowSDEValueWarmupEvaluationTest(unittest.TestCase):
    def test_metrics_cover_calibration_auc_margin_and_temporal_order(self):
        samples = (
            _sample(0, 0, 2, True, 0.8, 0.6),
            _sample(0, 1, 2, True, 1.0, 0.9),
            _sample(1, 0, 1, True, 1.0, 0.8),
            _sample(2, 0, 2, False, 0.0, 0.1),
            _sample(2, 1, 2, False, 0.0, 0.2),
            _sample(3, 0, 1, False, 0.0, -0.1),
        )
        result = evaluate_value_predictions(samples)

        self.assertEqual(result["scope"], "training_dataset_diagnostic")
        self.assertEqual(result["counts"]["chunk_boundaries"], 6)
        self.assertEqual(result["counts"]["success_episodes"], 2)
        self.assertEqual(result["counts"]["failure_episodes"], 2)
        self.assertEqual(result["discrimination"]["episode_mean_roc_auc"], 1.0)
        self.assertEqual(result["discrimination"]["episode_balanced_chunk_roc_auc"], 1.0)
        self.assertGreater(result["discrimination"]["episode_mean_margin"], 0.0)
        self.assertGreater(result["discrimination"]["terminal_chunk_margin"], 0.0)
        self.assertEqual(result["success_temporal"]["eligible_episode_count"], 1)
        self.assertEqual(result["success_temporal"]["spearman"]["mean"], 1.0)
        self.assertEqual(
            result["success_temporal"]["adjacent_nondecrease_rate"]["mean"], 1.0
        )
        self.assertGreater(result["baselines"]["mse_skill_over_best_constant"], 0.0)

    def test_episode_balancing_prevents_long_episode_domination(self):
        samples = []
        for chunk in range(10):
            samples.append(_sample(0, chunk, 10, True, 1.0, 0.0))
        samples.extend(
            (
                _sample(1, 0, 1, True, 1.0, 1.0),
                _sample(2, 0, 1, False, 0.0, 0.0),
            )
        )
        result = evaluate_value_predictions(samples)

        self.assertAlmostEqual(result["episode_balanced_metrics"]["mse"], 0.25)
        self.assertAlmostEqual(result["raw_chunk_metrics"]["mse"], 10.0 / 12.0)

    def test_tied_success_and_failure_scores_have_chance_auc(self):
        result = evaluate_value_predictions(
            (
                _sample(0, 0, 1, True, 1.0, 0.5),
                _sample(1, 0, 1, False, 0.0, 0.5),
            )
        )
        self.assertEqual(result["discrimination"]["episode_mean_roc_auc"], 0.5)
        self.assertEqual(result["discrimination"]["episode_balanced_chunk_roc_auc"], 0.5)

    def test_predictions_bind_to_records_in_exact_order(self):
        records = (
            ChunkBoundaryRecord(0, 0, 0, 0, 0, 2, True, 0.9),
            ChunkBoundaryRecord(0, 0, 4, 16, 1, 2, True, 1.0),
        )
        samples = samples_from_records(records, torch.tensor([0.8, 0.95]))
        self.assertEqual([sample.chunk_index for sample in samples], [0, 1])
        self.assertAlmostEqual(samples[1].prediction, 0.95, places=6)
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            samples_from_records(records, [0.8])

    def test_current_head_contract_and_exact_reload(self):
        torch.manual_seed(11)
        conditioning_dim = 7
        head = MultiTaskDiTValueHead(
            conditioning_dim, hidden_dims=CURRENT_VALUE_HEAD_HIDDEN_DIMS
        ).eval()
        conditioning = [torch.randn(3, conditioning_dim), torch.randn(2, conditioning_dim)]
        with torch.no_grad():
            predictions = [head(batch) for batch in conditioning]

        contract = validate_current_value_head_state_dict(
            head.state_dict(), conditioning_dim=conditioning_dim
        )
        self.assertFalse(contract["architecture_serialized_in_checkpoint"])
        reload_result = assert_exact_value_head_reload(
            head.state_dict(),
            conditioning,
            predictions,
            conditioning_dim=conditioning_dim,
            device=torch.device("cpu"),
        )
        self.assertTrue(reload_result["exact_prediction_match"])
        self.assertEqual(reload_result["prediction_max_abs_error"], 0.0)

        malformed = dict(head.state_dict())
        malformed["network.0.weight"] = malformed["network.0.weight"][:2]
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            validate_current_value_head_state_dict(
                malformed, conditioning_dim=conditioning_dim
            )

    def test_cli_defaults_to_stdout_and_sidecar_must_be_outside_bundle(self):
        args = build_parser().parse_args(["--bundle", "/bundle"])
        self.assertIsNone(args.output_json)
        self.assertEqual(args.batch_size, 8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = root / "bundle"
            bundle.mkdir()
            with self.assertRaisesRegex(ValueError, "outside the immutable"):
                _validate_output_path(bundle, bundle / "evaluation.json")
            output = _validate_output_path(bundle, root / "evaluation.json")
            self.assertEqual(output, root / "evaluation.json")
            _atomic_json(output, {"finite": 1.0})
            self.assertEqual(output.read_text(encoding="utf-8").strip(), '{\n  "finite": 1.0\n}')

    def test_bundle_and_checkpoint_integrity_gates_fail_closed(self):
        policy_hash = "sha256:" + "a" * 64
        manifest = {
            "format": "cyclo.flow_sde_ppo.value_warmup.bundle.v1",
            "status": "complete",
            "artifacts": {
                "model_path": "pretrained_model",
                "checkpoint_path": "training_state/value_warmup.pt",
                "progress_path": "progress.jsonl",
            },
            "config": {"steps": 10},
            "result": {
                "completed_steps": 10,
                "policy_sha256_before": policy_hash,
                "policy_sha256_after": policy_hash,
            },
            "base": {"policy_sha256": policy_hash},
            "dataset_contract": {"chunk_boundary_count": 4},
            "datasets": [{"path": "/workspace/lerobot/test"}],
        }
        checkpoint = {
            "format": "cyclo.flow_sde_ppo.value_warmup.v1",
            "status": "complete",
            "config": manifest["config"],
            "completed_steps": 10,
            "dataset_contract": manifest["dataset_contract"],
            "base_identity": manifest["base"],
            "dataset_identities": manifest["datasets"],
            "policy_sha256_before": policy_hash,
            "policy_sha256_after": policy_hash,
            "value_optimizer": {"state": {0: {}}, "param_groups": [{}]},
            "value_head": {"weight": torch.ones(1)},
        }
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "pretrained_model").mkdir()
            (bundle / "training_state").mkdir()
            (bundle / "training_state" / "value_warmup.pt").touch()
            (bundle / "progress.jsonl").touch()
            _validate_bundle_manifest(bundle, manifest)
            self.assertIs(_validate_checkpoint(checkpoint, manifest), checkpoint["value_head"])

            changed = dict(manifest)
            changed["result"] = {**manifest["result"], "policy_sha256_after": "sha256:changed"}
            with self.assertRaisesRegex(ValueError, "policy mutation"):
                _validate_bundle_manifest(bundle, changed)

            missing_optimizer = dict(checkpoint)
            missing_optimizer["value_optimizer"] = {"state": {}, "param_groups": [{}]}
            with self.assertRaisesRegex(ValueError, "optimizer state is empty"):
                _validate_checkpoint(missing_optimizer, manifest)


if __name__ == "__main__":
    unittest.main()
