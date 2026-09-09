#!/usr/bin/env python3

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


GROOT_ROOT = Path(__file__).resolve().parents[1]
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

from runtime import rlt_stage2_training_cli as cli  # noqa: E402
from runtime.rlt_stage2_training_cli import (  # noqa: E402
    _bind_training_round_with_lineage,
    _sample_indices,
    _spec_from_encoder,
    build_parser,
)
from runtime.rlt_provenance import GR00TRLTProvenance  # noqa: E402


class RLTStage2TrainingCLITests(unittest.TestCase):
    def test_resume_bind_enforces_parent_replay_lineage_before_binding(self) -> None:
        parent_round = {"round": "parent"}
        current_round = {"round": "current"}
        stage2 = mock.Mock()
        stage2.initialization_mode = "resume"
        stage2.training_round = parent_round
        replay_root = Path("/workspace/replay")

        with mock.patch.object(
            cli,
            "validate_stage2_replay_lineage",
        ) as validate:
            _bind_training_round_with_lineage(
                stage2,
                current_round,
                replay_root=replay_root,
            )

        validate.assert_called_once_with(parent_round, current_round)
        stage2.bind_training_round.assert_called_once_with(
            current_round,
            replay_root=replay_root,
        )

    def test_new_bind_does_not_apply_resume_lineage_guard(self) -> None:
        stage2 = mock.Mock()
        stage2.initialization_mode = "new"
        stage2.training_round = None
        current_round = {"round": "current"}
        replay_root = Path("/workspace/replay")

        with mock.patch.object(
            cli,
            "validate_stage2_replay_lineage",
        ) as validate:
            _bind_training_round_with_lineage(
                stage2,
                current_round,
                replay_root=replay_root,
            )

        validate.assert_not_called()
        stage2.bind_training_round.assert_called_once_with(
            current_round,
            replay_root=replay_root,
        )

    def test_parser_matches_supervisor_new_and_resume_contract(self) -> None:
        common = [
            "--dataset-root",
            "/workspace/lerobot/a-v21",
            "--output-dir",
            "/workspace/checkpoint/rlt/stage2/run",
            "--job-id",
            "run",
            "--steps",
            "100",
            "--batch-size",
            "4",
            "--save-freq",
            "25",
        ]
        new = build_parser().parse_args(
            [
                "--initialization-mode",
                "new",
                "--groot-checkpoint",
                "/workspace/model/groot/showroom_groot",
                "--rl-token-encoder",
                "/workspace/checkpoint/rlt/stage1/run/artifacts/rl_token_encoder.pt",
                *common,
            ]
        )
        resumed = build_parser().parse_args(
            [
                "--initialization-mode",
                "resume",
                "--rlt-bundle",
                "/workspace/checkpoint/rlt/stage2/round-1",
                *common,
            ]
        )
        self.assertEqual(new.initialization_mode, "new")
        self.assertEqual(resumed.initialization_mode, "resume")
        self.assertEqual(new.batch_size, 4)

    def test_sampler_is_deterministic_and_spec_is_10x19(self) -> None:
        first = list(_sample_indices(5, 3, 4, seed=7))
        second = list(_sample_indices(5, 3, 4, seed=7))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(len(batch) == 3 for batch in first))

        encoder = SimpleNamespace(
            representation_contract_fingerprint="a" * 64,
            artifact_fingerprint="b" * 64,
            config=SimpleNamespace(embedding_dim=2048),
        )
        provenance = GR00TRLTProvenance(
            weight_fingerprint="c" * 64,
            model_config_fingerprint="d" * 64,
            processor_fingerprint="e" * 64,
            checkpoint_fingerprint="f" * 64,
            action_normalization_id=f"sha256:{'1' * 64}",
            action_codec_id=f"sha256:{'2' * 64}",
        )
        spec = _spec_from_encoder(
            encoder,
            action_hz=15.0,
            provenance=provenance,
        )
        self.assertEqual(spec.rl_token_dim, 2048)
        self.assertEqual((spec.chunk_length, spec.action_dim), (10, 19))
        self.assertEqual(spec.reference_horizon, 16)
        self.assertEqual(
            spec.action_normalization_id,
            provenance.action_normalization_id,
        )
        self.assertEqual(spec.action_codec_id, provenance.action_codec_id)

    def test_main_reports_machine_readable_error_event(self) -> None:
        argv = [
            "--initialization-mode",
            "new",
            "--dataset-root",
            "/workspace/lerobot/a-v21",
            "--groot-checkpoint",
            "/workspace/model/groot/showroom_groot",
            "--rl-token-encoder",
            "/workspace/checkpoint/rlt/stage1/run/artifacts/rl_token_encoder.pt",
            "--output-dir",
            "/workspace/checkpoint/rlt/stage2/run",
            "--job-id",
            "run",
            "--steps",
            "100",
            "--batch-size",
            "4",
            "--save-freq",
            "25",
        ]
        stderr = io.StringIO()
        with mock.patch.object(cli, "run", side_effect=PermissionError("denied")):
            with redirect_stderr(stderr):
                with self.assertRaises(PermissionError):
                    cli.main(argv)

        payload = json.loads(stderr.getvalue().splitlines()[0])
        self.assertEqual(payload["event"], "error")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["job_id"], "run")
        self.assertIn("PermissionError: denied", payload["message"])


if __name__ == "__main__":
    unittest.main()
