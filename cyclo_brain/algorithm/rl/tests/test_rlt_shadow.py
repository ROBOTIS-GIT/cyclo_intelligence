"""Focused contract checks for the existing GR00T RLT showroom bundle."""

import os
from pathlib import Path
import unittest

import torch

from cyclo_brain.algorithm.rl.rlt import load_groot_rlt_shadow_policy


_REPOSITORY = Path(__file__).resolve().parents[4]
_REPOSITORY_WORKSPACE = _REPOSITORY / "docker/workspace"
_WORKSPACE = Path(
    os.environ.get("CYCLO_WORKSPACE_ROOT", str(_REPOSITORY_WORKSPACE))
)
if not _WORKSPACE.is_dir() and Path("/workspace").is_dir():
    _WORKSPACE = Path("/workspace")
_RLT_ROOT = _WORKSPACE / "checkpoint/rlt"
_ENCODER = (
    _RLT_ROOT
    / "showroom_groot_rlt_stage1_2k_v1/artifacts/rl_token_encoder.pt"
)
_ACTOR = (
    _RLT_ROOT
    / "showroom_groot_rlt_stage2_train_pilot_v2/artifacts/rlt_actor.pt"
)


@unittest.skipUnless(
    _ENCODER.is_file() and _ACTOR.is_file(),
    "local showroom GR00T RLT artifacts are not installed",
)
class ShowroomGR00TRLTShadowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_groot_rlt_shadow_policy(_ENCODER, _ACTOR)

    def test_artifact_contract_is_the_expected_frozen_10_by_19_bundle(self):
        self.assertEqual(self.policy.spec.rl_token_dim, 2048)
        self.assertEqual(self.policy.spec.proprio_dim, 19)
        self.assertEqual(self.policy.spec.reference_horizon, 16)
        self.assertEqual(self.policy.spec.chunk_length, 10)
        self.assertEqual(self.policy.spec.action_dim, 19)
        self.assertEqual(self.policy.spec.action_hz, 15.0)
        self.assertEqual(
            self.policy.actor_qualification,
            "training_only_not_deployment_validated",
        )
        self.assertFalse(self.policy.encoder.training)
        self.assertFalse(
            any(parameter.requires_grad for parameter in self.policy.encoder.parameters())
        )

        self.policy.train()
        self.assertFalse(self.policy.encoder.training)

    def test_shadow_forward_returns_exact_finite_action_chunk(self):
        generator = torch.Generator().manual_seed(20260825)
        tokens = torch.randn(2, 5, 2048, generator=generator)
        token_valid = torch.ones(2, 5, dtype=torch.bool)
        image_token = torch.tensor(
            [[True, True, False, True, False], [True, False, True, False, False]]
        )
        proprio = torch.randn(2, 19, generator=generator)
        reference = torch.randn(2, 16, 19, generator=generator)

        output = self.policy(
            tokens,
            token_valid,
            image_token,
            proprio,
            reference,
        )

        self.assertEqual(tuple(output.z_rl.shape), (2, 2048))
        self.assertEqual(tuple(output.reference_prefix.shape), (2, 10, 19))
        self.assertEqual(tuple(output.action_mean.shape), (2, 10, 19))
        self.assertTrue(bool(torch.isfinite(output.action_mean).all()))
        self.assertFalse(output.action_mean.requires_grad)
        torch.testing.assert_close(
            output.reference_prefix,
            reference[:, :10],
            rtol=0.0,
            atol=0.0,
        )

    def test_shadow_forward_rejects_a_wrong_reference_horizon(self):
        with self.assertRaisesRegex(ValueError, "reference_actions"):
            self.policy(
                torch.zeros(1, 2, 2048),
                torch.ones(1, 2, dtype=torch.bool),
                torch.ones(1, 2, dtype=torch.bool),
                torch.zeros(1, 19),
                torch.zeros(1, 10, 19),
            )

    def test_shadow_forward_can_select_the_tt_rtc_shifted_reference(self):
        reference = torch.arange(16 * 19, dtype=torch.float32).reshape(1, 16, 19)

        output = self.policy(
            torch.ones(1, 2, 2048),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 19),
            reference,
            reference_offset_steps=6,
        )

        torch.testing.assert_close(
            output.reference_prefix,
            reference[:, 6:16],
            rtol=0.0,
            atol=0.0,
        )

    def test_shadow_forward_rejects_an_out_of_range_tt_rtc_shift(self):
        with self.assertRaisesRegex(ValueError, "reference_offset_steps"):
            self.policy(
                torch.ones(1, 2, 2048),
                torch.ones(1, 2, dtype=torch.bool),
                torch.ones(1, 2, dtype=torch.bool),
                torch.zeros(1, 19),
                torch.zeros(1, 16, 19),
                reference_offset_steps=7,
            )


if __name__ == "__main__":
    unittest.main()
