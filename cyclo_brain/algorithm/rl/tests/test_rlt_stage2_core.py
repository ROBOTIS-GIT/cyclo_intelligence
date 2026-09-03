"""Focused CPU tests for the compact active RLT Stage-2 core."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import torch

from cyclo_brain.algorithm.rl.rlt import (
    RLTokenAutoencoder,
    RLTokenConfig,
    RLTokenStage1Trainer,
    RLTStage2Batch,
    RLTStage2Config,
    RLTStage2Run,
    RLTStage2Spec,
    load_frozen_rl_token_encoder,
    load_groot_rlt_shadow_policy,
    stage2_spec_fingerprint,
)


def _encoder_artifact(root: Path) -> tuple[Path, object]:
    representation = {
        "format": "groot-final-layer-tokens/v1",
        "embeddings": {"width": 8, "dtype": "float32"},
        "token_selection": "image",
    }
    model = RLTokenAutoencoder(
        RLTokenConfig(
            embedding_dim=8,
            max_tokens=4,
            num_heads=2,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=16,
            dropout=0.0,
            token_selection="image",
        )
    )
    path = RLTokenStage1Trainer(model, representation).export_encoder(
        root / "rl_token_encoder.pt"
    )
    return path, load_frozen_rl_token_encoder(path)


def _spec(encoder) -> RLTStage2Spec:
    return RLTStage2Spec(
        reference_contract_fingerprint=(
            encoder.representation_contract_fingerprint
        ),
        rl_token_artifact_fingerprint=encoder.artifact_fingerprint,
        rl_token_dim=8,
        proprio_dim=19,
        reference_horizon=16,
        chunk_length=10,
        action_dim=19,
        action_hz=15.0,
        action_normalization_id="showroom-normalized-19d/v1",
        action_codec_id="ffw_sg2_rev1-recorder-order/v1",
        model_domain="normalized",
        schema_version=1,
    )


def _run(root: Path) -> RLTStage2Run:
    encoder_path, encoder = _encoder_artifact(root)
    return RLTStage2Run.new(
        encoder_path,
        spec=_spec(encoder),
        groot_checkpoint="/workspace/model/showroom_groot",
        # Deliberately distinct from the representation-contract digest.
        groot_checkpoint_fingerprint="a" * 64,
        representation_contract_fingerprint=(
            encoder.representation_contract_fingerprint
        ),
        config=RLTStage2Config(
            fixed_standard_deviation=0.05,
            policy_constraint_weight=0.1,
            target_update_rate=0.1,
            reference_dropout_probability=0.25,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            critic_updates_per_actor=2,
        ),
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        random_seed=23,
    )


def _batch(run: RLTStage2Run) -> RLTStage2Batch:
    generator = torch.Generator().manual_seed(31)
    batch_size = 3
    return RLTStage2Batch(
        spec_fingerprint=stage2_spec_fingerprint(run.learner.spec),
        z_rl=torch.randn(batch_size, 8, generator=generator),
        proprio=torch.randn(batch_size, 19, generator=generator),
        reference_actions=torch.randn(batch_size, 10, 19, generator=generator),
        executed_actions=torch.randn(batch_size, 10, 19, generator=generator),
        reward=torch.tensor([[1.0], [0.0], [0.2]]),
        bootstrap_discount=torch.tensor([[0.0], [0.9], [0.9]]),
        next_z_rl=torch.randn(batch_size, 8, generator=generator),
        next_proprio=torch.randn(batch_size, 19, generator=generator),
        next_reference_actions=torch.randn(
            batch_size, 10, 19, generator=generator
        ),
    )


def _assert_tree_equal(testcase: unittest.TestCase, left, right) -> None:
    if isinstance(left, torch.Tensor):
        testcase.assertIsInstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        testcase.assertEqual(set(left), set(right))
        for key in left:
            _assert_tree_equal(testcase, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        testcase.assertEqual(type(left), type(right))
        testcase.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right, strict=True):
            _assert_tree_equal(testcase, left_item, right_item)
    else:
        testcase.assertEqual(left, right)


def _canonical_fingerprint(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class RLTStage2CoreTest(unittest.TestCase):
    def test_new_update_save_and_inference_artifact_are_10_by_19(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _run(root)
            first = run.learner.update(_batch(run))
            second = run.learner.update(_batch(run))
            self.assertFalse(first.actor_updated)
            self.assertTrue(second.actor_updated)
            self.assertEqual(second.completed_critic_updates, 2)
            self.assertEqual(second.completed_actor_updates, 1)

            bundle = run.save(root / "bundle")
            self.assertTrue((bundle / "artifacts/rl_token_encoder.pt").is_file())
            self.assertTrue((bundle / "artifacts/rlt_actor.pt").is_file())
            self.assertTrue((bundle / "training_state/rlt_stage2.pt").is_file())
            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertEqual(manifest["initialization"]["mode"], "new")
            self.assertEqual(manifest["completed_critic_updates"], 2)

            policy = load_groot_rlt_shadow_policy(
                bundle / "artifacts/rl_token_encoder.pt",
                bundle / "artifacts/rlt_actor.pt",
            )
            output = policy(
                torch.randn(2, 4, 8),
                torch.ones(2, 4, dtype=torch.bool),
                torch.ones(2, 4, dtype=torch.bool),
                torch.randn(2, 19),
                torch.randn(2, 16, 19),
            )
            self.assertEqual(tuple(output.action_mean.shape), (2, 10, 19))

    def test_resume_restores_actor_critic_optimizers_counters_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _run(root)
            batch = _batch(original)
            original.learner.update(batch)
            original.learner.update(batch)
            bundle = original.save(root / "round_1")

            resumed = RLTStage2Run.resume(
                bundle,
                expected_groot_checkpoint_fingerprint="a" * 64,
            )
            self.assertEqual(resumed.initialization_mode, "resume")
            _assert_tree_equal(
                self,
                original.learner.state_dict(),
                resumed.learner.state_dict(),
            )
            # Two updates cross the delayed actor boundary, proving that the
            # actor/critic optimizers and all three RNG streams resume exactly.
            for _ in range(2):
                original_result = original.learner.update(batch)
                resumed_result = resumed.learner.update(batch)
                self.assertEqual(original_result, resumed_result)
                _assert_tree_equal(
                    self,
                    original.learner.state_dict(),
                    resumed.learner.state_dict(),
                )

            second = resumed.save(root / "round_2")
            second_manifest = json.loads((second / "manifest.json").read_text())
            self.assertEqual(second_manifest["initialization"]["mode"], "resume")
            self.assertEqual(
                second_manifest["initialization"]["parent_bundle_fingerprint"],
                json.loads((bundle / "manifest.json").read_text())[
                    "manifest_fingerprint"
                ],
            )

    def test_new_rejects_wrong_contracts_and_live_feature_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder_path, encoder = _encoder_artifact(root)
            wrong = deepcopy(_spec(encoder))
            object.__setattr__(wrong, "action_dim", 18)
            with self.assertRaisesRegex(ValueError, "10x19"):
                RLTStage2Run.new(
                    encoder_path,
                    spec=wrong,
                    groot_checkpoint="/workspace/model/groot",
                    groot_checkpoint_fingerprint="a" * 64,
                    representation_contract_fingerprint=(
                        encoder.representation_contract_fingerprint
                    ),
                )
            with self.assertRaisesRegex(ValueError, "representation"):
                RLTStage2Run.new(
                    encoder_path,
                    spec=_spec(encoder),
                    groot_checkpoint="/workspace/model/groot",
                    groot_checkpoint_fingerprint="a" * 64,
                    representation_contract_fingerprint="b" * 64,
                )

            run = _run(root / "valid")
            batch = _batch(run)
            object.__setattr__(batch, "z_rl", batch.z_rl.requires_grad_(True))
            with self.assertRaisesRegex(ValueError, "detached"):
                run.learner.update(batch)

    def test_resume_rejects_artifact_digest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _run(root)
            bundle = run.save(root / "bundle")
            actor = bundle / "artifacts/rlt_actor.pt"
            actor.write_bytes(actor.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "digest disagrees"):
                RLTStage2Run.resume(bundle)

    def test_resume_rejects_invalid_new_resume_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = _run(root).save(root / "bundle")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["initialization"]["mode"] = "resume"
            unsigned = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_fingerprint"
            }
            manifest["manifest_fingerprint"] = _canonical_fingerprint(unsigned)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "parent bundle fingerprint"):
                RLTStage2Run.resume(bundle)


if __name__ == "__main__":
    unittest.main()
