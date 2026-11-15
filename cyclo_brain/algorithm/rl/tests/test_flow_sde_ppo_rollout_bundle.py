"""Persistence and split collection/update tests for Flow-SDE PPO."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from cyclo_brain.algorithm.rl.flow_sde_ppo import (
    ROLLOUT_PAYLOAD_NAME,
    SOURCE_POLICY_FORMAT,
    SOURCE_TRAINER_STATE_NAME,
    collect_one_episode,
    load_rollout_bundle,
    mark_rollout_bundle_consumed,
    save_rollout_bundle,
    update_rollout_bundle,
)


def _source_policy() -> dict:
    return {
        "format": SOURCE_POLICY_FORMAT,
        "checkpoint_path": "/workspace/checkpoint/test_multi_task_dit",
        "artifacts": {
            "config.json": "sha256:" + "1" * 64,
            "model.safetensors": "sha256:" + "2" * 64,
            "policy_preprocessor.json": "sha256:" + "3" * 64,
            "policy_postprocessor.json": "sha256:" + "4" * 64,
        },
        "frozen_policy_sha256": "sha256:" + "5" * 64,
        "policy_contract": {"horizon": 2, "action_dim": 2},
        "critic_contract": {
            "type": "multi_task_dit_value_head",
            "conditioning_dim": 3,
        },
        "task_instruction": "pick up the jelly bag",
        "robot_type": "ffw_sg2_rev1",
    }
from cyclo_brain.algorithm.rl.tests.test_flow_sde_ppo_runner import (
    _ThreeDecisionSource,
    _assert_tree_exact,
    _make_trainer,
)


def _assert_episode_exact(
    test: unittest.TestCase,
    expected,
    actual,
) -> None:
    test.assertEqual(len(actual.transitions), len(expected.transitions))
    test.assertEqual(actual.bootstrap_value, expected.bootstrap_value)
    test.assertEqual(actual.episode_return, expected.episode_return)
    for expected_transition, actual_transition in zip(
        expected.transitions, actual.transitions, strict=True
    ):
        torch.testing.assert_close(
            actual_transition.conditioning,
            expected_transition.conditioning,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual_transition.rollout.chains,
            expected_transition.rollout.chains,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual_transition.rollout.denoise_indices,
            expected_transition.rollout.denoise_indices,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual_transition.rollout.old_log_probs,
            expected_transition.rollout.old_log_probs,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual_transition.rollout.action_mask,
            expected_transition.rollout.action_mask,
            rtol=0.0,
            atol=0.0,
        )
        test.assertEqual(actual_transition.reward, expected_transition.reward)
        test.assertEqual(actual_transition.terminated, expected_transition.terminated)
        test.assertEqual(actual_transition.truncated, expected_transition.truncated)
        test.assertEqual(actual_transition.old_value, expected_transition.old_value)


class FlowSDEPPORolloutBundleTest(unittest.TestCase):
    def test_collect_only_does_not_mutate_actor_critic_or_optimizers(self):
        torch.manual_seed(101)
        trainer = _make_trainer()
        actor_before = copy.deepcopy(trainer._actor_module().state_dict())
        critic_before = copy.deepcopy(trainer.value_head.state_dict())
        actor_optimizer_before = copy.deepcopy(trainer.actor_optimizer.state_dict())
        critic_optimizer_before = copy.deepcopy(trainer.value_optimizer.state_dict())

        episode = collect_one_episode(trainer, _ThreeDecisionSource())

        self.assertEqual(len(episode.transitions), 3)
        self.assertEqual(trainer.update_step, 0)
        _assert_tree_exact(self, actor_before, trainer._actor_module().state_dict())
        _assert_tree_exact(self, critic_before, trainer.value_head.state_dict())
        _assert_tree_exact(
            self, actor_optimizer_before, trainer.actor_optimizer.state_dict()
        )
        _assert_tree_exact(
            self, critic_optimizer_before, trainer.value_optimizer.state_dict()
        )

    def test_bundle_roundtrip_preserves_every_rollout_value(self):
        torch.manual_seed(103)
        trainer = _make_trainer()
        episode = collect_one_episode(trainer, _ThreeDecisionSource())
        identity = trainer.rollout_policy_identity()
        metadata = {
            "task_instruction": "pick up the jelly bag",
            "robot_type": "ffw_sg2_rev1",
            "outcome": "success",
        }

        with tempfile.TemporaryDirectory() as temporary:
            bundle_path = save_rollout_bundle(
                Path(temporary) / "rollout_001",
                [episode],
                policy_identity=identity,
                source_policy=_source_policy(),
                source_training_state=trainer.training_state_dict(),
                metadata=metadata,
            )
            # The persisted payload is deliberately compatible with the
            # restricted loader; no pickled project dataclass is required.
            raw = torch.load(
                bundle_path / ROLLOUT_PAYLOAD_NAME,
                map_location="cpu",
                weights_only=True,
            )
            self.assertIsInstance(raw, dict)
            loaded = load_rollout_bundle(
                bundle_path,
                expected_policy_identity=identity,
            )

            self.assertEqual(loaded.policy_identity, identity)
            self.assertEqual(loaded.metadata, metadata)
            self.assertEqual(len(loaded.episodes), 1)
            _assert_episode_exact(self, episode, loaded.episodes[0])
            with self.assertRaises(FileExistsError):
                save_rollout_bundle(
                    bundle_path,
                    [episode],
                    policy_identity=identity,
                    source_policy=_source_policy(),
                    source_training_state=trainer.training_state_dict(),
                )

    def test_bundle_rejects_policy_identity_and_payload_tampering(self):
        torch.manual_seed(107)
        trainer = _make_trainer()
        episode = collect_one_episode(trainer, _ThreeDecisionSource())
        identity = trainer.rollout_policy_identity()

        with tempfile.TemporaryDirectory() as temporary:
            bundle_path = save_rollout_bundle(
                Path(temporary) / "rollout_002",
                [episode],
                policy_identity=identity,
                source_policy=_source_policy(),
                source_training_state=trainer.training_state_dict(),
            )
            wrong_identity = copy.deepcopy(identity)
            wrong_identity["critic_sha256"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(ValueError, "different actor or critic"):
                load_rollout_bundle(
                    bundle_path,
                    expected_policy_identity=wrong_identity,
                )

            with (bundle_path / SOURCE_TRAINER_STATE_NAME).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "source trainer failed"):
                load_rollout_bundle(bundle_path)

            payload_bundle = save_rollout_bundle(
                Path(temporary) / "rollout_002_payload",
                [episode],
                policy_identity=identity,
                source_policy=_source_policy(),
                source_training_state=trainer.training_state_dict(),
            )
            with (payload_bundle / ROLLOUT_PAYLOAD_NAME).open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_rollout_bundle(payload_bundle)

    def test_loaded_rollout_produces_bit_exact_update(self):
        torch.manual_seed(109)
        collector = _make_trainer()
        episode = collect_one_episode(collector, _ThreeDecisionSource())
        identity = collector.rollout_policy_identity()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = collector.save_checkpoint(root / "source_training_state")
            bundle_path = save_rollout_bundle(
                root / "rollout_003",
                [episode],
                policy_identity=identity,
                source_policy=_source_policy(),
                source_training_state=collector.training_state_dict(),
            )
            loaded = load_rollout_bundle(
                bundle_path,
                expected_policy_identity=identity,
            )
            self.assertTrue(loaded.source_trainer_checkpoint.is_file())

            direct = _make_trainer()
            direct.load_checkpoint(checkpoint)
            direct_metrics = direct.update([episode])

            restored = _make_trainer()
            wrong_source_policy = _source_policy()
            wrong_source_policy["frozen_policy_sha256"] = "sha256:" + "0" * 64
            with self.assertRaisesRegex(RuntimeError, "source policy"):
                update_rollout_bundle(
                    restored,
                    loaded,
                    expected_source_policy=wrong_source_policy,
                )
            restored_metrics = update_rollout_bundle(
                restored,
                loaded,
                expected_source_policy=_source_policy(),
            )

            self.assertEqual(restored_metrics.as_dict(), direct_metrics.as_dict())
            self.assertEqual(restored.update_step, 1)
            _assert_tree_exact(
                self,
                direct._actor_module().state_dict(),
                restored._actor_module().state_dict(),
            )
            _assert_tree_exact(
                self,
                direct.value_head.state_dict(),
                restored.value_head.state_dict(),
            )
            _assert_tree_exact(
                self,
                direct.actor_optimizer.state_dict(),
                restored.actor_optimizer.state_dict(),
            )
            _assert_tree_exact(
                self,
                direct.value_optimizer.state_dict(),
                restored.value_optimizer.state_dict(),
            )
            result_checkpoint = restored.save_checkpoint(root / "result_training_state")
            receipt = mark_rollout_bundle_consumed(
                loaded,
                result_policy_identity=restored.rollout_policy_identity(),
                metrics=restored_metrics.as_dict(),
                trainer_checkpoint=result_checkpoint,
            )
            self.assertTrue(receipt.is_file())
            consumed = load_rollout_bundle(bundle_path)
            self.assertIsNotNone(consumed.consumption_receipt)
            with self.assertRaisesRegex(RuntimeError, "already consumed"):
                update_rollout_bundle(
                    _make_trainer(),
                    consumed,
                    expected_source_policy=_source_policy(),
                )


if __name__ == "__main__":
    unittest.main()
