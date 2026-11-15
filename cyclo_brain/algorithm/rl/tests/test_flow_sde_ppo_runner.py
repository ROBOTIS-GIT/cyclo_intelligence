"""One-episode update and checkpoint tests for the Flow-SDE PPO runner."""

from __future__ import annotations

import copy
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from cyclo_brain.algorithm.rl.flow_sde_ppo import (
    FlowSDEEpisode,
    FlowSDEOnPolicyBuffer,
    FlowSDEPPOConfig,
    FlowSDEPPOTrainer,
    FlowSDETransition,
    collect_one_episode_and_update,
)
from cyclo_brain.model.multi_task_dit.flow_sde_adapter import (
    CYCLO_SG2_CAMERA_KEYS,
    MultiTaskDiTFlowAdapter,
)
from cyclo_brain.model.multi_task_dit.value_head import MultiTaskDiTValueHead


class _TinyObservationEncoder(nn.Module):
    conditioning_dim = 3

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def encode(self, _batch):
        raise AssertionError("the injected source supplies already-frozen conditioning")


class _TinyNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_scale = nn.Parameter(torch.tensor(0.15))
        self.conditioning = nn.Linear(3, 2)

    def forward(self, latent, progress, conditioning):
        bias = self.conditioning(conditioning)[:, None, :]
        return self.latent_scale * latent + bias + progress[:, None, None]


class _TinyMultiTaskPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        image_features = OrderedDict(
            (key, SimpleNamespace(shape=(3, 8, 8))) for key in CYCLO_SG2_CAMERA_KEYS
        )
        self.config = SimpleNamespace(
            objective="flow_matching",
            sigma_min=0.0,
            image_features=image_features,
            robot_state_feature=SimpleNamespace(shape=(2,)),
            action_feature=SimpleNamespace(shape=(2,)),
            horizon=2,
            n_obs_steps=1,
            n_action_steps=2,
        )
        self.observation_encoder = _TinyObservationEncoder()
        self.noise_predictor = _TinyNoisePredictor()

    @staticmethod
    def _prepare_batch(batch):
        return batch

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text("{}\n", encoding="utf-8")
        torch.save(self.state_dict(), output_dir / "model.safetensors")


class _TinyProcessor:
    def __init__(self, filename):
        self.filename = filename

    def save_pretrained(self, output_dir):
        (Path(output_dir) / self.filename).write_text("{}\n", encoding="utf-8")


class _ThreeDecisionSource:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(31)

    def collect_episode(self, runner):
        transitions = []
        rewards = (0.0, 0.25, 1.0)
        for index, reward in enumerate(rewards):
            conditioning = torch.tensor(
                [[0.2 + index * 0.1, -0.3, 0.5]],
                dtype=torch.float32,
            )
            decision = runner.sample_from_conditioning(
                conditioning,
                generator=self.generator,
                denoise_indices=torch.tensor([index % 2]),
            )
            transitions.append(
                decision.as_transition(
                    reward=reward,
                    terminated=index == len(rewards) - 1,
                    truncated=False,
                )
            )
        return FlowSDEEpisode(tuple(transitions))


def _make_trainer() -> FlowSDEPPOTrainer:
    policy = _TinyMultiTaskPolicy()
    adapter = MultiTaskDiTFlowAdapter(policy)
    value_head = MultiTaskDiTValueHead(adapter.conditioning_dim, hidden_dims=(8,))
    config = FlowSDEPPOConfig(
        ppo_epochs=2,
        minibatch_size=2,
        actor_learning_rate=1.0e-2,
        value_learning_rate=1.0e-2,
    )
    return FlowSDEPPOTrainer(adapter, value_head, config=config)


def _assert_tree_exact(test: unittest.TestCase, expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        test.assertIsInstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, dict):
        test.assertIsInstance(actual, dict)
        test.assertEqual(set(actual), set(expected))
        for key in expected:
            _assert_tree_exact(test, expected[key], actual[key])
        return
    if isinstance(expected, (list, tuple)):
        test.assertIsInstance(actual, type(expected))
        test.assertEqual(len(actual), len(expected))
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_tree_exact(test, expected_item, actual_item)
        return
    test.assertEqual(actual, expected)


class FlowSDEPPORunnerTest(unittest.TestCase):
    def test_one_injected_episode_updates_actor_and_value(self):
        torch.manual_seed(7)
        trainer = _make_trainer()
        actor_before = {
            name: value.detach().clone()
            for name, value in trainer._actor_module().state_dict().items()
        }
        value_before = {
            name: value.detach().clone()
            for name, value in trainer.value_head.state_dict().items()
        }

        episode, metrics = collect_one_episode_and_update(trainer, _ThreeDecisionSource())

        self.assertEqual(len(episode.transitions), 3)
        self.assertEqual(metrics.episodes, 1)
        self.assertEqual(metrics.transitions, 3)
        self.assertEqual(metrics.minibatches, 4)
        self.assertEqual(metrics.update_step, 1)
        self.assertAlmostEqual(metrics.episode_return_mean, 1.25, places=6)
        self.assertTrue(all(torch.isfinite(torch.tensor(list(metrics.as_dict().values())))))
        self.assertTrue(
            any(
                not torch.equal(actor_before[name], value)
                for name, value in trainer._actor_module().state_dict().items()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(value_before[name], value)
                for name, value in trainer.value_head.state_dict().items()
            )
        )

    def test_truncation_bootstraps_but_terminal_does_not(self):
        trainer = _make_trainer()
        source_episode = _ThreeDecisionSource().collect_episode(trainer)
        first_rollout = source_episode.transitions[0].rollout
        conditioning = source_episode.transitions[0].conditioning

        def transition(reward, *, terminated=False, truncated=False):
            return FlowSDETransition(
                conditioning=conditioning,
                rollout=first_rollout,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                old_value=0.0,
            )

        truncated_episode = FlowSDEEpisode(
            (
                transition(1.0),
                transition(2.0, truncated=True),
            ),
            bootstrap_value=4.0,
        )
        terminal_episode = FlowSDEEpisode(
            (
                transition(1.0),
                transition(2.0, terminated=True),
            ),
            bootstrap_value=4.0,
        )
        truncated_buffer = FlowSDEOnPolicyBuffer()
        truncated_buffer.add_episode(truncated_episode)
        terminal_buffer = FlowSDEOnPolicyBuffer()
        terminal_buffer.add_episode(terminal_episode)

        truncated_batch = truncated_buffer.build_batch(discount=1.0, gae_lambda=1.0)
        terminal_batch = terminal_buffer.build_batch(discount=1.0, gae_lambda=1.0)

        torch.testing.assert_close(truncated_batch.returns, torch.tensor([7.0, 6.0]))
        torch.testing.assert_close(terminal_batch.returns, torch.tensor([3.0, 2.0]))

    def test_training_checkpoint_and_actor_export_roundtrip(self):
        torch.manual_seed(11)
        trainer = _make_trainer()
        collect_one_episode_and_update(trainer, _ThreeDecisionSource())
        expected_actor = {
            name: value.detach().clone()
            for name, value in trainer._actor_module().state_dict().items()
        }
        expected_value = {
            name: value.detach().clone()
            for name, value in trainer.value_head.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = trainer.save_checkpoint(root / "checkpoint")
            export_path = trainer.export_actor(root / "export")
            self.assertEqual(checkpoint_path.name, "trainer_state.pt")
            self.assertEqual(export_path.name, "flow_sde_actor.pt")

            with torch.no_grad():
                for parameter in trainer.actor_parameters:
                    parameter.add_(10.0)
                for parameter in trainer.value_parameters:
                    parameter.sub_(10.0)
            trainer.update_step = 99
            restored_step = trainer.load_checkpoint(root / "checkpoint")

            self.assertEqual(restored_step, 1)
            self.assertTrue(trainer.actor_optimizer.state)
            self.assertTrue(trainer.value_optimizer.state)
            for name, value in trainer._actor_module().state_dict().items():
                torch.testing.assert_close(value, expected_actor[name], rtol=0.0, atol=0.0)
            for name, value in trainer.value_head.state_dict().items():
                torch.testing.assert_close(value, expected_value[name], rtol=0.0, atol=0.0)

            exported = torch.load(export_path, map_location="cpu", weights_only=True)
            self.assertEqual(exported["format"], trainer.EXPORT_FORMAT)
            self.assertNotIn("actor_optimizer", exported)
            self.assertEqual(exported["source_update_step"], 1)

            pretrained_dir = trainer.export_pretrained_policy(
                root / "pretrained_model",
                preprocessor=_TinyProcessor("policy_preprocessor.json"),
                postprocessor=_TinyProcessor("policy_postprocessor.json"),
            )
            self.assertTrue((pretrained_dir / "config.json").is_file())
            self.assertTrue((pretrained_dir / "model.safetensors").is_file())
            self.assertTrue((pretrained_dir / "flow_sde_ppo_export.json").is_file())
            with self.assertRaises(FileExistsError):
                trainer.export_pretrained_policy(
                    pretrained_dir,
                    preprocessor=_TinyProcessor("policy_preprocessor.json"),
                    postprocessor=_TinyProcessor("policy_postprocessor.json"),
                )

    def test_cross_job_resume_preserves_full_state_and_exact_next_update(self):
        torch.manual_seed(23)
        original = _make_trainer()
        provenance = {"format": "test.value.initialization.v1", "source": "warmup"}
        original.record_value_initialization_provenance(provenance)
        collect_one_episode_and_update(original, _ThreeDecisionSource())

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = original.save_checkpoint(Path(temporary) / "first-job")
            expected_actor = copy.deepcopy(original._actor_module().state_dict())
            expected_value = copy.deepcopy(original.value_head.state_dict())
            expected_actor_optimizer = copy.deepcopy(original.actor_optimizer.state_dict())
            expected_value_optimizer = copy.deepcopy(original.value_optimizer.state_dict())

            resumed = _make_trainer()
            self.assertEqual(resumed.load_checkpoint(checkpoint), 1)
            self.assertEqual(resumed.value_initialization_provenance, provenance)
            _assert_tree_exact(self, expected_actor, resumed._actor_module().state_dict())
            _assert_tree_exact(self, expected_value, resumed.value_head.state_dict())
            _assert_tree_exact(
                self,
                expected_actor_optimizer,
                resumed.actor_optimizer.state_dict(),
            )
            _assert_tree_exact(
                self,
                expected_value_optimizer,
                resumed.value_optimizer.state_dict(),
            )

            # Both branches consume the same on-policy rollout and the RNG state
            # serialized at the first job boundary. Their next PPO update must
            # therefore be bit-for-bit identical.
            continuation_episode = _ThreeDecisionSource().collect_episode(original)
            original.load_checkpoint(checkpoint)
            original_metrics = original.update([continuation_episode])
            resumed.load_checkpoint(checkpoint)
            resumed_metrics = resumed.update([continuation_episode])
            self.assertEqual(original_metrics.as_dict(), resumed_metrics.as_dict())
            self.assertEqual(original.update_step, 2)
            self.assertEqual(resumed.update_step, 2)
            _assert_tree_exact(
                self,
                original._actor_module().state_dict(),
                resumed._actor_module().state_dict(),
            )
            _assert_tree_exact(
                self,
                original.value_head.state_dict(),
                resumed.value_head.state_dict(),
            )
            _assert_tree_exact(
                self,
                original.actor_optimizer.state_dict(),
                resumed.actor_optimizer.state_dict(),
            )
            _assert_tree_exact(
                self,
                original.value_optimizer.state_dict(),
                resumed.value_optimizer.state_dict(),
            )

    def test_resume_rejects_optimizer_lr_mismatch_before_model_mutation(self):
        torch.manual_seed(37)
        trained = _make_trainer()
        collect_one_episode_and_update(trained, _ThreeDecisionSource())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = trained.save_checkpoint(root / "source")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["value_optimizer"]["param_groups"][0]["lr"] = 9.0e-4
            corrupt = root / "corrupt.pt"
            torch.save(payload, corrupt)

            target = _make_trainer()
            actor_before = copy.deepcopy(target._actor_module().state_dict())
            value_before = copy.deepcopy(target.value_head.state_dict())
            with self.assertRaisesRegex(ValueError, "learning rate"):
                target.load_checkpoint(corrupt, strict_config=True)
            _assert_tree_exact(self, actor_before, target._actor_module().state_dict())
            _assert_tree_exact(self, value_before, target.value_head.state_dict())


if __name__ == "__main__":
    unittest.main()
