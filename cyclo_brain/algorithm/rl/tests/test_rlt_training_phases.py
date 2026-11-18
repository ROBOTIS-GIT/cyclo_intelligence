"""Exercise phase boundaries with real optimizers and round-trip bundles."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from cyclo_brain.algorithm.rl.rlt import RLTStage2Run
from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run, _batch, _assert_tree_equal


class RLTPhaseTests(unittest.TestCase):
    def test_actor_warmup_imitates_recorded_executed_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp), action_dim=16)
            learner = run.learner
            original_batch = _batch(run)
            batch = replace(
                original_batch,
                reference_actions=torch.zeros_like(original_batch.reference_actions),
                executed_actions=torch.ones_like(original_batch.executed_actions),
            )
            with torch.no_grad():
                for parameter in learner.actor.parameters():
                    parameter.zero_()

            learner.begin_phase("actor_warmup")
            update = learner.update_phase(batch)

            # A VLA-reference target would produce zero loss here. The unit
            # loss proves warm-up uses the recorded behavior chunk instead.
            self.assertAlmostEqual(update.actor_loss, 1.0, places=6)

    def test_three_phases_weights_optimizers_counters_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root, action_dim=16)
            learner = run.learner
            batch = _batch(run)
            with self.assertRaisesRegex(ValueError, "Critic warmup"):
                learner.begin_phase("rl")
            learner.begin_phase("actor_warmup")
            actor_warmup_data = {
                "scope": "success_only",
                "replay_transition_count": 3,
                "eligible_transition_count": 2,
                "eligible_indices_sha256": "a" * 64,
            }
            learner.configure_actor_warmup_data(actor_warmup_data)
            before = deepcopy(learner.state_dict())
            learner.update_phase(batch)
            self.assertTrue(any(not torch.equal(v, learner.actor.state_dict()[k]) for k, v in before['actor'].items()))
            for key in ('critic', 'critic_target', 'critic_optimizer', 'target_generator'):
                _assert_tree_equal(self, before[key], learner.state_dict()[key])
            with self.assertRaisesRegex(ValueError, "Actor warmup"):
                learner.begin_phase('critic_warmup')
            learner.complete_phase()
            bundle = run.save(root / 'actor')
            resumed = RLTStage2Run.resume(bundle, expected_replay_root=root / "replay_cache")
            _assert_tree_equal(self, learner.state_dict(), resumed.learner.state_dict())
            self.assertEqual(
                resumed.learner.state_dict()["phase_state"]["actor_warmup_data"],
                actor_warmup_data,
            )
            learner = resumed.learner
            learner.begin_phase('critic_warmup')
            before = deepcopy(learner.state_dict())
            for _ in range(3):
                update = learner.update_phase(batch)
                self.assertFalse(update.actor_updated)
            for key in ('actor', 'actor_optimizer', 'actor_generator', 'dropout_generator'):
                _assert_tree_equal(self, before[key], learner.state_dict()[key])
            self.assertTrue(any(not torch.equal(v, learner.critic.state_dict()[k]) for k, v in before['critic'].items()))
            learner.complete_phase()
            bundle = resumed.save(root / 'critic')
            restored = RLTStage2Run.resume(bundle, expected_replay_root=root / "replay_cache").learner
            _assert_tree_equal(self, learner.state_dict(), restored.state_dict())
            learner.begin_phase('rl')
            restored.begin_phase('rl')
            batch = replace(
                batch,
                episode_success=torch.tensor([True, False, True]),
            )
            for i in range(4):
                update = learner.update_phase(batch)
                self.assertEqual(update, restored.update_phase(batch))
                self.assertEqual(update.actor_updated, i % 2 == 1)
            self.assertEqual(learner.completed_actor_updates, 3)
            self.assertEqual(learner.completed_critic_updates, 7)
            _assert_tree_equal(self, learner.state_dict(), restored.state_dict())
            self.assertEqual(learner.actor(batch.z_rl, batch.proprio, batch.reference_actions).shape, (3, 10, 16))

    def test_rl_success_rows_keep_executed_action_as_actor_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp), action_dim=16)
            learner = run.learner
            learner.completed_phases.append("critic_warmup")
            learner.config = replace(learner.config, critic_updates_per_actor=1)
            batch = _batch(run)
            batch = replace(
                batch,
                reference_actions=torch.zeros_like(batch.reference_actions),
                executed_actions=torch.ones_like(batch.executed_actions),
                reward=torch.zeros_like(batch.reward),
                bootstrap_discount=torch.zeros_like(batch.bootstrap_discount),
                episode_success=torch.tensor([True, False, True]),
            )
            with torch.no_grad():
                for module in (learner.actor, learner.critic, learner.critic_target):
                    for parameter in module.parameters():
                        parameter.zero_()
                learner.actor.fixed_standard_deviation.zero_()
            learner.begin_phase("rl")

            update = learner.update_phase(batch)

            # Two successful rows contribute 10*16 unit errors; the failure
            # row targets its frozen zero Actor and contributes no error.
            self.assertTrue(update.actor_updated)
            self.assertAlmostEqual(
                update.actor_reference_constraint,
                2 * 10 * 16 / 3,
                places=5,
            )

    def test_explicit_rl_rejects_a_batch_without_episode_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp), action_dim=16)
            learner = run.learner
            learner.completed_phases.append("critic_warmup")
            learner.begin_phase("rl")

            with self.assertRaisesRegex(ValueError, "episode_success provenance"):
                learner.update_phase(_batch(run))

    def test_legacy_state_can_resume_and_invalid_phase_counters_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            batch = _batch(run)
            for _ in range(2):
                run.learner.update(batch)
            state = deepcopy(run.learner.state_dict())
            state.pop('phase_state')
            run.learner.load_state_dict(state)
            with self.assertRaisesRegex(ValueError, 'Critic warmup'):
                run.learner.begin_phase('rl')
            with self.assertRaisesRegex(ValueError, 'already trained'):
                run.learner.begin_phase('actor_warmup')
            state = deepcopy(run.learner.state_dict())
            state['phase_state']['critic_warmup_updates'] = 100
            with self.assertRaisesRegex(ValueError, 'counters'):
                run.learner.load_state_dict(state)

