"""Behavioral checks for the immutable warmup-policy anchor."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from cyclo_brain.algorithm.rl.rlt import RLTStage2Run
from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run, _batch, _assert_tree_equal


class FrozenActorTests(unittest.TestCase):
    def test_failure_target_is_frozen_actor_not_failed_action_or_vla(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp), action_dim=16)
            learner = run.learner
            learner.config = replace(learner.config, critic_updates_per_actor=1)
            with torch.no_grad():
                for module in (learner.actor, learner.critic, learner.critic_target):
                    for parameter in module.parameters():
                        parameter.zero_()
                learner.actor.network[-1].bias.fill_(1)
            learner.completed_phases.append('critic_warmup')
            learner.begin_phase('rl')
            with torch.no_grad():
                learner.actor.network[-1].bias.fill_(3)
            batch = _batch(run)
            labels = torch.full_like(batch.executed_actions, 2)
            labels[1] = 99
            batch = replace(batch, executed_actions=labels,
                            reference_actions=torch.full_like(labels, -50),
                            bootstrap_discount=torch.zeros_like(batch.bootstrap_discount),
                            reward=torch.zeros_like(batch.reward),
                            episode_success=torch.tensor([True, False, True]))
            update = learner.update_phase(batch)
            # Two human-target rows have unit squared error, one anchor row 4.
            self.assertAlmostEqual(update.actor_reference_constraint, 320, places=5)
            self.assertGreater(update.policy_deviation_max, 1.9)
            self.assertLessEqual(update.policy_deviation_max, 2)
            self.assertEqual(update.policy_q_abs_max, 0)
            self.assertEqual(update.target_abs_max, 0)
            self.assertTrue(torch.all(learner.actor.network[-1].bias < 3))
            self.assertFalse(learner.actor_anchor.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None
                                for p in learner.actor_anchor.parameters()))
            torch.testing.assert_close(learner.actor_anchor.network[-1].bias,
                                       torch.ones(160), rtol=0, atol=0)

    def test_anchor_uses_full_reference_when_trainable_input_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp), action_dim=16)
            l = run.learner
            l.config = replace(l.config, critic_updates_per_actor=1,
                               reference_dropout_probability=.999999)
            with torch.no_grad():
                for module in (l.actor, l.critic, l.critic_target):
                    for p in module.parameters():
                        p.zero_()
                # Output[0] = relu(reference[0]); all remaining outputs = 0.
                l.actor.network[0].weight[0, 24] = 1
                l.actor.network[2].weight[0, 0] = 1
                l.actor.network[4].weight[0, 0] = 1
            l.completed_phases.append('critic_warmup')
            l.begin_phase('rl')
            with torch.no_grad():
                l.actor.network[-1].bias.fill_(.5)
            b = _batch(run)
            b = replace(b, reference_actions=torch.full_like(b.reference_actions, 2),
                        episode_success=torch.zeros(3, dtype=torch.bool),
                        reward=torch.zeros_like(b.reward),
                        bootstrap_discount=torch.zeros_like(b.bootstrap_discount))
            u = l.update_phase(b)
            self.assertAlmostEqual(u.actor_reference_constraint, 42, places=5)

    def test_anchor_survives_updates_reentry_and_exact_bundle_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _run(root, action_dim=16)
            l = run.learner
            b = replace(_batch(run), episode_success=torch.tensor([True, False, True]))
            l.begin_phase('actor_warmup'); l.update_phase(b); l.complete_phase()
            l.begin_phase('critic_warmup'); l.update_phase(b); l.complete_phase()
            # Old warmup checkpoints have no anchor and are still valid inputs.
            old_warmup = deepcopy(l.state_dict())
            old_warmup.pop('actor_anchor', None)
            l.load_state_dict(old_warmup)
            warmup_actor = deepcopy(l.actor.state_dict())
            l.begin_phase('rl')
            for _ in range(4):
                l.update_phase(b)
            l.begin_phase('rl')
            _assert_tree_equal(self, warmup_actor, l.actor_anchor.state_dict())
            self.assertTrue(any(not torch.equal(v, l.actor.state_dict()[k])
                                for k, v in warmup_actor.items()))
            bundle = run.save(root / 'rl')
            restored = RLTStage2Run.resume(bundle, expected_replay_root=run.replay_root).learner
            restored.begin_phase('rl')
            for _ in range(4):
                self.assertEqual(l.update_phase(b), restored.update_phase(b))
            _assert_tree_equal(self, l.state_dict(), restored.state_dict())
            _assert_tree_equal(self, warmup_actor, restored.actor_anchor.state_dict())
            # Never adopt a trained/divergent policy as a missing warmup anchor.
            old_rl = deepcopy(l.state_dict())
            old_rl.pop('actor_anchor')
            restored.load_state_dict(old_rl)
            with self.assertRaisesRegex(ValueError, 'Critic warmup'):
                restored.begin_phase('rl')
            with self.assertRaisesRegex(ValueError, 'Critic warmup'):
                restored.update_phase(b)

    def test_corrupt_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            l = _run(Path(tmp)).learner
            l.completed_phases.append('critic_warmup')
            l.completed_actor_warmup_updates = l.completed_actor_updates = 1
            l.completed_critic_warmup_updates = l.completed_critic_updates = 1
            l.begin_phase('rl')
            state = l.state_dict()
            state['actor_anchor']['network.0.weight'].fill_(float('nan'))
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                l.load_state_dict(state)

    def test_diagnostics_measure_policy_after_optimizer_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp), action_dim=16)
            l = run.learner
            l.config = replace(l.config, critic_updates_per_actor=1)
            with torch.no_grad():
                for module in (l.actor, l.critic, l.critic_target):
                    for p in module.parameters():
                        p.zero_()
            l.actor_optimizer.param_groups[0]['lr'] = 100
            l.completed_phases.append('critic_warmup')
            l.begin_phase('rl')
            b = _batch(run)
            b = replace(b, executed_actions=torch.ones_like(b.executed_actions),
                        episode_success=torch.ones(3, dtype=torch.bool),
                        reward=torch.zeros_like(b.reward),
                        bootstrap_discount=torch.zeros_like(b.bootstrap_discount))
            u = l.update_phase(b)
            # Error before the step is 1, but the large real Adam step overshoots.
            self.assertGreater(u.policy_deviation_max, 90)

    def test_split_update_requires_success_provenance_before_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _run(Path(tmp))
            l = run.learner
            l.completed_phases.append('critic_warmup')
            l.begin_phase('rl')
            before = deepcopy(l.state_dict())
            with self.assertRaisesRegex(ValueError, 'episode_success'):
                l.update_critic(_batch(run))
            _assert_tree_equal(self, before, l.state_dict())
