from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from cyclo_brain.algorithm.rl.rlt import RLTAsyncLearner
from cyclo_brain.algorithm.rl.rlt.policy_update import RLTPolicyUpdate
from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run, _batch, _assert_tree_equal


class RLTPolicyUpdateTests(unittest.TestCase):
    def test_auto_off_and_failed_copy_leave_live_policy_and_pending_version_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _run(Path(directory))
            actor = deepcopy(run.learner.actor)
            updates = RLTPolicyUpdate(actor, run.learner.spec)
            updates.publish(run.learner)
            updates.set_auto_apply(True)
            updates.set_auto_apply(False)
            self.assertIs(updates.apply_pending(actor)[0], actor)
            updates.request_apply()
            with mock.patch('cyclo_brain.algorithm.rl.rlt.policy_update.deepcopy', side_effect=RuntimeError('copy failed')):
                with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                    updates.apply_pending(actor)
            self.assertEqual(updates.status()['inference_version'], 0)
            self.assertEqual(updates.status()['pending_version'], 1)

    def test_manual_apply_pins_snapshot_and_does_not_modify_training_state(self):
        for dim in (16, 19):
            with self.subTest(dim=dim), tempfile.TemporaryDirectory() as directory:
                run = _run(Path(directory), action_dim=dim)
                learner = run.learner
                actor = deepcopy(learner.actor).eval().requires_grad_(False)
                updates = RLTPolicyUpdate(actor, learner.spec)
                batch = _batch(run)
                with torch.no_grad():
                    old_chunk = actor(batch.z_rl, batch.proprio, batch.reference_actions).clone()
                learner.update(batch)
                learner.update(batch)
                expected_first = deepcopy(learner.actor.state_dict())
                updates.publish(learner)
                self.assertFalse(updates.status()['auto_apply'])
                unchanged, identity = updates.apply_pending(actor)
                self.assertIs(unchanged, actor)
                self.assertIsNone(identity)
                updates.request_apply()
                learner.update(batch)
                learner.update(batch)
                updates.publish(learner)
                training_state = learner.state_dict()
                applied, identity = updates.apply_pending(actor)
                self.assertIsNot(applied, actor)
                self.assertEqual(identity['async_policy_version'], 1)
                _assert_tree_equal(self, expected_first, applied.state_dict())
                self.assertTrue(updates.status()['can_apply'])
                self.assertTrue(all(not value.requires_grad for value in applied.parameters()))
                with torch.no_grad():
                    torch.testing.assert_close(old_chunk, actor(batch.z_rl, batch.proprio, batch.reference_actions))
                updates.set_auto_apply(True)
                latest, identity = updates.apply_pending(applied)
                self.assertEqual(identity['async_policy_version'], 2)
                _assert_tree_equal(self, learner.actor.state_dict(), latest.state_dict())
                _assert_tree_equal(self, training_state, learner.state_dict())
                self.assertFalse(updates.status()['can_apply'])

    def test_rejects_partial_incompatible_and_nonfinite_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _run(Path(directory))
            learner = run.learner
            updates = RLTPolicyUpdate(learner.actor, learner.spec)
            with self.assertRaisesRegex(RuntimeError, 'No new'):
                updates.request_apply()
            learner.update_critic(_batch(run))
            with self.assertRaisesRegex(RuntimeError, 'Finish'):
                updates.publish(learner)
            learner.finish_update()
            other = deepcopy(learner)
            other.spec = replace(other.spec, action_codec_id='different')
            with self.assertRaisesRegex(ValueError, 'spec differs'):
                updates.publish(other)
            with torch.no_grad():
                next(learner.actor.parameters()).fill_(float('nan'))
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                updates.publish(learner)
            self.assertEqual(updates.status()['training_version'], 0)

    def test_async_update_callback_publishes_only_completed_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _run(Path(directory))
            learner = run.learner
            updates = RLTPolicyUpdate(learner.actor, learner.spec)

            def completed(update):
                if update.actor_updated:
                    updates.publish(learner)
                    worker.set_enabled(False)

            worker = RLTAsyncLearner(learner, lambda: _batch(run), after_update=completed)
            try:
                worker.set_enabled(True)
                with worker._condition:
                    self.assertTrue(worker._condition.wait_for(
                        lambda: not worker._enabled and not worker._busy, timeout=5,
                    ))
                self.assertIsNone(worker.status()['error'])
                self.assertEqual(updates.status()['training_version'], 1)
                self.assertEqual(updates.status()['inference_version'], 0)
                self.assertEqual(learner.completed_critic_updates, 2)
            finally:
                worker.close()


if __name__ == '__main__':
    unittest.main()
