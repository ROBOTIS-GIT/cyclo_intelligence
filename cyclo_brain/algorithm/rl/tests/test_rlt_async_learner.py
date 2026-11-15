"""CPU-only checks for cooperative RLT scheduling; no robot is connected."""

from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

from cyclo_brain.algorithm.rl.rlt import RLTAsyncLearner
from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import (
    _assert_tree_equal, _batch, _run,
)


class RLTAsyncLearnerTests(unittest.TestCase):
    def test_budget_pauses_after_complete_updates_and_resume_keeps_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _run(Path(directory))
            learner, batch = run.learner, _batch(run)
            expected = deepcopy(learner)
            worker = RLTAsyncLearner(learner, lambda: batch)
            self.addCleanup(worker.close)
            for limit in (1, 2, 3):
                with worker.inference():
                    worker.set_enabled(True, max_updates=limit)
                    # A duplicate ON must not reset the running budget.
                    worker.set_enabled(True, max_updates=99)
                self.wait_for(worker, lambda: not worker._enabled and not worker._busy)
                for _ in range(limit):
                    expected.update(batch)
                status = worker.status()
                self.assertTrue(status['limit_reached'])
                self.assertEqual(status['updates_this_run'], limit)
                self.assertEqual(status['max_updates'], limit)
                self.assertFalse(learner.update_pending)
                _assert_tree_equal(self, expected.state_dict(), learner.state_dict())

    def test_budget_counts_paused_partial_update_without_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _run(Path(directory))
            learner, batch = run.learner, _batch(run)
            learner.update_critic(batch)
            sample = Mock(return_value=batch)
            worker = RLTAsyncLearner(learner, sample)
            self.addCleanup(worker.close)
            worker.set_enabled(True, max_updates=1)
            self.wait_for(worker, lambda: not worker._enabled and not worker._busy)
            self.assertTrue(worker.status()['limit_reached'])
            self.assertFalse(learner.update_pending)
            sample.assert_not_called()

    def test_invalid_budget_never_starts_training(self):
        worker = RLTAsyncLearner(Mock(), Mock())
        self.addCleanup(worker.close)
        for value in (0, -1, True, 1.5, '10'):
            with self.assertRaises(ValueError):
                worker.set_enabled(True, max_updates=value)
        self.assertFalse(worker.status()['enabled'])

    def test_inference_then_background_before_enabled_training(self):
        learner = Mock(update_pending=False)
        order = []
        sample = Mock()
        worker = RLTAsyncLearner(learner, sample)
        self.addCleanup(worker.close)

        def critic(_):
            order.append('training')
            worker.set_enabled(False)
        learner.update_critic.side_effect = critic

        def enter(name, background):
            with worker.inference(background=background):
                order.append(name)

        threads = [threading.Thread(target=enter, args=(name, background))
                   for name, background in [('save', True), ('inference', False)]]
        with worker.inference():
            worker.set_enabled(True)
            for thread in threads:
                thread.start()
            self.wait_for(worker, lambda: worker._waiting_background == 1
                          and worker._waiting_inference == 1)
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.wait_for(worker, lambda: not worker._enabled and not worker._busy)
        self.assertEqual(order, ['inference', 'save', 'training'])
        self.assertEqual(worker._waiting_background, 0)
        sample.assert_called_once()

    def test_background_exception_releases_slot_for_training(self):
        learner = Mock(update_pending=False)
        worker = RLTAsyncLearner(learner, Mock())
        self.addCleanup(worker.close)
        learner.update_critic.side_effect = lambda _: worker.set_enabled(False)
        with self.assertRaisesRegex(ValueError, 'save failed'):
            with worker.inference(background=True):
                worker.set_enabled(True)
                raise ValueError('save failed')
        self.wait_for(worker, lambda: not worker._enabled and not worker._busy)
        self.assertEqual(worker._waiting_background, 0)
        learner.update_critic.assert_called_once()

    def test_inference_waiter_has_priority_over_feature_preparation(self):
        worker = RLTAsyncLearner(Mock(update_pending=False), None)
        self.addCleanup(worker.close)
        order = []
        def enter(name, background):
            with worker.inference(background=background):
                order.append(name)
        with worker.inference():
            preparation = threading.Thread(target=enter, args=('prepare', True))
            inference = threading.Thread(target=enter, args=('inference', False))
            preparation.start()
            inference.start()
            self.wait_for(worker, lambda: worker._waiting_inference == 1)
        inference.join(2)
        preparation.join(2)
        self.assertEqual(order, ['inference', 'prepare'])

    def test_replay_swap_finishes_pending_update_and_preserves_learner(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _run(Path(directory))
            learner = run.learner
            old_batch = _batch(run)
            learner.update_critic(old_batch)
            expected = deepcopy(learner)
            expected.finish_update()
            old_sampler = Mock(return_value=old_batch)
            worker = RLTAsyncLearner(learner, old_sampler)
            self.addCleanup(worker.close)
            worker.replace_replay(None, {'paths': [], 'transitions': 0})
            worker.set_enabled(True)
            self.wait_for(worker, lambda: not learner.update_pending and worker._sample_batch is None and not worker._busy)
            self.assertTrue(worker.status()['enabled'])
            self.assertTrue(worker.status()['waiting_data'])
            old_sampler.assert_not_called()
            _assert_tree_equal(self, learner.state_dict(), expected.state_dict())
            new_sampler = Mock(return_value=old_batch)
            worker._after_update = lambda *_: worker.set_enabled(False)
            worker.replace_replay(new_sampler, {'paths': ['/new'], 'transitions': 3})
            self.wait_for(worker, lambda: not worker._enabled and not worker._busy)
            self.assertIs(worker.learner, learner)
            new_sampler.assert_called_once()
            self.assertEqual(worker.status()['replay']['paths'], ['/new'])

    def wait_for(self, worker, predicate):
        with worker._condition:
            self.assertTrue(worker._condition.wait_for(predicate, timeout=5))

    def test_default_off_does_not_sample_or_start_thread(self):
        learner = Mock(update_pending=False)
        sample = Mock()
        worker = RLTAsyncLearner(learner, sample)
        self.addCleanup(worker.close)
        self.assertIsNone(worker._thread)
        self.assertFalse(worker.status()["enabled"])
        with worker.inference():
            pass
        worker.set_enabled(False)
        sample.assert_not_called()
        learner.update_critic.assert_not_called()

    def test_waiting_inference_runs_before_next_training_phase(self):
        critic_started = threading.Event()
        release_critic = threading.Event()
        inference_started = threading.Event()
        release_inference = threading.Event()
        order = []
        learner = Mock(update_pending=False)

        def critic(_):
            order.append("critic")
            critic_started.set()
            if not release_critic.wait(5):
                raise RuntimeError("test did not release critic")
            learner.update_pending = True

        def actor():
            order.append("actor")
            worker.set_enabled(False)
            learner.update_pending = False

        learner.update_critic.side_effect = critic
        learner.finish_update.side_effect = actor
        worker = RLTAsyncLearner(learner, Mock())
        self.addCleanup(worker.close)
        self.addCleanup(release_inference.set)
        self.addCleanup(release_critic.set)
        worker.set_enabled(True)
        self.assertTrue(critic_started.wait(5))

        def infer():
            with worker.inference():
                order.append("inference")
                inference_started.set()
                release_inference.wait(5)

        threads = [threading.Thread(target=infer) for _ in range(2)]
        for thread in threads:
            thread.start()
        self.wait_for(worker, lambda: worker._waiting_inference == 2)
        release_critic.set()
        self.assertTrue(inference_started.wait(5))
        self.assertEqual(order, ["critic", "inference"])
        release_inference.set()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.wait_for(worker, lambda: not worker._enabled and not worker._busy)
        self.assertEqual(order, ["critic", "inference", "inference", "actor"])

    def test_pause_and_resume_preserve_batch_optimizer_and_rng(self):
        for action_dim in (16, 19):
            with self.subTest(action_dim=action_dim), tempfile.TemporaryDirectory() as directory:
                run = _run(Path(directory), action_dim=action_dim)
                learner = run.learner
                # Make the next update include an actor step.
                batch = _batch(run)
                learner.update(batch)
                serial = deepcopy(learner)
                expected = serial.update(batch)
                sample = Mock(return_value=batch)
                worker = RLTAsyncLearner(learner, sample)
                original_critic = learner.update_critic
                original_finish = learner.finish_update

                def critic(value):
                    original_critic(value)
                    worker.set_enabled(False)

                def finish():
                    result = original_finish()
                    worker.set_enabled(False)
                    return result

                learner.update_critic = critic
                learner.finish_update = finish
                try:
                    worker.set_enabled(True)
                    self.wait_for(worker, lambda: not worker._enabled and not worker._busy)
                    self.assertTrue(learner.update_pending)
                    self.assertIsNone(worker.status()["last_update"])
                    # An unrelated inference failure must also release its slot.
                    with self.assertRaisesRegex(ValueError, "inference test"):
                        with worker.inference():
                            raise ValueError("inference test")
                    worker.set_enabled(True)
                    self.wait_for(worker, lambda: not worker._enabled and not worker._busy)
                    self.assertFalse(learner.update_pending)
                    sample.assert_called_once()
                    self.assertEqual(worker.status()["last_update"]["actor_loss"], expected.actor_loss)
                    _assert_tree_equal(self, learner.state_dict(), serial.state_dict())
                finally:
                    worker.close()

    def test_failed_training_does_not_block_inference(self):
        learner = Mock(update_pending=False)
        sample = Mock(side_effect=ValueError("replay unavailable"))
        worker = RLTAsyncLearner(learner, sample)
        self.addCleanup(worker.close)
        worker.set_enabled(True)
        self.wait_for(worker, lambda: worker._error is not None)
        self.assertFalse(worker.status()["enabled"])
        with worker.inference():
            pass
        with self.assertRaisesRegex(RuntimeError, "replay unavailable"):
            worker.set_enabled(True)
        learner.update_critic.assert_not_called()

    def test_close_while_inference_owns_slot_stops_without_training(self):
        learner = Mock(update_pending=False)
        sample = Mock()
        worker = RLTAsyncLearner(learner, sample)
        with worker.inference():
            worker.set_enabled(True)
            worker.close()
        self.assertFalse(worker._thread.is_alive())
        sample.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.set_enabled(True)
        # Closing the learner is not an inference stop command.
        with worker.inference():
            pass


if __name__ == "__main__":
    unittest.main()
