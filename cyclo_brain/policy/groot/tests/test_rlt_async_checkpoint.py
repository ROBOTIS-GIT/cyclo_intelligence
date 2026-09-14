"""CPU bundle round trips, with no GR00T load or robot communication."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.rlt_async_checkpoint import RLTAsyncCheckpoint
from runtime.rlt_stage2_dataset import RLTStage2FeatureReplay
from cyclo_brain.algorithm.rl.rlt import RLTAsyncLearner, RLTStage2Run
from cyclo_brain.algorithm.rl.tests.test_rlt_stage2_core import _run, _batch, _assert_tree_equal


class AsyncCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(os.geteuid() == 0, 'Docker root ownership check')
    def test_root_writer_preserves_workspace_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            os.chown(parent, 1000, 1000)
            run, worker, *_ = self.setup_worker(parent / 'original')
            adapter = SimpleNamespace(bundle=SimpleNamespace(root=parent / 'original'),
                _async_run=run, _async_training=worker, inference_slot=worker.inference)
            saver = RLTAsyncCheckpoint(adapter)
            saver.start()
            saver.close()
            self.assertIsNone(saver.status()['save_error'])
            output = Path(saver.status()['saved_bundle_path'])
            for path in (output, *output.rglob('*')):
                self.assertEqual((path.stat().st_uid, path.stat().st_gid), (1000, 1000))

    def setup_worker(self, root):
        run = _run(root)
        batch = _batch(run)
        metadata = json.loads((run.replay_root / 'manifest.json').read_text())['metadata']
        metadata.update(dataset_roots=['/selected'], usable_episode_count=2,
                        reference_extraction={'seed': None, 'feature_batch_size': 1})
        replay = RLTStage2FeatureReplay(run.learner.spec,
            {name: getattr(batch, name) for name in RLTStage2FeatureReplay._TENSOR_NAMES}, metadata=metadata)
        generator = torch.Generator().manual_seed(91)
        def sample():
            return replay.batch(torch.randint(len(replay), (3,), generator=generator).tolist(), device='cpu')
        worker = RLTAsyncLearner(run.learner, None)
        self.addCleanup(worker.close)
        worker._after_update = lambda update: worker.set_enabled(False) if update.completed_critic_updates == 2 else None
        worker.replace_replay(sample, {'paths': ['/selected'], 'batch_size': 3},
                              checkpoint_context={'replay': replay, 'generator': generator, 'sampling_seed': 91})
        worker.set_enabled(True)
        with worker._condition:
            self.assertTrue(worker._condition.wait_for(lambda: not worker._enabled and not worker._busy, timeout=5))
        return run, worker, replay, generator, sample

    def test_save_completes_while_training_stays_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'original'
            run, worker, *_ = self.setup_worker(root)
            adapter = SimpleNamespace(bundle=SimpleNamespace(root=root),
                _async_run=run, _async_training=worker, inference_slot=worker.inference)
            saver = RLTAsyncCheckpoint(adapter)
            worker._after_update = None
            worker.set_enabled(True)
            try:
                saver.start()
                saver._thread.join(15)
                self.assertFalse(saver._thread.is_alive(), 'saving starved by training')
                self.assertTrue(worker.status()['enabled'])
                status = saver.status()
                self.assertFalse(status['saving'])
                self.assertIsNone(status['save_error'])
                restored = RLTStage2Run.resume(status['saved_bundle_path'])
                self.assertEqual(restored.learner.completed_critic_updates,
                                 status['saved_critic_updates'])
                self.assertEqual(restored.learner.completed_actor_updates,
                                 status['saved_actor_updates'])
                self.assertGreaterEqual(status['saved_critic_updates'], 2)
            finally:
                worker.set_enabled(False)
                saver.close()

    def test_saved_bundle_resumes_weights_optimizer_rng_and_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'original'
            run, worker, replay, generator, sample = self.setup_worker(root)
            original_files = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            adapter = SimpleNamespace(bundle=SimpleNamespace(root=root), _async_run=run,
                _async_training=worker, inference_slot=worker.inference)
            saver = RLTAsyncCheckpoint(adapter)
            saver.start()
            saver.close()
            status = saver.status()
            self.assertIsNone(status['save_error'])
            output = Path(status['saved_bundle_path'])
            self.assertEqual(output.parent, root.parent)
            self.assertNotEqual(output, root)
            restored = RLTStage2Run.resume(output)
            _assert_tree_equal(self, restored.learner.state_dict(), run.learner.state_dict())
            self.assertEqual(restored.async_state['history'], worker.replay_history)
            self.assertEqual(restored.training_round['reference_extraction']['seed'], None)
            self.assertEqual(restored.training_round['optimization']['sampling_seed'], 91)
            from runtime.rlt_adapter import GR00TRLTInferenceAdapter
            resumed_adapter = GR00TRLTInferenceAdapter(
                SimpleNamespace(model=SimpleNamespace(device='cpu')),
                SimpleNamespace(actor=deepcopy(restored.learner.actor), spec=restored.learner.spec),
                SimpleNamespace(root=output))
            resumed_adapter._groot_checkpoint_fingerprint = run.source.groot_checkpoint_fingerprint
            try:
                resumed_adapter.set_async_training(True)
                resumed_adapter._async_prepare.join(5)
                self.assertFalse(resumed_adapter._async_prepare.is_alive())
                self.assertIsNone(resumed_adapter.async_training_status()['training_error'])
                self.assertTrue(resumed_adapter.async_training_status()['waiting_data'])
                _assert_tree_equal(self, resumed_adapter._async_run.learner.state_dict(), run.learner.state_dict())
                torch.testing.assert_close(resumed_adapter._async_replay.generator.get_state(), generator.get_state())
                self.assertEqual(resumed_adapter._async_training.replay_history, worker.replay_history)
                self.assertEqual(len(resumed_adapter._async_replay_builder.restored), len(replay))
            finally:
                resumed_adapter.close_async_training(dispose=True)
            saver.start()
            saver.close()
            self.assertIsNone(saver.status()['save_error'])
            second_output = saver.status()['saved_bundle_path']
            self.assertNotEqual(second_output, str(output))
            replay2 = RLTStage2FeatureReplay.load(restored.replay_root)
            generator2 = torch.Generator()
            generator2.set_state(restored.async_state['sampling_generator'])
            batch1 = sample()
            batch2 = replay2.batch(torch.randint(len(replay2), (3,), generator=generator2).tolist(), device='cpu')
            run.learner.update_critic(batch1)
            run.learner.finish_update()
            restored.learner.update_critic(batch2)
            restored.learner.finish_update()
            _assert_tree_equal(self, restored.learner.state_dict(), run.learner.state_dict())
            self.assertEqual(original_files, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
            saver.start()
            saver.close()
            # The preceding manual update was outside the worker and is rightly
            # rejected because its replay history wasn't updated.
            self.assertIn('history disagrees', saver.status()['save_error'])
            self.assertEqual(saver.status()['saved_bundle_path'], second_output)
            self.assertEqual(list(root.parent.glob('.rlt-save-*')), [])

    def test_snapshot_finishes_pending_work_when_off_and_survives_empty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            run, worker, replay, generator, sample = self.setup_worker(Path(directory))
            run.learner.update_critic(sample())
            expected = deepcopy(run.learner)
            expected.finish_update()
            # Selection is pending; checkpoint still describes the trained data.
            worker.replace_replay(None, {'paths': []})
            snapshot = worker.checkpoint_snapshot()
            self.assertFalse(worker.status()['enabled'])
            self.assertFalse(run.learner.update_pending)
            self.assertIs(snapshot['replay'], replay)
            _assert_tree_equal(self, snapshot['learner'], expected.state_dict())
            self.assertEqual(snapshot['history'][-1]['ending_critic_updates'], 3)

    def test_background_writes_do_not_hold_inference_slot_or_read_live_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            run, worker, replay, generator, sample = self.setup_worker(Path(directory) / 'original')
            snapshot = worker.checkpoint_snapshot()
            adapter = SimpleNamespace(bundle=SimpleNamespace(root=Path(directory) / 'original'),
                _async_run=run, _async_training=worker, inference_slot=worker.inference)
            save_replay = replay.save
            def write(path):
                self.assertFalse(worker._busy)
                # A training update during disk I/O must not change the snapshot.
                run.learner.update_critic(sample())
                run.learner.finish_update()
                return save_replay(path)
            saver = RLTAsyncCheckpoint(adapter)
            with patch.object(worker, 'checkpoint_snapshot', return_value=snapshot), \
                 patch.object(replay, 'save', side_effect=write):
                saver.start()
                saver.close()
            self.assertIsNone(saver.status()['save_error'])
            restored = RLTStage2Run.resume(saver.status()['saved_bundle_path'])
            _assert_tree_equal(self, restored.learner.state_dict(), snapshot['learner'])
            self.assertEqual(restored.learner.completed_critic_updates, 2)
            self.assertEqual(run.learner.completed_critic_updates, 3)

    def test_no_updates_or_write_failure_never_publishes_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            run, worker, replay, generator, sample = self.setup_worker(Path(directory) / 'original')
            adapter = SimpleNamespace(bundle=SimpleNamespace(root=Path(directory) / 'original'),
                _async_run=run, _async_training=worker, inference_slot=worker.inference)
            saver = RLTAsyncCheckpoint(adapter)
            with patch.object(replay, 'save', side_effect=OSError('disk full')):
                saver.start()
                saver.close()
            self.assertIn('disk full', saver.status()['save_error'])
            self.assertIsNone(saver.status()['saved_bundle_path'])
            self.assertEqual(list(Path(directory).glob('async_rlt_*')), [])
            empty = RLTAsyncLearner(run.learner, None)
            self.addCleanup(empty.close)
            with self.assertRaisesRegex(RuntimeError, 'No Async RL updates'):
                empty.checkpoint_snapshot()


if __name__ == '__main__':
    unittest.main()
