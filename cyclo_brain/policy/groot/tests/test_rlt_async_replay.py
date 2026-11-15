"""Replay selection races are tested without GR00T or any robot connection."""
import sys
import threading
import unittest
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.rlt_async_replay import RLTAsyncReplay, RLTAsyncReplayBuilder


class ReplaySelectionTests(unittest.TestCase):
    def test_builder_reuses_unchanged_datasets_and_preserves_random_state(self):
        from cyclo_brain.policy.groot.tests.test_rlt_stage2_dataset import (
            RLTStage2DatasetTests, _FakeExtractor, _spec,
        )
        from runtime import rlt_stage2_dataset as dataset
        fixtures = RLTStage2DatasetTests()
        with tempfile.TemporaryDirectory() as directory:
            sources = {}
            for name in ('a', 'b'):
                root = Path(directory) / name
                fixtures._make_dataset(root)
                sources[str(root)] = dataset.RLTStage2LeRobotV21Source(
                    root, parquet_reader=fixtures._parquet_reader, video_reader=fixtures._video_reader)
            extractor = _FakeExtractor()
            extract = extractor.extract
            def noisy(observation):
                torch.rand(3)
                return extract(observation)
            extractor.extract = noisy
            adapter = SimpleNamespace(spec=_spec(), inference_slot=lambda **_: nullcontext(),
                policy=SimpleNamespace(model=SimpleNamespace(device='cpu')),
                shadow_policy=SimpleNamespace(encoder=None))
            builder = RLTAsyncReplayBuilder(adapter)
            a, b = sources
            rng = torch.get_rng_state().clone()
            try:
                with patch.object(dataset, 'open_rlt_stage2_source', side_effect=lambda path, **_: sources[path]), \
                     patch.object(dataset, 'GR00TRLTStage2Extractor', return_value=extractor):
                    first = builder([a], lambda: None)
                    count = extractor.extracted_samples
                    builder.restored = first
                    self.assertIs(builder([a], lambda: None), first)
                    self.assertEqual(extractor.extracted_samples, count)
                    both = builder([a, b], lambda: None)
                    self.assertEqual(extractor.extracted_samples, count * 2)
                    self.assertEqual(len(both), len(first) * 2)
                    builder([b], lambda: None)
                    self.assertEqual(extractor.extracted_samples, count * 2)
                    # Historical MCAP caches have no LeRobot dataset_roots.
                    builder.restored = SimpleNamespace(metadata={})
                    self.assertIsNot(builder([a], lambda: None), builder.restored)
                    self.assertEqual(extractor.extracted_samples, count * 2)
                torch.testing.assert_close(torch.get_rng_state(), rng)
            finally:
                builder.close()

    def test_stale_preparation_never_replaces_empty_selection(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        def build(paths, check):
            started.set()
            release.wait(3)
            try:
                check()
            finally:
                finished.set()
            self.fail('stale preparation should be cancelled')
        worker = Mock()
        selection = RLTAsyncReplay(worker, build, 32, 0)
        try:
            selection.set_enabled(True)
            selection.select(['/first'])
            self.assertTrue(started.wait(2))
            selection.select([])
            release.set()
            self.assertTrue(finished.wait(2))
        finally:
            release.set()
            selection.close()
        worker.replace_replay.assert_called_once_with(None, {'paths': [], 'transitions': 0, 'episodes': 0})
        self.assertIsNone(selection.status()['replay_error'])

    def test_failed_new_selection_preserves_existing_replay_and_can_recover(self):
        worker = Mock()
        installed = threading.Event()
        worker.replace_replay.side_effect = lambda *_, **__: installed.set()
        class Replay:
            metadata = {'usable_episode_count': 2}
            def __len__(self): return 4
        def build(paths, check):
            check()
            if paths == ('/bad',): raise ValueError('labels missing')
            return Replay()
        selection = RLTAsyncReplay(worker, build, 2, 0)
        try:
            selection.set_enabled(True)
            selection.select(['/bad'])
            # Use the selection condition, rather than waiting for GPU work.
            with selection._condition:
                self.assertTrue(selection._condition.wait_for(lambda: selection._error is not None, timeout=2))
            worker.replace_replay.assert_not_called()
            selection.select(['/good'])
            self.assertTrue(installed.wait(2))
            self.assertIsNone(selection.status()['replay_error'])
        finally:
            selection.close()

if __name__ == '__main__':
    unittest.main()
