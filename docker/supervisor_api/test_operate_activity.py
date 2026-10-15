"""Operate journal stores bounded metadata without camera frames."""
import tempfile
import unittest
from pathlib import Path

from operate_activity import OperateActivity


class OperateActivityTests(unittest.TestCase):
    def test_append_and_recent_are_map_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = OperateActivity(directory)
            for index in range(3):
                journal.append('warehouse', {'kind': 'decision', 'title': 'AI · wait', 'detail': str(index)})
            journal.append('other', {'kind': 'operation', 'title': 'Mission completed', 'detail': 'Done'})
            self.assertEqual([row['detail'] for row in journal.recent('warehouse', 2)], ['1', '2'])
            self.assertEqual(len(journal.recent('other')), 1)
            self.assertTrue(Path(directory, 'warehouse.jsonl').exists())

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = OperateActivity(directory)
            with self.assertRaises(ValueError):
                journal.append('../other', {'title': 'bad'})
            with self.assertRaises(ValueError):
                journal.recent('other/subdir')
