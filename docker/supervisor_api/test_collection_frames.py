"""Bulk deletion contracts use temporary datasets; no live data or GPU calls."""
import base64
from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

parent = str(Path(__file__).resolve().parent.parent)
if parent not in sys.path:
    sys.path.insert(0, parent)
from supervisor_api import collection as c
from supervisor_api import collection_frames as f
from supervisor_api import segmentation as s

JPEG = b'\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03' + b'\x00' * 10 + b'\xff\xd9'


class FrameDeletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = c.CollectionStore(Path(self.temp.name) / 'data', lambda _: [])
        self.ep = self.store.start(c.EpisodeRequest(map_name='house'))
        self.eid = self.ep['id']
        self.frames = [self.store.capture(self.eid, {
            'image_base64': base64.b64encode(JPEG).decode(), 'captured_at': index + 1,
            'frame_stamp': str(index), 'camera_topic': '/head'}, {}) for index in range(4)]
        self.ids = [frame['frame_id'] for frame in self.frames]
        self.store.end(self.eid)
        self.directory = self.store.path('raw', self.eid)
        self.sam_service = s.SegmentationService(self.store, Mock(), Mock(return_value={'training': {'status': 'idle'}}))
        self.sam = SimpleNamespace(lock=threading.Lock(), service=self.sam_service,
                                   job={'status':'idle'}, training_job={'status':'idle'})
        self.qwen = SimpleNamespace(lock=threading.RLock(), jobs={})
        self.service = f.FrameDeletionService(self.store, self.sam, self.qwen)
        app = FastAPI()
        app.include_router(f.create_router(self.service), prefix='/api')
        self.client = TestClient(app)

    def delete(self, ids):
        return self.client.post(f'/api/collection/episodes/{self.eid}/frames/delete', json={'frame_ids':ids})

    def snapshot(self):
        return {str(path.relative_to(self.directory)):path.read_bytes()
                for path in self.directory.rglob('*') if path.is_file()}

    def add_mask(self, fid, oid, revisions=2):
        for revision in range(1, revisions + 1):
            path = self.directory / 'masks' / oid / f'{revision}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'mask')
            self.store.add_record(self.eid, 'regions', {'frame_id':fid, 'object_id':oid,
                'object_label':'dish', 'review_status':'approved', 'revision':revision,
                'mask':{'path':f'masks/{oid}/{revision}.png'}})

    def test_deletes_range_assets_and_records_while_preserving_order_and_ids(self):
        first, second, third, fourth = self.ids
        self.add_mask(second, 'removed')
        self.add_mask(fourth, 'kept')
        for fid in (second, fourth):
            self.store.annotate(self.eid, c.AnnotationRequest(frame_id=fid, decision='wait',
                reason='Wait.', annotator='user', review_status='approved'))
            self.store.add_record(self.eid, 'predictions', {'frame_id':fid,'prediction':{'decision':'wait'}})
        response = self.delete([third, second])
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body['deleted_frame_ids'], [second, third])
        self.assertEqual([row['frame_id'] for row in body['episode']['observations']], [first, fourth])
        self.assertEqual(body['episode']['frame_count'], 2)
        self.assertEqual([row['frame_ids'] for row in body['episode']['annotations']], [[fourth]])
        self.assertEqual([row['frame_id'] for row in body['episode']['predictions']], [fourth])
        self.assertEqual([row['object_id'] for row in body['episode']['regions']], ['kept'])
        self.assertFalse((self.directory/'images'/f'{second}.jpg').exists())
        self.assertFalse((self.directory/'images'/f'{third}.jpg').exists())
        self.assertFalse((self.directory/'masks'/'removed').exists())
        self.assertEqual((self.directory/'masks'/'kept'/'1.png').read_bytes(), b'mask')
        self.assertEqual(self.store.image(self.eid, fourth).read_bytes(), JPEG)

    def test_unknown_id_does_not_partially_delete_valid_frames(self):
        before = self.snapshot()
        response = self.delete([self.ids[0], 'missing'])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.snapshot(), before)

    def test_invalid_empty_duplicate_and_path_ids_are_rejected(self):
        before = self.snapshot()
        for ids in ([], [self.ids[0], self.ids[0]], ['../outside'], [1], ['a/b']):
            with self.subTest(ids=ids):
                self.assertEqual(self.delete(ids).status_code, 422)
                self.assertEqual(self.snapshot(), before)

    def test_request_supports_episodes_larger_than_capture_default(self):
        request = f.DeleteFramesRequest(frame_ids=[f'frame_{index}' for index in range(3000)])
        self.assertEqual(len(request.frame_ids), 3000)

    def test_finished_sam_training_does_not_leave_an_offline_deletion_block(self):
        self.sam.training_job = {'status':'training','map_name':'house'}
        self.sam_service.request.return_value = {'training':{'status':'completed','map_name':'house'}}
        self.assertEqual(self.delete(self.ids[:1]).status_code, 200)
        self.sam_service.request.side_effect = HTTPException(503, 'offline')
        self.assertEqual(self.delete(self.ids[1:2]).status_code, 200)

    def test_all_frames_can_be_deleted_without_deleting_episode(self):
        response = self.delete(self.ids)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['episode']['observations'], [])
        self.assertEqual(response.json()['episode']['frame_count'], 0)
        self.assertTrue((self.directory/'episode.json').exists())
        self.assertEqual((self.directory/'samples.jsonl').read_text(), '')
        self.assertEqual(self.store.list()[0]['frame_count'], 0)

    def test_sample_selections_and_prepared_training_are_invalidated(self):
        metadata = self.store.episode(self.eid)
        metadata.update(training={'counts':{'train':4}},
            sam_sample={'schema_version':'user_count_v1','frame_ids':self.ids[:2],'requested_count':2},
            qwen_sample={'schema_version':'user_count_v1','frame_ids':self.ids[:1],'requested_count':1})
        c.write_json(self.directory/'episode.json', metadata)
        for name in f.TRAINING_FILES:
            (self.directory/name).write_text('stale training')
        self.assertEqual(self.delete(self.ids[:1]).status_code, 200)
        saved = self.store.episode(self.eid)
        self.assertNotIn('training', saved)
        self.assertNotIn('qwen_sample', saved)
        self.assertEqual(saved['sam_sample']['frame_ids'], self.ids[1:2])
        self.assertEqual(saved['sam_sample']['requested_count'], 1)
        self.assertTrue(all(not (self.directory/name).exists() for name in f.TRAINING_FILES))

    def test_active_episode_cannot_be_modified(self):
        metadata = self.store.episode(self.eid)
        metadata['ended_at'] = None
        c.write_json(self.directory/'episode.json', metadata)
        before = self.snapshot()
        self.assertEqual(self.delete(self.ids[:1]).status_code, 409)
        self.assertEqual(self.snapshot(), before)

    def test_all_qwen_job_types_block_affected_episode_only(self):
        before = self.snapshot()
        for kind in ('training', 'reasoning_training', 'batch'):
            with self.subTest(kind=kind):
                self.qwen.jobs = {kind:{'status':'running','episode_ids':[self.eid]}}
                self.assertEqual(self.delete(self.ids[:1]).status_code, 409)
                self.assertEqual(self.snapshot(), before)
        self.qwen.jobs = {'training':{'status':'training','episode_ids':['other']}}
        self.assertEqual(self.delete(self.ids[:1]).status_code, 200)

    def test_sam_batch_training_and_preview_block_deletion(self):
        before = self.snapshot()
        self.sam.job = {'status':'running','episode_ids':[self.eid]}
        self.assertEqual(self.delete(self.ids[:1]).status_code, 409)
        self.sam.job = {'status':'idle'}
        self.sam_service.request.return_value = {'training':{'status':'training','map_name':'house'}}
        self.assertEqual(self.delete(self.ids[:1]).status_code, 409)
        self.sam_service.request.return_value = {'training':{'status':'idle'}}
        self.sam_service.inference_lock.acquire()
        try:
            self.assertEqual(self.delete(self.ids[:1]).status_code, 409)
        finally:
            self.sam_service.inference_lock.release()
        self.assertEqual(self.snapshot(), before)

    def test_offline_sam_does_not_block_unless_training_is_known_active(self):
        self.sam_service.request.side_effect = HTTPException(503, 'offline')
        self.sam.training_job = {'status':'training','episode_ids':[self.eid]}
        self.assertEqual(self.delete(self.ids[:1]).status_code, 409)
        self.sam.training_job = {'status':'completed'}
        self.assertEqual(self.delete(self.ids[:1]).status_code, 200)

    def test_shared_image_and_mask_revisions_are_preserved(self):
        samples = c.rows(self.directory/'samples.jsonl')
        samples[1]['observation']['image']['path'] = samples[0]['observation']['image']['path']
        self.store.save_samples(self.eid, samples)
        self.add_mask(self.ids[0], 'shared')
        self.add_mask(self.ids[1], 'shared')
        self.assertEqual(self.delete(self.ids[:1]).status_code, 200)
        self.assertTrue((self.directory/'images'/f'{self.ids[0]}.jpg').exists())
        self.assertTrue((self.directory/'masks'/'shared'/'1.png').exists())

    def test_unsafe_asset_reference_cannot_delete_outside_file(self):
        outside = Path(self.temp.name)/'outside.jpg'
        outside.write_bytes(b'keep')
        for relative in ('../../outside.jpg', '/tmp/outside.jpg', 'images/../episode.json', 'images\\outside.jpg'):
            with self.subTest(path=relative):
                samples = c.rows(self.directory/'samples.jsonl')
                samples[0]['observation']['image']['path'] = relative
                self.store.save_samples(self.eid, samples)
                before = self.snapshot()
                self.assertEqual(self.delete(self.ids[:1]).status_code, 422)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(outside.read_bytes(), b'keep')

    def test_symlink_revision_is_rejected_before_any_deletion(self):
        self.add_mask(self.ids[0], 'unsafe')
        outside = Path(self.temp.name)/'outside.png'
        outside.write_bytes(b'keep')
        (self.directory/'masks'/'unsafe'/'3.png').symlink_to(outside)
        before = self.snapshot()
        self.assertEqual(self.delete(self.ids[:1]).status_code, 400)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(outside.read_bytes(), b'keep')

    def test_preview_cache_entries_for_deleted_frames_are_removed(self):
        self.sam_service.previews = OrderedDict([
            ('delete', {'episode_id':self.eid, 'frame_id':self.ids[0]}),
            ('keep', {'episode_id':self.eid, 'frame_id':self.ids[1]}),
            ('other', {'episode_id':'other', 'frame_id':self.ids[0]}),
        ])
        self.assertEqual(self.delete(self.ids[:1]).status_code, 200)
        self.assertEqual(list(self.sam_service.previews), ['keep', 'other'])

    def test_metadata_write_failure_does_not_remove_frames_or_assets(self):
        before = self.snapshot()
        with patch.object(f, 'write_json', side_effect=PermissionError('blocked')):
            self.assertEqual(self.delete(self.ids[:1]).status_code, 507)
        self.assertEqual(self.snapshot(), before)

    def test_sample_publication_failure_rolls_metadata_back(self):
        metadata = self.store.episode(self.eid)
        metadata['qwen_sample'] = {'schema_version':'user_count_v1','frame_ids':self.ids[:1],'requested_count':1}
        c.write_json(self.directory/'episode.json', metadata)
        before = self.snapshot()
        with patch.object(self.store, 'save_samples', side_effect=PermissionError('blocked')):
            self.assertEqual(self.delete(self.ids[:1]).status_code, 507)
        self.assertEqual(self.snapshot(), before)

    def test_asset_cleanup_failure_returns_committed_episode_with_warning(self):
        target = self.directory/'images'/f'{self.ids[0]}.jpg'
        original_unlink = Path.unlink
        def unlink(path, *args, **kwargs):
            if path == target:
                raise PermissionError('blocked cleanup')
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', unlink):
            response = self.delete(self.ids[:1])
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body['cleanup_pending'])
        self.assertIn('Frames deleted', body['warning'])
        self.assertEqual([row['frame_id'] for row in body['episode']['observations']], self.ids[1:])
        self.assertEqual([row['observation']['frame_id'] for row in c.rows(self.directory/'samples.jsonl')], self.ids[1:])
        self.assertTrue(target.exists())

    def test_deletion_holds_workflow_submission_locks(self):
        def status(_):
            self.assertTrue(self.sam.lock.locked())
            # Called on the deletion worker thread while store operations execute.
            self.assertTrue(self.qwen.lock._is_owned())
            self.assertTrue(self.store.lock._is_owned())
            return {'training': {'status':'idle'}}
        self.sam_service.request.side_effect = status
        self.assertEqual(self.delete(self.ids[:1]).status_code, 200)


if __name__ == '__main__':
    unittest.main()
