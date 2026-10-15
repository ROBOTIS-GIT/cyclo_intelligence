"""Dataset contracts, durability, label validation and export; no robot/GPU calls."""
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value

c = module('collection'); r = module('reasoning'); m = module('navigation_missions')
# Minimal JPEG header fixture with SOF dimensions (storage validates the header).
JPEG = b'\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03' + b'\x00' * 10 + b'\xff\xd9'


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.catalog = [{'mission_id': 'm2', 'mission_name': 'Mission2', 'revision': 1, 'description': 'Pick stick'}]
        self.store = c.CollectionStore(Path(self.temp.name) / 'data', lambda _: self.catalog)
        self.ep = self.start()
        self.frame = {'image_base64': base64.b64encode(JPEG).decode(), 'captured_at': c.now()/1000,
                      'frame_stamp': '12.5', 'camera_topic': '/head'}

    def start(self, group='scene-a'):
        return self.store.start(c.EpisodeRequest(scenario='Shelf', group_id=group, robot_type='ffw_sg2',
                                                map_name='warehouse', environment='simulation'))

    def capture(self, ep=None, context=None):
        return self.store.capture((ep or self.ep)['id'], self.frame, context or {'active_mission': None})

    def label(self, row, ep=None, **values):
        data = dict(frame_id=row['frame_id'], decision='wait', reason='No task object visible.',
                    annotator='reviewer', review_status='approved')
        data.update(values)
        return self.store.annotate((ep or self.ep)['id'], c.AnnotationRequest(**data))

    def test_annotation_updates_only_selected_frame_metadata_with_revision_guard(self):
        first = self.capture()
        second = self.capture()
        metadata = dict(scenario='Empty plate', group_id='table-a', robot_type='ffw_sg2', environment='simulation')
        saved = self.label(first, metadata=metadata)
        by_id = {f['frame_id']: f for f in saved['observations']}
        self.assertEqual(by_id[first['frame_id']]['metadata'], metadata)
        self.assertEqual(by_id[second['frame_id']]['metadata']['scenario'], 'Shelf')
        with self.assertRaises(HTTPException) as error:
            self.label(first, metadata={**metadata, 'scenario': 'Stale edit'})
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(self.store.get(self.ep['id'])['observations'][0]['metadata'], metadata)
        self.assertEqual(saved['annotations'][0]['episode_type'], 'observation')

    def test_readable_names_reuse_deleted_numbers_with_new_ids(self):
        self.assertEqual(Path(self.ep['storage_path']).name, 'raw_episode_0001')
        self.assertNotEqual(self.ep['id'], 'raw_episode_0001')
        self.store.end(self.ep['id'])
        self.store.delete(self.ep['id'])
        second = self.start()
        self.assertEqual(Path(second['storage_path']).name, 'raw_episode_0001')
        self.assertNotEqual(second['id'], self.ep['id'])
        self.assertEqual(self.store.get(second['id'])['id'], second['id'])

    def test_smallest_gap_is_reused_after_restart_despite_legacy_counter(self):
        second = self.start()
        third = self.start()
        fourth = self.start()
        for episode in (second, fourth):
            self.store.end(episode['id'])
            self.store.delete(episode['id'])
        c.write_json(self.store.root / 'warehouse' / '.episode_sequence.json', {'last': 99})
        self.store = c.CollectionStore(self.store.root, lambda _: self.catalog)
        replacement = self.start()
        self.assertEqual(Path(replacement['storage_path']).name, 'raw_episode_0002')
        self.assertEqual(Path(self.start()['storage_path']).name, 'raw_episode_0004')
        self.assertEqual(Path(self.start()['storage_path']).name, 'raw_episode_0005')
        self.assertEqual(self.store.get(third['id'])['storage_path'], third['storage_path'])
        with self.assertRaises(HTTPException):
            self.store.get(second['id'])

    def test_numbering_skips_incomplete_directories_and_is_map_local(self):
        (self.store.root / 'warehouse' / 'observation' / 'raw_episode_0002').mkdir()
        self.assertEqual(Path(self.start()['storage_path']).name, 'raw_episode_0003')
        other = self.store.start(c.EpisodeRequest(scenario='Other', group_id='other',
                                 robot_type='robot', map_name='other_map'))
        self.assertEqual(Path(other['storage_path']).name, 'raw_episode_0001')
        self.assertFalse((self.store.root / 'other_map' / '.episode_sequence.json').exists())

    def test_mission_and_observation_folders_share_map_numbering_and_id_lookup(self):
        mission = self.store.start(c.EpisodeRequest(
            map_name='warehouse', robot_type='ffw_sg2',
            recording_mission_name='Mission2', episode_type='mission'))
        self.assertEqual(Path(self.ep['storage_path']).parent.name, 'observation')
        self.assertEqual(Path(mission['storage_path']).parent.name, 'mission')
        self.assertEqual(Path(mission['storage_path']).name, 'raw_episode_0002')
        self.assertEqual(self.store.get(mission['id'])['episode_type'], 'mission')
        self.assertEqual(len(self.store.list()), 2)
        self.store.end(self.ep['id'])
        self.store.delete(self.ep['id'])
        replacement = self.start()
        self.assertEqual(Path(replacement['storage_path']).name, 'raw_episode_0001')
        self.assertEqual(self.store.get(mission['id'])['id'], mission['id'])
        with self.assertRaises(HTTPException):
            self.store.start(c.EpisodeRequest(map_name='warehouse', robot_type='ffw_sg2',
                recording_mission_name='Mission2', episode_type='observation'))

    def test_only_two_source_files_and_records_survive_restart(self):
        directory = self.store.path('raw', self.ep['id'])
        frame = self.capture(); self.label(frame)
        self.store.prediction(self.ep['id'], frame['frame_id'], {'text':'scene'}, {'model':'test'})
        self.store.instruction(self.ep['id'], 'Inspect shelf')
        self.assertEqual({p.name for p in directory.iterdir() if p.is_file()}, {'episode.json', 'samples.jsonl'})
        sample = c.rows(directory / 'samples.jsonl')[0]
        self.assertEqual(sample['observation']['frame_id'], frame['frame_id'])
        self.assertEqual(len(sample['annotations']), 1)
        self.assertEqual(sample['predictions'][0]['prediction']['text'], 'scene')
        reopened = c.CollectionStore(self.store.root).get(self.ep['id'])
        self.assertEqual(reopened['catalog']['missions'], self.catalog)
        self.assertEqual(reopened['instructions'][0]['text'], 'Inspect shelf')
        self.assertEqual(reopened['annotations'][0]['frame_ids'], [frame['frame_id']])

    def test_reason_presets_are_shared_only_across_saved_episodes_on_the_same_map(self):
        first = self.capture()
        self.label(first, reason='A person remains visible.')
        self.store.end(self.ep['id'])

        second = self.start(group='scene-b')
        second_frame = self.capture(second)
        self.label(second_frame, ep=second, reason='A person remains visible.')
        self.store.end(second['id'])

        other = self.store.start(c.EpisodeRequest(
            scenario='Other', group_id='other', robot_type='robot',
            map_name='other_map', environment='simulation'))
        other_frame = self.capture(other)
        self.label(other_frame, ep=other, reason='A person remains visible.')
        self.store.end(other['id'])

        active = self.start(group='active')
        active_frame = self.capture(active)
        self.label(active_frame, ep=active, reason='Active episode reason.')

        self.assertEqual(self.store.reason_presets('warehouse'), [{
            'reasoning_task': 'mission_selection', 'decision': 'wait', 'mission_id': None,
            'reason': 'A person remains visible.', 'count': 2,
        }])
        self.assertEqual(self.store.reason_presets('missing_map'), [])

    def test_map_layout_and_export_isolation(self):
        directory = self.store.root / 'warehouse' / 'observation' / 'raw_episode_0001'
        self.assertTrue((directory / 'episode.json').is_file())
        self.assertEqual(self.store.path('raw', self.ep['id']), directory)
        self.assertFalse((self.store.root / 'raw').exists())
        other = self.store.start(c.EpisodeRequest(scenario='Other', group_id='other', robot_type='robot', map_name='other_map'))
        self.assertEqual(len(self.store.list()), 2)
        row = self.capture()
        self.label(row)
        self.store.end(self.ep['id']); self.store.end(other['id'])
        with self.assertRaises(HTTPException):
            self.store.export([self.ep['id'], other['id']])
        export = self.store.export([self.ep['id']])
        self.assertEqual(Path(export['storage_path']), self.store.path('raw', self.ep['id']))
        self.assertEqual(len(self.store.list()), 2)

    def test_episode_and_export_need_no_clock_fields(self):
        self.ep = self.store.start(c.EpisodeRequest(map_name='warehouse', episode_type='mission', recording_mission_name='Mission2'))
        self.assertNotIn('epoch_id', self.ep)
        self.assertNotIn('clock_domain', self.ep)
        # Old files may contain temporal fields; new exports must exclude them.
        ep_path = self.store.path('raw', self.ep['id'], 'episode.json')
        legacy = c.read_json(ep_path)
        legacy.update(clock_domain='ros_sim', epoch_id='legacy-clock')
        c.write_json(ep_path, legacy)
        row = self.capture(context={'reasoning_task': 'mission_supervision', 'received_at_ms': 123, 'client_sampled_at_ms': 122,
            'pose': {'x': 1, 'stamp': 120}, 'localized': True,
            'active_mission': {'mission_id': 'm2', 'status': 'running', 'started_at_ms': 99},
            'operator_instruction': {'text': 'Inspect the shelf', 'received_at_ms': 100}})
        self.assertNotIn('epoch_id', row['capture'])
        self.label(row, decision='continue', mission_id='m2')
        self.store.end(self.ep['id'])
        result = self.store.export([self.ep['id']])
        samples = sum((c.rows(Path(result['storage_path']) / (s + '.jsonl')) for s in ('train','val','test')), [])
        content = json.loads(samples[0]['messages'][1]['content'][1]['text'])
        self.assertEqual(content['context'], {'active_mission': {'mission_id': 'm2', 'status': 'running'},
                                             'operator_instruction': 'Inspect the shelf', 'reasoning_task': 'mission_supervision',
                                             'candidate_mission_ids': ['m2']})
        self.assertEqual(result['input_policy'], 'continuous_autonomy_v4')
        self.assertEqual(content['missions'], self.catalog)

    def legacy_episode(self):
        row = self.capture(); self.label(row)
        self.store.end(self.ep['id'])
        path = self.store.path('raw', self.ep['id'], 'episode.json')
        ep = c.read_json(path); ep['map_name'] = None
        c.write_json(path, ep)
        return row

    def test_delete_episode_removes_raw_files_but_keeps_exports_and_copies(self):
        frame = self.legacy_episode()
        copy = self.store.copy_to_map(self.ep['id'],c.MapCopyRequest(map_name='warehouse'))
        exported = self.store.export([self.ep['id']])
        export_root = Path(exported['storage_path'])
        before = {str(p):p.read_bytes() for p in export_root.rglob('*') if p.is_file()}
        source_dir = self.store.path('raw',self.ep['id'])
        c.atomic(source_dir / 'masks' / 'test.png', b'mask')
        outside = self.store.root / 'keep.txt'; outside.write_text('keep')
        (source_dir / 'external-link').symlink_to(outside)
        api = FastAPI(); api.include_router(c.create_router(self.store))
        response = TestClient(api).delete('/collection/episodes/' + self.ep['id'])
        self.assertEqual(response.status_code,200)
        self.assertFalse(response.json()['cleanup_pending'])
        self.assertFalse(source_dir.exists())
        self.assertEqual(outside.read_text(),'keep')
        self.assertEqual(self.store.image(copy['id'],frame['frame_id']).read_bytes(),JPEG)
        self.assertFalse(export_root.exists())
        self.assertNotIn(self.ep['id'],[e['id'] for e in self.store.list()])

    def test_delete_active_episode_is_rejected_and_missing_is_404(self):
        self.capture()
        with self.assertRaises(HTTPException) as err:self.store.delete(self.ep['id'])
        self.assertEqual(err.exception.status_code,409)
        self.assertTrue(self.store.path('raw',self.ep['id'],'episode.json').exists())
        with self.assertRaises(HTTPException) as err:self.store.delete('missing')
        self.assertEqual(err.exception.status_code,404)
        with self.assertRaises(HTTPException):self.store.delete('../exports')

    def test_delete_cleanup_failure_is_isolated_and_retryable(self):
        self.capture(); self.store.end(self.ep['id'])
        with patch.object(c.shutil,'rmtree',side_effect=OSError('permission denied')):
            with self.assertRaises(HTTPException) as caught:
                self.store.delete(self.ep['id'])
        self.assertEqual(caught.exception.status_code, 507)
        self.assertEqual(len(self.store.list()), 1)
        self.assertFalse(self.store.path('.deleted').exists())
        self.assertFalse(self.store.delete(self.ep['id'])['cleanup_pending'])

    def test_new_collection_defaults_and_reason_only_annotation(self):
        ep=self.store.start(c.EpisodeRequest(map_name='warehouse',robot_type='f1'))
        self.assertEqual(ep['group_id'],f"episode-{ep['id']}")
        self.assertEqual(ep['robot_type'],'f1')
        self.assertEqual(ep['environment'],'unknown')
        self.assertNotIn('scenario',ep)
        frame=self.capture(ep)
        self.assertEqual(frame['metadata'],{k:ep[k] for k in ('group_id','robot_type','environment')})
        self.store.end(ep['id'])
        saved=self.label(frame,ep=ep)
        self.assertEqual(set(saved['annotations'][0]['target']),{'decision','mission_id','reason'})
        self.assertEqual(saved['observations'][0]['metadata'],frame['metadata'])

    def test_new_episode_requires_map(self):
        api = FastAPI(); api.include_router(c.create_router(self.store))
        client = TestClient(api)
        fields = {'scenario':'Shelf','group_id':'scene','robot_type':'ffw'}
        for invalid in (fields, {**fields,'map_name':None}, {**fields,'map_name':' '}):
            self.assertEqual(client.post('/collection/episodes',json=invalid).status_code,422)

    def test_mapped_copy_preserves_source_images_masks_and_requires_label_review(self):
        frame = self.legacy_episode()
        directory = self.store.path('raw', self.ep['id'])
        mask = directory / 'masks' / 'object1' / '1.png'
        c.atomic(mask, b'mask-fixture')
        self.store.add_record(self.ep['id'], 'regions', {'episode_id':self.ep['id'],'frame_id':frame['frame_id'],
            'object_id':'object1','revision':1,'review_status':'approved','mission_id':None,
            'mask':{'path':'masks/object1/1.png','sha256':c.hashlib.sha256(mask.read_bytes()).hexdigest()}})
        before = {str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        self.catalog = [{**self.catalog[0], 'description':'Current Design mission'}]
        copy = self.store.copy_to_map(self.ep['id'], c.MapCopyRequest(map_name='warehouse'))
        self.assertNotEqual(copy['id'],self.ep['id'])
        self.assertEqual(copy['map_name'],'warehouse')
        self.assertEqual(copy['group_id'],self.ep['group_id'])
        self.assertTrue(copy['ended_at'])
        self.assertEqual(copy['catalog']['missions'],self.catalog)
        self.assertEqual(copy['annotations'][0]['review_status'],'draft')
        self.assertNotEqual(copy['annotations'][0]['sample_id'],self.store.get(self.ep['id'])['annotations'][0]['sample_id'])
        self.assertEqual(copy['observations'][0]['catalog_snapshot_id'],copy['catalog_snapshot_id'])
        self.assertEqual(self.store.image(copy['id'],frame['frame_id']).read_bytes(),JPEG)
        self.assertEqual(self.store.path('raw',copy['id'],'masks','object1','1.png').read_bytes(),b'mask-fixture')
        self.assertEqual(self.store.records(copy['id'], 'regions')[0]['episode_id'],copy['id'])
        after = {str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(copy['map_assignment']['source_episode_id'],self.ep['id'])

    def test_map_copy_failure_leaves_no_partial_episode(self):
        self.legacy_episode()
        with patch.object(c, 'write_json', side_effect=OSError('disk full')):
            with self.assertRaises(HTTPException):
                self.store.copy_to_map(self.ep['id'],c.MapCopyRequest(map_name='warehouse'))
        self.assertEqual(len(self.store.list()),1)
        self.assertEqual(list(self.store.path('_staging').glob('*')),[])

    def test_map_copy_rejects_linked_or_active_episode(self):
        with self.assertRaises(HTTPException):
            self.store.copy_to_map(self.ep['id'],c.MapCopyRequest(map_name='other'))

    def test_capture_stores_exact_bytes_and_snapshot_before_prediction(self):
        row = self.capture()
        self.assertEqual(self.store.image(self.ep['id'], row['frame_id']).read_bytes(), JPEG)
        self.assertEqual(row['image']['width'], 3)
        self.assertEqual(self.store.get(self.ep['id'])['predictions'], [])
        self.catalog[0]['description'] = 'changed'
        self.assertEqual(self.store.get(self.ep['id'])['catalog']['missions'][0]['description'], 'Pick stick')

    def test_annotation_requires_saved_image_catalog_and_idle_start_context(self):
        row = self.capture()
        for kwargs in ({'decision': 'start', 'mission_id': 'invented'}, {'mission_id': 'm2'}, {'reason': ' '}):
            with self.assertRaises(HTTPException): self.label(row, **kwargs)
        with self.assertRaises(HTTPException): self.label(row, decision='continue', mission_id='m2')
        with self.assertRaises(HTTPException): self.label(row, decision='cancel', mission_id='m2')
        live = self.capture(context={'active_mission': {'mission_id': 'm2', 'status': 'running'}})
        self.label(live, decision='wait', mission_id=None)
        with self.assertRaises(HTTPException): self.label(live, decision='start', mission_id='m2', expected_revision=1)
        self.label(row, decision='start', mission_id='m2')
        with self.assertRaises(HTTPException): self.label({'frame_id': 'not-a-frame'})

    def test_revisions_append_and_stale_review_is_rejected(self):
        row = self.capture()
        self.label(row, review_status='draft')
        with self.assertRaises(HTTPException) as caught: self.label(row)
        self.assertEqual(caught.exception.status_code, 409)
        ep = self.label(row, expected_revision=1)
        self.assertEqual(ep['annotations'][0]['revision'], 2)
        self.assertEqual(len(self.store.records(self.ep['id'], 'annotations')), 2)

    def test_disk_failure_does_not_publish_an_observation(self):
        with patch.object(self.store, 'save_samples', side_effect=OSError('disk full')):
            with self.assertRaises(HTTPException) as caught: self.capture()
        self.assertEqual(caught.exception.status_code, 507)
        self.assertEqual(self.store.get(self.ep['id'])['observations'], [])
        self.assertEqual(list(self.store.path('raw', self.ep['id'], 'images').iterdir()), [])

    def test_closed_episodes_reject_capture_but_allow_review(self):
        row = self.capture(); self.store.end(self.ep['id'])
        with self.assertRaises(HTTPException): self.capture()
        self.label(row)
        restored = c.CollectionStore(self.store.root)
        self.assertEqual(restored.get(self.ep['id'])['annotations'][0]['target']['decision'], 'wait')

    def test_export_approved_only_same_scene_same_split_and_valid_paths(self):
        row = self.capture(); self.label(row)
        self.label(self.capture(), review_status='draft')
        self.store.end(self.ep['id'])
        second = self.start(); self.label(self.capture(second), second); self.store.end(second['id'])
        result = self.store.export([self.ep['id'], second['id']])
        self.assertEqual(sum(result['counts'].values()), 2)
        self.assertEqual(len([v for v in result['counts'].values() if v]), 1)
        root = Path(result.get('datasets', [result])[0]['storage_path'])
        for split in ('train', 'val', 'test'):
            for sample in c.rows(root / (split + '.jsonl')):
                self.assertNotIn('images', sample)
                self.assertEqual([b['type'] for b in sample['messages'][1]['content']], ['image', 'text'])
                self.assertEqual(json.loads(sample['messages'][2]['content'][0]['text'])['decision'], 'wait')
                self.assertTrue((root / sample['messages'][1]['content'][0]['image']).is_file())

    def test_preparation_reuses_images_and_label_edit_removes_training_files(self):
        frame = self.capture(); self.label(frame); self.store.end(self.ep['id'])
        directory = self.store.path('raw', self.ep['id'])
        before = {p.name:p.read_bytes() for p in (directory / 'images').iterdir()}
        self.store.export([self.ep['id']])
        self.assertEqual(before, {p.name:p.read_bytes() for p in (directory / 'images').iterdir()})
        self.assertIn('training', self.store.episode(self.ep['id']))
        self.label(frame, expected_revision=1, review_status='rejected')
        self.assertNotIn('training', self.store.episode(self.ep['id']))
        self.assertFalse((directory / 'train.jsonl').exists())
        for name in ('_exports', '_staging', '.deleted'):
            self.assertFalse(list(self.store.root.rglob(name)))

    def test_rejected_revision_excludes_previously_approved_label(self):
        row = self.capture(); self.label(row); self.label(row, expected_revision=1, review_status='rejected')
        self.store.end(self.ep['id'])
        with self.assertRaises(HTTPException): self.store.export([self.ep['id']])
        self.assertFalse(self.store.path('warehouse', '_exports').exists())

    def test_corrupted_image_aborts_export_and_traversal_is_rejected(self):
        row = self.capture(); self.label(row); self.store.end(self.ep['id'])
        self.store.path('raw', self.ep['id'], 'images', row['frame_id'] + '.jpg').write_bytes(b'changed')
        with self.assertRaises(HTTPException): self.store.export([self.ep['id']])
        for value in ('../escape', '..', '/tmp', 'x\\y'):
            with self.assertRaises(HTTPException): self.store.get(value)

    def test_instruction_after_capture_is_not_leaked_into_frame(self):
        with patch.object(c, 'now', return_value=int(self.frame['captured_at'] * 1000) + 1000):
            self.store.instruction(self.ep['id'], 'Stop')
        self.assertIsNone(self.capture()['context']['operator_instruction'])

    def test_finish_captures_without_qwen_then_ends(self):
        service = r.ReasoningService(Mock(), Mock(), self.store)
        request = r.CaptureRequest(context={'map_name':'warehouse'})
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request') as qwen:
            result = service.finish_episode(self.ep['id'], request)
        qwen.assert_not_called()
        service.start_container.assert_not_called()
        self.assertTrue(result['episode']['ended_at'])
        self.assertEqual(result['saved_frame']['frame_id'], result['episode']['observations'][-1]['frame_id'])
        self.assertTrue(self.store.image(self.ep['id'], result['saved_frame']['frame_id']).exists())

    def test_finish_capture_failure_keeps_episode_open(self):
        service = r.ReasoningService(Mock(), Mock(), self.store)
        request = r.CaptureRequest(context={'map_name':'warehouse'})
        with patch.object(r, 'capture_head', side_effect=HTTPException(503, 'Camera unavailable')):
            with self.assertRaises(HTTPException): service.finish_episode(self.ep['id'], request)
        self.assertIsNone(self.store.get(self.ep['id'])['ended_at'])
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(self.store, 'save_samples', side_effect=OSError('Disk full')):
            with self.assertRaises(HTTPException): service.finish_episode(self.ep['id'], request)
        self.assertIsNone(self.store.get(self.ep['id'])['ended_at'])
        self.assertEqual(self.store.get(self.ep['id'])['observations'], [])

    def test_frame_metadata_is_saved_per_capture_and_counts_track_labels(self):
        service = r.ReasoningService(Mock(), Mock(), self.store)
        with patch.object(r, 'capture_head', return_value=self.frame):
            first = service.capture_frame(self.ep['id'], r.CaptureRequest(context={'map_name':'warehouse'}, metadata={'scenario':'Red stick', 'group_id':'shelf', 'robot_type':'ffw', 'environment':'simulation'}))
            second = service.capture_frame(self.ep['id'], r.CaptureRequest(context={'map_name':'warehouse'}, metadata={'scenario':'No stick', 'group_id':'shelf', 'robot_type':'ffw', 'environment':'simulation'}))
        ep = self.store.get(self.ep['id'])
        self.assertEqual([f['metadata']['scenario'] for f in ep['observations']], ['Red stick', 'No stick'])
        self.assertEqual(self.store.list()[0]['unlabeled_count'], 2)
        self.label(first['saved_frame'])
        self.assertEqual(self.store.list()[0]['unlabeled_count'], 1)
        self.assertEqual(self.store.list()[0]['frame_count'], 2)

    def test_manual_capture_works_with_qwen_stopped(self):
        service = r.ReasoningService(Mock(), Mock(), self.store)
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request') as qwen:
            saved = service.capture_frame(self.ep['id'], r.CaptureRequest(context={'map_name':'warehouse', 'pose':{'x':1}}))
        qwen.assert_not_called(); service.start_container.assert_not_called()
        self.assertEqual(len(saved['episode']['observations']), 1)
        self.assertEqual(saved['episode']['predictions'], [])
        self.assertEqual(saved['saved_frame']['context']['pose'], {'x':1})

    def test_reasoning_does_not_save_until_explicitly_requested_and_keeps_original_context(self):
        service = r.ReasoningService(Mock(), Mock(), self.store); service.update(phase='ready', model='Qwen/4B')
        def generate(*args, **kwargs):
            self.assertEqual(len(self.store.get(self.ep['id'])['observations']), 0)
            return {'text':'Shelf', 'elapsed_seconds':1.2}
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', side_effect=generate):
            result = service.observe(r.ObserveRequest(episode_id=self.ep['id'],context={'pose':{'x':1},'active_mission':{'mission_id':'invented'}}))
        self.assertEqual(self.store.get(self.ep['id'])['observations'], [])
        service.stop()  # saving an already analyzed frame must not restart the model
        with patch.object(r, 'capture_head') as camera, patch.object(r, 'qwen_request') as qwen:
            saved = service.save_analyzed_frame(self.ep['id'], r.SaveAnalysisRequest(analysis_id=result['analysis_id']))
        camera.assert_not_called();qwen.assert_not_called()
        frame = saved['saved_frame']
        self.assertEqual(frame['context']['pose'], {'x':1})
        self.assertIsNone(frame['context']['active_mission'])
        self.assertEqual(frame['capture']['received_at_ms'],int(self.frame['captured_at']*1000))
        self.assertEqual(saved['episode']['predictions'][0]['prediction']['text'],'Shelf')
        again = service.save_analyzed_frame(self.ep['id'], r.SaveAnalysisRequest(analysis_id=result['analysis_id']))
        self.assertEqual(again['saved_frame']['frame_id'],frame['frame_id'])
        self.assertEqual(len(again['episode']['predictions']),1)

    def test_collect_uses_current_catalog_and_freezes_instruction_and_adapter(self):
        self.store.instruction(self.ep['id'], 'Inspect the shelf')
        service = r.ReasoningService(Mock(), Mock(), self.store)
        service.update(phase='ready', model='Qwen/4B', adapter='/model/tuned')
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value={
            'text':json.dumps({'decision':'start','mission_id':'m2','reason':'Stick visible'}), 'finish_reason':'stop'
        }) as qwen:
            result = service.observe(r.ObserveRequest(episode_id=self.ep['id']))
        payload = qwen.call_args.args[1]
        self.assertEqual(json.loads(payload['prompt'])['context']['operator_instruction'], 'Inspect the shelf')
        self.assertFalse(payload['do_sample'])
        self.assertEqual(self.store.get(self.ep['id'])['annotations'], [])
        saved = service.save_analyzed_frame(self.ep['id'], r.SaveAnalysisRequest(analysis_id=result['analysis_id']))
        self.assertEqual(saved['saved_frame']['context']['operator_instruction'], 'Inspect the shelf')
        pinned = service.analyses[result['analysis_id']]['prediction']
        self.assertEqual(pinned['config']['adapter'], '/model/tuned')
        self.assertEqual(saved['episode']['annotations'], [])
        # A current preview can be stored in an older episode without rewriting
        # the episode's immutable source catalog.
        self.catalog[0]['revision'] = 2
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value={'text':'bad'}):
            changed = service.observe(r.ObserveRequest(context={'map_name':'warehouse'}))
        current_saved = service.save_analyzed_frame(self.ep['id'], r.SaveAnalysisRequest(analysis_id=changed['analysis_id']))
        self.assertEqual(current_saved['saved_frame']['capture']['source_catalog_snapshot_id'],c.digest(self.catalog))
        self.assertEqual(current_saved['episode']['catalog_snapshot_id'],self.ep['catalog_snapshot_id'])
        # A Design edit after inference makes that analysis stale.
        self.catalog[0]['revision'] = 3
        with self.assertRaises(HTTPException) as caught:
            service.save_analyzed_frame(self.ep['id'], r.SaveAnalysisRequest(analysis_id=changed['analysis_id']))
        self.assertEqual(caught.exception.status_code, 409)

    def test_failed_inference_can_be_saved_but_never_saves_automatically(self):
        service=r.ReasoningService(Mock(),Mock(),self.store);service.update(phase='ready')
        with patch.object(r,'capture_head',return_value=self.frame),patch.object(r,'qwen_request',side_effect=HTTPException(503,'unavailable')):
            result=service.observe(r.ObserveRequest(episode_id=self.ep['id']))
        self.assertEqual(self.store.get(self.ep['id'])['observations'],[])
        saved=service.save_analyzed_frame(self.ep['id'],r.SaveAnalysisRequest(analysis_id=result['analysis_id']))
        self.assertEqual(saved['episode']['predictions'][0]['prediction']['error'],'unavailable')

    def test_analysis_before_episode_needs_no_clock_and_keeps_missing_instruction(self):
        service=r.ReasoningService(Mock(),Mock(),self.store);service.update(phase='ready')
        with patch.object(r,'capture_head',return_value=self.frame),patch.object(r,'qwen_request',return_value={'text':'Shelf'}):
            result=service.observe(r.ObserveRequest(context={'map_name':'warehouse'}))
        saved=service.save_analyzed_frame(self.ep['id'],r.SaveAnalysisRequest(analysis_id=result['analysis_id']))['saved_frame']
        self.assertNotIn('clock_domain', saved['capture'])
        self.assertNotIn('epoch_id', saved['capture'])
        self.assertIsNone(saved['context']['operator_instruction'])

    def test_expired_analysis_closed_episode_and_map_mismatch_rejected(self):
        service=r.ReasoningService(Mock(),Mock(),self.store);service.update(phase='ready')
        with patch.object(r,'capture_head',return_value=self.frame),patch.object(r,'qwen_request',return_value={'text':'Shelf'}):
            result=service.observe(r.ObserveRequest(context={'map_name':'different'}))
        req=r.SaveAnalysisRequest(analysis_id=result['analysis_id'])
        with self.assertRaises(HTTPException):service.save_analyzed_frame(self.ep['id'],req)
        service.analyses[result['analysis_id']]['expires']=0
        with self.assertRaises(HTTPException) as caught:service.save_analyzed_frame(self.ep['id'],req)
        self.assertEqual(caught.exception.status_code,410)
        self.store.end(self.ep['id'])
        with patch.object(r,'capture_head') as camera:
            with self.assertRaises(HTTPException):service.capture_frame(self.ep['id'],r.CaptureRequest())
            camera.assert_not_called()

    def test_observation_publish_failure_rolls_back_prediction_and_image(self):
        with patch.object(self.store,'save_samples',side_effect=OSError('disk full')):
            with self.assertRaises(HTTPException):self.store.capture(self.ep['id'],self.frame,{},prediction={'prediction':{'text':'Shelf'}})
        ep=self.store.get(self.ep['id'])
        self.assertEqual(ep['observations'],[]);self.assertEqual(ep['predictions'],[])
        self.assertEqual(list(self.store.path('raw',self.ep['id'],'images').iterdir()),[])

    def test_http_contract_and_no_execute_endpoint(self):
        app = FastAPI(); app.include_router(c.create_router(self.store)); client = TestClient(app)
        self.assertEqual(client.get('/collection/episodes').status_code, 200)
        self.assertEqual(client.post('/collection/prepare-training', json={'episode_ids': []}).status_code, 422)
        self.assertEqual(client.post('/collection/execute').status_code, 404)

    def test_mission_identity_survives_edit_rename_and_changes_for_duplicate(self):
        with patch.object(m, 'NAVIGATION_DATA_ROOT', Path(self.temp.name) / 'navigation'):
            first = m.save_mission('map', m.MissionSaveRequest(), mission_name='One')
            mid = first.metadata['reasoning_mission_id']
            edited = m.save_mission('map', m.MissionSaveRequest(expected_revision=first.revision, metadata={'description': 'New'}), mission_name='One')
            self.assertEqual(edited.metadata['reasoning_mission_id'], mid)
            renamed = m.rename_mission('map', m.MissionRenameRequest(mission_name='One', new_name='Two', expected_revision=edited.revision))
            self.assertEqual(renamed.metadata['reasoning_mission_id'], mid)
            duplicate = m.duplicate_mission('map', m.MissionDuplicateRequest(mission_name='Two', new_name='Three', expected_revision=renamed.revision))
            self.assertNotEqual(duplicate.metadata['reasoning_mission_id'], mid)
            self.assertEqual(len(m.collection_catalog('map')), 2)


if __name__ == '__main__': unittest.main()

class ThreeDecisionCollectionTests(CollectionTests):
    def test_running_preview_keeps_legacy_labels_but_is_excluded_from_training(self):
        service = r.ReasoningService(Mock(), Mock(), self.store)
        with patch.object(r, 'capture_head', return_value=self.frame):
            saved = service.capture_frame(self.ep['id'], r.CaptureRequest(context={
                'map_name': 'warehouse', 'preview_active_mission': {'mission_id': 'm2', 'step': 2, 'action': 'grasping'}}))
        frame = saved['saved_frame']
        self.assertEqual(frame['context']['runtime_source'], 'operator_supplied_preview')
        self.label(frame, decision='wait', mission_id=None)
        with self.assertRaises(HTTPException): self.label(frame, decision='start', mission_id='m2', expected_revision=1)
        self.store.end(self.ep['id'])
        with self.assertRaises(HTTPException) as caught:
            self.store.export([self.ep['id']])
        self.assertEqual(caught.exception.status_code, 422)
        self.assertIn('No approved samples', caught.exception.detail)
        saved = self.store.get(self.ep['id'])
        self.assertEqual(saved['reasoning_frame_count'], 0)
        self.assertEqual(saved['annotations'][0]['target']['decision'], 'wait')
        context = saved['observations'][0]['context']
        self.assertEqual(context['active_mission']['action'], 'grasping')
        self.assertEqual(context['active_mission']['step'], 2)
        self.assertFalse(list(self.store.path('raw', self.ep['id']).glob('train.jsonl')))

class MissionNameSynchronizationTests(unittest.TestCase):
    setUp = CollectionTests.setUp
    start = CollectionTests.start
    capture = CollectionTests.capture
    label = CollectionTests.label

    def recorded_episode(self, scenario='warehouse / Mission2'):
        return self.store.start(c.EpisodeRequest(map_name='warehouse', robot_type='robot',
            recording_mission_name='Mission2', scenario=scenario))

    def test_old_episode_names_follow_id_without_changing_training_snapshot(self):
        ep = self.recorded_episode()
        fid = self.capture(ep)
        self.label(fid, ep, decision='start', mission_id='m2')
        path = self.store.path('raw', ep['id'], 'episode.json')
        raw = c.read_json(path); raw.pop('recording_mission_id')
        c.write_json(path, raw)
        samples = self.store.path('raw', ep['id'], 'samples.jsonl').read_bytes()
        self.catalog = [{'mission_id':'m2','mission_name':'TrashDisposal','revision':2}]
        listing = next(e for e in self.store.list() if e['id'] == ep['id'])
        self.assertEqual(listing['recording_mission_name'], 'TrashDisposal')
        self.assertEqual(listing['scenario'], 'warehouse / TrashDisposal')
        saved = self.store.get(ep['id'])
        self.assertEqual(saved['display_missions'][0]['mission_name'], 'TrashDisposal')
        self.assertEqual(saved['recording_mission_id'], 'm2')
        self.assertEqual(saved['catalog'], ep['catalog'])
        self.assertEqual(saved['catalog_snapshot_id'], c.digest(saved['catalog']['missions']))
        self.assertEqual(self.store.path('raw', ep['id'], 'samples.jsonl').read_bytes(), samples)
        self.assertEqual(c.read_json(path)['scenario'], 'warehouse / TrashDisposal')
        self.catalog[0]['mission_name'] = 'EmptyTrash'
        self.assertEqual(self.store.get(ep['id'])['scenario'], 'warehouse / EmptyTrash')

    def test_custom_description_and_reused_mission_name_are_preserved(self):
        ep = self.recorded_episode('Mission2 collects a visible object near the shelf.')
        self.catalog = [{'mission_id':'m2','mission_name':'TrashDisposal','revision':2}]
        self.assertEqual(self.store.get(ep['id'])['scenario'], ep['scenario'])
        self.catalog = [{'mission_id':'new-id','mission_name':'Mission2','revision':1}]
        saved = self.store.get(ep['id'])
        self.assertEqual(saved['recording_mission_id'], 'm2')
        self.assertEqual(saved['recording_mission_name'], 'TrashDisposal')

    def test_new_episode_keeps_stable_recording_id(self):
        ep = self.recorded_episode()
        self.assertEqual(c.read_json(self.store.path('raw',ep['id'],'episode.json'))['recording_mission_id'],'m2')

    def test_live_catalog_addition_is_available_without_mutating_episode_source(self):
        ep = self.recorded_episode()
        frame = self.capture(ep)
        source_catalog = ep['catalog']
        self.catalog.append({'mission_id':'observe','mission_name':'ObservationMode','revision':1,
                             'description':'Observe the scene'})
        current = self.store.get(ep['id'])
        self.assertEqual(current['catalog'], source_catalog)
        self.assertEqual([m['mission_name'] for m in current['current_catalog']['missions']],
                         ['Mission2','ObservationMode'])
        saved = self.label(frame, ep, decision='start', mission_id='observe')
        label = saved['annotations'][0]
        self.assertEqual(label['target']['mission_id'],'observe')
        self.assertEqual(label['catalog_snapshot_id'],current['current_catalog']['snapshot_id'])
        self.assertEqual(label['catalog']['missions'][-1]['mission_name'],'ObservationMode')

    def test_export_preserves_recorded_catalog_and_rejects_deleted_target(self):
        ep = self.start()  # Mission selection is recorded in an Observation episode.
        frame = self.capture(ep)
        self.label(frame, ep, decision='start', mission_id='m2')
        self.store.end(ep['id'])
        self.catalog.append({'mission_id':'observe','mission_name':'ObservationMode','revision':1})
        output, manifest = self.store._training_files(self.store.get(ep['id']))
        samples = [row for name, rows in output.items() if name != 'training_regions.jsonl' for row in rows]
        prompt = json.loads(samples[0]['messages'][1]['content'][1]['text'])
        self.assertEqual([m['mission_id'] for m in prompt['missions']],['m2'])
        self.assertEqual(manifest['catalog']['snapshot_id'],c.digest(self.catalog))
        self.assertEqual(manifest['catalog']['mission_revisions'],{'m2':1,'observe':1})

        self.catalog = [self.catalog[-1]]
        current = self.store.get(ep['id'])
        self.assertEqual(current['display_missions'][0]['mission_id'],'m2')
        self.assertFalse(current['display_missions'][0]['available'])
        self.assertEqual(current['catalog_status']['deleted_mission_ids'],['m2'])
        with self.assertRaises(HTTPException) as error:
            self.store._training_files(current)
        self.assertIn('deleted Design mission',error.exception.detail)
