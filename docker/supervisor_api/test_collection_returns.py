"""Mission return recordings preserve one role per episode and a shared session."""
import base64
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from supervisor_api import collection as c, collection_returns as returns
from supervisor_api.reasoning_capture_context import context_for_frame
from supervisor_api.test_collection_episode_contract import CATALOG, JPEG, VIEW


class MissionReturnCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = c.CollectionStore(Path(self.tmp.name), lambda _: CATALOG)
        self.view = patch('supervisor_api.autonomy_config.AutonomyRepository.view', return_value=deepcopy(VIEW)).start()
        self.addCleanup(patch.stopall)
        self.source = self.store.start(c.EpisodeRequest(
            map_name='Home', episode_type='mission', capture_contract='task_aligned_v1',
            recording_mission_name='CleanTable', return_observation_spot_id='table',
            robot_type='ffw_sg2', environment='simulation', group_id='meal-session'))
        self.stamp = time.time() * 1000
        self.frame = {'image_base64': base64.b64encode(JPEG).decode(), 'captured_at': self.stamp / 1000,
                      'frame_stamp': '100.01', 'camera_topic': '/head'}

    def context(self, returning=False, **changes):
        role = ({'reasoning_task': 'patrol_supervision', 'phase': 'returning', 'from_spot_id': None, 'to_spot_id': 'table'}
                if returning else {'reasoning_task': 'mission_supervision', 'mission_id': 'clean', 'step': 1, 'action': 'navigation'})
        report = {'map_name': 'Home', 'status': 'running', 'context_version': 4,
                  'context_valid_from_ms': self.stamp - 100, 'sampled_at_ms': self.stamp, **role, **changes}
        return context_for_frame({'map_name': 'Home', 'recording_execution': report}, self.frame)

    def test_return_is_linked_patrol_with_shared_split_and_idempotent_http(self):
        mission_frame = self.store.capture(self.source['id'], self.frame, self.context())
        self.store.annotate(self.source['id'], c.AnnotationRequest(frame_id=mission_frame['frame_id'],
            decision='continue', mission_id='clean', reason='The task may proceed.', annotator='human', review_status='approved'))
        api = FastAPI(); api.include_router(c.create_router(self.store)); client = TestClient(api)
        response = client.post(f"/collection/episodes/{self.source['id']}/return")
        self.assertEqual(response.status_code, 200)
        child = response.json()
        self.assertEqual(child['episode_type'], 'patrol')
        self.assertEqual(child['capture_segment'], 'mission_return')
        self.assertEqual(child['patrol_spot_ids'], ['table'])
        self.assertEqual(child['source_episode_id'], self.source['id'])
        self.assertEqual(child['previous_episode_id'], self.source['id'])
        self.assertEqual(child['return_mission_id'], 'clean')
        self.assertEqual(child['return_observation_spot']['id'], 'table')
        self.assertEqual(child['group_id'], self.source['group_id'])
        self.assertEqual(child['robot_type'], self.source['robot_type'])
        self.assertEqual(Path(child['storage_path']).parent.name, 'patrol')
        source = self.store.get(self.source['id'])
        self.assertEqual(source['return_episode_id'], child['id'])
        self.assertEqual(source['next_episode_id'], child['id'])
        self.assertIsNotNone(source['ended_at'])
        self.assertEqual(client.post(f"/collection/episodes/{self.source['id']}/return").json()['id'], child['id'])
        self.assertEqual(len(self.store.list()), 2)
        frame = self.store.capture(child['id'], self.frame, self.context(returning=True, status='paused'))
        self.assertIsNone(frame['context']['active_mission'])
        self.assertEqual(frame['context']['observation']['phase'], 'returning')
        self.store.annotate(child['id'], c.AnnotationRequest(frame_id=frame['frame_id'], decision='pause',
            reason='A person blocks the return path.', annotator='human', review_status='approved'))
        self.store.end(child['id'])
        saved = self.store.start_return(self.source['id'])
        self.assertEqual(saved['ended_at'], self.store.get(child['id'])['ended_at'])
        self.assertEqual(saved['id'], child['id'])
        output = self.store.export([self.source['id'], child['id']])
        first, second = output['datasets']
        self.assertEqual(first['groups'], second['groups'])
        self.assertEqual({tuple(d['reasoning_tasks']) for d in output['datasets']},
                         {('mission_supervision',), ('patrol_supervision',)})
        for ep in (self.source, child):
            names = {p.name for p in self.store.path('raw', ep['id']).iterdir() if p.is_file()}
            self.assertTrue(names <= {'episode.json', 'samples.jsonl', 'train.jsonl', 'val.jsonl', 'test.jsonl'})

    def test_user_creation_cannot_bypass_patrol_route_requirement(self):
        with self.assertRaises(HTTPException):
            self.store.start(c.EpisodeRequest(map_name='Home', episode_type='patrol',
                capture_contract='task_aligned_v1', patrol_spot_ids=[]))
        with self.assertRaises(HTTPException):
            self.store.start(c.EpisodeRequest(map_name='Home', episode_type='observation',
                observation_spot_id='table', return_observation_spot_id='table'))
        with self.assertRaises(HTTPException):
            self.store.start(c.EpisodeRequest(map_name='Home', episode_type='mission',
                recording_mission_name='CleanTable', return_observation_spot_id='missing'))

    def test_return_capture_rejects_task_motion_wrong_target_and_arrived_selection(self):
        child = self.store.start_return(self.source['id'])
        for context in (self.context(), self.context(returning=True, to_spot_id='laundry'),
                        self.context(returning=True, phase='moving', from_spot_id='laundry')):
            with self.subTest(context=context), self.assertRaises(HTTPException):
                self.store.capture(child['id'], self.frame, context)
        with self.assertRaises(HTTPException):
            self.store.capture(self.source['id'], self.frame, self.context())
        self.assertEqual(self.store.get(child['id'])['observations'], [])

    def test_missing_or_deleted_return_spot_does_not_close_source_or_create_child(self):
        self.view.return_value = {'plan': {'observations': []}, 'spots': []}
        with self.assertRaises(HTTPException) as caught:
            self.store.start_return(self.source['id'])
        self.assertEqual(caught.exception.status_code, 409)
        self.assertIsNone(self.store.get(self.source['id'])['ended_at'])
        self.assertEqual(len(self.store.list()), 1)
        no_target = self.store.start(c.EpisodeRequest(map_name='Home', episode_type='mission', recording_mission_name='CleanTable'))
        with self.assertRaises(HTTPException) as caught:
            self.store.start_return(no_target['id'])
        self.assertEqual(caught.exception.status_code, 422)

    def test_deleted_child_or_source_is_never_recreated_by_retry(self):
        child = self.store.start_return(self.source['id'])
        self.store.end(child['id']); self.store.delete(child['id'])
        with self.assertRaises(HTTPException) as caught:
            self.store.start_return(self.source['id'])
        self.assertEqual(caught.exception.status_code, 410)
        self.assertEqual(len(self.store.list()), 1)
        self.store.delete(self.source['id'])
        with self.assertRaises(HTTPException) as caught:
            self.store.start_return(self.source['id'])
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(self.store.list(), [])

    def test_source_link_write_failure_recovers_same_child(self):
        original = returns.write_json
        source_path = self.store.path('raw', self.source['id'], 'episode.json')
        def fail_link(path, value):
            if path == source_path:
                raise OSError('disk full while linking source')
            original(path, value)
        with patch.object(returns, 'write_json', side_effect=fail_link), self.assertRaises(HTTPException) as caught:
            self.store.start_return(self.source['id'])
        self.assertEqual(caught.exception.status_code, 507)
        before = self.store.list()
        self.assertEqual(len(before), 2)
        child = next(ep for ep in before if ep['episode_type'] == 'patrol')
        retry = self.store.start_return(self.source['id'])
        self.assertEqual(retry['id'], child['id'])
        self.assertEqual(self.store.get(self.source['id'])['return_episode_id'], child['id'])
        self.assertEqual(len(self.store.list()), 2)

    def test_return_display_name_follows_stable_mission_id_without_changing_capture_catalog(self):
        child = self.store.start_return(self.source['id'])
        captured_catalog = deepcopy(child['catalog'])
        self.store.catalog_provider = lambda _: [{'mission_id': 'clean', 'mission_name': 'ClearDishes', 'revision': 2}]
        updated = self.store.get(child['id'])
        self.assertEqual(updated['return_mission_name'], 'ClearDishes')
        self.assertEqual(updated['return_mission_id'], 'clean')
        self.assertNotIn('recording_mission_name', updated)
        self.assertEqual(updated['catalog'], captured_catalog)
        listed = next(ep for ep in self.store.list() if ep['id'] == child['id'])
        self.assertEqual(listed['return_mission_name'], 'ClearDishes')
        self.assertEqual(self.store.get(self.source['id'])['return_observation_spot_id'], 'table')

    def test_return_preserves_start_time_spot_pose_and_captures_arrival_before_end(self):
        source_path = self.store.path('raw', self.source['id'], 'episode.json')
        source = c.read_json(source_path)
        source['return_observation_spot']['pose'] = {'x': 1.0, 'y': 2.0, 'yaw': 0.5}
        source['return_observation_spot']['arrival_bt'] = '<root><LookAtTable/></root>'
        source['recording_plan']['observations'][0]['arrival_bt'] = '<root><LookAtTable/></root>'
        c.write_json(source_path, source)
        self.view.return_value['spots'][0]['pose'] = {'x': 10.0, 'y': 20.0, 'yaw': 1.5}
        self.view.return_value['plan']['observations'][0]['arrival_bt'] = '<root><DifferentPose/></root>'
        child = self.store.start_return(source['id'])
        self.assertEqual(child['return_observation_spot']['pose'], {'x': 1.0, 'y': 2.0, 'yaw': 0.5})
        self.assertEqual(child['return_observation_spot']['arrival_bt'], '<root><LookAtTable/></root>')
        self.assertEqual(child['recording_plan'], source['recording_plan'])
        context = self.context(returning=True, phase='settling')
        self.assertEqual(context['observation']['phase'], 'settling')
        self.assertEqual(context['observation']['at_spot_id'], 'table')
        row = self.store.capture(child['id'], self.frame, context)
        self.assertEqual(row['context']['reasoning_task'], 'patrol_supervision')
        self.assertIsNone(row['context']['active_mission'])
        ended = self.store.end(child['id'])
        self.assertEqual(ended['observations'][0]['frame_id'], row['frame_id'])
