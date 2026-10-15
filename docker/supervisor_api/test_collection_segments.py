"""Execution changes split independent Record captures without losing data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from supervisor_api import collection as c, collection_segments as segments
from supervisor_api.test_collection_episode_contract import CATALOG, VIEW


class CollectionSegmentsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = c.CollectionStore(Path(self.tmp.name), lambda _: CATALOG)
        self.config = deepcopy(VIEW)
        self.config['plan']['revision'] = 4
        self.view = patch('supervisor_api.autonomy_config.AutonomyRepository.view', return_value=self.config)
        self.view.start(); self.addCleanup(self.view.stop)
        self.source = self.store.start(self.request('patrol'))

    def request(self, kind, **changes):
        fields = dict(map_name='Home', episode_type=kind, capture_contract='task_aligned_v1',
                      plan_revision=4, capture_mode='continuous', capture_interval_ms=500)
        if kind == 'patrol': fields['patrol_spot_ids'] = ['table', 'laundry']
        else: fields['observation_spot_id'] = 'table'
        if kind == 'mission': fields.update(recording_mission_name='CleanTable', recording_mission_id='clean')
        return c.EpisodeRequest(**{**fields, **changes})

    def test_http_segment_links_roles_in_one_session_and_retry_is_idempotent(self):
        api = FastAPI(); api.include_router(c.create_router(self.store)); client = TestClient(api)
        request = self.request('observation').model_dump()
        path = f"/collection/episodes/{self.source['id']}/segment"
        response = client.post(path, json=request)
        self.assertEqual(response.status_code, 200, response.text)
        child = response.json()
        self.assertEqual(child['previous_episode_id'], self.source['id'])
        self.assertEqual(child['group_id'], self.source['group_id'])
        self.assertEqual(child['episode_type'], 'observation')
        self.assertEqual(child['candidate_mission_ids'], ['clean'])
        self.assertEqual(child['capture_mode'], 'continuous')
        self.assertEqual(child['capture_interval_ms'], 500)
        self.assertIsNone(child['ended_at'])
        source = self.store.get(self.source['id'])
        self.assertEqual(source['next_episode_id'], child['id'])
        self.assertEqual(source['ended_at'], child['started_at'])
        self.assertEqual(client.post(path, json=request).json()['id'], child['id'])
        self.assertEqual(client.post(path, json=self.request('mission').model_dump()).status_code, 409)
        self.assertEqual(len(self.store.list()), 2)

    def test_mission_return_then_observation_preserves_group_and_pinned_target(self):
        mission = self.store.start_segment(self.source['id'], self.request('mission'))
        returned = self.store.start_return(mission['id'])
        observed = self.store.start_segment(returned['id'], self.request('observation'))
        self.assertEqual(returned['capture_segment'], 'mission_return')
        self.assertEqual(returned['return_observation_spot_id'], 'table')
        self.assertEqual(observed['previous_episode_id'], returned['id'])
        self.assertEqual({ep['group_id'] for ep in self.store.list()}, {self.source['group_id']})
        self.assertEqual(len(self.store.list()), 4)
        with self.assertRaises(HTTPException):
            self.store.start_segment(mission['id'], self.request('patrol'))

    def test_validation_failure_keeps_source_open_without_orphan_child(self):
        for changes in ({'map_name': 'Other'}, {'plan_revision': 3}, {'observation_spot_id': 'missing'},
                        {'group_id': 'other-session'}, {'capture_contract': None}):
            with self.subTest(changes=changes), self.assertRaises(HTTPException):
                self.store.start_segment(self.source['id'], self.request('observation', **changes))
            self.assertIsNone(self.store.get(self.source['id'])['ended_at'])
            self.assertEqual(len(self.store.list()), 1)
        self.store.end(self.source['id'])
        with self.assertRaises(HTTPException):
            self.store.start_segment(self.source['id'], self.request('observation'))

    def test_metadata_is_inherited_from_the_recording_session(self):
        source = self.store.start(self.request('patrol', robot_type='ffw_sg2', environment='simulation'))
        child = self.store.start_segment(source['id'], self.request('observation', robot_type='different', environment='real'))
        self.assertEqual(child['robot_type'], 'ffw_sg2')
        self.assertEqual(child['environment'], 'simulation')

    def test_failed_source_link_recovers_published_child_without_duplicate(self):
        original = segments.write_json
        source_path = self.store.path('raw', self.source['id'], 'episode.json')
        def fail(path, value):
            if path == source_path: raise OSError('disk full')
            original(path, value)
        with patch.object(segments, 'write_json', side_effect=fail), self.assertRaises(HTTPException) as caught:
            self.store.start_segment(self.source['id'], self.request('observation'))
        self.assertEqual(caught.exception.status_code, 507)
        self.assertIsNone(self.store.get(self.source['id'])['ended_at'])
        published = next(ep for ep in self.store.list() if ep.get('previous_episode_id'))
        retry = self.store.start_segment(self.source['id'], self.request('observation'))
        self.assertEqual(retry['id'], published['id'])
        self.assertEqual(len(self.store.list()), 2)

    def test_failed_segment_publication_cannot_be_branched_via_return(self):
        mission = self.store.start_segment(self.source['id'], self.request('mission'))
        with patch.object(segments, 'write_json', side_effect=OSError('disk full')), self.assertRaises(HTTPException):
            self.store.start_segment(mission['id'], self.request('observation'))
        with self.assertRaises(HTTPException) as caught:
            self.store.start_return(mission['id'])
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(len(self.store.list()), 3)
        saved = self.store.start_segment(mission['id'], self.request('observation'))
        self.assertEqual(self.store.get(mission['id'])['next_episode_id'], saved['id'])

    def test_deleted_child_is_not_recreated_and_return_cannot_branch_after_segment(self):
        child = self.store.start_segment(self.source['id'], self.request('mission'))
        observed = self.store.start_segment(child['id'], self.request('observation'))
        with self.assertRaises(HTTPException): self.store.start_return(child['id'])
        self.store.end(observed['id']); self.store.delete(observed['id'])
        with self.assertRaises(HTTPException) as caught:
            self.store.start_segment(child['id'], self.request('observation'))
        self.assertEqual(caught.exception.status_code, 410)
        self.assertEqual(len(self.store.list()), 2)
