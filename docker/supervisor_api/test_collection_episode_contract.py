"""Three capture roles remain homogeneous from Record reports to saved labels."""
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from supervisor_api import collection as c
from supervisor_api.collection_episode_contract import frame_matches_episode
from supervisor_api.reasoning_capture_context import context_for_frame

JPEG = b'\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03' + b'\x00' * 10 + b'\xff\xd9'
CATALOG = [{'mission_id': 'clean', 'mission_name': 'CleanTable', 'revision': 1}]
VIEW = {'plan': {'observations': [{'spot_id': 'table', 'mission_ids': ['clean']},
                                {'spot_id': 'laundry', 'mission_ids': []}]},
        'spots': [{'id': 'table', 'label': 'Table'}, {'id': 'laundry', 'label': 'Laundry'}]}


class ThreeTypeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = c.CollectionStore(Path(self.temp.name), lambda _: CATALOG)
        self.view = patch('supervisor_api.autonomy_config.AutonomyRepository.view', return_value=VIEW)
        self.view.start()
        self.addCleanup(self.view.stop)
        self.stamp = time.time() * 1000
        self.frame = {'image_base64': base64.b64encode(JPEG).decode(), 'captured_at': self.stamp / 1000,
                      'frame_stamp': '100.01', 'camera_topic': '/head'}

    def start(self, kind):
        details = {'observation': {'observation_spot_id': 'table'},
                   'patrol': {'patrol_spot_ids': ['table', 'laundry']},
                   'mission': {'recording_mission_name': 'CleanTable'}}[kind]
        return self.store.start(c.EpisodeRequest(map_name='Home', episode_type=kind,
                                                capture_contract='task_aligned_v1', **details))

    def report(self, kind, **changes):
        details = {
            'observation': dict(reasoning_task='mission_selection', observation_spot_id='table',
                                status='arrived', localized=True),
            'patrol': dict(reasoning_task='patrol_supervision', from_spot_id='table', to_spot_id='laundry', status='running'),
            'mission': dict(reasoning_task='mission_supervision', mission_id='clean', status='running', step=1, action='navigation'),
        }[kind]
        return {**dict(map_name='Home', context_version=2, sampled_at_ms=self.stamp,
                       context_valid_from_ms=self.stamp - 500), **details, **changes}

    def context(self, kind, **changes):
        return context_for_frame({'map_name': 'Home', 'candidate_mission_ids': ['clean'],
                                  'recording_execution': self.report(kind, **changes)}, self.frame)

    def test_three_folders_share_numbers_and_support_lookup_delete_reuse(self):
        episodes = [self.start(kind) for kind in ('observation', 'patrol', 'mission')]
        for i, (kind, ep) in enumerate(zip(('observation', 'patrol', 'mission'), episodes), 1):
            path = Path(ep['storage_path'])
            self.assertEqual(path.parent.name, kind)
            self.assertEqual(path.name, f'raw_episode_{i:04d}')
            self.assertEqual(self.store.get(ep['id'])['schema_version'], 'reasoning_episode_v4')
        self.assertEqual(len(self.store.list()), 3)
        self.store.end(episodes[1]['id'])
        self.store.delete(episodes[1]['id'])
        self.assertEqual(Path(self.start('observation')['storage_path']).name, 'raw_episode_0002')

    def test_new_contract_requires_type_and_role_specific_target(self):
        for request in (
            dict(), dict(episode_type='observation'), dict(episode_type='patrol'),
            dict(episode_type='patrol', patrol_spot_ids=['table', 'table']),
            dict(episode_type='patrol', patrol_spot_ids=['table', 'missing']),
            dict(episode_type='mission'),
        ):
            with self.subTest(request=request), self.assertRaises(HTTPException):
                self.store.start(c.EpisodeRequest(map_name='Home', capture_contract='task_aligned_v1', **request))
        self.assertEqual(self.store.list(), [])

    def test_observation_requires_confirmed_arrival_to_selected_spot(self):
        ep = self.start('observation')
        with self.assertRaises(HTTPException):
            self.context('observation', status='moving')
        for context in (self.context('patrol'), self.context('observation', observation_spot_id='laundry'),
                        {'reasoning_task': 'mission_selection', 'map_name': 'Home', 'localized': True}):
            with self.subTest(context=context), self.assertRaises(HTTPException):
                self.store.capture(ep['id'], self.frame, context)
        self.assertEqual(self.store.get(ep['id'])['observations'], [])
        context = {**self.context('observation'), 'candidate_mission_ids': ['invented']}
        row = self.store.capture(ep['id'], self.frame, context)
        self.assertEqual(row['context']['candidate_mission_ids'], ['clean'])
        self.assertEqual(row['episode_type'], 'observation')
        self.assertEqual(row['context']['observation']['at_spot_id'], 'table')
        self.assertEqual(self.store.get(ep['id'])['reasoning_frame_count'], 1)

    def test_paused_patrol_preserves_task_and_cannot_accept_mission_selection(self):
        ep = self.start('patrol')
        row = self.store.capture(ep['id'], self.frame, self.context('patrol', status='paused'))
        self.assertEqual(row['context']['reasoning_task'], 'patrol_supervision')
        self.assertEqual(row['context']['observation']['status'], 'paused')
        self.assertIsNone(row['context']['active_mission'])
        self.store.annotate(ep['id'], c.AnnotationRequest(frame_id=row['frame_id'], decision='pause',
            reason='A person blocks the path.', annotator='tester', review_status='approved'))
        with self.assertRaises(HTTPException):
            self.store.capture(ep['id'], self.frame, self.context('observation'))
        with self.assertRaises(HTTPException):
            self.store.capture(ep['id'], self.frame, self.context('patrol', to_spot_id='other'))
        self.store.end(ep['id'])
        self.assertEqual(self.store.reason_presets('Home')[0]['reasoning_task'], 'patrol_supervision')
        self.assertEqual(self.store.list()[0]['reasoning_approved_count'], 1)

    def test_single_target_approach_is_patrol_without_an_invented_source(self):
        ep = self.store.start(c.EpisodeRequest(map_name='Home', episode_type='patrol',
            capture_contract='task_aligned_v1', patrol_spot_ids=['table']))
        context = self.context('patrol', phase='approaching', from_spot_id=None, to_spot_id='table')
        row = self.store.capture(ep['id'], self.frame, context)
        self.assertEqual(row['context']['observation']['phase'], 'approaching')
        self.assertIsNone(row['context']['observation']['from_spot_id'])
        self.store.annotate(ep['id'], c.AnnotationRequest(frame_id=row['frame_id'], decision='continue',
            reason='The route is clear.', annotator='tester', review_status='approved'))
        self.assertEqual(self.store.get(ep['id'])['reasoning_approved_count'], 1)
        for changes in ({'phase': 'moving'}, {'from_spot_id': 'laundry'}):
            with self.subTest(changes=changes), self.assertRaises(HTTPException):
                self.context('patrol', **{'phase': 'approaching', 'from_spot_id': None, 'to_spot_id': 'table', **changes})

    def test_capture_can_begin_mid_return_without_inventing_a_mission_episode(self):
        ep = self.store.start(c.EpisodeRequest(map_name='Home', episode_type='patrol',
            capture_contract='task_aligned_v1', patrol_spot_ids=['table']))
        row = self.store.capture(ep['id'], self.frame,
            self.context('patrol', phase='returning', from_spot_id=None, to_spot_id='table'))
        self.assertEqual(row['context']['observation']['phase'], 'returning')
        self.assertEqual(ep['episode_type'], 'patrol')
        self.assertNotIn('source_episode_id', ep)
        self.assertNotIn('return_mission_id', ep)

    def test_mission_requires_actual_selected_mission_while_paused_capture_continues(self):
        ep = self.start('mission')
        row = self.store.capture(ep['id'], self.frame, self.context('mission', status='paused'))
        self.assertEqual(row['context']['active_mission']['status'], 'paused')
        for context in (self.context('mission', mission_id='other'), self.context('patrol'),
                        self.context('mission', mission_id=None)):
            with self.subTest(context=context), self.assertRaises(HTTPException):
                self.store.capture(ep['id'], self.frame, context)
        self.assertEqual(len(self.store.get(ep['id'])['observations']), 1)

    def test_unmatched_report_never_becomes_observation_training_data(self):
        ep = self.start('patrol')
        context = self.context('patrol', context_valid_from_ms=self.stamp + 1)
        self.assertEqual(context['reasoning_task'], 'patrol_supervision')
        self.assertEqual(context['runtime_source'], 'unverified')
        with self.assertRaises(HTTPException):
            self.store.capture(ep['id'], self.frame, context)
        self.assertFalse(frame_matches_episode({'episode_type': 'observation'}, {'context': context}))

    def test_server_at_capture_overrides_reported_role(self):
        ep = self.start('observation')
        snapshot = dict(map_name='Home', reasoning_task='patrol_supervision', context_version=9,
                        active_mission=None, observation={'phase': 'moving', 'from_spot_id': 'table', 'target_spot_id': 'laundry'},
                        context_valid_from=self.stamp / 1000 - 1, context_valid_until=self.stamp / 1000 + 1)
        context = context_for_frame({'map_name': 'Home', 'recording_execution': self.report('observation')},
                                    self.frame, lambda **_: snapshot)
        self.assertEqual(context['source'], 'server_at_capture')
        with self.assertRaises(HTTPException):
            self.store.capture(ep['id'], self.frame, context)

    def test_legacy_mixed_frames_remain_readable_but_not_in_reasoning_counts(self):
        ep = self.store.start(c.EpisodeRequest(map_name='Home'))
        valid = self.store.capture(ep['id'], self.frame, {'reasoning_task': 'mission_selection'})
        mixed = self.store.capture(ep['id'], self.frame, self.context('mission'))
        for frame, decision, mid in ((valid, 'wait', None), (mixed, 'continue', 'clean')):
            self.store.annotate(ep['id'], c.AnnotationRequest(frame_id=frame['frame_id'], decision=decision,
                mission_id=mid, reason='Visible evidence.', annotator='tester', review_status='approved'))
        stored = self.store.episode(ep['id'])
        stored['qwen_sample'] = {'schema_version': 'user_count_v1', 'frame_ids': [valid['frame_id'], mixed['frame_id']]}
        c.write_json(self.store.path('raw', ep['id'], 'episode.json'), stored)
        for view in (self.store.get(ep['id']), self.store.list()[0]):
            self.assertEqual(view['reasoning_frame_count'], 1)
            self.assertEqual(view['reasoning_approved_count'], 1)
            self.assertEqual(view['qwen_sample_eligible_count'], 1)
            self.assertEqual(view['qwen_sample_approved_count'], 1)
        self.assertEqual(len(self.store.get(ep['id'])['observations']), 2)

    def test_report_rejects_invalid_patrol_targets_and_observation_localization(self):
        for kind, changes in (('patrol', {'to_spot_id': 'table'}), ('patrol', {'mission_id': 'clean'}),
                              ('patrol', {'status': 'idle'}), ('observation', {'localized': False})):
            with self.subTest(kind=kind, changes=changes), self.assertRaises(HTTPException):
                self.context(kind, **changes)

    def test_legacy_return_frames_remain_readable_but_are_not_mission_training(self):
        ep = self.store.start(c.EpisodeRequest(map_name='Home', episode_type='mission', recording_mission_name='CleanTable'))
        contexts = [
            {**self.context('mission'), 'active_mission': {'mission_id': 'clean', 'status': 'returning'}},
            {**self.context('mission'), 'active_mission': {'mission_id': 'clean', 'status': 'paused', 'action': 'return-navigation'}},
            {**self.context('mission'), 'active_mission': {'mission_id': 'clean', 'status': 'running', 'action': 'return-posture'}},
            {**self.context('mission'), 'observation': {'phase': 'returning'}},
        ]
        for context in contexts:
            row = self.store.capture(ep['id'], self.frame, context)
            self.assertFalse(frame_matches_episode(ep, row))
        saved = self.store.get(ep['id'])
        self.assertEqual(len(saved['observations']), 4)
        self.assertEqual(saved['reasoning_frame_count'], 0)
        self.assertTrue(frame_matches_episode(ep, {'context': self.context('mission', status='paused')}))
