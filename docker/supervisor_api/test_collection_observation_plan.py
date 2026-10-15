"""The Record observation plan is the immutable source of collection context."""
from copy import deepcopy
import unittest
from fastapi import HTTPException
from supervisor_api import collection as c
from supervisor_api.test_collection_episode_contract import ThreeTypeCaptureTests, VIEW, CATALOG


class ObservationPlanBindingTests(unittest.TestCase):
    report = ThreeTypeCaptureTests.report
    context = ThreeTypeCaptureTests.context
    def setUp(self):
        ThreeTypeCaptureTests.setUp(self)
        self.config = deepcopy(VIEW)
        self.config['plan']['revision'] = 7
        self.config['plan']['excluded_mission_ids'] = []
        self.view.stop()
        from unittest.mock import patch
        self.view = patch('supervisor_api.autonomy_config.AutonomyRepository.view', return_value=self.config)
        self.view.start()
        self.addCleanup(self.view.stop)

    def bound(self, kind='observation', **changes):
        fields = dict(map_name='Home', episode_type=kind, capture_contract='task_aligned_v1', plan_revision=7)
        if kind == 'patrol': fields['patrol_spot_ids'] = ['table', 'laundry']
        else: fields['observation_spot_id'] = 'table'
        if kind == 'mission': fields.update(recording_mission_name='CleanTable', recording_mission_id='clean')
        return self.store.start(c.EpisodeRequest(**{**fields, **changes}))

    def test_mission_uses_its_source_observation_as_return_target(self):
        source = self.bound('mission')
        self.assertEqual(source['source_observation_spot_id'], 'table')
        self.assertEqual(source['return_observation_spot_id'], 'table')
        self.assertEqual(source['candidate_mission_ids'], ['clean'])
        child = self.store.start_return(source['id'])
        self.assertEqual(child['return_observation_spot_id'], 'table')
        self.assertEqual(child['source_observation_spot_id'], 'table')
        self.assertEqual(child['plan_revision'], 7)
        self.assertEqual(child['catalog'], source['catalog'])

    def test_invalid_binding_is_rejected_before_creating_data(self):
        for changes in ({'observation_spot_id': None}, {'observation_spot_id': 'laundry'},
                        {'return_observation_spot_id': 'laundry'}, {'recording_mission_id': 'changed'},
                        {'plan_revision': 6}):
            with self.subTest(changes=changes), self.assertRaises(HTTPException):
                self.bound('mission', **changes)
        self.assertEqual(self.store.list(), [])

    def test_same_mission_can_return_to_different_trigger_observations(self):
        self.config['plan']['observations'][1]['mission_ids'] = ['clean']
        first = self.bound('mission')
        second = self.bound('mission', observation_spot_id='laundry')
        self.assertEqual(first['return_observation_spot_id'], 'table')
        self.assertEqual(second['return_observation_spot_id'], 'laundry')

    def test_plan_and_candidate_snapshot_stay_fixed_after_plan_edits(self):
        ep = self.bound()
        self.config['plan']['revision'] = 8
        self.config['plan']['observations'][0]['mission_ids'] = []
        self.store.catalog_provider = lambda _: [{'mission_id': 'other', 'mission_name': 'DifferentMission', 'revision': 1}]
        frame = self.store.capture(ep['id'], self.frame, {**self.context('observation'), 'candidate_mission_ids': ['other']})
        self.assertEqual(frame['context']['candidate_mission_ids'], ['clean'])
        self.assertEqual(frame['context']['plan_revision'], 7)
        saved = self.store.get(ep['id'])
        self.assertEqual(saved['observation_plans'][frame['context']['plan_snapshot_id']]['revision'], 7)
        self.store.annotate(ep['id'], c.AnnotationRequest(frame_id=frame['frame_id'], decision='start', mission_id='clean',
            reason='The dishes need clearing.', annotator='human', review_status='approved'))
        with self.assertRaises(HTTPException):
            self.store.annotate(ep['id'], c.AnnotationRequest(frame_id=frame['frame_id'], decision='start', mission_id='other',
                reason='Wrong candidate.', annotator='human', review_status='approved'))

    def test_patrol_requires_saved_plan_membership(self):
        self.config['spots'].append({'id': 'extra', 'metadata': {'role': 'observation'}})
        with self.assertRaises(HTTPException): self.bound('patrol', patrol_spot_ids=['table', 'extra'])
        ep = self.bound('patrol')
        self.assertEqual(ep['recording_plan']['revision'], 7)

    def test_empty_candidate_observation_can_collect_wait_training(self):
        ep = self.bound(observation_spot_id='laundry')
        frame = self.store.capture(ep['id'], self.frame, self.context('observation', observation_spot_id='laundry'))
        self.assertEqual(frame['context']['candidate_mission_ids'], [])
        self.store.annotate(ep['id'], c.AnnotationRequest(frame_id=frame['frame_id'], decision='wait',
            reason='No task is available here.', annotator='human', review_status='approved'))


    def test_patrol_arrival_posture_keeps_its_previous_stop(self):
        ep = self.bound('patrol')
        frame = self.store.capture(ep['id'], self.frame, self.context('patrol', phase='settling'))
        self.assertEqual(frame['context']['observation']['from_spot_id'], 'table')
        self.assertEqual(frame['context']['observation']['at_spot_id'], 'laundry')
