"""Continuous reasoning contracts without robot, GPU or dataset side effects."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from fastapi import HTTPException

from supervisor_api import mission_decisions as contract
from supervisor_api import reasoning as r
from supervisor_api import collection as c
from supervisor_api import qwen_annotation_workflow as q
from supervisor_api.reasoning_capture_context import context_for_frame
from supervisor_api.qwen_annotation_contract import prompt_for, sample_for, parse_target
from supervisor_api.qwen_training_balance import balance_frames

CATALOG = [{'mission_id': 'clean', 'mission_name': 'CleanTable', 'revision': 1},
           {'mission_id': 'serve', 'mission_name': 'ServeDish', 'revision': 1}]
JPEG = b'\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03' + b'\x00' * 10 + b'\xff\xd9'
FRAME = {'image_base64': base64.b64encode(JPEG).decode(), 'captured_at': 12,
         'frame_stamp': '12.0', 'camera_topic': '/head'}


def supervision(status='running'):
    return {'reasoning_task': 'mission_supervision', 'map_name': 'Home',
            'active_mission': {'mission_id': 'clean', 'status': status, 'step': 2, 'action': 'navigation'},
            'observation': {'phase': 'returning' if status == 'returning' else 'executing'},
            'candidate_mission_ids': ['serve']}


def target(action='pause', mid='clean'):
    return {'decision': action, 'mission_id': mid, 'reason': 'A person is crossing the route.'}


class ContractTests(unittest.TestCase):
    def test_tasks_never_inferred_from_active_mission_or_old_labels(self):
        legacy = {'active_mission': {'mission_id': 'clean', 'status': 'running'}}
        self.assertEqual(contract.task_for(legacy), 'mission_selection')
        for action in ('continue', 'cancel', 'pause'):
            with self.assertRaises(ValueError):
                parse_target(json.dumps(target(action)), legacy)

    def test_explicit_empty_scope_is_not_all_missions(self):
        context = {'candidate_mission_ids': [], 'allowed_mission_ids': ['clean']}
        self.assertEqual(contract.build_input(context, CATALOG)['missions'], [])
        with self.assertRaises(ValueError):
            contract.validate_target(target('start'), context, CATALOG)
        self.assertEqual(contract.candidates_for({'allowed_mission_ids': ['clean']}, {'missions': CATALOG}), [CATALOG[0]])

    def test_supervision_is_active_mission_only_in_all_runtime_states(self):
        for status in ('running', 'paused', 'returning'):
            context = supervision(status)
            self.assertEqual(contract.candidates_for(context, CATALOG), [CATALOG[0]])
            for action in (('pause', 'continue') if status == 'returning' else ('pause', 'continue', 'cancel')):
                contract.validate_target(target(action), context, CATALOG)
                with self.assertRaises(ValueError):
                    contract.validate_target(target(action, 'serve'), context, CATALOG)
            for action in (('start', 'wait', 'unknown', 'cancel') if status == 'returning' else ('start', 'wait', 'unknown')):
                with self.assertRaises(ValueError):
                    contract.validate_target(target(action), context, CATALOG)
        for context in ({'reasoning_task': 'mission_supervision'}, supervision('completed')):
            with self.assertRaises(ValueError):
                contract.build_input(context, CATALOG)

    def test_live_worker_and_training_use_identical_inputs(self):
        store = Mock(); store.catalog_provider.return_value = CATALOG
        service = r.ReasoningService(Mock(), Mock(), store)
        service.activity = Mock(); service.update(phase='ready')
        context = supervision('paused')
        result = {'finish_reason': 'stop', 'text': json.dumps(target('continue'))}
        with patch.object(r, 'capture_head', return_value=FRAME), patch.object(r, 'qwen_request', return_value=result) as generate:
            output = service.decide(r.MissionDecisionRequest(localized=True, **context))
        training = sample_for({'context': context}, {'missions': CATALOG}, target('continue'), 'images/a.jpg')
        self.assertEqual(generate.call_args.args[1]['system_prompt'], training['messages'][0]['content'][0]['text'])
        self.assertEqual(generate.call_args.args[1]['prompt'], prompt_for({'context': context}, CATALOG))
        self.assertEqual(generate.call_args.args[1]['prompt'], training['messages'][1]['content'][1]['text'])
        self.assertTrue(output['action_allowed'])
        self.assertEqual(output['reasoning_task'], 'mission_supervision')

    def test_transit_start_activity_is_a_contextual_proposal_not_dispatch(self):
        store = Mock(); store.catalog_provider.return_value = CATALOG
        context = {'map_name': 'Home', 'reasoning_task': 'mission_selection', 'active_mission': None,
                   'candidate_mission_ids': ['clean'], 'observation': {'phase': 'moving', 'target_spot_id': 'B'},
                   'runtime_source': 'autonomy', 'context_version': 5,
                   'context_valid_from': 11, 'context_valid_until': 13}
        service = r.ReasoningService(Mock(), Mock(), store, context_provider=lambda **_: context)
        service.activity = Mock(); service.update(phase='ready')
        with patch.object(r, 'capture_head', return_value=FRAME), patch.object(r, 'qwen_request', return_value={
                'finish_reason': 'stop', 'text': json.dumps(target('start'))}):
            service.decide(r.MissionDecisionRequest(map_name='Home', localized=True,
                observation=context['observation'], allowed_mission_ids=['clean']))
        event = service.activity.append.call_args.args[1]
        self.assertEqual(event['title'], 'AI · start proposal')
        self.assertEqual(event['reasoning_task'], 'mission_selection')
        self.assertEqual(event['phase'], 'moving')
        self.assertEqual(event['context_version'], 5)
        self.assertEqual(event['target_spot_id'], 'B')
        self.assertFalse(event['execution_dispatched'])
        self.assertNotIn('image_base64', event)

    def test_balance_never_silently_drops_supervision(self):
        def row(action):
            context = supervision() if action in ('pause', 'continue', 'cancel') else {}
            value = target(action, None if action in ('wait', 'unknown') else 'clean')
            return {'sample': contract.sample_for(context, CATALOG, value, 'a.jpg')}
        source = [row(a) for a in ('start', 'wait', 'wait', 'pause', 'continue', 'cancel', 'unknown')]
        selected, report = balance_frames(source, True)
        self.assertEqual(report['after']['decisions'], {'start': 1, 'wait': 1, 'pause': 1, 'continue': 1, 'cancel': 1, 'unknown': 1})
        self.assertEqual(report['supervision_retained'], 3)
        _, report = balance_frames([row('pause'), row('continue')], True)
        self.assertNotIn('error', report)


class CaptureContextTests(unittest.TestCase):
    def snapshot(self):
        return {**supervision('paused'), 'runtime_source': 'autonomy', 'context_version': 7,
                'context_valid_from': 11, 'context_valid_until': 13, 'localized': True,
                'mission_catalog_snapshot': {'missions': CATALOG},
                'plan_snapshot': {'revision': 2, 'spots': [{'id': 'table'}]}}

    def test_verified_server_snapshot_overrides_ui_but_no_runtime_preserves_preview(self):
        raw = {'map_name': 'Home', 'reasoning_task': 'mission_selection',
               'preview_active_mission': {'mission_id': 'serve'}}
        provider = Mock(return_value=self.snapshot())
        context = context_for_frame(raw, FRAME, provider)
        provider.assert_called_once_with(snapshot_time=12)
        self.assertEqual(context['reasoning_task'], 'mission_supervision')
        self.assertEqual(context['active_mission']['mission_id'], 'clean')
        context = context_for_frame(raw, FRAME, lambda **_: None)
        self.assertEqual(context['active_mission']['mission_id'], 'serve')
        self.assertEqual(context['runtime_source'], 'operator_supplied_preview')

    def test_record_runner_report_is_explicitly_client_reported_and_correlated(self):
        now = time.time() * 1000
        report = {'map_name': 'Home', 'mission_id': 'clean', 'status': 'paused', 'step': 2,
                  'action': 'navigation', 'context_version': 3,
                  'context_valid_from_ms': now - 2000, 'sampled_at_ms': now}
        frame = {**FRAME, 'captured_at': (now + 50) / 1000}
        context = context_for_frame({'map_name': 'Home', 'recording_execution': report}, frame, lambda **_: None)
        self.assertEqual(context['runtime_source'], 'record_runner_reported')
        self.assertEqual(context['reasoning_task'], 'mission_supervision')
        self.assertEqual(context['active_mission']['status'], 'paused')
        self.assertEqual(context['context_confidence'], 'client_reported')
        self.assertEqual(context['capture_report_delta_ms'], 50)
        for captured_at in ((now - 3000) / 1000, (now + 2000) / 1000):
            unmatched = context_for_frame({'map_name': 'Home', 'recording_execution': report},
                                          {**frame, 'captured_at': captured_at})
            self.assertEqual(unmatched['runtime_source'], 'unverified')
            self.assertIsNone(unmatched['active_mission'])
        missing_id = context_for_frame({'map_name': 'Home', 'recording_execution': {**report, 'mission_id': None}}, frame)
        self.assertEqual(missing_id['runtime_source'], 'unverified')
        self.assertIsNone(missing_id['active_mission'])
        # An invalid client report must never override valid server history.
        overridden = context_for_frame({'map_name': 'Home', 'recording_execution': {'mission_id': 'invented'}},
                                       FRAME, lambda **_: self.snapshot())
        self.assertEqual(overridden['runtime_source'], 'autonomy')
        with self.assertRaises(HTTPException):
            context_for_frame({'map_name': 'Other', 'recording_execution': report}, frame)

    def test_mismatched_future_or_missing_context_cannot_fabricate_runtime(self):
        for update in ({'context_valid_from': 14}, {'context_valid_until': 11}, {'context_version': None}):
            context = context_for_frame({'map_name': 'Home'}, FRAME, lambda **_: {**self.snapshot(), **update})
            self.assertEqual(context['runtime_source'], 'unverified')
            self.assertIsNone(context['active_mission'])
            self.assertEqual(context['candidate_mission_ids'], [])
        context = context_for_frame({'map_name': 'Home'}, FRAME, Mock(side_effect=RuntimeError('Unavailable')))
        self.assertEqual(context['runtime_source'], 'unverified')

    def test_transition_before_frame_capture_rejects_old_request(self):
        store = Mock(); store.catalog_provider.return_value = CATALOG
        service = r.ReasoningService(Mock(), Mock(), store, context_provider=lambda **_: self.snapshot())
        service.update(phase='ready')
        with patch.object(r, 'capture_head', return_value=FRAME), patch.object(r, 'qwen_request') as generate:
            with self.assertRaises(HTTPException) as caught:
                service.decide(r.MissionDecisionRequest(map_name='Home', localized=True))
        self.assertEqual(caught.exception.status_code, 409)
        generate.assert_not_called()

    def test_result_from_previous_runtime_version_is_not_executable(self):
        store = Mock(); store.catalog_provider.return_value = CATALOG
        calls = iter([self.snapshot(), {**self.snapshot(), 'context_version': 8}])
        service = r.ReasoningService(Mock(), Mock(), store, context_provider=lambda **_: next(calls))
        service.activity = Mock(); service.update(phase='ready')
        with patch.object(r, 'capture_head', return_value=FRAME), patch.object(r, 'qwen_request', return_value={
                'finish_reason': 'stop', 'text': json.dumps(target('continue'))}):
            result = service.decide(r.MissionDecisionRequest(localized=True, **supervision('paused')))
        self.assertFalse(result['action_allowed'])
        self.assertIn('context changed', result['error'])

    def test_running_session_input_and_saved_frame_keep_the_same_catalog_after_design_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            current = deepcopy(CATALOG)
            store = c.CollectionStore(folder, lambda _: current)
            episode = store.start(c.EpisodeRequest(map_name='Home'))
            snapshot = self.snapshot()
            service = r.ReasoningService(Mock(), Mock(), store, context_provider=lambda **_: snapshot)
            service.update(phase='ready'); service.activity = Mock()
            # Design changed after this session's catalog was frozen but before inference.
            current[0].update(revision=2, mission_name='Changed task', description='New task conditions')
            result = {'finish_reason': 'stop', 'text': json.dumps(target('continue'))}
            with patch.object(r, 'capture_head', return_value=FRAME), patch.object(r, 'qwen_request', return_value=result) as generate:
                live = service.decide(r.MissionDecisionRequest(localized=True, **supervision('paused')))
                preview = service.observe(r.ObserveRequest(context={'map_name': 'Home'}))
                saved = service.save_analyzed_frame(episode['id'], r.SaveAnalysisRequest(analysis_id=preview['analysis_id']))
                captured = service.capture_frame(episode['id'], r.CaptureRequest(context={'map_name': 'Home'}))
            self.assertTrue(live['action_allowed'])
            for call in generate.call_args_list:
                self.assertEqual(json.loads(call.args[1]['prompt'])['missions'], [CATALOG[0]])
            for frame in (saved['saved_frame'], captured['saved_frame']):
                self.assertEqual(frame['catalog']['missions'], CATALOG)
                self.assertEqual(frame['catalog_snapshot_id'], c.digest(CATALOG))
            self.assertEqual(saved['saved_frame']['capture']['source_catalog_snapshot_id'], c.digest(CATALOG))
            self.assertEqual(saved['saved_frame']['capture']['observed_live_catalog_snapshot_id'], c.digest(current))

    def test_deleted_candidate_in_session_snapshot_is_not_executable(self):
        snapshot = {**self.snapshot(), 'reasoning_task': 'mission_selection', 'active_mission': None,
                    'observation': {'phase': 'observing'}, 'candidate_mission_ids': ['clean']}
        store = Mock(); store.catalog_provider.return_value = [CATALOG[1]]
        service = r.ReasoningService(Mock(), Mock(), store, context_provider=lambda **_: snapshot)
        service.update(phase='ready'); service.activity = Mock()
        with patch.object(r, 'capture_head', return_value=FRAME), patch.object(r, 'qwen_request', return_value={
                'finish_reason': 'stop', 'text': json.dumps(target('start'))}):
            result = service.decide(r.MissionDecisionRequest(map_name='Home', localized=True))
        self.assertFalse(result['action_allowed'])
        self.assertIn('no longer', result['error'])

    def test_capture_and_export_preserve_plan_and_task_without_new_source_files(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = deepcopy(CATALOG)
            store = c.CollectionStore(folder, lambda _: catalog)
            ep = store.start(c.EpisodeRequest(map_name='Home', episode_type='mission', recording_mission_name='CleanTable'))
            service = r.ReasoningService(Mock(), Mock(), store, context_provider=lambda **_: self.snapshot())
            with patch.object(r, 'capture_head', return_value=FRAME):
                saved = service.capture_frame(ep['id'], r.CaptureRequest(context={'map_name': 'Home'}))
            frame = saved['saved_frame']
            self.assertEqual(frame['context']['context_version'], 7)
            self.assertEqual(frame['context']['active_mission']['status'], 'paused')
            self.assertEqual(frame['catalog']['missions'], CATALOG)
            self.assertEqual(len(saved['episode']['observation_plans']), 1)
            store.annotate(ep['id'], c.AnnotationRequest(frame_id=frame['frame_id'], **target(),
                annotator='human', review_status='approved'))
            store.end(ep['id'])
            catalog[0]['mission_name'] = 'Renamed task'
            output, _ = store._training_files(store.get(ep['id']))
            samples = output['train.jsonl'] + output['val.jsonl'] + output['test.jsonl']
            user = json.loads(samples[0]['messages'][1]['content'][1]['text'])
            self.assertEqual(user['context']['reasoning_task'], 'mission_supervision')
            self.assertEqual(user['missions'][0]['mission_name'], 'CleanTable')
            presets = store.reason_presets('Home')
            self.assertEqual(presets[0]['reasoning_task'], 'mission_supervision')
            self.assertEqual(presets[0]['mission_id'], 'clean')
            directory = Path(ep['storage_path'])
            self.assertEqual({p.name for p in directory.iterdir() if p.is_file()}, {'episode.json', 'samples.jsonl'})


class SupervisionWorkflowTests(unittest.TestCase):
    def test_batch_supervision_saves_draft_then_trains_with_same_task(self):
        with tempfile.TemporaryDirectory() as folder:
            store = c.CollectionStore(Path(folder) / 'data', lambda _: CATALOG)
            ep = store.start(c.EpisodeRequest(map_name='Home', episode_type='mission', recording_mission_name='CleanTable'))
            frames = [store.capture(ep['id'], FRAME, supervision(status)) for status in ('running', 'paused')]
            store.end(ep['id'])
            workflow = q.QwenAnnotationWorkflow(store, Mock(), Path(folder) / 'models', recover=False)
            model = {'schema': q.SCHEMA, 'id': 'a'*32, 'map_name': 'Home', 'model_name': 'Supervisor',
                     'catalog_snapshot_id': c.digest(CATALOG)}
            c.write_json(workflow.root / 'Home' / 'Supervisor' / 'model.json', model)
            with patch.object(q.threading, 'Thread'):
                job = workflow.batch(q.BatchRequest(model_id=model['id'], episode_ids=[ep['id']]))
            request = c.read_json(Path(workflow.jobs['batch']['request_path']))
            for row in request['frames']:
                self.assertEqual(workflow._prediction(row, {'text': json.dumps(target())}, job['id']), 'created')
                label = next(a for a in store.get(ep['id'])['annotations'] if a['frame_ids'] == [row['frame_id']])
                self.assertEqual(label['reasoning_task'], 'mission_supervision')
                self.assertEqual(label['review_status'], 'draft')
                store.annotate(ep['id'], c.AnnotationRequest(frame_id=row['frame_id'], **target(),
                    annotator='human', review_status='approved', expected_revision=1))
            workflow.jobs['batch']['status'] = 'completed'
            with patch.object(q.threading, 'Thread'):
                trained = workflow.train_reasoning(q.ReasoningTrainRequest(map_name='Home', episode_ids=[ep['id']],
                    model_name='Supervisor v2', balance_decisions=True))
            training = c.read_json(Path(workflow.jobs['reasoning_training']['request_path']))
            self.assertEqual(trained['reasoning_tasks'], ['mission_supervision'])
            self.assertEqual(trained['task_sample_counts'], {'mission_selection': 0, 'patrol_supervision': 0, 'mission_supervision': 2})
            self.assertEqual(trained['supervision_decision_counts'], {'pause': 2, 'continue': 0, 'cancel': 0})
            self.assertEqual(trained['training_catalog_snapshot_ids'], [c.digest(CATALOG)])
            self.assertEqual(trained['input_policy'], contract.PROMPT_VERSION)
            self.assertEqual(training['sampling']['after']['decisions'], {'pause': 2})
            for row in training['frames']:
                messages = row['sample']['messages']
                self.assertEqual(messages[0]['content'][0]['text'], contract.SUPERVISION_SYSTEM)
                self.assertEqual(json.loads(messages[2]['content'][0]['text'])['decision'], 'pause')
