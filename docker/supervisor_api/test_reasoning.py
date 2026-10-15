"""Observation bridge tests: no Docker, GPU, camera or robot required."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location('reasoning_under_test', Path(__file__).with_name('reasoning.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ReasoningTests(unittest.TestCase):
    def setUp(self):
        self.start_container = Mock()
        self.stop_container = Mock()
        store = Mock()
        store.catalog_provider.return_value = [{'mission_id': 'm2', 'mission_name': 'Mission2'}]
        self.service = r.ReasoningService(self.start_container, self.stop_container, store)
        activity_dir = tempfile.TemporaryDirectory()
        self.addCleanup(activity_dir.cleanup)
        self.service.activity = r.OperateActivity(activity_dir.name)
        self.service.update(phase='ready')
        self.frame = {'frame_stamp': '12.3', 'image_base64': 'abc', 'captured_at': 42, 'camera_topic': '/head'}

    def test_wait_with_start_in_reason_never_becomes_start(self):
        import json
        text = json.dumps({'decision': 'wait', 'mission_id': None,
                           'reason': 'The scene supports starting Mission2; the decision is start.'})
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(
                r, 'qwen_request', return_value={'finish_reason': 'stop', 'text': text}):
            output = self.service.decide(r.MissionDecisionRequest(map_name='Home', localized=True))
        self.assertEqual(output['decision']['decision'], 'wait')
        self.assertIsNone(output['selected_mission'])
        event = self.service.activity.recent('Home')[-1]
        self.assertEqual(event['raw_response'], text)
        self.assertEqual(event['prompt_version'], r.PROMPT_VERSION)

    def test_observation_scope_rejects_out_of_scope_mission(self):
        import json
        self.service.collection_store.catalog_provider.return_value = [
            {'mission_id': 'm2', 'mission_name': 'Mission2'},
            {'mission_id': 'm3', 'mission_name': 'Mission3'}]
        result = {'finish_reason': 'stop', 'text': json.dumps(
            {'decision': 'start', 'mission_id': 'm3', 'reason': 'Ready'})}
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value=result) as request:
            output = self.service.decide(r.MissionDecisionRequest(map_name='Home', localized=True, allowed_mission_ids=['m2']))
        self.assertFalse(output['action_allowed'])
        self.assertTrue(output['error'])
        self.assertEqual([m['mission_id'] for m in json.loads(request.call_args.args[1]['prompt'])['missions']], ['m2'])

    def test_record_preview_resolves_candidate_missions_from_the_saved_observation_plan(self):
        import json, time
        stamp = time.time()
        self.frame['captured_at'] = stamp
        self.service.collection_store.catalog_provider.return_value = [
            {'mission_id': 'm2', 'mission_name': 'Mission2'}, {'mission_id': 'm3', 'mission_name': 'OtherSpotMission'}]
        self.service.collection_store.observation_view.return_value = {'plan': {'revision': 4,
            'observations': [{'spot_id': 'table', 'mission_ids': ['m2']}]}, 'spots': [{'id': 'table'}]}
        report = {'map_name': 'Home', 'reasoning_task': 'mission_selection', 'observation_spot_id': 'table',
                  'status': 'arrived', 'localized': True, 'context_version': 1,
                  'context_valid_from_ms': stamp * 1000 - 100, 'sampled_at_ms': stamp * 1000}
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value={
                'finish_reason': 'stop', 'text': json.dumps({'decision': 'wait', 'mission_id': None, 'reason': 'Wait.'})}) as request:
            self.service.observe(r.ObserveRequest(context={'map_name': 'Home', 'localized': True, 'recording_execution': report}))
        content = json.loads(request.call_args.args[1]['prompt'])
        self.assertEqual([mission['mission_id'] for mission in content['missions']], ['m2'])

    def test_frame_captured_during_navigation_cannot_trigger_a_task(self):
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request') as request:
            with self.assertRaises(HTTPException) as caught:
                self.service.decide(r.MissionDecisionRequest(map_name='Home', localized=True, frame_captured_after=43))
        self.assertEqual(caught.exception.status_code, 409)
        request.assert_not_called()

    def test_prepare_loads_only_when_needed(self):
        self.service.prepare_lock.acquire()
        with patch.object(r, 'qwen_request', side_effect=[{'ready': False}, {}, {'ready': True, 'model': 'Qwen'}]) as request:
            self.service.prepare()
        self.start_container.assert_called_once()
        self.assertEqual(self.service.status()['phase'], 'ready')
        self.assertEqual(request.call_args_list[1].args, ('/load', {}))
        self.assertFalse(self.service.prepare_lock.locked())

    def test_preparation_failure_is_visible_and_retryable(self):
        self.start_container.side_effect = RuntimeError('Docker unavailable')
        self.service.prepare_lock.acquire()
        self.service.prepare()
        self.assertEqual(self.service.status()['error'], 'Docker unavailable')
        self.assertFalse(self.service.prepare_lock.locked())

    def test_start_deduplicates_loading(self):
        with patch.object(r.threading, 'Thread') as thread:
            self.service.start()
            self.service.start()
            thread.assert_called_once()
        self.service.prepare_lock.release()

    def test_observe_uses_actual_frame_and_preserves_incomplete_response(self):
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value={'text':'장면', 'finish_reason':'length'}) as request:
            output = self.service.observe(r.ObserveRequest(context={'map_name':'warehouse'}))
        self.assertEqual(output['frame_stamp'], '12.3')
        self.assertEqual(output['finish_reason'], 'length')
        self.assertEqual(request.call_args.args[1]['image_base64'], 'abc')
        self.assertFalse(request.call_args.args[1]['enable_thinking'])

    def test_missing_or_repeated_frame_does_not_infer(self):
        for capture in [Mock(side_effect=HTTPException(503, 'No camera')), Mock(return_value=self.frame)]:
            with patch.object(r, 'capture_head', capture), patch.object(r, 'qwen_request') as request:
                with self.assertRaises(HTTPException):
                    self.service.observe(r.ObserveRequest(previous_frame_stamp='12.3', context={'map_name':'warehouse'}))
                request.assert_not_called()
                self.assertFalse(self.service.observe_lock.locked())

    def test_parallel_observation_is_rejected(self):
        self.service.observe_lock.acquire()
        with patch.object(r, 'capture_head') as capture:
            with self.assertRaises(HTTPException) as caught:
                self.service.observe(r.ObserveRequest(context={'map_name':'warehouse'}))
            self.assertEqual(caught.exception.status_code, 409)
            capture.assert_not_called()
        self.service.observe_lock.release()

    def test_routes_are_observation_only_and_validate_input(self):
        app = FastAPI(); app.include_router(r.create_router(Mock(), Mock()))
        client = TestClient(app)
        self.assertEqual(client.get('/reasoning/status').json()['phase'], 'idle')
        self.assertEqual(client.post('/reasoning/observe', json={'previous_frame_stamp':'x'*101}).status_code, 422)
        self.assertEqual(client.post('/reasoning/execute').status_code, 404)

    def test_stop_invalidates_inflight_load_and_does_not_restore_ready(self):
        entered = r.threading.Event(); release = r.threading.Event()
        def request(path, *args, **kwargs):
            if path == '/load':
                entered.set(); self.assertTrue(release.wait(3)); return {}
            return {'ready': False}
        self.service.update(phase='idle')
        with patch.object(r, 'qwen_request', side_effect=request):
            self.service.start()
            self.assertTrue(entered.wait(3))
            self.assertEqual(self.service.stop()['phase'], 'stopped')
            self.stop_container.assert_called_once()
            release.set()
            self.assertTrue(self.service.prepare_lock.acquire(timeout=3))
            self.service.prepare_lock.release()
        self.assertEqual(self.service.status()['phase'], 'stopped')

    def test_stop_waits_for_container_creation_then_stops_it(self):
        entered = r.threading.Event(); release = r.threading.Event(); calls = []
        def start_container():
            entered.set(); release.wait(3); calls.append('start')
        self.service.start_container = start_container
        self.service.stop_container = lambda: calls.append('stop')
        self.service.update(phase='idle')
        with patch.object(r, 'qwen_request', return_value={'ready': True}):
            self.service.start(); self.assertTrue(entered.wait(3))
            worker = r.threading.Thread(target=self.service.stop); worker.start()
            deadline = r.time.monotonic() + 3
            while self.service.status()['phase'] != 'stopping' and r.time.monotonic() < deadline:
                r.time.sleep(.001)
            with self.assertRaises(HTTPException): self.service.start()
            release.set(); worker.join(3)
            self.assertFalse(worker.is_alive())
        self.assertEqual(calls, ['start', 'stop'])
        self.assertEqual(self.service.status()['phase'], 'stopped')

    def test_stop_failure_is_visible_and_can_be_retried(self):
        self.stop_container.side_effect = RuntimeError('Docker unavailable')
        with self.assertRaises(HTTPException): self.service.stop()
        self.assertEqual(self.service.status()['phase'], 'error')
        self.stop_container.side_effect = None
        self.assertEqual(self.service.stop()['phase'], 'stopped')

    def test_stopped_service_rejects_observation(self):
        self.service.stop()
        with patch.object(r, 'capture_head') as capture:
            with self.assertRaises(HTTPException): self.service.observe(r.ObserveRequest(context={'map_name':'warehouse'}))
            capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()


class MissionDecisionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [{'mission_id':'m2','map_name':'warehouse','mission_name':'Mission2','revision':1}]
        self.store = Mock(); self.store.catalog_provider.return_value = self.catalog
        self.service = r.ReasoningService(Mock(), Mock(), self.store); self.service.update(phase='ready')
        self.req = r.MissionDecisionRequest(map_name='warehouse', localized=True)
        self.frame = {'frame_stamp':'1','image_base64':'abc','captured_at':1,'camera_topic':'head'}
    def result(self, decision='start', mid='m2'):
        import json
        return {'finish_reason':'stop', 'text':json.dumps({'decision':decision,'mission_id':mid,'reason':'visible evidence'})}
    def test_start_uses_live_frame_catalog_and_no_ground_truth(self):
        with patch.object(r,'capture_head',return_value=self.frame), patch.object(r,'qwen_request',return_value=self.result()) as call:
            out=self.service.decide(self.req)
        self.assertTrue(out['action_allowed']); self.assertEqual(out['selected_mission']['mission_id'],'m2')
        payload=call.call_args.args[1]
        self.assertEqual(payload['image_base64'],'abc'); self.assertFalse(payload['do_sample'])
        self.assertTrue(payload['json_object'])
        self.assertNotIn('assistant',payload['prompt'])
    def test_multiple_json_answers_are_not_silently_accepted_by_execution(self):
        result = self.result()
        result['text'] += '\nuser\n' + self.result(mid='foreign')['text']
        with patch.object(r,'capture_head',return_value=self.frame), patch.object(r,'qwen_request',return_value=result):
            out=self.service.decide(self.req)
        self.assertFalse(out['action_allowed'])
        self.assertIn('Extra data',out['error'])

    def test_wrong_map_id_truncation_and_removed_execution_decisions_never_execute(self):
        for result in [self.result(mid='foreign'), self.result('cancel'), self.result('unknown','m2'),
                       {**self.result(),'finish_reason':'length'}]:
            with patch.object(r,'capture_head',return_value=self.frame), patch.object(r,'qwen_request',return_value=result):
                out=self.service.decide(self.req)
            self.assertFalse(out['action_allowed']); self.assertTrue(out['error'])
    def test_repeat_start_is_allowed_but_running_mission_blocks_start(self):
        for status in ('completed', 'failed', 'cancelled'):
            self.req.recent_missions=[{'mission_id': 'm2', 'status': status}]
            self.assertTrue(r.validate_mission_decision(self.result(),self.catalog,self.req)['action_allowed'])
        self.req.active_mission={'mission_id':'m2','status':'running'}
        self.assertFalse(r.validate_mission_decision(self.result(),self.catalog,self.req)['action_allowed'])
        for removed in ('continue','cancel'):
            with self.assertRaises(ValueError):
                r.validate_mission_decision(self.result(removed),self.catalog,self.req)
    def test_stop_invalidates_inflight_inference(self):
        def complete(*args,**kwargs):
            self.service.generation+=1
            return self.result()
        with patch.object(r,'capture_head',return_value=self.frame), patch.object(r,'qwen_request',side_effect=complete):
            with self.assertRaises(HTTPException):self.service.decide(self.req)
        self.assertFalse(self.service.observe_lock.locked())
    def test_localization_and_catalog_changes_gate_execution(self):
        self.req.localized=False
        with self.assertRaises(HTTPException):self.service.decide(self.req)
        self.req.localized=True;self.store.catalog_provider.side_effect=[self.catalog,[]]
        with patch.object(r,'capture_head',return_value=self.frame), patch.object(r,'qwen_request',return_value=self.result()):
            self.assertFalse(self.service.decide(self.req)['action_allowed'])


class CollectDecisionPreviewTests(unittest.TestCase):
    setUp = MissionDecisionTests.setUp
    result = MissionDecisionTests.result
    def test_collect_and_operate_send_identical_idle_inputs_but_collect_never_allows_execution(self):
        self.service.update(model='Qwen/4B', adapter='/model/tuned')
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value=self.result()) as call:
            operate = self.service.decide(self.req)
            operate_payload = call.call_args.args[1]
            for localized in (True, False):
                collect = self.service.observe(r.ObserveRequest(context={'map_name':'warehouse', 'localized':localized}))
                self.assertEqual(call.call_args.args[1], {**operate_payload, 'attention_preview': True})
                self.assertNotIn('attention_preview', operate_payload)
                self.assertTrue(collect['preview_only'])
                self.assertFalse(collect['action_allowed'])
                self.assertTrue(collect['decision_valid'])
                self.assertEqual(collect['decision'], operate['decision'])
                self.assertEqual(collect['adapter'], '/model/tuned')
                self.assertIn(collect['analysis_id'], self.service.analyses)
        self.store.capture.assert_not_called()
        self.service.start_container.assert_not_called()
        self.service.stop_container.assert_not_called()

    def test_collect_requires_map_and_catalog(self):
        with patch.object(r, 'qwen_request') as call:
            with self.assertRaises(HTTPException): self.service.observe(r.ObserveRequest())
            self.store.catalog_provider.return_value = []
            with self.assertRaises(HTTPException): self.service.observe(r.ObserveRequest(context={'map_name':'warehouse'}))
            call.assert_not_called()

    def test_collect_invalid_output_and_catalog_change_remain_non_executable(self):
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value=self.result(mid='foreign')):
            out = self.service.observe(r.ObserveRequest(context={'map_name':'warehouse'}))
            self.assertFalse(out['decision_valid']); self.assertFalse(out['action_allowed'])
            self.assertIsNone(out['selected_mission'])
        self.store.catalog_provider.side_effect = [self.catalog, []]
        with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value=self.result()):
            out = self.service.observe(r.ObserveRequest(context={'map_name':'warehouse'}))
            self.assertFalse(out['decision_valid']); self.assertIn('catalog changed', out['error'])

class ThreeDecisionTests(MissionDecisionTests):
    def test_all_three_decisions_use_one_contract_in_collect_and_operate(self):
        for action in ('start', 'wait', 'unknown'):
            with self.subTest(action=action):
                req = r.MissionDecisionRequest(map_name='warehouse', localized=True)
                context = {'map_name': 'warehouse', 'localized': True}
                mid = 'm2' if action == 'start' else None
                with patch.object(r, 'capture_head', return_value=self.frame), patch.object(r, 'qwen_request', return_value=self.result(action, mid)) as call:
                    live = self.service.decide(req)
                    payload = call.call_args.args[1]
                    preview = self.service.observe(r.ObserveRequest(context=context))
                    self.assertEqual(call.call_args.args[1], {**payload, 'attention_preview': True})
                    self.assertEqual(live['decision']['decision'], action)
                    self.assertTrue(preview['decision_valid'])
                    self.assertFalse(preview['action_allowed'])

    def test_preview_rejects_foreign_mission_before_inference(self):
        with patch.object(r, 'qwen_request') as call:
            with self.assertRaises(HTTPException):
                self.service.observe(r.ObserveRequest(context={'map_name': 'warehouse', 'preview_active_mission': {'mission_id': 'foreign'}}))
            call.assert_not_called()
