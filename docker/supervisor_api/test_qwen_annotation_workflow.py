"""Offline temporary datasets and fake workers: no GPU, robot or production data."""
import base64
import json
from pathlib import Path
import tempfile
import threading
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch
from fastapi import HTTPException, FastAPI
from fastapi.testclient import TestClient
from test_segmentation import JPEG
from supervisor_api import collection as c
from supervisor_api import qwen_annotation_workflow as q
from supervisor_api.qwen_annotation_contract import parse_target

class QwenWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.catalog=[{'mission_id':'m1','mission_name':'Mission1','revision':1}]
        self.store=c.CollectionStore(Path(self.temp.name)/'data',lambda _:self.catalog)
        self.ep=self.store.start(c.EpisodeRequest(map_name='warehouse',scenario='Table',group_id='table-a',robot_type='ffw',environment='simulation'))
        self.frames=[self.store.capture(self.ep['id'],{'image_base64':base64.b64encode(JPEG).decode(),'camera_topic':'/head','captured_at':1,'frame_stamp':'1'}, {}) for _ in range(3)]
        self.store.end(self.ep['id'])
        self.runtime=Mock()
        self.workflow=q.QwenAnnotationWorkflow(self.store,self.runtime,Path(self.temp.name)/'models',recover=False)
        self.target={'decision':'start','mission_id':'m1','reason':'A person is visible.'}

    def test_training_steps_accept_positive_integers_without_a_fixed_upper_limit(self):
        from pydantic import ValidationError
        for request_type in (q.TrainRequest, q.ReasoningTrainRequest):
            for steps in (1, 100, 2000, 5000, 10000):
                with self.subTest(request=request_type.__name__, steps=steps):
                    request = request_type(map_name='house', episode_ids=['ep'], model_name='Test', steps=steps)
                    self.assertEqual(request.steps, steps)
            for steps in (0, -1, 1.5, True):
                with self.subTest(request=request_type.__name__, invalid=steps):
                    with self.assertRaises(ValidationError):
                        request_type(map_name='house', episode_ids=['ep'], model_name='Test', steps=steps)

    def queued_training(self,kind='training'):
        for frame in self.frames[:2]:self.label(frame)
        self.workflow.sample(self.ep['id'],q.SampleRequest(count=3))
        with patch.object(q.threading,'Thread'):
            if kind=='training':self.workflow.train(q.TrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],model_name='Cancelled training'))
            else:self.workflow.train_reasoning(q.ReasoningTrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],model_name='Cancelled reasoning'))
        return deepcopy(self.workflow.jobs[kind])

    def test_cancel_queued_training_and_reasoning_jobs_preserves_dataset(self):
        for kind in q.TRAINING_KINDS:
            with self.subTest(kind=kind):
                initial=self.queued_training(kind)
                result=self.workflow.cancel_job(kind,initial['id'])
                self.assertEqual(result[kind]['status'],'cancelled')
                self.assertNotIn('error',result[kind])
                self.workflow._run(kind,initial)
                self.runtime.execute.assert_not_called()
                self.assertEqual(len(self.store.get(self.ep['id'])['observations']),3)
                self.assertFalse(Path(c.read_json(Path(initial['request_path']))['model_path']).exists())
                self.workflow._idle()
                self.workflow.reset_job(kind)

    def test_retry_cancelled_training_reuses_display_name_but_not_partial_weights(self):
        for kind in q.TRAINING_KINDS:
            initial=self.queued_training(kind)
            first=c.read_json(Path(initial['request_path']))
            adapter=Path(first['adapter_path']);adapter.mkdir(parents=True)
            (adapter/'training_report.json').write_text('{"incomplete":true}')
            self.workflow.cancel_job(kind,initial['id'])
            next_job=self.queued_training(kind)
            next_request=c.read_json(Path(next_job['request_path']))
            self.assertEqual(initial['model_name'],next_job['model_name'])
            self.assertNotEqual(first['adapter_path'],next_request['adapter_path'])
            self.assertTrue((adapter/'training_report.json').exists())
            self.assertFalse((Path(next_request['adapter_path'])/'training_report.json').exists())
            self.workflow.cancel_job(kind,next_job['id'])

    def test_cancel_running_worker_never_publishes_adapter_or_failure(self):
        initial=self.queued_training()
        def worker(path,job_id,event):
            event({'event':'progress','step':12})
            self.workflow.cancel_job('training',job_id)
            event({'event':'progress','step':100})
        self.runtime.execute.side_effect=worker
        self.workflow._run('training',initial)
        job=self.workflow.status()['training']
        self.assertEqual(job['status'],'cancelled');self.assertEqual(job['step'],12)
        self.assertNotIn('error',job)
        self.assertEqual(self.workflow.models(),[])
        self.runtime.cancel.assert_called_once_with(initial['id'])
        self.workflow._idle()

    def test_cancel_is_job_scoped_idempotent_and_rejects_finished_jobs(self):
        initial=self.queued_training()
        with self.assertRaises(HTTPException):self.workflow.cancel_job('training','f'*32)
        self.runtime.cancel.assert_not_called()
        first=self.workflow.cancel_job('training',initial['id'])
        self.assertEqual(first,self.workflow.cancel_job('training',initial['id']))
        self.runtime.cancel.assert_called_once()
        self.workflow.jobs['training']['status']='completed'
        with self.assertRaises(HTTPException):self.workflow.cancel_job('training',initial['id'])
        with self.assertRaises(HTTPException):self.workflow.cancel_job('batch',initial['id'])

    def test_cancelling_keeps_lock_and_status_remains_readable_until_stop(self):
        initial=self.queued_training();entered=threading.Event();release=threading.Event()
        def stop(job_id):entered.set();release.wait(2)
        self.runtime.cancel.side_effect=stop
        worker=threading.Thread(target=self.workflow.cancel_job,args=('training',initial['id']))
        worker.start();self.assertTrue(entered.wait(1))
        try:
            self.assertEqual(self.workflow.status()['training']['status'],'cancelling')
            with self.assertRaises(HTTPException):self.workflow._idle()
            with self.assertRaises(HTTPException):self.workflow.reset_job('training')
        finally:release.set();worker.join(2)
        self.workflow._idle()

    def test_failed_stop_can_retry_and_does_not_release_training_lock(self):
        initial=self.queued_training()
        self.runtime.cancel.side_effect=RuntimeError('Docker unavailable')
        with self.assertRaises(HTTPException) as error:self.workflow.cancel_job('training',initial['id'])
        self.assertEqual(error.exception.status_code,503)
        self.assertEqual(self.workflow.status()['training']['status'],'cancelling')
        with self.assertRaises(HTTPException):self.workflow._idle()
        self.runtime.cancel.side_effect=None
        self.assertEqual(self.workflow.cancel_job('training',initial['id'])['training']['status'],'cancelled')

    def test_late_worker_cannot_update_replacement_job(self):
        initial=self.queued_training()
        replacement={'id':'new-job','status':'queued','step':0}
        def worker(path,job_id,event):
            self.workflow.jobs['training']=deepcopy(replacement)
            event({'event':'progress','step':100})
        self.runtime.execute.side_effect=worker
        self.workflow._run('training',initial)
        self.assertEqual(self.workflow.jobs['training'],replacement)

    def test_cancel_api_requires_current_job_id(self):
        initial=self.queued_training();app=FastAPI();app.include_router(q.create_router(self.workflow));client=TestClient(app)
        path='/collection/qwen/jobs/training/cancel'
        self.assertEqual(client.post(path,json={}).status_code,422)
        self.assertEqual(client.post(path,json={'job_id':'f'*32}).status_code,409)
        response=client.post(path,json={'job_id':initial['id']})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['training']['status'],'cancelled')

    def test_reset_failed_job_preserves_data_and_models_and_allows_retraining(self):
        self.model()
        self.workflow.jobs['training']={'status':'failed','error':'Stopped','step':120}
        result=self.workflow.reset_job('training')
        self.assertEqual(result['training'],{'status':'idle'})
        self.assertEqual(len(result['models']),1)
        self.assertEqual(len(self.store.get(self.ep['id'])['observations']),3)
        self.assertEqual(c.read_json(self.workflow.root/'workflow.json')['training'],{'status':'idle'})
        self.workflow._idle()

    def test_reset_refuses_active_or_successful_jobs(self):
        for status in ('queued','training','running','completed'):
            self.workflow.jobs['training']={'status':status}
            with self.assertRaises(HTTPException) as error:self.workflow.reset_job('training')
            self.assertEqual(error.exception.status_code,409)
            self.assertEqual(self.workflow.jobs['training']['status'],status)
        with self.assertRaises(HTTPException):self.workflow.reset_job('missing')

    def label(self,frame,status='approved'):
        ep=self.store.get(self.ep['id'])
        old=next((a for a in ep['annotations'] if a['frame_ids']==[frame['frame_id']]),None)
        return self.store.annotate(self.ep['id'],c.AnnotationRequest(frame_id=frame['frame_id'],
            decision='wait',mission_id=None,reason='No request.',annotator='collect-ui',review_status=status,
            expected_revision=old['revision'] if old else 0))

    def model(self):
        catalog=self.store.current_catalog('warehouse')
        row={'schema':q.SCHEMA,'id':'a'*32,'map_name':'warehouse','model_name':'Table v1',
             'base_model':'Qwen/Qwen3.5-4B','revision':'pinned','adapter_path':str(self.workflow.root/'warehouse/table/adapter'),
             'catalog_snapshot_id':catalog['snapshot_id'],'mission_ids':['m1'],'mission_revisions':{'m1':1}}
        c.write_json(self.workflow.root/'warehouse/table/model.json',row)
        return row

    def submit_batch(self):
        self.model()
        with patch.object(q.threading,'Thread'):
            self.workflow.batch(q.BatchRequest(model_id='a'*32,episode_ids=[self.ep['id']]))
        job=self.workflow.jobs['batch']
        return job,c.read_json(Path(job['request_path']))

    def test_three_type_sampling_training_and_batch_exclude_mixed_frames(self):
        self.label(self.frames[0])
        self.workflow.sample(self.ep['id'],q.SampleRequest(count=3))
        mission=self.store.start(c.EpisodeRequest(map_name='warehouse', robot_type='ffw',
            recording_mission_name='Mission1', episode_type='mission'))
        context={'reasoning_task':'mission_supervision',
                 'active_mission':{'mission_id':'m1','status':'running'}}
        mission_frames=[self.store.capture(mission['id'],
            {'image_base64':base64.b64encode(JPEG).decode(),'camera_topic':'/head','captured_at':1,'frame_stamp':'1'},context) for _ in range(2)]
        mixed=self.store.capture(mission['id'],
            {'image_base64':base64.b64encode(JPEG).decode(),'camera_topic':'/head','captured_at':1,'frame_stamp':'1'}, {})
        self.store.annotate(mission['id'],c.AnnotationRequest(frame_id=mixed['frame_id'],
            decision='wait',reason='Legacy selection frame in a mission episode.',annotator='human',review_status='approved'))
        self.store.end(mission['id'])
        self.store.annotate(mission['id'],c.AnnotationRequest(frame_id=mission_frames[0]['frame_id'],
            decision='pause',mission_id='m1',reason='A person blocks the robot.',
            annotator='human',review_status='approved'))
        with self.assertRaises(HTTPException):
            self.workflow.sample(mission['id'],q.SampleRequest(count=3))
        sampled=self.workflow.sample(mission['id'],q.SampleRequest(count=2))
        self.assertNotIn(mixed['frame_id'],sampled['qwen_sample']['frame_ids'])
        patrol=self.store.start(c.EpisodeRequest(map_name='warehouse',episode_type='patrol'))
        patrol_context={'reasoning_task':'patrol_supervision','active_mission':None,
                        'observation':{'phase':'moving','from_spot_id':'table','target_spot_id':'laundry'}}
        image={'image_base64':base64.b64encode(JPEG).decode(),'camera_topic':'/head','captured_at':1,'frame_stamp':'1'}
        patrol_frames=[self.store.capture(patrol['id'],image,patrol_context) for _ in range(2)]
        unverified=self.store.capture(patrol['id'],image,{**patrol_context,'runtime_source':'unverified'})
        self.store.annotate(patrol['id'],c.AnnotationRequest(frame_id=patrol_frames[0]['frame_id'],
            decision='continue',reason='The route is clear.',annotator='human',review_status='approved'))
        self.store.end(patrol['id'])
        with self.assertRaises(HTTPException):
            self.workflow.sample(patrol['id'],q.SampleRequest(count=3))
        sampled=self.workflow.sample(patrol['id'],q.SampleRequest(count=2))
        self.assertNotIn(unverified['frame_id'],sampled['qwen_sample']['frame_ids'])
        with patch.object(q.threading,'Thread'):
            self.workflow.train(q.TrainRequest(map_name='warehouse',
                episode_ids=[self.ep['id'],mission['id'],patrol['id']],model_name='Mixed'))
        job=self.workflow.jobs['training']
        request=c.read_json(Path(job['request_path']))
        self.assertEqual(request['episode_types'],['mission','observation','patrol'])
        self.assertNotIn('episode_type',request)
        self.assertEqual(request['reasoning_tasks'],['mission_selection','mission_supervision','patrol_supervision'])
        self.assertEqual(len(request['frames']),3)
        self.assertEqual(request['task_sample_counts'],
            {'mission_selection':1,'patrol_supervision':1,'mission_supervision':1})
        self.assertEqual(request['task_decision_counts'],
            {'mission_selection':{'wait':1},'patrol_supervision':{'continue':1},'mission_supervision':{'pause':1}})
        self.assertNotIn(mixed['frame_id'],{row['frame_id'] for row in request['frames']})
        targets=[json.loads(row['sample']['messages'][-1]['content'][0]['text']) for row in request['frames']]
        self.assertEqual([target['decision'] for target in targets],['wait','pause','continue'])
        self.assertEqual(request['frames'][1]['frame']['context']['active_mission']['mission_id'],'m1')
        self.assertNotEqual(request['frames'][0]['sample']['messages'][0],request['frames'][1]['sample']['messages'][0])
        def execute(path,jid,event):
            adapter=Path(request['adapter_path']);adapter.mkdir(parents=True)
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'test')
        self.runtime.execute.side_effect=execute
        self.workflow._run('training',job)
        model=c.read_json(Path(request['model_path']))
        self.assertEqual(model['episode_types'],['mission','observation','patrol'])
        with patch.object(q.threading,'Thread'):
            self.workflow.batch(q.BatchRequest(model_id=model['id'],episode_ids=[self.ep['id'],mission['id'],patrol['id']]))
        batch=c.read_json(Path(self.workflow.jobs['batch']['request_path']))
        self.assertEqual({row['episode_id'] for row in batch['frames']},{self.ep['id'],mission['id'],patrol['id']})
        self.assertEqual(len(batch['frames']),4)
        self.assertNotIn(mixed['frame_id'],{row['frame_id'] for row in batch['frames']})
        self.assertNotIn(unverified['frame_id'],{row['frame_id'] for row in batch['frames']})
        self.assertEqual({row['frame']['context'].get('reasoning_task') for row in batch['frames']},
                         {'mission_selection','patrol_supervision','mission_supervision'})

    def test_legacy_single_type_model_keeps_its_generation_scope(self):
        model=self.model()
        model['episode_type']='mission'
        c.write_json(self.workflow.root/'warehouse/table/model.json',model)
        with self.assertRaises(HTTPException) as error:
            self.workflow.batch(q.BatchRequest(model_id=model['id'],episode_ids=[self.ep['id']]))
        self.assertEqual(error.exception.status_code,422)

    def test_reviewed_training_does_not_require_random_samples(self):
        for frame in self.frames[:2]: self.label(frame)
        self.label(self.frames[2],status='draft')
        with patch.object(q.threading,'Thread'):
            job=self.workflow.train(q.TrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],
                model_name='Reviewed v2',sampled_only=False))
        request=c.read_json(Path(self.workflow.jobs['training']['request_path']))
        self.assertEqual(job['sample_count'],2)
        self.assertEqual(job['episode_type'],'observation')
        self.assertEqual({r['frame_id'] for r in request['frames']},{f['frame_id'] for f in self.frames[:2]})

    def test_regeneration_targets_automatic_drafts_not_manual_or_approved(self):
        self.label(self.frames[0])
        self.label(self.frames[1],status='draft')
        self.store.annotate(self.ep['id'],c.AnnotationRequest(frame_id=self.frames[2]['frame_id'],
            decision='wait',mission_id=None,reason='Draft',annotator='qwen:previous',
            review_status='draft',expected_revision=0))
        job,request=self.submit_batch()
        self.assertEqual([r['frame_id'] for r in request['frames']],[self.frames[2]['frame_id']])
        self.workflow.jobs['batch']['status']='idle'
        with self.assertRaises(HTTPException):
            self.workflow.batch(q.BatchRequest(model_id='a'*32,episode_ids=[self.ep['id']],replace_unreviewed=False))

    def test_balanced_training_matches_preview_and_preserves_source(self):
        self.store.annotate(self.ep['id'],c.AnnotationRequest(frame_id=self.frames[0]['frame_id'],
            decision='start',mission_id='m1',reason='A person.',annotator='test',review_status='approved',expected_revision=0))
        for frame in self.frames[1:]: self.label(frame)
        path=self.store.path('raw',self.ep['id'],'samples.jsonl')
        before=path.read_bytes()
        args=dict(map_name='warehouse',episode_ids=[self.ep['id']],balance_decisions=True)
        preview=self.workflow.preview_reasoning(q.ReasoningDatasetRequest(**args))
        self.assertEqual(preview['before']['decisions'],{'start':1,'wait':2})
        self.assertEqual(preview['after']['decisions'],{'start':1,'wait':1})
        with patch.object(q.threading,'Thread'):
            job=self.workflow.train_reasoning(q.ReasoningTrainRequest(**args,model_name='Balanced'))
        request=c.read_json(Path(self.workflow.jobs['reasoning_training']['request_path']))
        self.assertEqual(len(request['frames']),2)
        self.assertEqual(job['sampling']['after'],preview['after'])
        self.assertEqual(path.read_bytes(),before)

    def test_reasoning_training_uses_pinned_2b_base_when_selected(self):
        for frame in self.frames[:2]: self.label(frame)
        with patch.object(q.threading, 'Thread'):
            job = self.workflow.train_reasoning(q.ReasoningTrainRequest(
                map_name='warehouse', episode_ids=[self.ep['id']],
                model_name='Small reasoning', base_model='Qwen/Qwen3.5-2B'))
        self.assertEqual(job['base_model'], 'Qwen/Qwen3.5-2B')
        self.assertEqual(job['revision'], '15852e8c16360a2fea060d615a32b45270f8a8fc')

    def test_reasoning_training_uses_pinned_08b_base_when_selected(self):
        for frame in self.frames[:2]: self.label(frame)
        with patch.object(q.threading, 'Thread'):
            job = self.workflow.train_reasoning(q.ReasoningTrainRequest(
                map_name='warehouse', episode_ids=[self.ep['id']],
                model_name='Tiny reasoning', base_model='Qwen/Qwen3.5-0.8B'))
        self.assertEqual(job['base_model'], 'Qwen/Qwen3.5-0.8B')
        self.assertEqual(job['revision'], '2fc06364715b967f1860aea9cf38778875588b17')

    def test_balancing_preserves_single_class_with_warning(self):
        for frame in self.frames: self.label(frame)
        before=self.store.path('raw',self.ep['id'],'samples.jsonl').read_bytes()
        with patch.object(q.threading,'Thread'):
            job=self.workflow.train_reasoning(q.ReasoningTrainRequest(map_name='warehouse',
                episode_ids=[self.ep['id']],model_name='Wait baseline',balance_decisions=True))
        self.assertEqual(job['sampling']['after']['decisions'],{'wait':3})
        self.assertIn('only one decision',job['sampling']['warnings'][0])
        request=c.read_json(Path(self.workflow.jobs['reasoning_training']['request_path']))
        self.assertEqual(len(request['frames']),3)
        self.assertEqual(self.store.path('raw',self.ep['id'],'samples.jsonl').read_bytes(),before)

    def test_sampling_is_separate_from_sam_and_keeps_labels_and_frames(self):
        meta=self.store.episode(self.ep['id']);meta['sam_sample']={'frame_ids':[self.frames[0]['frame_id']]}
        c.write_json(self.store.path('raw',self.ep['id'],'episode.json'),meta)
        self.label(self.frames[0]); before=self.store.path('raw',self.ep['id'],'samples.jsonl').read_bytes()
        result=self.workflow.sample(self.ep['id'],q.SampleRequest(count=2))
        self.assertEqual(len(set(result['qwen_sample']['frame_ids'])),2)
        self.assertEqual(result['qwen_sample']['schema_version'],q.SAMPLE_SCHEMA)
        self.assertEqual(result['sam_sample'],meta['sam_sample'])
        self.assertEqual(self.store.path('raw',self.ep['id'],'samples.jsonl').read_bytes(),before)
        with self.assertRaises(HTTPException):self.workflow.sample(self.ep['id'],q.SampleRequest(count=4))

    def test_training_rejects_a_legacy_fixed_size_sample(self):
        meta=self.store.episode(self.ep['id'])
        meta['qwen_sample']={'frame_ids':[frame['frame_id'] for frame in self.frames[:2]],
                             'requested_count':20,'strategy':'stratified_random_v1'}
        c.write_json(self.store.path('raw',self.ep['id'],'episode.json'),meta)
        for frame in self.frames[:2]:self.label(frame)
        with self.assertRaises(HTTPException):
            self.workflow.train(q.TrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],model_name='Legacy'))

    def test_training_uses_approved_subset_and_typed_image_messages(self):
        ep=self.workflow.sample(self.ep['id'],q.SampleRequest(count=3))
        req=q.TrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],model_name='Table v1')
        with self.assertRaises(HTTPException):self.workflow.train(req)
        self.label(self.frames[0]); self.label(self.frames[1]); self.label(self.frames[2],status='draft')
        with patch.object(q.threading,'Thread'):
            job=self.workflow.train(req)
        request=c.read_json(Path(self.workflow.jobs['training']['request_path']))
        self.assertEqual(len(request['frames']),2)
        self.assertEqual({row['frame_id'] for row in request['frames']},
                         {frame['frame_id'] for frame in self.frames[:2]})
        for frame in request['frames']:
            messages=frame['sample']['messages']
            self.assertEqual([m['role'] for m in messages],['system','user','assistant'])
            self.assertEqual([b['type'] for b in messages[1]['content']],['image','text'])
            target=json.loads(messages[2]['content'][0]['text'])
            self.assertEqual(set(target),{'decision','mission_id','reason'})
            self.assertNotIn('scenario',json.loads(messages[1]['content'][1]['text']))
        self.assertEqual(job['model_name'],'Table v1')
        self.assertEqual(job['catalog_snapshot_id'],c.digest(self.catalog))
        self.assertEqual(job['mission_ids'],['m1'])
        self.assertEqual(job['mission_revisions'],{'m1':1})
        self.assertEqual(self.workflow.models(),[])

    def test_training_combines_approved_samples_from_saved_episodes_on_one_map(self):
        first=self.workflow.sample(self.ep['id'],q.SampleRequest(count=2))
        for frame in self.frames:
            if frame['frame_id'] in first['qwen_sample']['frame_ids']: self.label(frame)
        second=self.store.start(c.EpisodeRequest(map_name='warehouse',group_id='table-b'))
        second_frames=[self.store.capture(second['id'],{'image_base64':base64.b64encode(JPEG).decode(),
            'camera_topic':'/head','captured_at':2,'frame_stamp':'2'}, {}) for _ in range(2)]
        self.store.end(second['id'])
        self.workflow.sample(second['id'],q.SampleRequest(count=2))
        for frame in second_frames:
            self.store.annotate(second['id'],c.AnnotationRequest(frame_id=frame['frame_id'],
                decision='wait',reason='People have not sat at the table.',annotator='human',review_status='approved'))
        summaries={ep['id']:ep for ep in self.store.list()}
        self.assertEqual(summaries[self.ep['id']]['qwen_sample_approved_count'],2)
        self.assertEqual(summaries[second['id']]['qwen_sample_approved_count'],2)
        with patch.object(q.threading,'Thread'):
            job=self.workflow.train(q.TrainRequest(map_name='warehouse',
                episode_ids=[self.ep['id'],second['id']],model_name='Combined'))
        request=c.read_json(Path(self.workflow.jobs['training']['request_path']))
        self.assertEqual(job['sample_count'],4)
        self.assertEqual(job['episode_ids'],[self.ep['id'],second['id']])
        self.assertEqual({row['episode_id'] for row in request['frames']},{self.ep['id'],second['id']})
        self.assertEqual(len(request['frames']),4)
        self.assertEqual(self.workflow.models(),[])

    def test_training_rejects_incomplete_or_other_map_sampled_episodes(self):
        first=self.workflow.sample(self.ep['id'],q.SampleRequest(count=2))
        for frame in self.frames:
            if frame['frame_id'] in first['qwen_sample']['frame_ids']: self.label(frame)
        second=self.store.start(c.EpisodeRequest(map_name='warehouse'))
        second_frames=[self.store.capture(second['id'],{'image_base64':base64.b64encode(JPEG).decode(),
            'camera_topic':'/head','captured_at':2,'frame_stamp':'2'}, {}) for _ in range(2)]
        self.store.end(second['id'])
        self.workflow.sample(second['id'],q.SampleRequest(count=2))
        self.store.annotate(second['id'],c.AnnotationRequest(frame_id=second_frames[0]['frame_id'],
            decision='wait',reason='Not ready.',annotator='human',review_status='approved'))
        with patch.object(q.threading,'Thread'):
            partial=self.workflow.train(q.TrainRequest(map_name='warehouse',
                episode_ids=[self.ep['id'],second['id']],model_name='Incomplete'))
        self.assertEqual(partial['sample_count'],3)
        self.assertEqual(next(ep for ep in self.store.list() if ep['id']==second['id'])['qwen_sample_approved_count'],1)
        self.workflow.jobs['training']['status']='idle'
        other=self.store.start(c.EpisodeRequest(map_name='other'))
        self.store.end(other['id'])
        with self.assertRaises(HTTPException) as mixed:
            self.workflow.train(q.TrainRequest(map_name='warehouse',
                episode_ids=[self.ep['id'],other['id']],model_name='Mixed'))
        self.assertEqual(mixed.exception.status_code,422)
        self.assertTrue((self.workflow.root/'warehouse'/'Incomplete').exists())
        self.assertFalse((self.workflow.root/'warehouse'/'Mixed').exists())

    def test_training_and_auto_label_preserve_recorded_design_catalog(self):
        ep=self.workflow.sample(self.ep['id'],q.SampleRequest(count=2))
        for frame in self.frames:
            if frame['frame_id'] in ep['qwen_sample']['frame_ids']:self.label(frame)
        self.catalog.append({'mission_id':'observe','mission_name':'ObservationMode','revision':3})
        with patch.object(q.threading,'Thread'):
            job=self.workflow.train(q.TrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],model_name='Live catalog'))
        request=c.read_json(Path(self.workflow.jobs['training']['request_path']))
        prompt=json.loads(request['frames'][0]['sample']['messages'][1]['content'][1]['text'])
        self.assertEqual([m['mission_id'] for m in prompt['missions']],['m1'])
        self.assertEqual(job['mission_revisions'],{'m1':1,'observe':3})

    def test_reasoning_training_uses_all_approved_labels_from_selected_map_episodes(self):
        self.label(self.frames[0])
        self.label(self.frames[1],status='draft')
        second=self.store.start(c.EpisodeRequest(map_name='warehouse',group_id='table-b'))
        second_frames=[self.store.capture(second['id'],{'image_base64':base64.b64encode(JPEG).decode(),
            'camera_topic':'/head','captured_at':2,'frame_stamp':'2'}, {}) for _ in range(2)]
        self.store.end(second['id'])
        for frame in second_frames:
            current=self.store.get(second['id'])
            self.store.annotate(second['id'],c.AnnotationRequest(frame_id=frame['frame_id'],
                decision='wait',reason='No request.',annotator='collect-ui',review_status='approved'))
        with patch.object(q.threading,'Thread'):
            job=self.workflow.train_reasoning(q.ReasoningTrainRequest(map_name='warehouse',
                episode_ids=[self.ep['id'],second['id']],model_name='Warehouse reasoning v1'))
        request=c.read_json(Path(self.workflow.jobs['reasoning_training']['request_path']))
        self.assertEqual(request['kind'],'reasoning_training')
        self.assertEqual(request['purpose'],'reasoning')
        self.assertTrue(Path(request['adapter_path']).is_relative_to(self.workflow.reasoning_root))
        self.assertFalse(Path(request['adapter_path']).is_relative_to(self.workflow.root))
        self.assertEqual(request['episode_ids'],[self.ep['id'],second['id']])
        self.assertEqual(len(request['frames']),3)
        self.assertEqual(job['sample_count'],3)
        self.assertTrue(all(json.loads(row['sample']['messages'][2]['content'][0]['text'])['decision']=='wait'
                            for row in request['frames']))

    def test_reasoning_training_rejects_other_map_or_insufficient_approved_data(self):
        with self.assertRaises(HTTPException):
            self.workflow.train_reasoning(q.ReasoningTrainRequest(map_name='warehouse',
                episode_ids=[self.ep['id']],model_name='Too small'))
        other=self.store.start(c.EpisodeRequest(map_name='other'))
        self.store.end(other['id'])
        with self.assertRaises(HTTPException):
            self.workflow.train_reasoning(q.ReasoningTrainRequest(map_name='warehouse',
                episode_ids=[self.ep['id'],other['id']],model_name='Mixed maps'))

    def test_auto_label_rejects_model_after_design_catalog_changes(self):
        self.model()
        self.catalog.append({'mission_id':'observe','mission_name':'ObservationMode','revision':1})
        with self.assertRaises(HTTPException) as error:
            self.workflow.batch(q.BatchRequest(model_id='a'*32,episode_ids=[self.ep['id']]))
        self.assertEqual(error.exception.status_code,409)

    def test_completed_training_publishes_named_model_only_after_weights_exist(self):
        self.workflow.sample(self.ep['id'],q.SampleRequest(count=2))
        for frame in self.frames:self.label(frame)
        with patch.object(q.threading,'Thread'):
            self.workflow.train(q.TrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],model_name='My table'))
        job=self.workflow.jobs['training'];request=c.read_json(Path(job['request_path']))
        def execute(path,jid,event):
            adapter=Path(request['adapter_path']);adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'test weights')
            event({'event':'progress','step':200,'loss':.1,'phase':'Training'})
        self.runtime.execute.side_effect=execute
        self.workflow._run('training',job)
        self.assertEqual(self.workflow.models()[0]['model_name'],'My table')
        self.assertEqual(self.workflow.models()[0]['id'],request['id'])
        self.assertEqual(self.workflow.status()['training']['status'],'completed')
        with self.assertRaises(HTTPException):
            self.workflow.train(q.TrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],model_name='My table'))

    def test_failed_training_does_not_appear_as_a_selectable_model(self):
        self.workflow.sample(self.ep['id'],q.SampleRequest(count=2))
        for frame in self.frames:self.label(frame)
        with patch.object(q.threading,'Thread'):
            self.workflow.train(q.TrainRequest(map_name='warehouse',episode_ids=[self.ep['id']],model_name='Broken model'))
        self.workflow._run('training',self.workflow.jobs['training'])
        self.assertEqual(self.workflow.status()['training']['status'],'failed')
        self.assertEqual(self.workflow.models(),[])

    def test_batch_skips_approved_labels_and_imports_drafts_with_fixed_metadata(self):
        self.label(self.frames[0]);job,request=self.submit_batch()
        self.assertEqual(len(request['frames']),2)
        self.assertNotIn(self.frames[0]['frame_id'],[r['frame_id'] for r in request['frames']])
        def execute(path,jid,event):
            for frame in request['frames']:
                event({'event':'prediction','episode_id':frame['episode_id'],'frame_id':frame['frame_id'],'text':json.dumps(self.target)})
        self.runtime.execute.side_effect=execute
        self.workflow._run('batch',job)
        current=self.store.get(self.ep['id'])
        self.assertEqual(self.workflow.status()['batch']['status'],'completed')
        self.assertEqual(self.workflow.status()['batch']['created'],2)
        self.assertEqual(sum(a['review_status']=='approved' for a in current['annotations']),1)
        for frame in current['observations'][1:]:
            self.assertEqual(frame['metadata'],{'scenario':'Table','group_id':'table-a','robot_type':'ffw','environment':'simulation'})
        self.assertTrue(all(a['review_status']=='draft' for a in current['annotations'][1:]))

    def test_late_prediction_does_not_overwrite_a_human_save(self):
        job,request=self.submit_batch()
        row=request['frames'][0]
        self.label(self.frames[0])
        self.assertEqual(self.workflow._prediction(row,{'text':json.dumps(self.target)},job['id']),'skipped')
        self.assertEqual(self.store.get(self.ep['id'])['annotations'][0]['target']['decision'],'wait')

    def test_invalid_mission_or_runtime_context_is_not_saved(self):
        job,request=self.submit_batch();row=request['frames'][0]
        with self.assertRaises(HTTPException):
            self.workflow._prediction(row,{'text':json.dumps({**self.target,'mission_id':'missing'})},job['id'])
        for decision in ('continue','cancel'):
            with self.assertRaises(ValueError):
                self.workflow._prediction(row,{'text':json.dumps({**self.target,'decision':decision})},job['id'])
        self.assertEqual(self.store.get(self.ep['id'])['annotations'],[])

    def test_worker_error_counts_failed_frames_and_does_not_claim_success(self):
        job,request=self.submit_batch()
        def execute(path,jid,event):
            for frame in request['frames']:
                event({'event':'prediction','episode_id':frame['episode_id'],'frame_id':frame['frame_id'],'text':'not json'})
        self.runtime.execute.side_effect=execute
        self.workflow._run('batch',job)
        result=self.workflow.status()['batch']
        self.assertEqual(result['status'],'completed_with_errors')
        self.assertEqual(result['failed'],3)
        self.assertEqual(self.store.get(self.ep['id'])['annotations'],[])

    def test_incomplete_worker_does_not_publish_a_completed_job(self):
        job,_=self.submit_batch()
        self.workflow._run('batch',job)
        self.assertEqual(self.workflow.status()['batch']['status'],'failed')

    def test_restart_preserves_processed_frame_results_without_duplicates(self):
        job,request=self.submit_batch()
        frame=request['frames'][0]
        event={'event':'prediction','episode_id':frame['episode_id'],'frame_id':frame['frame_id'],'text':json.dumps(self.target)}
        self.workflow._prediction(frame,event,job['id'])
        self.assertEqual(self.workflow._prediction(frame,event,job['id']),'created')
        self.assertEqual(len(self.store.records(self.ep['id'],'annotations')),1)
        restored=q.QwenAnnotationWorkflow(self.store,self.runtime,self.workflow.root,recover=False)
        self.assertEqual(restored.jobs['batch']['id'],job['id'])

    def test_api_validates_sample_count_and_returns_jobs_without_worker_inputs(self):
        app=FastAPI();app.include_router(q.create_router(self.workflow));client=TestClient(app)
        self.assertEqual(q.SampleRequest(count=80).count,80)
        for body in ({},{'count':0},{'count':2.5}):
            response=client.post(f'/collection/episodes/{self.ep["id"]}/qwen/sample',json=body)
            self.assertEqual(response.status_code,422)
        self.submit_batch()
        status=client.get('/collection/qwen/workflow').json()
        self.assertNotIn('request_path',status['batch'])
        self.assertNotIn('results',status['batch'])

    def test_annotation_training_matches_operate_export_without_frame_information(self):
        ep = self.store.start(c.EpisodeRequest(map_name='warehouse'))
        frames = [self.store.capture(ep['id'], {'image_base64':base64.b64encode(JPEG).decode(),
            'camera_topic':'/head','captured_at':1,'frame_stamp':'1'},
            {'active_mission':None,'operator_instruction':{'text':'Wait','received_at_ms':1},
             'recent_missions':[{'mission_id':'m1','status':'succeeded','secret':'ignored'}]}) for _ in range(2)]
        self.store.end(ep['id'])
        for f in frames:
            self.store.annotate(ep['id'],c.AnnotationRequest(frame_id=f['frame_id'],decision='wait',
                reason='No task object is visible.',annotator='human',review_status='approved'))
        self.workflow.sample(ep['id'],q.SampleRequest(count=2))
        with patch.object(q.threading,'Thread'):
            self.workflow.train(q.TrainRequest(map_name='warehouse',episode_ids=[ep['id']],model_name='No metadata'))
        request=c.read_json(Path(self.workflow.jobs['training']['request_path']))
        exported=self.store._training_files(self.store.get(ep['id']))[0]
        exported_rows=[row for name,values in exported.items() if name in ('train.jsonl','val.jsonl','test.jsonl') for row in values]
        self.assertEqual(len(exported_rows),2)
        for trained,exported in zip(request['frames'],exported_rows):
            for idx in (0,2):self.assertEqual(trained['sample']['messages'][idx],exported['messages'][idx])
            self.assertEqual(trained['sample']['messages'][1]['content'][1],exported['messages'][1]['content'][1])
            self.assertNotIn('scenario',trained['frame']['metadata'])

    def test_contract_rejects_missing_or_extra_output_fields(self):
        for row in ({'scenario':'A table'}, {**self.target,'robot_type':'invented'}, {**self.target,'reason':''}, {**self.target,'scenario':'Extra field'}, {**self.target,'decision':'continue'}, {**self.target,'decision':'cancel'}):
            with self.assertRaises(ValueError):parse_target(json.dumps(row))



    def test_delete_reasoning_model_blocks_in_use_and_preserves_other_models_and_data(self):
        directory=self.workflow.reasoning_root/'warehouse'/'Reasoning'
        adapter=str(directory/'adapter')
        c.write_json(directory/'model.json',{'schema':q.SCHEMA,'id':'reasoning-model',
            'purpose':'reasoning','adapter_path':adapter})
        c.write_json(directory/'adapter'/'weights.json',{'test':True})
        other=self.workflow.root/'warehouse'/'Auto'
        c.write_json(other/'model.json',{'schema':q.SCHEMA,'id':'auto-model','purpose':'auto_labeling'})
        self.workflow.jobs['reasoning_training']={'status':'completed','adapter_path':adapter}
        self.workflow.model_in_use=lambda _:True
        with self.assertRaises(HTTPException) as error:self.workflow.delete_model('reasoning-model')
        self.assertEqual(error.exception.status_code,409)
        self.assertTrue(directory.exists())
        self.workflow.model_in_use=lambda _:False
        self.workflow.jobs['reasoning_training']['status']='training'
        with self.assertRaises(HTTPException):self.workflow.delete_model('reasoning-model')
        self.workflow.jobs['reasoning_training']['status']='completed'
        self.assertEqual(self.workflow.delete_model('reasoning-model'),{'deleted':'reasoning-model'})
        self.assertFalse(directory.exists())
        self.assertTrue(other.exists())
        self.assertEqual(self.workflow.jobs['reasoning_training'],{'status':'idle'})
        self.assertEqual(len(self.store.get(self.ep['id'])['observations']),3)
        with self.assertRaises(HTTPException) as error:self.workflow.delete_model('reasoning-model')
        self.assertEqual(error.exception.status_code,404)

if __name__=='__main__':unittest.main()
