"""No GPU or robot calls: draft provenance, review, filtering and collision guards."""
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi import HTTPException
from test_segmentation import SegmentationTests
from supervisor_api import segmentation_workflow as w

class WorkflowTests(unittest.TestCase):
    setUp = SegmentationTests.setUp

    def workflow(self):
        self.reply.update(strategy='sam_multiclass_instances_v1',object_label='stick')
        self.store.end(self.ep['id'])
        workflow=w.SegmentationWorkflow(self.service)
        workflow.job={'status':'running','processed':0,'created':0,'empty':0,'failed':0}
        return workflow

    def test_cancel_training_forwards_job_identity_without_starting_container(self):
        workflow=self.workflow();self.start.reset_mock()
        cancelled={'id':'a'*32,'status':'cancelling'}
        self.service.request=Mock(side_effect=lambda path,payload=None: cancelled if path=='/adaptation/cancel' else {'training':cancelled,'models':[]})
        result=workflow.cancel_training('a'*32)
        self.service.request.assert_any_call('/adaptation/cancel',{'job_id':'a'*32})
        self.assertEqual(result['training']['status'],'cancelling')
        self.start.assert_not_called()

    def test_regeneration_replaces_drafts_and_preserves_reviewed_masks(self):
        workflow=self.workflow()
        model={'object_labels':['stick']}
        workflow.batch('a'*32,model,[(self.ep['id'],self.frame)])
        original=self.service.list_regions(self.ep['id'])[0]
        workflow.job['replace_unreviewed']=True
        workflow.batch('b'*32,model,[(self.ep['id'],self.frame)])
        regions=self.service.list_regions(self.ep['id'])
        self.assertEqual(len(regions),1)
        self.assertNotEqual(regions[0]['object_id'],original['object_id'])
        approved=workflow.approve(self.ep['id'],regions[0]['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        workflow.batch('b'*32,model,[(self.ep['id'],self.frame)])
        self.assertEqual(self.service.list_regions(self.ep['id']),[approved])

    def test_empty_regeneration_clears_drafts_but_failed_inference_keeps_them(self):
        workflow=self.workflow()
        model={'object_labels':['stick']}
        workflow.batch('a'*32,model,[(self.ep['id'],self.frame)])
        workflow.job['replace_unreviewed']=True
        self.service.request=Mock(side_effect=RuntimeError('offline'))
        workflow.batch('b'*32,model,[(self.ep['id'],self.frame)])
        self.assertEqual(len(self.service.list_regions(self.ep['id'])),1)
        self.service.request=Mock(return_value={'strategy':'sam_multiclass_instances_v1','instances':[]})
        workflow.batch('b'*32,model,[(self.ep['id'],self.frame)])
        self.assertEqual(self.service.list_regions(self.ep['id']),[])

    def test_reviewed_training_without_sample_selection(self):
        workflow=self.workflow()
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        region=self.service.list_regions(self.ep['id'])[0]
        workflow.approve(self.ep['id'],region['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        second={**region,'object_id':'second','review_status':'approved'}
        self.store.add_record(self.ep['id'],'regions',second)
        workflow.job['status']='idle'
        self.service.request=Mock(return_value={'status':'queued'})
        workflow.train(w.TrainingRequest(map_name='warehouse',episode_ids=[self.ep['id']],sampled_only=False))
        payload=self.service.request.call_args.args[1]
        self.assertEqual(len(payload['examples']),2)
        summary=self.store.list()[0]
        self.assertEqual(summary['sam_trainable_instance_count'],2)
        self.assertEqual(summary['sam_sample_trainable_instance_count'],0)

    def test_generation_excludes_entire_training_frames_even_when_missing_classes_or_retrying(self):
        import base64
        from test_segmentation import JPEG
        second=self.store.capture(self.ep['id'],{'image_base64':base64.b64encode(JPEG).decode(),
            'captured_at':2,'frame_stamp':'2','camera_topic':'/head'}, {})
        workflow=self.workflow(); workflow.job['status']='idle'
        model={'id':'a'*32,'map_name':'warehouse','object_labels':['stick','dish'],
               'strategy':'sam_multiclass_instances_v1',
               'training_samples':[{'episode_id':self.ep['id'],'frame_id':self.frame['frame_id'],'object_label':'stick'}]}
        state={'training':{'status':'idle'},'models':[model]}
        self.service.request=Mock(side_effect=lambda path,payload=None: state if path=='/adaptation' else self.reply)
        before=self.store.path('raw',self.ep['id'],'samples.jsonl').read_bytes()
        with patch.object(w.threading,'Thread') as thread:
            result=workflow.start_batch(w.BatchRequest(model_id='a'*32,episode_ids=[self.ep['id']],retry_frame_ids=[self.frame['frame_id']]))
        targets=thread.call_args.kwargs['args'][2]
        self.assertEqual([f['frame_id'] for _,f in targets],[second['frame_id']])
        self.assertEqual(result['total'],1)
        self.assertEqual(result['excluded_training'],1)
        self.assertEqual(self.store.path('raw',self.ep['id'],'samples.jsonl').read_bytes(),before)

    def test_deleted_masks_do_not_leave_generation_marker_blocking_retry(self):
        workflow=self.workflow()
        workflow.job['status']='idle'
        frame_id=self.frame['frame_id']
        path=self.store.path('raw',self.ep['id'],'samples.jsonl')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['observation']['auto_labeling']={'sam':{'status':'generated'}}
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        model={'id':'a'*32,'map_name':'warehouse','object_labels':['stick'],
               'strategy':'sam_multiclass_instances_v1','training_samples':[]}
        state={'training':{'status':'idle'},'models':[model]}
        self.service.request=Mock(return_value=state)
        with patch.object(w.threading,'Thread') as thread:
            result=workflow.start_batch(w.BatchRequest(model_id='a'*32,episode_ids=[self.ep['id']]))
        self.assertEqual(result['total'],1)
        self.assertEqual(thread.call_args.kwargs['args'][2][0][1]['frame_id'],frame_id)

    def test_automatic_masks_are_drafts_until_explicit_review(self):
        workflow=self.workflow()
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        regions=self.service.list_regions(self.ep['id'])
        self.assertEqual(len(regions),1)
        region=regions[0]
        self.assertEqual(region['review_status'],'draft')
        self.assertEqual(region['annotator'],'sam-auto')
        self.assertEqual(workflow.summary(),[])
        approved=workflow.approve(self.ep['id'],region['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        self.assertEqual(approved['review_status'],'approved')
        self.assertEqual(workflow.summary()[0]['frame_count'],1)
        with self.assertRaises(HTTPException):
            workflow.approve(self.ep['id'],region['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))

    def test_batch_preserves_existing_masks(self):
        workflow=self.workflow()
        for _ in range(2): workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        self.assertEqual(len(self.service.list_regions(self.ep['id'])),1)

    def test_training_combines_reviewed_sample_masks_from_saved_episodes_on_one_map(self):
        import base64
        from test_segmentation import JPEG
        from supervisor_api import collection as c
        workflow=self.workflow()
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        first=self.service.list_regions(self.ep['id'])[0]
        workflow.approve(self.ep['id'],first['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        workflow.sample(self.ep['id'],w.SampleRequest(count=1))
        second=self.store.start(c.EpisodeRequest(scenario='Shelf 2',group_id='shelf',robot_type='ffw',map_name='warehouse'))
        frame=self.store.capture(second['id'],{'image_base64':base64.b64encode(JPEG).decode(),
            'captured_at':2,'frame_stamp':'2','camera_topic':'/head'}, {})
        self.store.end(second['id'])
        self.reply['object_label']='dish'
        workflow.batch('a'*32,{'object_labels':['dish']},[(second['id'],frame)])
        region=self.service.list_regions(second['id'])[0]
        workflow.approve(second['id'],region['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        workflow.sample(second['id'],w.SampleRequest(count=1))
        summaries={ep['id']:ep for ep in self.store.list()}
        for eid in (self.ep['id'],second['id']):
            self.assertEqual(summaries[eid]['sam_sample_approved_frame_count'],1)
            self.assertEqual(summaries[eid]['sam_sample_approved_instance_count'],1)
            self.assertEqual(summaries[eid]['sam_sample_pending_frame_count'],0)
        workflow.job['status']='idle'
        workflow.train(w.TrainingRequest(map_name='warehouse',episode_ids=[self.ep['id'],second['id']],
            model_name='Combined masks',sampled_only=True))
        call=next(call for call in reversed(self.service.request.call_args_list)
                  if call.args[0]=='/adaptation/train')
        examples=call.args[1]['examples']
        self.assertEqual(len(examples),2)
        self.assertEqual({item['episode_id'] for item in examples},{self.ep['id'],second['id']})
        self.assertEqual({item['object_label'] for item in examples},{'stick','dish'})


    def test_training_uses_reviewed_subset_and_skips_draft_frames(self):
        import base64
        from test_segmentation import JPEG
        from supervisor_api import collection as c
        second=self.store.capture(self.ep['id'],{'image_base64':base64.b64encode(JPEG).decode(),
            'captured_at':2,'frame_stamp':'2','camera_topic':'/head'}, {})
        third=self.store.capture(self.ep['id'],{'image_base64':base64.b64encode(JPEG).decode(),
            'captured_at':3,'frame_stamp':'3','camera_topic':'/head'}, {})
        workflow=self.workflow()
        for frame in (self.frame,second,third):
            workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],frame)])
        for region in self.service.list_regions(self.ep['id']):
            if region['frame_id'] != third['frame_id']:
                workflow.approve(self.ep['id'],region['object_id'],
                    w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        workflow.sample(self.ep['id'],w.SampleRequest(count=3))
        summary=next(ep for ep in self.store.list() if ep['id']==self.ep['id'])
        self.assertEqual(summary['sam_sample_trainable_frame_count'],2)
        self.assertEqual(summary['sam_sample_trainable_instance_count'],2)
        workflow.job['status']='idle'
        workflow.train(w.TrainingRequest(map_name='warehouse',episode_ids=[self.ep['id']],
            model_name='Partial masks',sampled_only=True))
        call=next(call for call in reversed(self.service.request.call_args_list)
                  if call.args[0]=='/adaptation/train')
        self.assertEqual({item['frame_id'] for item in call.args[1]['examples']},
                         {self.frame['frame_id'],second['frame_id']})

    def test_training_rejects_incomplete_or_other_map_selected_episodes(self):
        import base64
        from test_segmentation import JPEG
        from supervisor_api import collection as c
        workflow=self.workflow()
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        region=self.service.list_regions(self.ep['id'])[0]
        workflow.approve(self.ep['id'],region['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        workflow.sample(self.ep['id'],w.SampleRequest(count=1))
        incomplete=self.store.start(c.EpisodeRequest(scenario='Incomplete',group_id='shelf',robot_type='ffw',map_name='warehouse'))
        self.store.capture(incomplete['id'],{'image_base64':base64.b64encode(JPEG).decode(),
            'captured_at':2,'frame_stamp':'2','camera_topic':'/head'}, {})
        self.store.end(incomplete['id'])
        workflow.sample(incomplete['id'],w.SampleRequest(count=1))
        workflow.job['status']='idle'
        with self.assertRaises(HTTPException) as error:
            workflow.train(w.TrainingRequest(map_name='warehouse',episode_ids=[self.ep['id'],incomplete['id']],
                model_name='Incomplete masks',sampled_only=True))
        self.assertEqual(error.exception.status_code,422)
        other=self.store.start(c.EpisodeRequest(scenario='Other map',group_id='other',robot_type='ffw',map_name='other'))
        self.store.end(other['id'])
        with self.assertRaises(HTTPException) as error:
            workflow.train(w.TrainingRequest(map_name='warehouse',episode_ids=[self.ep['id'],other['id']],
                model_name='Mixed maps',sampled_only=True))
        self.assertEqual(error.exception.status_code,422)

    def test_unreviewed_examples_cannot_start_training(self):
        workflow=self.workflow()
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        with self.assertRaises(HTTPException) as error:
            workflow.train(w.TrainingRequest(map_name='warehouse',object_label='stick'))
        self.assertEqual(error.exception.status_code,422)

    def test_empty_model_prediction_is_not_saved_as_approved_background(self):
        workflow=self.workflow()
        self.reply['rle']['counts']=[6]
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        self.assertEqual(workflow.job['empty'],1)
        self.assertEqual(self.service.list_regions(self.ep['id']),[])

    def test_other_map_model_cannot_label_episode(self):
        workflow=self.workflow();workflow.job['status']='idle'
        self.service.request=Mock(return_value={'training':{'status':'idle'},'models':[{'id':'a'*32,'map_name':'other','object_labels':['stick'],'strategy':'sam_multiclass_instances_v1'}]})
        with self.assertRaises(HTTPException) as error:
            workflow.start_batch(w.BatchRequest(model_id='a'*32,episode_ids=[self.ep['id']]))
        self.assertEqual(error.exception.status_code,422)

    def test_two_same_class_instances_remain_independent(self):
        workflow=self.workflow()
        self.service.request=Mock(return_value={'instances':[
            {'rle':{'size':[2,3],'counts':[0,2,4]},'score':.9,'object_label':'stick'},
            {'rle':{'size':[2,3],'counts':[4,2]},'score':.8,'object_label':'stick'}],
            'strategy':'sam_multiclass_instances_v1'})
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        regions=self.service.list_regions(self.ep['id'])
        self.assertEqual(len(regions),2)
        self.assertEqual(len({r['object_id'] for r in regions}),2)
        self.assertEqual({r['instance_number'] for r in regions},{1,2})
        self.assertEqual(workflow.job['created'],2)
        for r in regions:
            approved=workflow.approve(self.ep['id'],r['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
            self.assertEqual(approved['instance_number'],r['instance_number'])
        summary=workflow.summary()[0]
        self.assertEqual((summary['frame_count'],summary['instance_count']),(1,2))
        workflow.train(w.TrainingRequest(map_name='warehouse',object_label='stick'))
        call=next(call for call in self.service.request.call_args_list if call.args[0]=='/adaptation/train')
        examples=call.args[1]['examples']
        self.assertEqual(len(examples),2)
        self.assertNotEqual(examples[0]['object_id'],examples[1]['object_id'])
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        self.assertEqual(len(self.service.list_regions(self.ep['id'])),2)

    def test_legacy_binary_model_requires_retraining(self):
        workflow=self.workflow();workflow.job['status']='idle'
        self.service.request=Mock(return_value={'training':{'status':'idle'},'models':[{'id':'a'*32,'map_name':'warehouse','object_labels':['stick'],'strategy':'full_image_box_binary_decoder'}]})
        with self.assertRaises(HTTPException) as error:
            workflow.start_batch(w.BatchRequest(model_id='a'*32,episode_ids=[self.ep['id']]))
        self.assertIn('Retrain',error.exception.detail)

    def test_sampling_uses_requested_count_and_persists(self):
        workflow=self.workflow()
        original_get=self.store.get
        ep=original_get(self.ep['id'])
        frames=[{'frame_id':str(i)} for i in range(300)]
        self.store.get=Mock(return_value={**ep,'observations':frames})
        workflow.sample(self.ep['id'],w.SampleRequest(count=17))
        selection=self.store.episode(self.ep['id'])['sam_sample']['frame_ids']
        self.assertEqual(len(set(selection)),17)
        self.assertEqual(self.store.episode(self.ep['id'])['sam_sample']['schema_version'],w.SAMPLE_SCHEMA)
        for i,fid in enumerate(selection): self.assertTrue(i*300//17 <= int(fid) < (i+1)*300//17)
        self.assertEqual(self.store.episode(self.ep['id'])['sam_sample']['requested_count'],17)
        self.store.get=original_get
        self.assertEqual(self.store.get(self.ep['id'])['sam_sample']['frame_ids'],selection)

    def test_small_episode_samples_all_frames_without_duplicates(self):
        workflow=self.workflow()
        result=workflow.sample(self.ep['id'],w.SampleRequest(count=1))
        self.assertEqual(result['sam_sample']['frame_ids'],[self.frame['frame_id']])

    def test_sample_only_training_excludes_approved_unsampled_frames(self):
        workflow=self.workflow()
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        region=self.service.list_regions(self.ep['id'])[0]
        workflow.approve(self.ep['id'],region['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        with self.assertRaises(HTTPException):
            workflow.train(w.TrainingRequest(map_name='warehouse',object_label='stick',episode_id=self.ep['id'],sampled_only=True))
        self.assertEqual(self.store.episode(self.ep['id']).get('sam_sample'),None)

    def test_object_labels_persist_and_deduplicate(self):
        workflow=self.workflow()
        for label in ['stick',' stick ','box']:
            result=workflow.add_object_label(self.ep['id'],w.ObjectLabelRequest(label=label))
        self.assertEqual(result['labels'],['box','stick'])
        self.assertEqual(self.store.episode(self.ep['id'])['object_labels'],['box','stick'])
        with self.assertRaises(HTTPException):
            workflow.add_object_label(self.ep['id'],w.ObjectLabelRequest(label='  '))

    def test_status_never_scans_dataset_or_takes_storage_lock(self):
        workflow=self.workflow()
        self.service.request=Mock(return_value={'training':{'status':'idle'},'models':[]})
        with patch.object(workflow,'summary',side_effect=AssertionError('must not scan')):
            self.assertEqual(workflow.status()['batch']['status'],'running')

    def test_status_lists_persisted_models_while_sam_is_stopped(self):
        root=Path(self.temp.name)/'adapters'
        model={'id':'a'*32,'model_name':'Kitchen v1','map_name':'warehouse',
               'strategy':'sam_multiclass_instances_v1','object_labels':['person'],
               'created_at_ms':123}
        path=root/model['id']/'model.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(model),encoding='utf-8')
        workflow=w.SegmentationWorkflow(self.service,adapter_root=root)
        self.service.request=Mock(side_effect=HTTPException(503,'SAM is stopped'))
        state=workflow.status()
        self.assertEqual(state['training']['status'],'unavailable')
        self.assertEqual(state['models'],[model])

    def test_status_merges_live_and_persisted_models_without_duplicates(self):
        root=Path(self.temp.name)/'adapters'
        stored={'id':'a'*32,'model_name':'stored','map_name':'warehouse','created_at_ms':1}
        path=root/stored['id']/'model.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(stored),encoding='utf-8')
        live={**stored,'model_name':'live'}
        workflow=w.SegmentationWorkflow(self.service,adapter_root=root)
        self.service.request=Mock(return_value={'training':{'status':'idle'},'models':[live]})
        self.assertEqual(workflow.status()['models'],[live])

    def test_auto_frame_reads_and_publishes_samples_once(self):
        workflow=self.workflow()
        from supervisor_api import segmentation as seg
        prepared=[({'rle':{'size':[2,3],'counts':counts}},seg.mask_artifacts({'size':[2,3],'counts':counts},3,2))
                  for counts in ([0,2,4],[4,2])]
        with patch.object(seg,'rows',wraps=seg.rows) as reads, patch.object(self.store,'save_samples',wraps=self.store.save_samples) as writes:
            self.assertEqual(self.service.save_predictions(self.ep['id'],self.frame,'stick',prepared),2)
            self.assertEqual(reads.call_count,1)
            self.assertEqual(writes.call_count,1)
        original=self.service.list_regions(self.ep['id'])
        self.assertEqual(self.service.save_predictions(self.ep['id'],self.frame,'stick',prepared,resume=True),0)
        self.assertEqual(self.service.list_regions(self.ep['id']),original)

    def test_resume_adds_missing_predictions_without_overwriting_saved_ids(self):
        workflow=self.workflow()
        from supervisor_api import segmentation as seg
        prepared=[({'rle':{'size':[2,3],'counts':counts}},seg.mask_artifacts({'size':[2,3],'counts':counts},3,2))
                  for counts in ([0,2,4],[4,2])]
        self.service.save_predictions(self.ep['id'],self.frame,'stick',prepared[:1])
        first=self.service.list_regions(self.ep['id'])[0]
        self.assertEqual(self.service.save_predictions(self.ep['id'],self.frame,'stick',prepared,resume=True),1)
        self.assertIn(first,self.service.list_regions(self.ep['id']))

    def test_all_labels_train_and_generate_in_one_model(self):
        workflow=self.workflow()
        self.service.request=Mock(return_value={'strategy':'sam_multiclass_instances_v1','instances':[
            {'object_label':label,'rle':{'size':[2,3],'counts':counts},'score':.9}
            for label,counts in [('person',[0,2,4]),('dish',[2,2,2]),('empty dish',[4,2])]]})
        model={'object_labels':['dish','empty dish','person']}
        workflow.batch('a'*32,model,[(self.ep['id'],self.frame)])
        self.assertEqual(workflow.job['failed'],0)
        regions=self.service.list_regions(self.ep['id'])
        self.assertEqual({r['object_label'] for r in regions},set(model['object_labels']))
        for region in regions:
            workflow.approve(self.ep['id'],region['object_id'],w.ReviewRequest(expected_revision=1,annotator='reviewer'))
        workflow.sample(self.ep['id'],w.SampleRequest(count=1))
        workflow.train(w.TrainingRequest(model_name='  Kitchen v1  ',map_name='warehouse',episode_id=self.ep['id'],sampled_only=True))
        call=next(c for c in self.service.request.call_args_list if c.args[0]=='/adaptation/train')
        self.assertEqual({e['object_label'] for e in call.args[1]['examples']},set(model['object_labels']))
        self.assertEqual(call.args[1]['model_name'],'Kitchen v1')

    def test_existing_person_does_not_block_missing_dish(self):
        workflow=self.workflow()
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        original=self.service.list_regions(self.ep['id'])[0]
        self.service.request=Mock(return_value={'strategy':'sam_multiclass_instances_v1','instances':[
            {'object_label':'dish','rle':{'size':[2,3],'counts':[0,2,4]},'score':.9},
            {'object_label':'stick','rle':{'size':[2,3],'counts':[2,2,2]},'score':.9}]})
        workflow.batch('a'*32,{'object_labels':['stick','dish']},[(self.ep['id'],self.frame)])
        regions=self.service.list_regions(self.ep['id'])
        self.assertEqual(len(regions),2);self.assertIn(original,regions)

    def test_untrained_label_rejected_before_any_frame_write(self):
        workflow=self.workflow();self.reply['object_label']='untrained'
        workflow.batch('a'*32,{'object_labels':['person']},[(self.ep['id'],self.frame)])
        self.assertEqual(workflow.job['failed'],1)
        self.assertEqual(self.service.list_regions(self.ep['id']),[])

    def test_sample_count_is_required_and_cannot_exceed_available_frames(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):w.SampleRequest()
        self.assertEqual(w.SampleRequest(count=80).count,80)
        workflow=self.workflow()
        with self.assertRaises(HTTPException) as error:
            workflow.sample(self.ep['id'],w.SampleRequest(count=2))
        self.assertEqual(error.exception.status_code,422)
        self.assertNotIn('sam_sample',self.store.episode(self.ep['id']))
