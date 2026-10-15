"""Automatic labeling policy tests use temporary datasets and fake workers only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from fastapi import HTTPException
from supervisor_api.labeling_automation import allocate_name, automatic_steps, remove_model, AutomaticTrainingRequest
from supervisor_api import collection as c
from supervisor_api import qwen_annotation_workflow as q
from test_qwen_annotation_workflow import QwenWorkflowTests
from test_segmentation_workflow import WorkflowTests

class PolicyTests(unittest.TestCase):
    def test_names_keep_increasing_after_deletion_and_use_actual_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first=allocate_name(root,'qwen','house',[], 'Qwen/Qwen3.5-0.8B')
            self.assertEqual(first,'qwen0.8b_house_1')
            c.write_json(root/'house'/first/'model.json',{'id':'model','model_name':first})
            remove_model(root,'model',2)
            self.assertEqual(allocate_name(root,'qwen','house',[], 'Qwen/Qwen3.5-0.8B'),'qwen0.8b_house_2')
            self.assertEqual(allocate_name(root,'sam','house',[{'model_name':'sam_house_7'}]),'sam_house_8')

    def test_steps_scale_with_worker_samples_and_stay_bounded(self):
        self.assertEqual([automatic_steps('sam',n) for n in [2,30,80,4096]],[100,300,800,2000])
        self.assertEqual([automatic_steps('qwen',n) for n in [2,30,80,4000]],[100,150,400,2000])

    def test_deletion_cannot_escape_model_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'models'; outside=Path(tmp)/'outside'
            c.write_json(outside/'model.json',{'id':'outside'})
            root.mkdir();(root/'link').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(HTTPException):remove_model(root,'outside',1)
            self.assertTrue((outside/'model.json').exists())

class QwenAutomaticTests(unittest.TestCase):
    setUp=QwenWorkflowTests.setUp
    label=QwenWorkflowTests.label
    model=QwenWorkflowTests.model

    def test_auto_training_uses_all_same_map_manual_samples_and_auto_configuration(self):
        self.workflow.sample(self.ep['id'],q.SampleRequest(count=3))
        for frame in self.frames:self.label(frame)
        # A fourth frame from a second saved episode on the same map.
        import base64
        from test_segmentation import JPEG
        second=self.store.start(c.EpisodeRequest(map_name='warehouse'))
        frame=self.store.capture(second['id'],{'image_base64':base64.b64encode(JPEG).decode(),'camera_topic':'/head','captured_at':1,'frame_stamp':'1'}, {})
        self.store.end(second['id']);self.workflow.sample(second['id'],q.SampleRequest(count=1))
        self.store.annotate(second['id'],c.AnnotationRequest(frame_id=frame['frame_id'],decision='wait',reason='Wait.',annotator='collect-ui',review_status='approved'))
        with patch.object(q.threading,'Thread'):
            job=self.workflow.auto_train(AutomaticTrainingRequest(map_name='warehouse'))
        self.assertEqual(job['sample_count'],4)
        self.assertEqual(set(job['episode_ids']),{self.ep['id'],second['id']})
        self.assertEqual(job['steps'],100)
        self.assertRegex(job['model_name'],r'^qwen[0-9.]+b_warehouse_1$')

    def test_reviewed_auto_draft_never_becomes_manual_training_seed(self):
        self.workflow.sample(self.ep['id'],q.SampleRequest(count=3))
        for frame in self.frames[:2]: self.label(frame)
        third=self.frames[2]
        self.store.annotate(self.ep['id'],c.AnnotationRequest(frame_id=third['frame_id'],decision='wait',reason='Auto.',annotator='qwen:job',review_status='draft'))
        self.label(third)
        with patch.object(q.threading,'Thread'):
            job=self.workflow.auto_train(AutomaticTrainingRequest(map_name='warehouse'))
        self.assertEqual(job['sample_count'],2)
        evidence=self.store.get(self.ep['id'])['observations'][2]['auto_labeling']['qwen']
        self.assertEqual(evidence['status'],'generated')

    def test_model_deletion_blocks_active_jobs_or_loaded_model_and_keeps_dataset(self):
        self.model()
        self.workflow.jobs['batch']={'status':'running'}
        with self.assertRaises(HTTPException):self.workflow.delete_model('a'*32)
        self.workflow.jobs['batch']={'status':'idle'}
        self.workflow.model_in_use=lambda _: True
        with self.assertRaises(HTTPException):self.workflow.delete_model('a'*32)
        self.workflow.model_in_use=lambda _: False
        self.workflow.delete_model('a'*32)
        self.assertEqual(self.workflow.models(),[])
        self.assertEqual(len(self.store.get(self.ep['id'])['observations']),3)

class SamAutomaticTests(unittest.TestCase):
    setUp=WorkflowTests.setUp
    workflow=WorkflowTests.workflow

    def test_empty_predictions_persist_generation_evidence_and_no_extra_dataset_read(self):
        workflow=self.workflow()
        self.service.request=Mock(return_value={'strategy':'sam_multiclass_instances_v1','instances':[]})
        from supervisor_api import segmentation
        with patch.object(segmentation,'rows',wraps=segmentation.rows) as read:
            workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        self.assertEqual(read.call_count,1)
        self.assertEqual(workflow.job['failed'],0)
        frame=self.store.get(self.ep['id'])['observations'][0]
        self.assertEqual(frame['auto_labeling']['sam']['status'],'generated')
        self.assertEqual(self.service.list_regions(self.ep['id']),[])

    def test_failed_prediction_does_not_enable_review(self):
        workflow=self.workflow();self.service.request=Mock(side_effect=RuntimeError('offline'))
        workflow.batch('a'*32,{'object_labels':['stick']},[(self.ep['id'],self.frame)])
        self.assertEqual(workflow.job['failed'],1)
        self.assertNotIn('auto_labeling',self.store.get(self.ep['id'])['observations'][0])

    def test_sam_auto_training_uses_manual_samples_and_names_model(self):
        from test_segmentation import SegmentationTests
        from supervisor_api import segmentation_workflow as w
        preview=SegmentationTests.preview(self)
        SegmentationTests.save(self,preview,object_id='one')
        SegmentationTests.save(self,preview,object_id='two')
        self.store.end(self.ep['id'])
        workflow=w.SegmentationWorkflow(self.service,Path(self.temp.name)/'models')
        workflow.sample(self.ep['id'],w.SampleRequest(count=1))
        self.service.request=Mock(return_value={'status':'queued'})
        workflow.train(w.TrainingRequest(map_name='warehouse',sampled_only=True,automatic=True))
        payload=self.service.request.call_args.args[1]
        self.assertEqual(payload['model_name'],'sam_warehouse_1')
        self.assertEqual(payload['steps'],100)
        self.assertEqual(len(payload['examples']),2)
