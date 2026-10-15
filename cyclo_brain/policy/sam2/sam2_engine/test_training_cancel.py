"""Cancellation and engine ownership, without loading SAM or a GPU."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi import HTTPException
from sam2_engine.adaptation import Adaptation, TrainRequest, TrainingCancelled

class CancelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.worker=Adaptation(Mock(lock=threading.Lock()),Path(self.temp.name))
        example=dict(object_label='person',image_base64='AA==',image_sha256='a'*64,frame_id='frame',episode_id='ep',rle={})
        self.req=TrainRequest(map_name='house',model_name='Test',examples=[example,example],steps=2)
    def queued(self):
        with patch('sam2_engine.adaptation.threading.Thread'):return self.worker.start(self.req)
    def test_queued_cancel_never_loads_model_and_releases_engine_before_retry(self):
        job=self.queued();self.worker.cancel(job['id'])
        with self.assertRaises(HTTPException):self.worker.start(self.req)
        self.worker.train(self.req,job['id'])
        self.assertEqual(self.worker.job['status'],'cancelled')
        self.assertNotIn('error',self.worker.job)
        self.assertFalse(self.worker.engine.lock.locked())
        self.worker.engine.load.assert_not_called()
        self.assertEqual(self.worker.models(),[])
        next_job=self.queued();self.assertNotEqual(next_job['id'],job['id'])
        self.assertFalse(self.worker.cancel_requested.is_set());self.worker.engine.lock.release()
    def test_running_cancel_ends_without_failure_and_unpublishes_no_model(self):
        job=self.queued()
        def train(worker,req,model_id):
            worker.cancel(model_id)
            self.assertTrue(worker.engine.lock.locked())
            self.assertEqual(worker.job['status'],'cancelling')
            worker.check_cancelled()
        module=Mock(train_model=train)
        with patch.dict(sys.modules,{'sam2_engine.class_training':module}):self.worker.train(self.req,job['id'])
        self.assertEqual(self.worker.job['status'],'cancelled')
        self.assertFalse(self.worker.engine.lock.locked());self.assertEqual(self.worker.models(),[])
    def test_stale_cancel_and_finished_cancel_are_rejected(self):
        job=self.queued()
        with self.assertRaises(HTTPException):self.worker.cancel('f'*32)
        self.assertFalse(self.worker.cancel_requested.is_set())
        self.worker.job['status']='completed'
        with self.assertRaises(HTTPException):self.worker.cancel(job['id'])
        self.worker.engine.lock.release()
    def test_real_training_error_remains_failed(self):
        job=self.queued();module=Mock(train_model=Mock(side_effect=ValueError('Bad mask')))
        with patch.dict(sys.modules,{'sam2_engine.class_training':module}):self.worker.train(self.req,job['id'])
        self.assertEqual(self.worker.job['status'],'failed');self.assertEqual(self.worker.job['error'],'Bad mask')
        self.assertFalse(self.worker.engine.lock.locked())
