"""Naming persistence and job identity, without running GPU training."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from fastapi import HTTPException
from pydantic import ValidationError
from sam2_engine.adaptation import Adaptation,TrainRequest

class ModelNameTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.worker=Adaptation(Mock(lock=threading.Lock()),self.root)
        example=dict(object_label='person',image_base64='AA==',image_sha256='a'*64,frame_id='frame',episode_id='ep',rle={})
        self.args=dict(map_name='warehouse',examples=[example,example],steps=2)
    def stored(self,name=None):
        p=self.root/('a'*32);p.mkdir()
        metadata={'id':'a'*32,'map_name':'warehouse','created_at':1700000000}
        if name is not None:metadata['model_name']=name
        (p/'model.json').write_text(json.dumps(metadata))
        return p
    def test_trimmed_name_reaches_training_job_and_worker(self):
        with patch('sam2_engine.adaptation.threading.Thread') as thread:
            job=self.worker.start(TrainRequest(model_name='  Kitchen v1  ',**self.args))
        self.assertEqual(job['model_name'],'Kitchen v1')
        self.assertEqual(thread.call_args.kwargs['args'][0].model_name,'Kitchen v1')
        self.worker.engine.lock.release()
    def test_duplicate_name_is_rejected_before_training_or_overwrite(self):
        p=self.stored('Kitchen v1');before=(p/'model.json').read_bytes()
        with self.assertRaises(HTTPException) as error:
            self.worker.start(TrainRequest(model_name='kitchen V1',**self.args))
        self.assertEqual(error.exception.status_code,409)
        self.assertFalse(self.worker.engine.lock.locked())
        self.assertEqual((p/'model.json').read_bytes(),before)
    def test_saved_name_is_returned_and_legacy_name_has_no_token(self):
        p=self.stored('Kitchen v1')
        self.assertEqual(self.worker.models()[0]['model_name'],'Kitchen v1')
        metadata=json.loads((p/'model.json').read_text());metadata.pop('model_name')
        (p/'model.json').write_text(json.dumps(metadata))
        model=self.worker.models()[0]
        self.assertIn('warehouse',model['model_name']);self.assertNotIn(model['id'],model['model_name'])
    def test_empty_overlong_or_multiline_names_are_rejected(self):
        for name in ['   ','x'*81,'one\ntwo']:
            with self.subTest(name=name),self.assertRaises(ValidationError):
                TrainRequest(model_name=name,**self.args)
