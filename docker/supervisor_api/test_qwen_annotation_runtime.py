"""Job lifecycle and lossless result streaming without starting GPU containers."""
import threading
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import Mock
import docker
import test_segmentation  # Establish the supervisor_api package import path.
from supervisor_api.qwen_annotation_runtime import QwenAnnotationRuntime, JobCancelled

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.client=Mock()
        self.container=Mock(status='exited',labels={'cyclo.qwen.annotation.job':'job1'})
        self.client.containers.get.return_value=self.container
        self.container.wait.return_value={'StatusCode':0}
        self.runtime=QwenAnnotationRuntime(lambda:self.client,lambda:'/host/workspace',lambda:'/host/repo/docker',lambda:'/host/cache')

    def test_cancel_stops_only_matching_running_container_and_closes_client(self):
        self.container.status='running'
        self.runtime.cancel('job1')
        self.container.stop.assert_called_once_with(timeout=10)
        self.client.close.assert_called_once()
        self.container.remove.assert_not_called()

    def test_cancelled_queued_job_cannot_create_a_container(self):
        self.client.containers.get.side_effect=docker.errors.NotFound('missing')
        self.runtime.cancel('job1')
        with self.assertRaises(JobCancelled):
            self.runtime.execute(Path('/request.json'),'job1',Mock())
        self.client.containers.run.assert_not_called()

    def test_cancel_does_not_touch_a_replacement_job(self):
        self.container.status='running'
        self.container.labels={'cyclo.qwen.annotation.job':'replacement'}
        self.runtime.cancel('old')
        self.container.stop.assert_not_called()
        self.container.remove.assert_not_called()

    def test_cancel_during_log_stream_is_intentional_and_discards_late_events(self):
        self.container.status='running'
        def stopped(**kwargs):self.container.status='exited'
        self.container.stop.side_effect=stopped
        def logs(**kwargs):
            self.runtime.cancel('job1')
            yield b'{"event":"progress","step":99}\n'
        self.container.logs.side_effect=logs
        event=Mock()
        with self.assertRaises(JobCancelled):self.runtime.execute(Path('/request'),'job1',event)
        event.assert_not_called()
        self.container.remove.assert_called_once()

    def test_cancel_serializes_with_container_creation(self):
        self.client.containers.get.side_effect=docker.errors.NotFound('missing')
        entered=threading.Event();release=threading.Event();cancelled=threading.Event()
        self.container.status='running';self.container.logs.return_value=[]
        def create(*args,**kwargs):entered.set();release.wait(2);return self.container
        self.client.containers.run.side_effect=create
        def stop(**kwargs):self.container.status='exited'
        self.container.stop.side_effect=stop
        def cancel():self.runtime.cancel('job1');cancelled.set()
        def execute():
            try:self.runtime.execute(Path('/request'),'job1',Mock())
            except JobCancelled:pass
        worker=threading.Thread(target=execute);worker.start();self.assertTrue(entered.wait(1))
        # After creation, the cancel lookup finds this exact container.
        self.client.containers.get.side_effect=None;self.client.containers.get.return_value=self.container
        stopper=threading.Thread(target=cancel);stopper.start()
        self.assertFalse(cancelled.wait(.05))
        release.set();worker.join(2);stopper.join(2)
        self.assertFalse(worker.is_alive());self.assertFalse(stopper.is_alive())

    def test_reconnects_to_same_job_without_launching_a_second_container(self):
        self.container.logs.return_value=[b'{"event":"progress","step":1}\n']
        event=Mock()
        self.runtime.execute(Path('/workspace/request.json'),'job1',event)
        self.client.containers.run.assert_not_called()
        event.assert_called_once_with({'event':'progress','step':1})
        self.container.remove.assert_called_once()
        self.client.close.assert_called_once()

    def test_split_unicode_and_final_line_reach_storage_without_corruption(self):
        value={'event':'prediction','text':'사람이 보입니다.'}
        raw=json.dumps(value,ensure_ascii=False).encode()
        start=raw.index('사'.encode())
        self.container.logs.return_value=[raw[:start+1],raw[start+1:start+2],raw[start+2:]]
        event=Mock()
        self.runtime.execute(Path('/workspace/request.json'),'job1',event)
        event.assert_called_once_with(value)

    def test_callback_failure_stops_only_the_owned_job(self):
        self.container.status='running'
        self.container.logs.return_value=[b'{"event":"prediction"}\n']
        def stopped(**kwargs):self.container.status='exited'
        self.container.stop.side_effect=stopped
        with self.assertRaisesRegex(ValueError,'write failed'):
            self.runtime.execute(Path('/workspace/request.json'),'job1',Mock(side_effect=ValueError('write failed')))
        self.container.stop.assert_called_once_with(timeout=10)
        self.container.remove.assert_called_once()
        self.client.close.assert_called_once()

    def test_other_running_job_is_never_stopped_or_removed(self):
        self.container.status='running'
        self.container.labels={'cyclo.qwen.annotation.job':'other'}
        with self.assertRaisesRegex(RuntimeError,'Another Qwen'):
            self.runtime.execute(Path('/workspace/request.json'),'job1',Mock())
        self.container.stop.assert_not_called()
        self.container.remove.assert_not_called()
        self.client.close.assert_called_once()

    def test_nonzero_exit_keeps_failure_detail(self):
        self.container.logs.return_value=[b'CUDA out of memory\n']
        self.container.wait.return_value={'StatusCode':1}
        with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):
            self.runtime.execute(Path('/workspace/request.json'),'job1',Mock())
        self.container.remove.assert_called_once()

    def test_new_worker_mounts_annotation_and_reasoning_model_roots_writable(self):
        self.client.containers.get.side_effect=docker.errors.NotFound('missing')
        created=Mock(status='exited',labels={'cyclo.qwen.annotation.job':'job1'})
        created.logs.return_value=[];created.wait.return_value={'StatusCode':0}
        self.client.containers.run.return_value=created
        self.runtime.execute(Path('/workspace/model/qwen/annotation/jobs/job1/request.json'),'job1',Mock())
        volumes=self.client.containers.run.call_args.kwargs['volumes']
        self.assertEqual(volumes['/host/workspace/model/qwen/annotation']['mode'],'rw')
        self.assertEqual(volumes['/host/workspace/model/qwen/reasoning']['mode'],'rw')


    def test_failure_preserves_original_error_and_full_log_before_container_removal(self):
        lines=['CUDA original failure before worker traceback']+['traceback detail '+str(i) for i in range(20)]+['Qwen fine-tuning failed']
        self.container.logs.return_value=[(line+'\n').encode() for line in lines]
        self.container.wait.return_value={'StatusCode':1}
        with tempfile.TemporaryDirectory() as directory:
            request=Path(directory)/'request.json';request.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'CUDA original failure') as raised:
                self.runtime.execute(request,'job1',Mock())
            self.assertIn('exit code 1',str(raised.exception))
            self.assertEqual(request.with_name('worker.log').read_text(),'\n'.join(lines)+'\n')
        self.container.remove.assert_called_once()

    def test_full_log_preserves_split_unicode(self):
        raw='사람 이미지 처리\n'.encode()
        self.container.logs.return_value=[raw[:1],raw[1:2],raw[2:]]
        with tempfile.TemporaryDirectory() as directory:
            request=Path(directory)/'request.json';request.write_text('{}')
            self.runtime.execute(request,'job1',Mock())
            self.assertEqual(request.with_name('worker.log').read_text(),raw.decode())
