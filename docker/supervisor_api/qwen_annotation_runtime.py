"""Run offline annotation jobs in the existing Qwen training image."""
import codecs
import json
import os
import threading
from pathlib import Path


class JobCancelled(RuntimeError):
    """Intentional cancellation, separate from a failed training run."""


class QwenAnnotationRuntime:
    def __init__(self, client_factory, workspace, project, huggingface):
        self.client_factory, self.workspace = client_factory, workspace
        self.project, self.huggingface = project, huggingface
        self.lifecycle_lock = threading.RLock()
        self.cancelled = set()

    def _check_cancelled(self, job_id):
        with self.lifecycle_lock:
            if job_id in self.cancelled:
                raise JobCancelled('Qwen training cancelled')

    def cancel(self, job_id):
        """Serialize cancellation with creation and stop only this job's container."""
        import docker
        client = self.client_factory()
        try:
            with self.lifecycle_lock:
                self.cancelled.add(job_id)
                try:
                    container = client.containers.get('qwen_annotation_job')
                except docker.errors.NotFound:
                    return
                if container.labels.get('cyclo.qwen.annotation.job') != job_id:
                    return
                container.reload()
                if container.status in ('running', 'restarting', 'paused'):
                    container.stop(timeout=10)
        finally:
            client.close()

    def release(self, job_id):
        with self.lifecycle_lock:
            self.cancelled.discard(job_id)

    def execute(self, request_path, job_id, on_event):
        import docker
        client = self.client_factory()
        container = None
        log = None
        try:
            with self.lifecycle_lock:
                self._check_cancelled(job_id)
                name = 'qwen_annotation_job'
                try:
                    container = client.containers.get(name)
                except docker.errors.NotFound:
                    pass
                if container is not None and container.labels.get('cyclo.qwen.annotation.job') != job_id:
                    if container.status in ('running','restarting','paused'):
                        raise RuntimeError('Another Qwen annotation job is running')
                    container.remove(); container = None
                if container is None:
                    workspace, project, cache = self.workspace(), self.project(), self.huggingface()
                    if not all((workspace,project,cache)):
                        raise RuntimeError('Workspace, source and Hugging Face mounts are unavailable')
                    repo = Path(project).parent
                    container = client.containers.run(
                        os.getenv('QWEN_ANNOTATION_IMAGE','mission-reasoning/qwen-training:0.1.0'),
                        ['-u','/app/training/annotation_worker.py',str(request_path)],
                        entrypoint=['python'], name=name, detach=True, init=True, shm_size='2g',
                        labels={'cyclo.qwen.annotation.job':job_id},
                        environment={'PYTHONPATH':'/app:/app/contracts', 'NVIDIA_DRIVER_CAPABILITIES':'compute,utility'},
                        volumes={str(workspace):{'bind':'/workspace','mode':'ro'},
                            str(Path(workspace)/'model/qwen/annotation'):{'bind':'/workspace/model/qwen/annotation','mode':'rw'},
                            str(Path(workspace)/'model/qwen/reasoning'):{'bind':'/workspace/model/qwen/reasoning','mode':'rw'},
                            str(repo/'cyclo_brain/policy/qwen/training'):{'bind':'/app/training','mode':'ro'},
                            str(repo/'cyclo_brain/policy/qwen/qwen_engine'):{'bind':'/app/qwen_engine','mode':'ro'},
                            str(repo/'docker/supervisor_api'):{'bind':'/app/contracts','mode':'ro'},
                            str(cache):{'bind':'/root/.cache/huggingface','mode':'rw'}},
                        device_requests=[docker.types.DeviceRequest(
                            device_ids=[os.getenv('QWEN_GPU_DEVICE','0')],capabilities=[['gpu']])])
            if request_path.is_file():
                log = request_path.with_name('worker.log').open('a', encoding='utf-8')
            buffer = ''
            decoder = codecs.getincrementaldecoder('utf-8')(errors='strict')
            tail = []
            def dispatch(line):
                self._check_cancelled(job_id)
                tail.append(line)
                del tail[:-80]
                try: event = json.loads(line)
                except ValueError: return
                if isinstance(event, dict) and event.get('event'):
                    on_event(event)
            for chunk in container.logs(stream=True, follow=True):
                decoded = decoder.decode(chunk)
                if log is not None:
                    log.write(decoded); log.flush()
                buffer += decoded
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    dispatch(line)
            decoded = decoder.decode(b'', final=True)
            if log is not None:
                log.write(decoded); log.flush()
            buffer += decoded
            if buffer:
                dispatch(buffer)
            result = container.wait()
            self._check_cancelled(job_id)
            if result.get('StatusCode') != 0:
                raise RuntimeError(f"Qwen job failed (exit code {result.get('StatusCode')}): "+'\n'.join(tail)[-6000:])
        except Exception:
            if container is not None and container.labels.get('cyclo.qwen.annotation.job') == job_id:
                try:
                    container.reload()
                    if container.status in ('running','restarting','paused'):
                        container.stop(timeout=10)
                except docker.errors.NotFound:
                    pass
            raise
        finally:
            if log is not None:
                log.close()
            try:
                if container is not None and container.labels.get('cyclo.qwen.annotation.job') == job_id:
                    try:
                        container.reload()
                        if container.status not in ('running','restarting','paused'):
                            container.remove()
                    except docker.errors.NotFound:
                        pass
            finally:
                client.close()
