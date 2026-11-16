"""Background Worker image builds with bounded logs and one build at a time."""

from collections import deque
import os
import signal
import subprocess
import threading
import time


class WorkerBuilds:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs = {}
        self._thread = None
        self._process = None
        self._closed = False

    def status(self, name):
        with self._lock:
            job = self._jobs.get(name)
            if job is None:
                return {"state": "idle", "message": "", "log_tail": []}
            return {**job, "log_tail": list(job["log_tail"])}

    def start(self, name, command, env, verify, release):
        with self._lock:
            if self._closed:
                raise RuntimeError("Worker image builder is shutting down")
            if self._thread and self._thread.is_alive():
                raise RuntimeError("A Worker image build is already running")
            self._jobs[name] = {
                "state": "building", "message": "Building image",
                "started_at": time.time(), "finished_at": None,
                "returncode": None, "log_tail": deque(maxlen=200),
            }
            self._thread = threading.Thread(
                target=self._run, args=(name, command, env, verify, release),
                daemon=True, name="worker-image-build",
            )
            try:
                self._thread.start()
            except Exception as exc:
                self._thread = None
                self._jobs[name].update(state="failed", message=str(exc), finished_at=time.time())
                raise
            return self.status(name)

    def _run(self, name, command, env, verify, release):
        job = self._jobs[name]
        state, message, returncode = "failed", "Build interrupted", None
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Worker image builder is shutting down")
                process = self._process = subprocess.Popen(
                    command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, errors="replace", start_new_session=True,
                )
            with process.stdout:
                while line := process.stdout.readline(2048):
                    with self._lock:
                        job["log_tail"].append(line.rstrip())
            returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"Image build failed (exit {returncode}); check build log")
            verify()
            state, message = "succeeded", "Image build complete"
        except Exception as exc:
            message = str(exc)
        finally:
            try:
                release()
            finally:
                with self._lock:
                    self._process = None
                    job.update(state=state, message=message, returncode=returncode,
                               finished_at=time.time())

    def close(self):
        with self._lock:
            self._closed = True
            process, thread = self._process, self._thread
        if process:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if thread:
            thread.join(timeout=5)
            # A shell can exit while a child still holds the output pipe open.
            if thread.is_alive() and process:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            thread.join()
