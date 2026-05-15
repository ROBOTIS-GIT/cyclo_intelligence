# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""/data/hub service — HfOperation handler.

Step 3 Part C2c migrated the real HF pipeline here from
orchestrator.OrchestratorNode (atomic swap — ownership of HfApiWorker
and the /huggingface/status publisher moves in a single commit so the
worker is never alive in two places simultaneously).

Request routing (Part C2c-ui):
  * UI calls /data/hub directly with HfOperation (enum payload). It
    does not carry tokens; HubService falls back to the file-backed
    HFEndpointStore to resolve endpoint + token.
  * Programmatic callers (e.g. future orchestrator plumbing) may
    inline endpoint + token in the request to skip the store lookup.
  * HFEndpointStore is mutated only by the orchestrator-side services
    (set / get / list / select_hf_endpoint). The store uses its own
    internal file lock, so cyclo_data's read-only resolve() is safe.

HfApiWorker lifecycle (unchanged from orchestrator):
  * Eager start on service __init__.
  * 2 Hz status poll → HFOperationStatus publish on /huggingface/status
    (topic name preserved for UI backwards compat).
  * Auto-shutdown after 5 idle cycles; lazy restart on next request.
  * CANCEL runs non-blocking cleanup in a worker thread, then emits
    three "Idle / stop" heartbeats so the UI progress bar resets.
"""

import threading
import time
from typing import Optional

from cyclo_data.hub.api_worker import HfApiWorker
from cyclo_data.hub.endpoint_store import HFEndpointStore

from interfaces.msg import DataOperationStatus, HFOperationStatus
from interfaces.srv import HfOperation


_OPERATION_STR = {
    HfOperation.Request.UPLOAD: 'upload',
    HfOperation.Request.DOWNLOAD: 'download',
    HfOperation.Request.CANCEL: 'cancel',
}

_REPO_TYPE_STR = {
    HfOperation.Request.DATASET: 'dataset',
    HfOperation.Request.MODEL: 'model',
}


class HubService:
    SERVICE_NAME = '/data/hub'
    STATUS_TOPIC = '/huggingface/status'
    STATUS_PERIOD_SEC = 0.5
    IDLE_TICKS_BEFORE_SHUTDOWN = 5

    def __init__(self, node, status_publisher):
        self._node = node
        self._data_status_pub = status_publisher  # umbrella /data/status

        self._hf_status_pub = node.create_publisher(
            HFOperationStatus, self.STATUS_TOPIC, 10)

        # Read-only fallback resolver. When a caller (e.g. the UI) invokes
        # /data/hub without inlining endpoint + token we consult the
        # file-based store here. Orchestrator remains the sole writer
        # (set_hf_user / select_hf_endpoint / …), so concurrent reads are
        # safe under the store's internal lock.
        self._endpoint_store = HFEndpointStore()

        self._api_worker: Optional[HfApiWorker] = None
        self._status_timer = None
        self._idle_count = 0
        self._last_status = None
        self._cancel_in_progress = False
        # Service callback (_callback) and the 2 Hz _status_timer_callback
        # are both registered on io_callback_group which is a
        # ReentrantCallbackGroup — under MultiThreadedExecutor they can
        # fire concurrently. _state_lock serialises mutations of the
        # worker handle, idle counter, last-status snapshot and the
        # cancel flag so the timer can't tear `_api_worker = None` out
        # from under a check-then-use in _handle_transfer.
        self._state_lock = threading.Lock()

        self._server = node.create_service(
            HfOperation,
            self.SERVICE_NAME,
            self._callback,
            callback_group=node.io_callback_group,
        )
        node.get_logger().info(f'Service advertised: {self.SERVICE_NAME}')

        self._init_worker()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _init_worker(self):
        # Build the worker outside the lock — HfApiWorker() forks a child
        # process and start() blocks on the pipe handshake.
        try:
            worker = HfApiWorker()
            if not worker.start():
                self._node.get_logger().error('Failed to start HF API Worker')
                return
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f'Error initializing HF API Worker: {exc}')
            return

        timer = self._node.create_timer(
            self.STATUS_PERIOD_SEC,
            self._status_timer_callback,
            callback_group=self._node.io_callback_group,
        )
        with self._state_lock:
            self._api_worker = worker
            self._status_timer = timer
            self._idle_count = 0
        self._node.get_logger().info('HF API Worker started')

    def shutdown(self):
        """Explicit cleanup hook invoked by cyclo_data_node on shutdown."""
        self._cleanup_worker()

    # ------------------------------------------------------------------
    # Service callback
    # ------------------------------------------------------------------

    def _callback(self, request, response):
        op_str = _OPERATION_STR.get(request.operation)
        repo_type_str = _REPO_TYPE_STR.get(request.repo_type)

        if op_str is None:
            response.success = False
            response.job_id = ''
            response.message = f'Unknown HF operation: {request.operation}'
            return response
        if repo_type_str is None:
            response.success = False
            response.job_id = ''
            response.message = f'Unknown repo_type: {request.repo_type}'
            return response

        with self._state_lock:
            if self._cancel_in_progress:
                response.success = False
                response.job_id = ''
                response.message = 'HF API Worker is currently canceling'
                return response

        if request.operation == HfOperation.Request.CANCEL:
            return self._handle_cancel(response)

        return self._handle_transfer(request, response, op_str, repo_type_str)

    def _handle_cancel(self, response):
        with self._state_lock:
            self._cancel_in_progress = True
        try:
            self._cleanup_worker_with_threading()
            response.success = True
            response.job_id = ''
            response.message = 'Cancellation started.'
            self._publish_data_status(
                DataOperationStatus.CANCELLED, 'cancel', 'HF cancellation started.')
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f'Error during cancel: {exc}')
            response.success = False
            response.job_id = ''
            response.message = f'Cancel failed: {exc}'
        finally:
            with self._state_lock:
                self._cancel_in_progress = False
        return response

    def _handle_transfer(self, request, response, op_str, repo_type_str):
        if not request.repo_id or not request.local_dir:
            response.success = False
            response.job_id = ''
            response.message = 'repo_id and local_dir are required for UPLOAD/DOWNLOAD.'
            return response

        endpoint = request.endpoint
        token = request.token
        if not endpoint or not token:
            # UI (direct /data/hub caller) does not carry tokens in its
            # request; resolve via the endpoint store instead.
            entry = self._endpoint_store.resolve(endpoint or '')
            if entry is None:
                response.success = False
                response.job_id = ''
                response.message = (
                    'No HuggingFace endpoint registered '
                    f'(requested: {endpoint or "<active>"}). '
                    'Register a token from the UI first.'
                )
                return response
            endpoint = entry.endpoint
            token = entry.token

        # Snapshot the handle so the timer can't None it out between
        # the is_alive / is_busy / send_request calls below.
        with self._state_lock:
            worker = self._api_worker
        if worker is None or not worker.is_alive():
            self._node.get_logger().info('HF API Worker not running, restarting...')
            self._init_worker()
            with self._state_lock:
                worker = self._api_worker

        if worker is None:
            response.success = False
            response.job_id = ''
            response.message = 'HF API Worker could not be started.'
            return response

        if worker.is_busy():
            self._node.get_logger().warning('HF API Worker is currently busy with another task')
            response.success = False
            response.job_id = ''
            response.message = 'HF API Worker is currently busy with another task'
            return response

        request_data = {
            'mode': op_str,
            'repo_id': request.repo_id,
            'local_dir': request.local_dir,
            'repo_type': repo_type_str,
            'author': request.author,
            'endpoint': endpoint,
            'token': token,
        }
        if worker.send_request(request_data):
            self._node.get_logger().info(
                f'HF API request sent: {op_str} {request.repo_id} via {endpoint}')
            response.success = True
            response.job_id = ''  # HfApiWorker has no native async job_id
            response.message = (
                f'HF API request started: {op_str} for {request.repo_id} '
                f'via {endpoint}'
            )
            self._publish_data_status(
                DataOperationStatus.RUNNING,
                f'{op_str}_{repo_type_str}',
                response.message,
            )
        else:
            self._node.get_logger().error('Failed to send request to HF API Worker')
            response.success = False
            response.job_id = ''
            response.message = 'Failed to send request to HF API Worker'
        return response

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def _status_timer_callback(self):
        with self._state_lock:
            worker = self._api_worker
        if worker is None:
            return
        try:
            status = worker.check_task_status()
            self._publish_hf_status(status)

            current = status.get('status', 'Unknown')
            with self._state_lock:
                last_status = self._last_status
                last = (
                    last_status.get('status', 'Unknown')
                    if last_status else 'Unknown'
                )
                changed = last_status is not None and last != current
                self._last_status = status
                if current == 'Idle':
                    self._idle_count += 1
                    idle_count = self._idle_count
                else:
                    self._idle_count = 0
                    idle_count = 0

            if changed:
                self._node.get_logger().info(
                    f'HF API Status changed: {last} -> {current}')

            if current == 'Idle' and idle_count >= self.IDLE_TICKS_BEFORE_SHUTDOWN:
                self._node.get_logger().info(
                    f'HF API Worker idle for {self.IDLE_TICKS_BEFORE_SHUTDOWN} '
                    'cycles, shutting down worker and timer.')
                self._cleanup_worker()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f'Error in HF status timer callback: {exc}')

    def _publish_hf_status(self, status):
        msg = HFOperationStatus()
        msg.operation = status.get('operation', 'Unknown')
        msg.status = status.get('status', 'Unknown')
        msg.repo_id = status.get('repo_id', '')
        msg.local_path = status.get('local_path', '')
        msg.message = status.get('message', '')
        progress = status.get('progress', {})
        msg.progress_current = progress.get('current', 0)
        msg.progress_total = progress.get('total', 0)
        msg.progress_percentage = progress.get('percentage', 0.0)
        self._hf_status_pub.publish(msg)

    def _publish_data_status(self, status: int, stage: str, message: str) -> None:
        msg = DataOperationStatus()
        msg.operation_type = DataOperationStatus.OP_HF
        msg.status = status
        msg.job_id = ''
        msg.progress_percentage = 0.0
        msg.stage = stage
        msg.message = message
        self._data_status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_worker_with_threading(self):
        """Non-blocking cleanup — matches the orchestrator behaviour."""
        # Atomically detach the handles so a concurrent _callback /
        # _status_timer_callback can't grab them mid-teardown. The actual
        # worker.stop() runs on a daemon thread outside the lock because
        # it can block on the child-process join.
        with self._state_lock:
            timer = self._status_timer
            worker = self._api_worker
            self._status_timer = None
            self._api_worker = None

        if timer is None and worker is None:
            self._node.get_logger().info('No HF API components to cleanup')
            return

        def cleanup():
            try:
                if timer is not None:
                    timer.cancel()
                if worker is not None:
                    worker.stop()
                self._node.get_logger().info('HF API Worker cleaned up successfully')
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().error(f'Error in cleanup thread: {exc}')

        try:
            self._node.get_logger().info('Starting non-blocking HF API Worker cleanup...')
            thread = threading.Thread(target=cleanup, daemon=True)
            thread.start()

            # Three "Idle / stop" heartbeats so the UI progress widget resets
            # (preserves the original orchestrator behaviour).
            for _ in range(3):
                self._publish_hf_status({
                    'status': 'Idle',
                    'operation': 'stop',
                    'repo_id': '',
                    'local_path': '',
                    'message': 'Canceled by stop command',
                    'progress': {'current': 0, 'total': 0, 'percentage': 0.0},
                })
                time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(
                f'Error starting non-blocking HF API Worker cleanup: {exc}')
            # Inline fall-back — handles are already detached.
            try:
                if timer is not None:
                    timer.cancel()
                if worker is not None:
                    worker.stop()
            except Exception as inner:  # noqa: BLE001
                self._node.get_logger().error(
                    f'Error cleaning up HF API Worker (fallback): {inner}')

    def _cleanup_worker(self):
        with self._state_lock:
            timer = self._status_timer
            worker = self._api_worker
            self._status_timer = None
            self._api_worker = None
        try:
            if timer is not None:
                timer.cancel()
            if worker is not None:
                worker.stop()
            self._node.get_logger().info('HF API Worker cleaned up successfully')
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(f'Error cleaning up HF API Worker: {exc}')
