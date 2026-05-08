# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""/data/convert and /data/convert/status services.

Step 3 Part C2e migrated the real Mp4ConversionWorker pipeline here
from orchestrator.OrchestratorNode. Ownership of the worker and its
2 Hz status poll live in this service now; orchestrator's CONVERT_MP4
branch of send_command_callback forwards through CycloDataClient.

Lifecycle (mirrors HubService): eager start on service init →
auto-shutdown after 5 idle cycles → lazy restart on the next request.

Progress fan-out:
  * DataOperationStatus on /data/status (OP_CONVERSION) carries the
    percentage and stage. UI subscribes this topic directly for the
    conversion progress widget — conversion is a Data-Tools-side flow,
    distinct from the live recording session reflected on
    /data/recording/status (RecordingStatus). The two were briefly
    merged onto record_phase=CONVERTING in the original D18 split, but
    that put two publishers on /data/recording/status (RecordingService
    + ConversionService) and the UI saw their messages interleave once
    a recording session left a DataManager alive — see follow-up
    revert.
"""

from pathlib import Path
from typing import List, Optional
import threading
import uuid

from cyclo_data.converter.pipeline_worker import (
    LEROBOT_OUTPUT_ROOT,
    Mp4ConversionWorker,
)

from interfaces.msg import DataOperationStatus
from interfaces.srv import GetConversionStatus, StartConversion


_WORKER_STATUS_TO_SRV = {
    'Idle': GetConversionStatus.Response.QUEUED,
    'Converting': GetConversionStatus.Response.RUNNING,
    'Success': GetConversionStatus.Response.COMPLETED,
    'Failed': GetConversionStatus.Response.FAILED,
}

_WORKER_STATUS_TO_DATA_STATUS = {
    'Idle': DataOperationStatus.IDLE,
    'Converting': DataOperationStatus.RUNNING,
    'Success': DataOperationStatus.COMPLETED,
    'Failed': DataOperationStatus.FAILED,
}


class ConversionService:
    START_SERVICE_NAME = '/data/convert'
    STATUS_SERVICE_NAME = '/data/convert/status'
    STATUS_PERIOD_SEC = 0.5
    IDLE_TICKS_BEFORE_SHUTDOWN = 5

    def __init__(self, node, status_publisher):
        self._node = node
        self._data_status_pub = status_publisher

        self._worker: Optional[Mp4ConversionWorker] = None
        self._status_timer = None
        self._idle_count = 0
        self._last_status = None

        # Tracks the most recently accepted job so GetConversionStatus can
        # answer without trawling the worker history queue. Guarded by a
        # lock because _start_callback runs in the io callback group
        # while the status timer runs in the same group reentrantly.
        self._job_lock = threading.Lock()
        self._current_job_id: str = ''
        self._current_dataset_path: str = ''

        self._start_server = node.create_service(
            StartConversion,
            self.START_SERVICE_NAME,
            self._start_callback,
            callback_group=node.io_callback_group,
        )
        self._status_server = node.create_service(
            GetConversionStatus,
            self.STATUS_SERVICE_NAME,
            self._status_callback,
            callback_group=node.io_callback_group,
        )
        node.get_logger().info(f'Service advertised: {self.START_SERVICE_NAME}')
        node.get_logger().info(f'Service advertised: {self.STATUS_SERVICE_NAME}')

        self._init_worker()

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _init_worker(self):
        try:
            self._worker = Mp4ConversionWorker()
            if self._worker.start():
                self._node.get_logger().info('MP4 Conversion Worker started')
                self._idle_count = 0
                self._status_timer = self._node.create_timer(
                    self.STATUS_PERIOD_SEC,
                    self._status_timer_callback,
                    callback_group=self._node.io_callback_group,
                )
            else:
                self._node.get_logger().error('Failed to start MP4 Conversion Worker')
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(
                f'Error initializing MP4 Conversion Worker: {exc}')

    def shutdown(self):
        """Explicit cleanup hook invoked by cyclo_data_node on shutdown."""
        self._cleanup_worker()

    def _cleanup_worker(self):
        try:
            if self._status_timer is not None:
                self._status_timer.cancel()
                self._status_timer = None
            if self._worker is not None:
                self._worker.stop()
                self._worker = None
            self._node.get_logger().info('MP4 Conversion Worker cleaned up successfully')
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(
                f'Error cleaning up MP4 Conversion Worker: {exc}')

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------

    def _start_callback(self, request, response):
        if not request.dataset_path:
            response.success = False
            response.job_id = ''
            response.message = 'dataset_path is required.'
            return response

        if self._worker is None or not self._worker.is_alive():
            self._node.get_logger().info('MP4 Conversion Worker not running, restarting...')
            self._init_worker()
        if self._worker is None:
            response.success = False
            response.job_id = ''
            response.message = 'MP4 Conversion Worker could not be started.'
            return response

        if self._worker.is_busy():
            response.success = False
            response.job_id = ''
            response.message = 'MP4 conversion is already in progress'
            return response

        # convert_v21 / convert_v30 are bools on the srv. When both
        # are false (e.g. old UI / orchestrator that hasn't been
        # rebuilt against the new srv) we fall back to "run both" so
        # nothing silently regresses from doing all stages to doing
        # zero stages.
        convert_v21 = bool(getattr(request, 'convert_v21', False))
        convert_v30 = bool(getattr(request, 'convert_v30', False))
        if not convert_v21 and not convert_v30:
            convert_v21 = True
            convert_v30 = True

        # Refuse if any target LeRobot output already has a previous
        # conversion (meta/episodes.jsonl exists). The v21/v30 writers
        # *append* to episodes.jsonl, so a second run with the same
        # source name silently mixes the new conversion's metadata into
        # the old dataset. User has to delete or rename the existing
        # output dir to retry.
        existing = self._existing_lerobot_outputs(
            request.dataset_path, convert_v21, convert_v30)
        if existing:
            paths = ', '.join(str(p) for p in existing)
            response.success = False
            response.job_id = ''
            response.message = (
                f'Output already exists for this dataset. Delete or rename '
                f'before retrying: {paths}'
            )
            return response

        # Unpack the selection knobs from parallel arrays into Python
        # dicts. Empty / 0 means "use everything from robot_config" —
        # same as the legacy behaviour when these fields didn't exist.
        camera_rotations: dict = {}
        rot_keys = list(getattr(request, 'camera_rotation_keys', []) or [])
        rot_values = list(getattr(request, 'camera_rotation_values', []) or [])
        for i, key in enumerate(rot_keys):
            if not key:
                continue
            if i < len(rot_values):
                camera_rotations[key] = int(rot_values[i])

        resize_h = int(getattr(request, 'image_resize_height', 0) or 0)
        resize_w = int(getattr(request, 'image_resize_width', 0) or 0)
        image_resize = (
            (resize_h, resize_w)
            if resize_h > 0 and resize_w > 0
            else None
        )

        request_data = {
            'dataset_path': request.dataset_path,
            'robot_type': request.robot_type,
            'robot_config_path': request.robot_config_path,
            'source_folders': list(request.source_folders),
            # Conversion-time fps. 0 = caller wants the worker-side
            # default (recording is rate-agnostic; fps is purely a
            # conversion knob that drives MP4 encode rate + info.json).
            'fps': int(getattr(request, 'fps', 0) or 0),
            'convert_v21': convert_v21,
            'convert_v30': convert_v30,
            'selected_cameras': list(getattr(request, 'selected_cameras', []) or []),
            'camera_rotations': camera_rotations,
            'image_resize': image_resize,
            'selected_state_topics': list(getattr(request, 'selected_state_topics', []) or []),
            'selected_action_topics': list(getattr(request, 'selected_action_topics', []) or []),
            'selected_joints': list(getattr(request, 'selected_joints', []) or []),
        }

        if not self._worker.send_request(request_data):
            response.success = False
            response.job_id = ''
            response.message = 'Failed to send request to MP4 Conversion Worker'
            return response

        job_id = uuid.uuid4().hex
        with self._job_lock:
            self._current_job_id = job_id
            self._current_dataset_path = request.dataset_path

        self._node.get_logger().info(
            f'MP4 conversion started: job_id={job_id} dataset={request.dataset_path}'
            + (f' source_folders={request.source_folders}'
               if request.source_folders else '')
        )
        self._publish_data_status(
            job_id=job_id,
            status=DataOperationStatus.RUNNING,
            progress=0.0,
            stage='queued',
            message=f'MP4 conversion started for: {request.dataset_path}',
        )

        response.success = True
        response.job_id = job_id
        response.message = f'MP4 conversion started for: {request.dataset_path}'
        return response

    def _status_callback(self, request, response):
        # The worker exposes one active task at a time; if the caller
        # queries an older job_id we answer UNKNOWN.
        with self._job_lock:
            current_job_id = self._current_job_id

        if not current_job_id or (request.job_id and request.job_id != current_job_id):
            response.success = True
            response.status = GetConversionStatus.Response.UNKNOWN
            response.progress_percentage = 0.0
            response.episodes_processed = 0
            response.episodes_total = 0
            response.current_stage = ''
            response.message = (
                f'Unknown or superseded job_id: {request.job_id}'
                if request.job_id else 'No conversion job in flight.'
            )
            return response

        if self._worker is None:
            response.success = True
            response.status = GetConversionStatus.Response.UNKNOWN
            response.progress_percentage = 0.0
            response.episodes_processed = 0
            response.episodes_total = 0
            response.current_stage = ''
            response.message = 'Worker has shut down (likely idle-reaped).'
            return response

        worker_status = self._worker.check_task_status()
        status_str = worker_status.get('status', 'Unknown')
        progress = worker_status.get('progress', {})

        response.success = True
        response.status = _WORKER_STATUS_TO_SRV.get(
            status_str, GetConversionStatus.Response.UNKNOWN)
        response.progress_percentage = float(progress.get('percentage', 0.0))
        response.episodes_processed = int(progress.get('current', 0))
        response.episodes_total = int(progress.get('total', 0))
        response.current_stage = worker_status.get('stage', status_str)
        response.message = worker_status.get('message', '')
        return response

    # ------------------------------------------------------------------
    # Status polling → DataOperationStatus relay
    # ------------------------------------------------------------------

    def _status_timer_callback(self):
        if self._worker is None:
            return
        try:
            status = self._worker.check_task_status()
            current = status.get('status', 'Unknown')

            last = (
                self._last_status.get('status', 'Unknown')
                if self._last_status else 'Unknown'
            )
            if self._last_status is not None and last != current:
                self._node.get_logger().info(
                    f'MP4 Conversion Status changed: {last} -> {current}')
            self._last_status = status

            progress = status.get('progress', {})
            percentage = float(progress.get('percentage', 0.0))

            with self._job_lock:
                job_id = self._current_job_id

            if current == 'Converting':
                self._publish_data_status(
                    job_id=job_id,
                    status=DataOperationStatus.RUNNING,
                    progress=percentage,
                    stage=status.get('stage', 'converting'),
                    message=status.get('message', ''),
                )
            elif current == 'Success':
                self._publish_data_status(
                    job_id=job_id,
                    status=DataOperationStatus.COMPLETED,
                    progress=100.0,
                    stage='success',
                    message=status.get('message', ''),
                )
            elif current == 'Failed':
                self._publish_data_status(
                    job_id=job_id,
                    status=DataOperationStatus.FAILED,
                    progress=percentage,
                    stage='failed',
                    message=status.get('message', ''),
                )

            if current == 'Idle':
                self._idle_count += 1
                if self._idle_count >= self.IDLE_TICKS_BEFORE_SHUTDOWN:
                    self._node.get_logger().info(
                        f'MP4 Conversion Worker idle for {self.IDLE_TICKS_BEFORE_SHUTDOWN} '
                        'cycles, shutting down worker and timer.')
                    self._cleanup_worker()
            else:
                self._idle_count = 0
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(
                f'Error in MP4 status timer callback: {exc}')

    def _publish_data_status(
        self,
        job_id: str,
        status: int,
        progress: float,
        stage: str,
        message: str,
    ) -> None:
        msg = DataOperationStatus()
        msg.operation_type = DataOperationStatus.OP_CONVERSION
        msg.status = status
        msg.job_id = job_id
        msg.progress_percentage = progress
        msg.stage = stage
        msg.message = message
        self._data_status_pub.publish(msg)

    @staticmethod
    def _existing_lerobot_outputs(
        dataset_path: str,
        convert_v21: bool,
        convert_v30: bool,
    ) -> List[Path]:
        """Return target LeRobot output dirs that already hold a prior conversion.

        Existence is signalled by a non-empty meta/episodes.jsonl —
        empty/half-baked dirs (mid-crash leftovers) don't trigger the
        guard so the user can re-run without manual cleanup.
        """
        name = Path(dataset_path).name
        candidates: List[Path] = []
        if convert_v21:
            candidates.append(LEROBOT_OUTPUT_ROOT / f'{name}_lerobot_v21')
        if convert_v30:
            candidates.append(LEROBOT_OUTPUT_ROOT / f'{name}_lerobot_v30')
        return [
            p for p in candidates
            if (p / 'meta' / 'episodes.jsonl').exists()
            and (p / 'meta' / 'episodes.jsonl').stat().st_size > 0
        ]
