# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""/data/recording service — RecordingCommand handler.

Part C2d progression (REVIEW §9.6):
  * B2:   stub callback publishes DataOperationStatus and returns OK.
  * C2d-1: RecordingService owns RosbagControl (client + action_event pub).
  * C2d-2: RecordingService owns DataManager capability + 5 Hz status
           publisher on /data/recording/status.
  * C2d-3: _callback dispatches the full 10-command set (REFRESH_TOPICS
           / START / STOP / FINISH / MOVE_TO_NEXT / RERECORD / CANCEL /
           SKIP_TASK / PAUSE / RESUME).
  * C2d-4: orchestrator's recording branch becomes a forwarder and the
           orchestrator-side DataManager / TaskStatus publish goes away.
  * D18:   the relay through /task/status is retired; UI subscribes
           /data/recording/status (RecordingStatus) directly. The phase
           field split into orthogonal record_phase / inference_phase
           (PLAN §10.3 D18, supersedes REVIEW §9.4).

Session-state boundary (REVIEW §9.3):
  This service owns DataManager + rosbag control + action events only.
  on_recording / on_inference / robot_type lookup / inference_manager —
  those stay on the orchestrator node. The forwarder sets its own
  flags before invoking us and after our response returns.
"""

from pathlib import Path
from typing import Optional

from cyclo_data.recorder.rosbag_control import RosbagControl
from cyclo_data.recorder.session_manager import DataManager
from orchestrator.internal.device_manager.cpu_checker import CPUChecker
from orchestrator.internal.device_manager.ram_checker import RAMChecker
from orchestrator.internal.device_manager.storage_checker import StorageChecker

from interfaces.msg import DataOperationStatus, RecordingStatus
from interfaces.srv import RecordingCommand


_COMMAND_NAMES = {
    RecordingCommand.Request.START: 'START',
    RecordingCommand.Request.STOP: 'STOP',
    RecordingCommand.Request.PAUSE: 'PAUSE',
    RecordingCommand.Request.RESUME: 'RESUME',
    RecordingCommand.Request.FINISH: 'FINISH',
    RecordingCommand.Request.MOVE_TO_NEXT: 'MOVE_TO_NEXT',
    RecordingCommand.Request.RERECORD: 'RERECORD',
    RecordingCommand.Request.SKIP_TASK: 'SKIP_TASK',
    RecordingCommand.Request.CANCEL: 'CANCEL',
    RecordingCommand.Request.REFRESH_TOPICS: 'REFRESH_TOPICS',
}


class RecordingService:
    SERVICE_NAME = '/data/recording'
    STATUS_TOPIC = '/data/recording/status'
    STATUS_PERIOD_SEC = 0.2  # 5 Hz

    # Matches orchestrator.OrchestratorNode.DEFAULT_SAVE_ROOT_PATH so the
    # on-disk layout is identical during the C2d-3 → C2d-4 handoff.
    DEFAULT_SAVE_ROOT_PATH = Path.home() / '.cache/huggingface/lerobot'

    def __init__(self, node, status_publisher):
        self._node = node
        self._status_pub = status_publisher  # umbrella /data/status
        self._rosbag = RosbagControl(node)

        self._data_manager: Optional[DataManager] = None
        self._robot_type: str = ''

        # Idle-state metrics: filled into the 5 Hz status publish before any
        # session_manager exists so the UI's CPU/RAM/Storage panel keeps
        # rendering live values between recordings. Once a DataManager is
        # active, its own CPUChecker takes over (this one stays unused).
        self._cpu_checker = CPUChecker()

        self._recording_status_pub = node.create_publisher(
            RecordingStatus, self.STATUS_TOPIC, 10)
        self._status_timer = node.create_timer(
            self.STATUS_PERIOD_SEC,
            self._publish_recording_status,
            callback_group=node.io_callback_group,
        )

        self._server = node.create_service(
            RecordingCommand,
            self.SERVICE_NAME,
            self._callback,
            callback_group=node.state_callback_group,
        )
        node.get_logger().info(f'Service advertised: {self.SERVICE_NAME}')
        node.get_logger().info(
            f'Status topic: {self.STATUS_TOPIC} '
            f'({int(1.0 / self.STATUS_PERIOD_SEC)} Hz, '
            'system metrics published continuously)')

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._status_timer is not None:
            try:
                self._status_timer.cancel()
            except Exception:  # noqa: BLE001
                pass
            self._status_timer = None
        # Best-effort teardown of a live session before node destroy.
        if self._data_manager is not None:
            try:
                if self._data_manager.is_recording():
                    self._data_manager.stop_recording()
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f'DataManager stop on shutdown failed: {exc}')
            self._data_manager = None
        self._rosbag.shutdown()

    # ------------------------------------------------------------------
    # DataManager management
    # ------------------------------------------------------------------

    def _ensure_data_manager(self, task_info, robot_type: str) -> DataManager:
        self._robot_type = robot_type
        candidate = DataManager(
            save_root_path=self.DEFAULT_SAVE_ROOT_PATH,
            robot_type=robot_type,
            task_info=task_info,
        )
        if (self._data_manager is None
                or getattr(self._data_manager, '_save_repo_name', None)
                != candidate._save_repo_name):
            self._data_manager = candidate
            self._node.get_logger().info(
                f'DataManager initialised: repo={candidate._save_repo_name} '
                f'robot_type={robot_type}')
        else:
            # Same task as before — reuse existing manager but refresh
            # its task_info so per-session knobs (e.g. UI's
            # include_robotis_license checkbox) flipped between
            # episodes are picked up on the next save_robotis_metadata.
            self._data_manager.update_task_info(task_info)
        return self._data_manager

    def _clear_data_manager(self) -> None:
        if self._data_manager is not None:
            self._node.get_logger().info(
                f'DataManager cleared (repo={self._data_manager._save_repo_name})')
            self._data_manager = None

    # ------------------------------------------------------------------
    # Status fan-out
    # ------------------------------------------------------------------

    def _publish_recording_status(self) -> None:
        if self._data_manager is not None:
            try:
                status: RecordingStatus = (
                    self._data_manager.get_current_record_status()
                )
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warn(
                    f'DataManager.get_current_record_status() raised: {exc}')
                return
        else:
            # No active session — emit a minimal RecordingStatus carrying
            # only system metrics so the UI's resource panel has data
            # between recordings. record_phase=READY signals "idle" to UI
            # state machines (taskSlice / RecordPhase).
            status = RecordingStatus()
            status.record_phase = RecordingStatus.READY
            status.used_cpu = float(self._cpu_checker.get_cpu_usage())
            ram_total, ram_used = RAMChecker.get_ram_gb()
            status.used_ram_size = float(ram_used)
            status.total_ram_size = float(ram_total)
            total_storage, used_storage = StorageChecker.get_storage_gb('/')
            status.used_storage_size = float(used_storage)
            status.total_storage_size = float(total_storage)
        if self._robot_type and not status.robot_type:
            status.robot_type = self._robot_type
        self._recording_status_pub.publish(status)

    def _publish_umbrella_status(self, status: int, stage: str, message: str) -> None:
        msg = DataOperationStatus()
        msg.operation_type = DataOperationStatus.OP_RECORDING
        msg.status = status
        msg.job_id = ''
        msg.progress_percentage = 0.0
        msg.stage = stage
        msg.message = message
        self._status_pub.publish(msg)

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def _callback(self, request, response):
        command_name = _COMMAND_NAMES.get(request.command)
        if command_name is None:
            response.success = False
            response.message = f'Unknown recording command: {request.command}'
            self._node.get_logger().warn(response.message)
            return response

        task_num = request.task_info.task_num or '<unset>'
        self._node.get_logger().info(
            f'RecordingCommand.{command_name} received '
            f'(task_num={task_num}, robot_type={request.robot_type or "<unset>"})')

        cmd = request.command
        Req = RecordingCommand.Request

        try:
            if cmd == Req.REFRESH_TOPICS:
                return self._do_refresh_topics(request, response)
            if cmd == Req.START:
                return self._do_start(request, response)
            if cmd in (Req.STOP, Req.FINISH, Req.MOVE_TO_NEXT):
                return self._do_stop_and_save(
                    request, response, command_name, event='finish')
            if cmd == Req.RERECORD:
                return self._do_cancel_with_review(
                    request, response, event='cancel')
            if cmd == Req.CANCEL:
                return self._do_cancel(request, response)
            if cmd == Req.SKIP_TASK:
                return self._do_skip_task(request, response)
            if cmd == Req.PAUSE:
                return self._do_pause(request, response)
            if cmd == Req.RESUME:
                return self._do_resume(request, response)

            # Shouldn't reach here — command_name gate catches unknowns.
            response.success = False
            response.message = f'No dispatch for {command_name}'
            return response
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().error(
                f'RecordingCommand.{command_name} raised: {exc}')
            response.success = False
            response.message = f'{command_name} failed: {exc}'
            self._publish_umbrella_status(
                DataOperationStatus.FAILED, command_name, str(exc))
            return response

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _do_refresh_topics(self, request, response):
        topics = list(request.topics or [])
        if not topics:
            response.success = False
            response.message = 'REFRESH_TOPICS requires non-empty topics[]'
            return response
        if not self._rosbag.is_available():
            response.success = False
            response.message = 'rosbag_recorder service unavailable'
            return response
        self._rosbag.prepare_rosbag(topics=topics)
        response.success = True
        response.message = f'Topics refreshed ({len(topics)} topics)'
        return response

    def _do_start(self, request, response):
        if not request.robot_type:
            response.success = False
            response.message = 'START requires robot_type'
            return response
        if not self._rosbag.is_available():
            response.success = False
            response.message = 'rosbag_recorder service unavailable'
            return response

        dm = self._ensure_data_manager(request.task_info, request.robot_type)

        # Prepare first — mirrors orchestrator's START_RECORD flow
        # (Communicator.prepare_rosbag before start_rosbag).
        topics = list(request.topics or [])
        if topics:
            self._rosbag.prepare_rosbag(topics=topics)
        else:
            self._node.get_logger().warn(
                'START: topics[] empty — skipping prepare. '
                'Caller should populate from orchestrator.Communicator.get_all_topics().')

        rosbag_path = dm.get_save_rosbag_path(allow_idle=True)
        if not rosbag_path:
            response.success = False
            response.message = 'Failed to resolve rosbag path'
            return response

        self._rosbag.start_rosbag(rosbag_uri=rosbag_path)
        dm.start_recording()
        self._rosbag.publish_action_event('start')

        self._publish_umbrella_status(
            DataOperationStatus.RUNNING, 'START',
            f'Recording started at {rosbag_path}')

        response.success = True
        response.message = 'Recording started'
        return response

    def _do_stop_and_save(self, request, response, command_name: str, event: str):
        """STOP / FINISH / MOVE_TO_NEXT — save metadata, stop rosbag,
        stop DataManager, fire action_event.
        """
        if self._data_manager is None:
            response.success = False
            response.message = f'{command_name}: no active recording session'
            return response

        self._node.get_logger().info(
            f'{command_name}: episode={self._data_manager._record_episode_count} '
            f'status={self._data_manager.get_status()}')

        if request.urdf_path:
            self._data_manager.save_robotis_metadata(urdf_path=request.urdf_path)

        self._rosbag.stop_rosbag()
        self._data_manager.stop_recording()
        self._rosbag.publish_action_event(event)

        self._publish_umbrella_status(
            DataOperationStatus.COMPLETED, command_name,
            f'{command_name} saved — '
            f'next_episode={self._data_manager._record_episode_count}')

        response.success = True
        response.message = {
            'STOP': 'Recording stopped and saved',
            'FINISH': 'Recording finished and saved',
            'MOVE_TO_NEXT': 'Episode saved',
        }.get(command_name, f'{command_name} completed')
        return response

    def _do_cancel_with_review(self, request, response, event: str):
        """RERECORD — save current episode with needs_review=True, then stop."""
        if self._data_manager is None:
            response.success = False
            response.message = 'RERECORD: no active recording session'
            return response

        if request.urdf_path:
            self._data_manager.save_robotis_metadata(
                urdf_path=request.urdf_path, needs_review=True)

        self._rosbag.stop_rosbag()
        self._data_manager.stop_recording()
        self._rosbag.publish_action_event(event)

        self._publish_umbrella_status(
            DataOperationStatus.CANCELLED, 'RERECORD',
            'Recording cancelled — data saved with review flag')

        response.success = True
        response.message = 'Recording cancelled (saved with review flag)'
        return response

    def _do_cancel(self, request, response):
        """CANCEL — two modes (matches orchestrator):
          * Active recording → same as RERECORD (save with review flag).
          * No active recording → toggle previous episode's needs_review.
        """
        if self._data_manager is None:
            response.success = False
            response.message = 'CANCEL: no DataManager yet — nothing to toggle'
            return response

        if self._data_manager.is_recording():
            return self._do_cancel_with_review(request, response, event='cancel')

        toggled = self._data_manager.toggle_previous_episode_needs_review()
        if toggled is None:
            response.success = False
            response.message = 'No previous episode to toggle'
            return response
        self._rosbag.publish_action_event('review_on' if toggled else 'review_off')
        response.success = True
        response.message = f'Previous episode needs_review: {toggled}'
        self._publish_umbrella_status(
            DataOperationStatus.IDLE, 'CANCEL_TOGGLE_REVIEW',
            response.message)
        return response

    def _do_skip_task(self, request, response):
        # Orchestrator never defined SKIP_TASK dispatch in send_command —
        # the command exists in RecordingCommand.srv for UI completeness.
        # TODO(C2d-follow-up): define semantics with user (skip without save
        # + advance to next task? requires orchestrator coordination).
        response.success = True
        response.message = 'SKIP_TASK acknowledged — no-op (pending design)'
        self._publish_umbrella_status(
            DataOperationStatus.IDLE, 'SKIP_TASK', response.message)
        return response

    def _do_pause(self, request, response):
        # DataManager does not currently expose a pause() method. PAUSE
        # is new in RecordingCommand.srv (PLAN §10.3 D8). For now this
        # is a status-only acknowledgement.
        # TODO(C2d-follow-up): extend DataManager with pause/resume,
        # or gate pause via orchestrator's operation_mode transitions.
        response.success = True
        response.message = 'PAUSE acknowledged — no-op (DataManager pause pending)'
        self._publish_umbrella_status(
            DataOperationStatus.RUNNING, 'PAUSE', response.message)
        return response

    def _do_resume(self, request, response):
        response.success = True
        response.message = 'RESUME acknowledged — no-op (DataManager resume pending)'
        self._publish_umbrella_status(
            DataOperationStatus.RUNNING, 'RESUME', response.message)
        return response
