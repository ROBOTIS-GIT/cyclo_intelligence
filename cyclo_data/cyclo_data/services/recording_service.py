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

import shutil
import threading
from pathlib import Path
from typing import Optional

from cyclo_data.recorder.camera_info_snapshot import CameraInfoSnapshot
from cyclo_data.recorder.rosbag_control import RosbagControl
from cyclo_data.recorder.session_manager import DataManager
from cyclo_data.recorder.transcoder import TranscodeWorker
from cyclo_data.recorder.video_recorder import VideoRecorder
from orchestrator.internal.device_manager.cpu_checker import CPUChecker
from orchestrator.internal.device_manager.ram_checker import RAMChecker
from orchestrator.internal.device_manager.storage_checker import StorageChecker
from shared.robot_configs import schema as robot_schema

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
        # Recording format v2: per-camera MP4 + camera_info yaml. The
        # recorder/snapshot instances live from REFRESH_TOPICS (= robot_type
        # selection) through service shutdown — only the per-episode
        # writers toggle on START/STOP. ``_video_robot_type`` tracks the
        # robot_type the current subs were built for so reconfigure only
        # fires when it actually changes.
        self._video_recorder: Optional[VideoRecorder] = None
        self._camera_info: Optional[CameraInfoSnapshot] = None
        self._video_robot_type: str = ''
        self._last_image_topics: dict = {}
        self._last_camera_info_topics: dict = {}
        self._last_video_stats: dict = {}
        self._last_camera_info_files: dict = {}
        self._last_camera_rotations: dict = {}
        # rosbag_recorder's `prepare` always destroys + recreates its
        # subscriptions (service_bag_recorder.cpp:188-192), which resets
        # the topic monitor's EMA baseline and triggers a fresh wave of
        # zenoh liveliness declarations. We skip forwarding prepare when
        # the topic set hasn't changed since the last call so START
        # doesn't reissue what REFRESH_TOPICS already did.
        self._last_prepared_topics: tuple = ()
        # Background transcoder converts each episode's raw MJPEG MP4s
        # into H.264 after STOP. One pool per service instance, lazily
        # initialised on first STOP so process startup stays cheap.
        self._transcoder: Optional[TranscodeWorker] = None

        # The 5 Hz _publish_recording_status timer runs on io_callback_group
        # (Reentrant) while _callback runs on state_callback_group
        # (MutuallyExclusive). Under MultiThreadedExecutor the timer can
        # therefore observe a torn TOCTOU on _data_manager (one read sees
        # a manager, the next sees None as a callback completes teardown).
        # _session_lock just brackets the pointer reads/writes — DataManager
        # has its own internal _state_lock so we never need to nest locks.
        self._session_lock = threading.Lock()

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
        with self._session_lock:
            dm = self._data_manager
            self._data_manager = None
        if dm is not None:
            try:
                if dm.is_recording():
                    dm.stop_recording()
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f'DataManager stop on shutdown failed: {exc}')
        # Release persistent video/camera_info subscriptions held since
        # REFRESH_TOPICS. rclpy's node.destroy_node() would clean them
        # up anyway, but doing it explicitly lets leak audits see a
        # clean state.
        if self._video_recorder is not None:
            try:
                self._video_recorder.close()
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f'VideoRecorder.close failed: {exc}')
            self._video_recorder = None
        if self._camera_info is not None:
            try:
                self._camera_info.close()
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f'CameraInfoSnapshot.close failed: {exc}')
            self._camera_info = None
        if self._transcoder is not None:
            try:
                self._transcoder.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f'TranscodeWorker.shutdown failed: {exc}')
            self._transcoder = None
        self._rosbag.shutdown()

    # ------------------------------------------------------------------
    # DataManager management
    # ------------------------------------------------------------------

    def _ensure_data_manager(self, task_info, robot_type: str) -> DataManager:
        with self._session_lock:
            self._robot_type = robot_type
            existing = self._data_manager
        candidate = DataManager(
            save_root_path=self.DEFAULT_SAVE_ROOT_PATH,
            robot_type=robot_type,
            task_info=task_info,
        )
        if (existing is None
                or getattr(existing, '_save_repo_name', None)
                != candidate._save_repo_name):
            with self._session_lock:
                self._data_manager = candidate
            self._node.get_logger().info(
                f'DataManager initialised: repo={candidate._save_repo_name} '
                f'robot_type={robot_type}')
            return candidate
        # Same task as before — reuse existing manager but refresh
        # its task_info so per-session knobs (e.g. UI's
        # include_robotis_license checkbox) flipped between
        # episodes are picked up on the next save_robotis_metadata.
        existing.update_task_info(task_info)
        return existing

    def _clear_data_manager(self) -> None:
        with self._session_lock:
            dm = self._data_manager
            self._data_manager = None
        if dm is not None:
            self._node.get_logger().info(
                f'DataManager cleared (repo={dm._save_repo_name})')

    # ------------------------------------------------------------------
    # Status fan-out
    # ------------------------------------------------------------------

    def _publish_recording_status(self) -> None:
        # Snapshot once — a concurrent _callback teardown could otherwise
        # null self._data_manager between this check and the method call.
        with self._session_lock:
            dm = self._data_manager
            robot_type = self._robot_type
        if dm is not None:
            try:
                status: RecordingStatus = dm.get_current_record_status()
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
        if robot_type and not status.robot_type:
            status.robot_type = robot_type
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
        self._prepare_rosbag_topics(topics)

        # When the orchestrator forwards REFRESH_TOPICS from
        # set_robot_type_callback, it carries the freshly-selected
        # robot_type. That's our trigger to build the persistent
        # video/camera_info subscriptions so subsequent START commands
        # don't fire a zenoh declaration storm. Failures here don't
        # poison the response — rosbag is already prepared, and the
        # next REFRESH_TOPICS will retry.
        if request.robot_type:
            try:
                self._ensure_video_pipeline(request.robot_type)
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().error(
                    f'REFRESH_TOPICS video pipeline setup failed: {exc!r}')

        response.success = True
        response.message = f'Topics refreshed ({len(topics)} topics)'
        return response

    def _prepare_rosbag_topics(self, topics: list) -> None:
        """Forward `prepare` to rosbag_recorder only when the set changes.

        rosbag_recorder rebuilds its subscriptions on every prepare
        (service_bag_recorder.cpp:188-192) — that resets the topic
        monitor's EMA baseline and fires a fresh zenoh liveliness
        declaration wave. Caching by sorted-tuple lets the no-op case
        (REFRESH_TOPICS already prepared this set, and START hands us
        the same one) skip the round-trip entirely.
        """
        new_set = tuple(sorted(topics))
        if new_set == self._last_prepared_topics:
            return
        self._rosbag.prepare_rosbag(topics=topics)
        self._last_prepared_topics = new_set

    def _ensure_video_pipeline(self, robot_type: str) -> None:
        """Build or reconfigure the persistent video/camera_info subscriptions.

        Called from ``_do_refresh_topics`` whenever the orchestrator
        forwards a REFRESH_TOPICS with a robot_type. First call builds
        the subscriptions; subsequent calls with the same robot_type
        are no-ops; a different robot_type triggers reconfigure on both
        components.
        """
        if not robot_type:
            return
        if self._video_robot_type == robot_type and (
            self._video_recorder is not None or self._camera_info is not None
        ):
            return

        image_topics, camera_info_topics, rotations = self._resolve_video_topics(
            robot_type)
        self._last_image_topics = image_topics
        self._last_camera_info_topics = camera_info_topics
        self._last_camera_rotations = rotations

        if self._video_recorder is None:
            if image_topics:
                self._video_recorder = VideoRecorder(
                    node=self._node, cameras=image_topics,
                    callback_group=getattr(self._node, 'io_callback_group', None),
                )
        else:
            try:
                self._video_recorder.reconfigure(image_topics)
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().error(
                    f'VideoRecorder.reconfigure failed: {exc!r}')

        if self._camera_info is None:
            if camera_info_topics:
                self._camera_info = CameraInfoSnapshot(
                    node=self._node, camera_info_topics=camera_info_topics,
                    callback_group=getattr(self._node, 'io_callback_group', None),
                )
        else:
            try:
                self._camera_info.reconfigure(camera_info_topics)
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().error(
                    f'CameraInfoSnapshot.reconfigure failed: {exc!r}')

        self._video_robot_type = robot_type
        self._node.get_logger().info(
            f'Video pipeline ready for robot_type={robot_type!r} '
            f'(cameras={len(image_topics)}, '
            f'camera_info={len(camera_info_topics)})')

    def _resolve_video_topics(self, robot_type: str):
        """Return ``(image_topics, camera_info_topics, rotations)`` for a robot.

        ``image_topics`` and ``camera_info_topics`` are ``{cam_name: topic}``
        dicts. ``rotations`` is ``{cam_name: degrees}`` (0/90/180/270) so
        the recorder can stash it in ``episode_info.json`` and the
        background transcoder can apply ``-vf transpose=N`` later.

        Loads the robot section from the yaml on every call rather than
        caching — recording is infrequent enough that the IO is
        negligible.
        """
        try:
            section = robot_schema.load_robot_section(robot_type)
        except Exception as exc:
            self._node.get_logger().error(
                f'Failed to load robot section for {robot_type!r}: {exc!r}')
            return {}, {}, {}
        image_groups = robot_schema.get_image_topics(section)
        image_topics = {
            cam: cfg['topic'] for cam, cfg in image_groups.items()
        }
        rotations = {
            cam: int(cfg.get('rotation_deg', 0) or 0)
            for cam, cfg in image_groups.items()
        }
        camera_info_topics = robot_schema.get_camera_info_topics(section)
        return image_topics, camera_info_topics, rotations

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

        # rosbag_recorder is normally prepared at REFRESH_TOPICS time
        # (= robot_type selection) — _prepare_rosbag_topics short-circuits
        # when the topic set hasn't changed, so this call is a no-op in
        # the common case. If REFRESH_TOPICS never ran (tests, recovery)
        # and request.topics is empty, warn so the caller knows the bag
        # will be empty.
        topics = list(request.topics or [])
        if topics:
            self._prepare_rosbag_topics(topics)
        elif not self._last_prepared_topics:
            self._node.get_logger().warn(
                'START: topics[] empty and rosbag never prepared — '
                'caller should populate from '
                'orchestrator.Communicator.get_mcap_topics().')

        rosbag_path = dm.get_save_rosbag_path(allow_idle=True)
        if not rosbag_path:
            response.success = False
            response.message = 'Failed to resolve rosbag path'
            return response

        # Recording format v2: per-camera MP4 writers + one-shot
        # camera_info snapshotter live in ``videos/`` and ``camera_info/``
        # subdirs of the rosbag episode dir. Spin them up AFTER the
        # rosbag service starts — rosbag_recorder's storage plugin
        # treats the URI as a fresh bag root and may rewrite it on open,
        # which would wipe any subdirectory we created first.
        self._rosbag.start_rosbag(rosbag_uri=rosbag_path)

        episode_dir = Path(rosbag_path)
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Subscriptions were built up-front in REFRESH_TOPICS, so START
        # only opens the per-episode writers. Defensive ensure for the
        # rare case where START arrived without a preceding
        # REFRESH_TOPICS (tests, recovery paths) — same robot_type re-
        # entry is a no-op inside _ensure_video_pipeline.
        self._ensure_video_pipeline(request.robot_type)
        if self._video_recorder is not None:
            self._video_recorder.start_episode(episode_dir)
        if self._camera_info is not None:
            self._camera_info.start_episode(episode_dir)

        dm.start_recording()
        self._rosbag.publish_action_event('start')

        self._publish_umbrella_status(
            DataOperationStatus.RUNNING, 'START',
            f'Recording started at {rosbag_path}')

        response.success = True
        response.message = 'Recording started'
        return response

    def _ensure_transcoder(self) -> TranscodeWorker:
        if self._transcoder is None:
            self._transcoder = TranscodeWorker(logger=self._node.get_logger())
        return self._transcoder

    def _submit_transcode(self, episode_dir):
        """Fire-and-forget queue a finished episode for H.264 transcoding.

        Defensive: any failure to enqueue is logged but never propagated
        — STOP/FINISH must always succeed from the caller's perspective.
        The pending raw MJPEG remains on disk so a future ``submit_pending_recovery``
        call (on next service start) will retry.
        """
        try:
            worker = self._ensure_transcoder()
        except Exception as exc:
            self._node.get_logger().error(
                f"Transcoder pool unavailable: {exc!r}; "
                f"episode {episode_dir} will need manual transcode"
            )
            return
        try:
            worker.submit(episode_dir, on_complete=self._on_transcode_done)
        except Exception as exc:
            self._node.get_logger().error(
                f"Transcoder submit failed for {episode_dir}: {exc!r}"
            )

    def _on_transcode_done(self, result):
        if result.success:
            self._node.get_logger().info(
                f"Transcode done: {result.episode_dir.name} "
                f"({len(result.cameras_done)} cameras, "
                f"{result.elapsed_sec:.1f}s, {result.encoder})"
            )
        else:
            self._node.get_logger().error(
                f"Transcode failed: {result.episode_dir.name} "
                f"failures={result.cameras_failed} error={result.error}"
            )

    def resume_pending_transcodes(self, workspace_root):
        """Called by cyclo_data_node on startup — process any episodes
        left in pending/running state after a previous crash."""
        try:
            worker = self._ensure_transcoder()
            futures = worker.submit_pending_recovery(
                workspace_root, on_complete=self._on_transcode_done,
            )
            if futures:
                self._node.get_logger().info(
                    f"Resumed {len(futures)} pending transcode job(s) under {workspace_root}"
                )
        except Exception as exc:
            self._node.get_logger().error(
                f"Transcoder resume scan failed: {exc!r}"
            )

    def _stop_episode_writers(self):
        """End the current episode, keeping subscribers alive.

        Stats / produced-file lists are stashed on the instance so the
        next ``save_robotis_metadata`` call can include them in
        ``episode_info.json``. The recorder/snapshot instances remain
        for the next episode — only ``close()`` (on shutdown) or
        ``reconfigure()`` (on robot_type change) tears their subs down.
        """
        self._last_video_stats = {}
        self._last_camera_info_files = {}
        if self._video_recorder is not None:
            try:
                self._last_video_stats = self._video_recorder.stop_episode() or {}
            except Exception as exc:  # pragma: no cover - defensive
                self._node.get_logger().error(
                    f'VideoRecorder.stop_episode raised: {exc!r}')
        if self._camera_info is not None:
            try:
                produced = self._camera_info.stop_episode() or {}
                self._last_camera_info_files = {
                    cam: str(p) for cam, p in produced.items()
                }
            except Exception as exc:  # pragma: no cover - defensive
                self._node.get_logger().error(
                    f'CameraInfoSnapshot.stop_episode raised: {exc!r}')

    def _do_stop_and_save(self, request, response, command_name: str, event: str):
        """STOP / FINISH / MOVE_TO_NEXT — save metadata, stop rosbag,
        stop DataManager, fire action_event.

        No-op (without raising) when no recording is active. The
        inference page's Clear button forwards FINISH to land here even
        when only inference (no recording) was running; without this
        guard the 'finish' action_event would fire and the UI's
        ACTION_VOICE_MAP would play "Recording finished" — confusing in
        an inference-only context.
        """
        if self._data_manager is None:
            response.success = True
            response.message = f'{command_name}: no DataManager — no-op'
            return response
        if not self._data_manager.is_recording():
            response.success = True
            response.message = (
                f'{command_name}: no active recording — no-op'
            )
            return response

        self._node.get_logger().info(
            f'{command_name}: episode={self._data_manager._record_episode_count} '
            f'status={self._data_manager.get_status()}')

        episode_dir = Path(self._data_manager.get_save_rosbag_path() or '')
        self._rosbag.stop_rosbag()
        self._stop_episode_writers()

        if request.urdf_path:
            self._data_manager.save_robotis_metadata(
                urdf_path=request.urdf_path,
                video_stats=self._last_video_stats,
                camera_info_files=self._last_camera_info_files,
                camera_rotations=self._last_camera_rotations,
            )

        # Fire the H.264 transcode in the background. The episode dir is
        # captured before stop_recording() bumps the episode counter so
        # we hand off the right path.
        if episode_dir.exists() and (episode_dir / 'videos').exists():
            self._submit_transcode(episode_dir)

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
        """RERECORD — stop current episode and save (no review flag).

        Historically this also stamped ``needs_review=True`` on the
        episode_info.json, but that field has been removed (downstream
        tooling never consumed it). RERECORD now behaves like STOP
        save-wise; the path is kept distinct because the action_event
        the orchestrator publishes here is ``cancel`` rather than
        ``finish``, which other consumers may still discriminate on.
        """
        if self._data_manager is None:
            response.success = False
            response.message = 'RERECORD: no active recording session'
            return response

        self._rosbag.stop_rosbag()
        self._stop_episode_writers()

        if request.urdf_path:
            self._data_manager.save_robotis_metadata(
                urdf_path=request.urdf_path,
                video_stats=self._last_video_stats,
                camera_info_files=self._last_camera_info_files,
                camera_rotations=self._last_camera_rotations,
            )

        self._data_manager.stop_recording()
        self._rosbag.publish_action_event(event)

        self._publish_umbrella_status(
            DataOperationStatus.CANCELLED, 'RERECORD',
            'Recording cancelled — data saved')

        response.success = True
        response.message = 'Recording cancelled (data saved)'
        return response

    def _do_cancel(self, request, response):
        """CANCEL — discard the active episode entirely.

        On active recording: delete the on-disk bag + mp4/yaml
        siblings and leave the episode counter where it was so the
        slot is reused next START.

        On idle (no active recording): there's nothing to discard.
        Previously this path toggled the prior episode's
        ``needs_review`` flag, but that flag was removed (downstream
        never read it), so idle CANCEL now responds with a no-op.
        """
        if self._data_manager is None:
            response.success = False
            response.message = 'CANCEL: no DataManager yet'
            return response

        if self._data_manager.is_recording():
            return self._do_discard(request, response, event='cancel')

        response.success = True
        response.message = 'CANCEL: no active recording — nothing to discard'
        self._publish_umbrella_status(
            DataOperationStatus.IDLE, 'CANCEL', response.message)
        return response

    def _do_discard(self, request, response, event: str):
        """Active-recording CANCEL — drop the episode without saving.

        Order matters: drain VideoRecorder/CameraInfoSnapshot writers
        first so no ffmpeg subprocess is still holding files open in
        ``episode_dir``, then tell rosbag_recorder to stop and delete
        the bag (which removes ``episode_dir`` outright), then defensively
        rmtree anything that survived (e.g. if rosbag's delete was
        partial). Finally flip DataManager to idle *without* bumping
        the episode counter so the next START reuses the same slot.
        """
        if self._data_manager is None:
            response.success = False
            response.message = 'CANCEL: no active recording session'
            return response

        episode_dir = Path(self._data_manager.get_save_rosbag_path() or '')

        # 1. Close mp4/parquet writers + ffmpeg before the bag dir is
        #    deleted underneath them.
        self._stop_episode_writers()

        # 2. rosbag_recorder stops + removes its bag directory
        #    (= episode_dir). Synchronous so we don't race with step 3.
        try:
            self._rosbag.stop_and_delete_rosbag()
        except Exception as exc:  # noqa: BLE001
            self._node.get_logger().warning(
                f'stop_and_delete_rosbag failed: {exc!r}')

        # 3. Belt-and-braces: if anything (videos/, camera_info/, stray
        #    .mcap.tmp from a crash) survived, sweep it.
        if episode_dir.exists():
            try:
                shutil.rmtree(episode_dir)
            except Exception as exc:  # noqa: BLE001
                self._node.get_logger().warning(
                    f'episode_dir cleanup failed: {episode_dir}: {exc!r}')

        # 4. Flip session to idle without bumping the episode counter.
        self._data_manager.discard_recording()
        self._rosbag.publish_action_event(event)

        self._publish_umbrella_status(
            DataOperationStatus.CANCELLED, 'CANCEL',
            f'Recording discarded — episode removed: {episode_dir.name}')

        response.success = True
        response.message = 'Recording discarded'
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
