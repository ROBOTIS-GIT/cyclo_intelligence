# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dataset edit and read-only task-info services.

``/data/edit`` dispatches filesystem mutations on rosbag task folders.
``/dataset/get_info`` exposes robot-independent dataset status from the same
always-on data-plane service, so it remains available before a robot type is
selected.

Step 3 Part C2b migrated the real logic here (previously stubbed in
Part B2). The orchestrator's /dataset/edit forwarder was removed once
the UI roslib switched to /data/edit directly.
"""

from pathlib import Path

from cyclo_data.editor.episode_editor import DataEditor

from interfaces.msg import DataOperationStatus, DatasetInfo
from interfaces.srv import EditDataset, GetDatasetInfo
from shared.io.dataset_lock import DatasetOperationLock


_MODE_NAMES = {
    EditDataset.Request.MERGE: 'MERGE',
    EditDataset.Request.DELETE: 'DELETE',
    EditDataset.Request.PRUNE_OLDEST: 'PRUNE_OLDEST',
}


class EditService:
    SERVICE_NAME = '/data/edit'
    INFO_SERVICE_NAME = '/dataset/get_info'

    def __init__(self, node, status_publisher):
        self._node = node
        self._status_pub = status_publisher
        self._editor = DataEditor()
        self._server = node.create_service(
            EditDataset,
            self.SERVICE_NAME,
            self._callback,
            callback_group=node.io_callback_group,
        )
        self._info_server = node.create_service(
            GetDatasetInfo,
            self.INFO_SERVICE_NAME,
            self._get_info_callback,
            callback_group=node.io_callback_group,
        )
        node.get_logger().info(f'Service advertised: {self.SERVICE_NAME}')
        node.get_logger().info(f'Service advertised: {self.INFO_SERVICE_NAME}')

    def _get_info_callback(self, request, response):
        try:
            with DatasetOperationLock(exclusive=False):
                task_info = self._editor.get_rosbag_task_info(
                    Path(request.dataset_path)
                )

            info = DatasetInfo()
            info.robot_type = task_info.robot_type
            info.task_instruction = task_info.task_instruction
            info.episode_count = int(task_info.episode_count)
            info.total_duration_s = float(task_info.total_duration_s)
            info.fps = int(task_info.fps)
            info.success_count = int(task_info.success_count)
            info.failure_count = int(task_info.failure_count)
            info.unlabeled_count = int(task_info.unlabeled_count)
            info.success_episode_indices = task_info.success_episode_indices
            info.failure_episode_indices = task_info.failure_episode_indices
            info.unlabeled_episode_indices = task_info.unlabeled_episode_indices

            response.dataset_info = info
            response.success = True
            response.message = 'Task info retrieved successfully'
        except Exception as exc:  # noqa: BLE001 — surface any failure to UI
            self._node.get_logger().error(
                f'GetDatasetInfo failed: {exc}'
            )
            response.dataset_info = DatasetInfo()
            response.success = False
            response.message = f'Error: {exc}'
        return response

    def _callback(self, request, response):
        mode_name = _MODE_NAMES.get(request.mode)
        if mode_name is None:
            response.success = False
            response.message = f'Unknown edit mode: {request.mode}'
            response.affected_count = 0
            self._node.get_logger().warn(response.message)
            return response

        self._publish_status(DataOperationStatus.RUNNING, mode_name, '')

        try:
            with DatasetOperationLock(exclusive=True):
                if request.mode == EditDataset.Request.MERGE:
                    result = self._editor.merge_rosbag_task_folders(
                        [Path(p) for p in request.merge_source_task_dirs],
                        Path(request.merge_output_task_dir),
                        move=bool(request.merge_move_sources),
                    )
                    response.success = True
                    response.affected_count = int(result.total_episodes)
                    response.message = (
                        f'Merged {result.total_episodes} episodes into '
                        f'{result.output_dir} '
                        f'(mode={"move" if result.moved else "copy"})'
                    )

                elif request.mode == EditDataset.Request.DELETE:
                    result = self._editor.delete_rosbag_episodes(
                        Path(request.delete_task_dir),
                        [int(i) for i in request.delete_episode_num],
                        compact=bool(request.delete_compact),
                    )
                    response.success = True
                    response.affected_count = int(result.deleted_count)
                    response.message = (
                        f'Deleted {result.deleted_count} episodes from '
                        f'{result.task_dir} (compact={result.compacted}, '
                        f'remaining={result.remaining_count})'
                    )

                elif request.mode == EditDataset.Request.PRUNE_OLDEST:
                    result, selected = self._editor.prune_oldest_rosbag_episodes(
                        Path(request.delete_task_dir),
                        success_count=int(request.prune_oldest_success_count),
                        failure_count=int(request.prune_oldest_failure_count),
                    )
                    response.success = True
                    response.affected_count = int(result.deleted_count)
                    response.message = (
                        f'Deleted oldest episodes {selected} from {result.task_dir} '
                        f'(remaining={result.remaining_count}, indices preserved)'
                    )

            self._publish_status(
                DataOperationStatus.COMPLETED, mode_name, response.message)
            return response

        except Exception as exc:  # noqa: BLE001 — surface any failure to UI
            self._node.get_logger().error(f'EditDataset.{mode_name} failed: {exc}')
            response.success = False
            response.affected_count = 0
            response.message = f'{mode_name} failed: {exc}'
            self._publish_status(
                DataOperationStatus.FAILED, mode_name, str(exc))
            return response

    def _publish_status(self, status: int, stage: str, message: str) -> None:
        msg = DataOperationStatus()
        msg.operation_type = DataOperationStatus.OP_EDIT
        msg.status = status
        msg.job_id = ''
        msg.progress_percentage = 0.0
        msg.stage = stage
        msg.message = message
        self._status_pub.publish(msg)
