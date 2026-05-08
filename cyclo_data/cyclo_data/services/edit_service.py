# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""/data/edit service — EditDataset handler.

Dispatches MERGE / DELETE on rosbag task folders. Pure filesystem work;
no background worker needed, so the callback runs synchronously in the
io callback group.

Step 3 Part C2b migrated the real logic here (previously stubbed in
Part B2). The orchestrator's /dataset/edit forwarder was removed once
the UI roslib switched to /data/edit directly. Orchestrator still owns
DataEditor for the read-only /dataset/get_info path (separate service).
"""

from pathlib import Path

from cyclo_data.editor.episode_editor import DataEditor

from interfaces.msg import DataOperationStatus
from interfaces.srv import EditDataset


_MODE_NAMES = {
    EditDataset.Request.MERGE: 'MERGE',
    EditDataset.Request.DELETE: 'DELETE',
}


class EditService:
    SERVICE_NAME = '/data/edit'

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
        node.get_logger().info(f'Service advertised: {self.SERVICE_NAME}')

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
