// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import PageType from '../constants/pageType';

const stringArray = (items) => (
  Array.isArray(items) ? items.map((item) => String(item ?? '')) : []
);

const numberOrDefault = (value, fallback) => {
  if (value === '' || value == null) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const editableNumberOrDefault = (value, fallback) => (
  value === '' ? '' : numberOrDefault(value, fallback)
);

const actionRequestModeOrDefault = (value) => (
  String(value ?? '').trim().toLowerCase() === 'sync' ? 'sync' : 'async'
);

export const normalizeRecordTaskInfo = (taskInfo = {}) => ({
  taskNum: String(taskInfo.taskNum ?? '').trim(),
  taskName: String(taskInfo.taskName ?? '').trim(),
  taskType: String(taskInfo.taskType ?? 'record').trim() || 'record',
  taskInstruction: stringArray(taskInfo.taskInstruction),
  subtaskInstruction: stringArray(taskInfo.subtaskInstruction),
  includeRobotisLicense: Boolean(taskInfo.includeRobotisLicense),
  warmupTime: numberOrDefault(taskInfo.warmupTime ?? 0, 0),
  episodeTime: numberOrDefault(taskInfo.episodeTime ?? 0, 0),
  resetTime: numberOrDefault(taskInfo.resetTime ?? 0, 0),
  numEpisodes: numberOrDefault(taskInfo.numEpisodes ?? 0, 0),
  pushToHub: Boolean(taskInfo.pushToHub),
  privateMode: Boolean(taskInfo.privateMode),
  useOptimizedSave: Boolean(taskInfo.useOptimizedSave),
  recordRosBag2: Boolean(taskInfo.recordRosBag2),
});

export const normalizeInferenceTaskInfo = (taskInfo = {}) => ({
  taskType: 'inference',
  taskInstruction: stringArray(taskInfo.taskInstruction),
  policyPath: String(taskInfo.policyPath ?? '').trim(),
  recordInferenceMode: Boolean(taskInfo.recordInferenceMode),
  controlHz: editableNumberOrDefault(taskInfo.controlHz ?? 100, 100),
  inferenceHz: editableNumberOrDefault(taskInfo.inferenceHz ?? 15, 15),
  chunkAlignWindowS: editableNumberOrDefault(
    taskInfo.chunkAlignWindowS ?? 0.3,
    0.3
  ),
  serviceType: String(taskInfo.serviceType ?? '').trim(),
  policyType: String(taskInfo.policyType ?? '').trim(),
  inferenceMode: String(taskInfo.inferenceMode ?? 'simulation').trim() || 'simulation',
  actionRequestMode: actionRequestModeOrDefault(taskInfo.actionRequestMode),
  accelerationMode: String(taskInfo.accelerationMode ?? 'pytorch').trim(),
  accelerationEnginePath: String(taskInfo.accelerationEnginePath ?? '').trim(),
  initialPoseSync: Boolean(taskInfo.initialPoseSync),
  initialPoseSyncDurationS: numberOrDefault(
    taskInfo.initialPoseSyncDurationS ?? 5.0,
    5.0
  ),
});

export const getRecordTaskInfoKey = (taskInfo = {}) =>
  JSON.stringify(normalizeRecordTaskInfo(taskInfo));

export const getInferenceTaskInfoKey = (taskInfo = {}) =>
  JSON.stringify(normalizeInferenceTaskInfo(taskInfo));

export const rosTaskInfoToUiTaskInfo = (taskInfo = {}) => ({
  taskNum: taskInfo.task_num || '',
  taskName: taskInfo.task_name || '',
  taskType: taskInfo.task_type || '',
  taskInstruction: taskInfo.task_instruction || [],
  subtaskInstruction: taskInfo.subtask_instruction || [],
  policyPath: taskInfo.policy_path || '',
  recordInferenceMode: Boolean(taskInfo.record_inference_mode),
  serviceType: taskInfo.service_type || 'lerobot',
  inferenceMode: taskInfo.inference_mode || 'simulation',
  actionRequestMode: actionRequestModeOrDefault(taskInfo.action_request_mode),
  accelerationMode: taskInfo.acceleration_mode || 'pytorch',
  accelerationEnginePath: taskInfo.acceleration_engine_path || '',
  initialPoseSync: Boolean(taskInfo.initial_pose_sync),
  initialPoseSyncDurationS: numberOrDefault(
    taskInfo.initial_pose_sync_duration_s,
    5.0
  ),
  userId: taskInfo.user_id || '',
  controlHz: taskInfo.control_hz || 100,
  inferenceHz: taskInfo.inference_hz || 15,
  chunkAlignWindowS: taskInfo.chunk_align_window_s || 0.3,
  includeRobotisLicense: Boolean(taskInfo.include_robotis_license),
  warmupTime: taskInfo.warmup_time_s || 0,
  episodeTime: taskInfo.episode_time_s || 0,
  resetTime: taskInfo.reset_time_s || 0,
  numEpisodes: taskInfo.num_episodes || 0,
  pushToHub: Boolean(taskInfo.push_to_hub),
  privateMode: Boolean(taskInfo.private_mode),
  useOptimizedSave: Boolean(taskInfo.use_optimized_save_mode),
  recordRosBag2: Boolean(taskInfo.record_rosbag2),
});

export const hasRosTaskInfoPayload = (taskInfo = {}) => {
  const hasText = (value) => String(value ?? '').trim().length > 0;
  const hasTextArray = (items) => (
    Array.isArray(items) && items.some((item) => hasText(item))
  );
  return Boolean(taskInfo) && (
    hasText(taskInfo.task_name) ||
    hasText(taskInfo.task_type) ||
    hasText(taskInfo.policy_path) ||
    hasText(taskInfo.service_type) ||
    hasTextArray(taskInfo.task_instruction) ||
    hasTextArray(taskInfo.subtask_instruction)
  );
};

export const shouldApplyServerTaskInfoToPage = ({
  currentPage,
  initialTaskInfoSynced = false,
} = {}) => {
  if (currentPage === PageType.INFERENCE || currentPage === PageType.RECORD) {
    return true;
  }

  if (currentPage === PageType.HOME) {
    return !initialTaskInfoSynced;
  }

  return false;
};
