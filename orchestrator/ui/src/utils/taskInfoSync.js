import PageType, { isInferenceWorkspacePage } from '../constants/pageType';
import { supportsTensorRtInference } from '../constants/policyCapabilities';

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

export const normalizeActionRequestMode = (value) => {
  const normalized = String(value ?? '').trim().toLowerCase();
  return ['sync', 'tt_rtc'].includes(normalized) ? normalized : 'async';
};

const actionPolicyModeOrDefault = (value) => (
  String(value ?? '').trim().toLowerCase() === 'rlt' ? 'rlt' : 'base'
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

export const normalizeInferenceTaskInfo = (taskInfo = {}) => {
  const serviceType = String(taskInfo.serviceType ?? '').trim();
  const policyType = String(taskInfo.policyType ?? '').trim();
  const supportsTensorRt = supportsTensorRtInference(serviceType, policyType);
  return {
    taskType: 'inference',
    taskInstruction: stringArray(taskInfo.taskInstruction),
    policyPath: String(taskInfo.policyPath ?? '').trim(),
    recordInferenceMode: Boolean(taskInfo.recordInferenceMode),
    recordingFolder: String(taskInfo.recordingFolder ?? '').trim(),
    controlHz: numberOrDefault(taskInfo.controlHz ?? 100, 100),
    inferenceHz: numberOrDefault(taskInfo.inferenceHz ?? 15, 15),
    chunkAlignWindowS: numberOrDefault(taskInfo.chunkAlignWindowS ?? 0.3, 0.3),
    serviceType,
    policyType,
    inferenceMode: String(taskInfo.inferenceMode ?? 'simulation').trim() || 'simulation',
    actionRequestMode: normalizeActionRequestMode(taskInfo.actionRequestMode),
    accelerationMode: supportsTensorRt
      ? String(taskInfo.accelerationMode ?? 'pytorch').trim()
      : 'pytorch',
    accelerationEnginePath: supportsTensorRt
      ? String(taskInfo.accelerationEnginePath ?? '').trim()
      : '',
    rltEnabled: Boolean(taskInfo.rltEnabled),
    rltBundlePath: String(taskInfo.rltBundlePath ?? '').trim(),
    rltRobotOverride: Boolean(taskInfo.rltRobotOverride),
    actionPolicyMode: actionPolicyModeOrDefault(taskInfo.actionPolicyMode),
  };
};

export const getRecordTaskInfoKey = (taskInfo = {}) =>
  JSON.stringify(normalizeRecordTaskInfo(taskInfo));

export const getInferenceTaskInfoKey = (taskInfo = {}) =>
  JSON.stringify(normalizeInferenceTaskInfo(taskInfo));

const getTaggedPolicyType = (taskInfo = {}) => {
  const prefix = 'policy_type:';
  const tag = (taskInfo.tags || []).find(
    (item) => String(item || '').trim().startsWith(prefix)
  );
  return tag ? String(tag).trim().slice(prefix.length).trim() : '';
};

export const rosTaskInfoToUiTaskInfo = (taskInfo = {}) => {
  const serviceType = taskInfo.service_type || 'lerobot';
  const taggedPolicyType = getTaggedPolicyType(taskInfo);
  // N1.7 is the only deployed legacy GR00T policy. Old status messages do
  // not carry policy_type, so infer that one safe pair while leaving an
  // untagged LeRobot policy untouched in the local UI.
  const policyType = taggedPolicyType || (serviceType === 'groot' ? 'n17' : '');
  return {
    taskNum: taskInfo.task_num || '',
    taskName: taskInfo.task_name || '',
    taskType: taskInfo.task_type || '',
    taskInstruction: taskInfo.task_instruction || [],
    subtaskInstruction: taskInfo.subtask_instruction || [],
    policyPath: taskInfo.policy_path || '',
    recordInferenceMode: Boolean(taskInfo.record_inference_mode),
    serviceType,
    ...(policyType ? { policyType } : {}),
    inferenceMode: taskInfo.inference_mode || 'simulation',
    actionRequestMode: normalizeActionRequestMode(taskInfo.action_request_mode),
    accelerationMode: taskInfo.acceleration_mode || 'pytorch',
    accelerationEnginePath: taskInfo.acceleration_engine_path || '',
    rltEnabled: Boolean(taskInfo.rlt_enabled),
    rltBundlePath: taskInfo.rlt_bundle_path || '',
    rltRobotOverride: Boolean(taskInfo.rlt_robot_override),
    actionPolicyMode: actionPolicyModeOrDefault(taskInfo.action_policy_mode),
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
  };
};

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
    hasText(taskInfo.rlt_bundle_path) ||
    hasText(taskInfo.action_policy_mode) ||
    hasTextArray(taskInfo.task_instruction) ||
    hasTextArray(taskInfo.subtask_instruction)
  );
};

export const shouldApplyServerTaskInfoToPage = ({
  currentPage,
  initialTaskInfoSynced = false,
} = {}) => {
  if (isInferenceWorkspacePage(currentPage) || currentPage === PageType.RECORD) {
    return true;
  }

  if (currentPage === PageType.HOME) {
    return !initialTaskInfoSynced;
  }

  return false;
};
