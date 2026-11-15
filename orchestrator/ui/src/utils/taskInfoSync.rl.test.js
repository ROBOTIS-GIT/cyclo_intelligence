import { InferencePhase } from '../constants/taskPhases';
import PageType from '../constants/pageType';
import {
  getInferenceTaskInfoKey,
  hasRosTaskInfoPayload,
  normalizeActionRequestMode,
  rosTaskInfoToUiTaskInfo,
  shouldApplyServerTaskInfoToPage,
} from './taskInfoSync';

describe('taskInfoSync echo routing', () => {
  test('preserves the TT-RTC wire value and defaults unknown request modes', () => {
    expect(normalizeActionRequestMode(' tt_rtc ')).toBe('tt_rtc');
    expect(normalizeActionRequestMode('sync')).toBe('sync');
    expect(normalizeActionRequestMode('unknown')).toBe('async');
    expect(rosTaskInfoToUiTaskInfo({
      task_type: 'inference',
      service_type: 'groot',
      action_request_mode: 'tt_rtc',
    })).toEqual(expect.objectContaining({
      actionRequestMode: 'tt_rtc',
    }));
  });

  test('detects inference task info even without record identity fields', () => {
    expect(hasRosTaskInfoPayload({
      task_type: 'inference',
      task_instruction: ['pick up the cup'],
    })).toBe(true);
  });

  test('applies inference task info to inference pages', () => {
    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'inference' },
      currentPage: PageType.INFERENCE,
      inferencePhase: InferencePhase.READY,
    })).toBe(true);
  });

  test('applies inference task info to the Offline RL workspace', () => {
    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'inference' },
      currentPage: PageType.AUTONOMY_STUDIO,
      workspaceKind: 'rl_canvas',
      inferencePhase: InferencePhase.READY,
    })).toBe(true);
  });

  test('applies record task info to inference pages', () => {
    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'record' },
      currentPage: PageType.INFERENCE,
      inferencePhase: InferencePhase.READY,
    })).toBe(true);
  });

  test('applies inference task info to record pages while inference is active or idle', () => {
    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'inference' },
      currentPage: PageType.RECORD,
      inferencePhase: InferencePhase.INFERENCING,
    })).toBe(true);

    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'inference' },
      currentPage: PageType.RECORD,
      inferencePhase: InferencePhase.READY,
    })).toBe(true);
  });

  test('applies record task info to record pages while inference is idle or active', () => {
    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'record' },
      currentPage: PageType.RECORD,
      inferencePhase: InferencePhase.READY,
    })).toBe(true);

    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'record' },
      currentPage: PageType.RECORD,
      inferencePhase: InferencePhase.INFERENCING,
    })).toBe(true);
  });

  test('applies task info to home only for initial sync', () => {
    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'record' },
      currentPage: PageType.HOME,
      inferencePhase: InferencePhase.READY,
      initialTaskInfoSynced: false,
    })).toBe(true);

    expect(shouldApplyServerTaskInfoToPage({
      taskInfo: { taskType: 'record' },
      currentPage: PageType.HOME,
      inferencePhase: InferencePhase.READY,
      initialTaskInfoSynced: true,
    })).toBe(false);
  });

  test('preserves blank inference numeric drafts until explicitly corrected', () => {
    expect(getInferenceTaskInfoKey({
      taskType: 'inference',
      taskInstruction: ['pick'],
      policyPath: '/policy',
      serviceType: 'groot',
      controlHz: '',
      inferenceHz: '',
      chunkAlignWindowS: '',
    })).not.toBe(getInferenceTaskInfoKey({
      taskType: 'inference',
      taskInstruction: ['pick'],
      policyPath: '/policy',
      serviceType: 'groot',
      controlHz: 100,
      inferenceHz: 15,
      chunkAlignWindowS: 0.3,
    }));
  });

  test('hydrates RLT preload and runtime route fields from ROS task info', () => {
    expect(rosTaskInfoToUiTaskInfo({
      task_type: 'inference',
      rlt_enabled: true,
      rlt_bundle_path: '/workspace/checkpoint/rlt/showroom_bundle',
      rlt_robot_override: true,
      action_policy_mode: 'rlt',
    })).toEqual(expect.objectContaining({
      rltEnabled: true,
      rltBundlePath: '/workspace/checkpoint/rlt/showroom_bundle',
      rltRobotOverride: true,
      actionPolicyMode: 'rlt',
    }));
  });

  test('round-trips initial pose sync settings', () => {
    expect(rosTaskInfoToUiTaskInfo({
      task_type: 'inference',
      initial_pose_sync: true,
      initial_pose_sync_duration_s: 3.5,
    })).toEqual(expect.objectContaining({
      initialPoseSync: true,
      initialPoseSyncDurationS: 3.5,
    }));
  });

  test('hydrates policy type from the existing TaskInfo tags field', () => {
    expect(rosTaskInfoToUiTaskInfo({
      task_type: 'inference',
      service_type: 'lerobot',
      tags: ['inference_mode:simulation', 'policy_type:multi_task_dit'],
    })).toEqual(expect.objectContaining({
      serviceType: 'lerobot',
      policyType: 'multi_task_dit',
    }));
  });

  test('infers N1.7 for a legacy untagged GR00T status message', () => {
    expect(rosTaskInfoToUiTaskInfo({
      task_type: 'inference',
      service_type: 'groot',
      tags: ['inference_mode:simulation'],
    })).toEqual(expect.objectContaining({
      serviceType: 'groot',
      policyType: 'n17',
    }));
  });

  test('does not invent a policy type for a legacy untagged LeRobot message', () => {
    expect(rosTaskInfoToUiTaskInfo({
      task_type: 'inference',
      service_type: 'lerobot',
    })).not.toHaveProperty('policyType');
  });

  test('normalizes TensorRT fields out of unsupported policy sync keys', () => {
    expect(getInferenceTaskInfoKey({
      serviceType: 'lerobot',
      policyType: 'act',
      accelerationMode: 'tensorrt_dit',
      accelerationEnginePath: '/workspace/model/groot/dit.trt',
    })).toBe(getInferenceTaskInfoKey({
      serviceType: 'lerobot',
      policyType: 'act',
      accelerationMode: 'pytorch',
      accelerationEnginePath: '',
    }));
  });
});
