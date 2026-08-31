import { InferencePhase } from '../constants/taskPhases';
import PageType from '../constants/pageType';
import {
  getInferenceTaskInfoKey,
  hasRosTaskInfoPayload,
  normalizeInferenceTaskInfo,
  rosTaskInfoToUiTaskInfo,
  shouldApplyServerTaskInfoToPage,
} from './taskInfoSync';

describe('taskInfoSync echo routing', () => {
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

  test('keeps blank inference numeric fields distinct while editing', () => {
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

  test('preserves initial pose sync settings across ROS task info conversion', () => {
    const converted = rosTaskInfoToUiTaskInfo({
      task_type: 'inference',
      initial_pose_sync: true,
      initial_pose_sync_duration_s: 7.5,
    });

    expect(normalizeInferenceTaskInfo(converted)).toMatchObject({
      initialPoseSync: true,
      initialPoseSyncDurationS: 7.5,
    });
  });

  test('uses the initial pose sync default for legacy ROS task info', () => {
    const converted = rosTaskInfoToUiTaskInfo({ task_type: 'inference' });

    expect(converted.initialPoseSync).toBe(false);
    expect(converted.initialPoseSyncDurationS).toBe(5.0);
  });
});
