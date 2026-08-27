import {
  getEpisodeOutcomeForCommand,
  getActionPolicyCommandFields,
  getCommandRecordingFolder,
  getConversionCommandFields,
  getRecordCommandServiceTimeoutMs,
  isInferenceCommandRequest,
  normalizeEpisodeOutcome,
  transformReplayDataResult,
} from './useRosServiceCaller';
import { EpisodeOutcome } from '../constants/taskCommand';
import PageType from '../constants/pageType';

describe('command task-info source', () => {
  test('requires an explicit inference source on Offline RL pages', () => {
    expect(isInferenceCommandRequest(PageType.INFERENCE)).toBe(true);
    expect(isInferenceCommandRequest(PageType.OFFLINE_RL)).toBe(false);
    expect(isInferenceCommandRequest(
      PageType.OFFLINE_RL,
      'inference'
    )).toBe(true);
    expect(isInferenceCommandRequest(PageType.INFERENCE, 'record')).toBe(false);
  });
});

describe('recording folder command override', () => {
  const taskInfo = {
    recordingFolder: '/workspace/rosbag2/Task_stale_inference_MCAP',
  };

  test('uses an atomic command override before the Redux snapshot', () => {
    expect(getCommandRecordingFolder(taskInfo, {
      recordingFolder: '/workspace/rosbag2/Task_fresh_inference_MCAP',
    })).toBe('/workspace/rosbag2/Task_fresh_inference_MCAP');
    expect(getCommandRecordingFolder(taskInfo, {}))
      .toBe('/workspace/rosbag2/Task_stale_inference_MCAP');
    expect(getCommandRecordingFolder(taskInfo, { recordingFolder: '' }))
      .toBe('');
  });
});

describe('getRecordCommandServiceTimeoutMs', () => {
  test('does not time out recording save commands', () => {
    expect(getRecordCommandServiceTimeoutMs('stop_segment')).toBe(0);
    expect(getRecordCommandServiceTimeoutMs('finish_episode')).toBe(0);
    expect(getRecordCommandServiceTimeoutMs('stop_inference_record')).toBe(0);
  });

  test('keeps shorter defaults for non-save commands', () => {
    expect(getRecordCommandServiceTimeoutMs('refresh_topics')).toBe(10000);
    expect(getRecordCommandServiceTimeoutMs('start_inference')).toBe(30000);
  });

  test('allows callers to override the service timeout', () => {
    expect(getRecordCommandServiceTimeoutMs('stop_segment', {
      serviceTimeoutMs: 45000,
    })).toBe(45000);
  });
});

describe('RLT action-policy command fields', () => {
  const taskInfo = {
    rltEnabled: true,
    rltBundlePath: ' /workspace/checkpoint/rlt/showroom_bundle ',
    rltRobotOverride: true,
    actionPolicyMode: 'rlt',
  };

  test('forces a fresh Start Inference request onto base GR00T', () => {
    expect(getActionPolicyCommandFields(taskInfo, 'start_inference')).toEqual({
      rlt_enabled: true,
      rlt_bundle_path: '/workspace/checkpoint/rlt/showroom_bundle',
      rlt_robot_override: true,
      action_policy_mode: 'base',
    });
  });

  test('serializes an explicit hot-switch target without changing preload fields', () => {
    expect(getActionPolicyCommandFields(taskInfo, 'set_action_policy', {
      actionPolicyMode: 'rlt',
    })).toEqual({
      rlt_enabled: true,
      rlt_bundle_path: '/workspace/checkpoint/rlt/showroom_bundle',
      rlt_robot_override: true,
      action_policy_mode: 'rlt',
    });
  });

  test('allows an explicit Real Robot approval to override the stored safety flag', () => {
    expect(getActionPolicyCommandFields(
      { ...taskInfo, rltRobotOverride: false },
      'set_action_policy',
      { actionPolicyMode: 'rlt', rltRobotOverride: true }
    )).toEqual({
      rlt_enabled: true,
      rlt_bundle_path: '/workspace/checkpoint/rlt/showroom_bundle',
      rlt_robot_override: true,
      action_policy_mode: 'rlt',
    });
  });
});

describe('conversion command fields', () => {
  test('forwards a normalized LeRobot destination and preserves empty legacy default', () => {
    expect(getConversionCommandFields({
      conversionFps: 15,
      convertV21: true,
      convertV30: false,
      lerobotOutputRoot: ' /workspace/lerobot/round_1 ',
      deleteSourceAfterSuccess: true,
    })).toEqual({
      conversion_fps: 15,
      convert_v21: true,
      convert_v30: false,
      lerobot_output_root: '/workspace/lerobot/round_1',
      delete_source_after_success: true,
    });
    expect(getConversionCommandFields().lerobot_output_root).toBe('');
  });
});

describe('inference recording outcome', () => {
  test('accepts only Success and Failure values', () => {
    expect(normalizeEpisodeOutcome(EpisodeOutcome.SUCCESS))
      .toBe(EpisodeOutcome.SUCCESS);
    expect(normalizeEpisodeOutcome(EpisodeOutcome.FAILURE))
      .toBe(EpisodeOutcome.FAILURE);
    expect(normalizeEpisodeOutcome(99)).toBe(EpisodeOutcome.UNSPECIFIED);
  });

  test('forces general recording commands to UNSPECIFIED', () => {
    expect(getEpisodeOutcomeForCommand('stop', EpisodeOutcome.SUCCESS))
      .toBe(EpisodeOutcome.UNSPECIFIED);
    expect(getEpisodeOutcomeForCommand(
      'stop_inference_record',
      EpisodeOutcome.FAILURE
    )).toBe(EpisodeOutcome.FAILURE);
  });
});

describe('transformReplayDataResult', () => {
  test('preserves replay robot metadata for the 3D viewer', () => {
    const result = transformReplayDataResult(
      {
        success: true,
        robot_type: 'ffw_sh5_rev1',
        urdf_path: '/workspace/robot_configs/urdf/ffw_sh5_follower.urdf',
        end_effector_links: ['tool0'],
      },
      '/workspace/rosbag2/sh5/0'
    );

    expect(result.robot_type).toBe('ffw_sh5_rev1');
    expect(result.urdf_path).toBe(
      '/workspace/robot_configs/urdf/ffw_sh5_follower.urdf'
    );
    expect(result.end_effector_links).toEqual(['tool0']);
    expect(result.bag_path).toBe('/workspace/rosbag2/sh5/0');
  });
});
