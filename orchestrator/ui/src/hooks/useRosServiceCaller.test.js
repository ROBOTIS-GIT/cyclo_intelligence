import {
  useRosServiceCaller,
  buildInitialPoseSyncTaskInfo,
  getRecordCommandServiceTimeoutMs,
  transformReplayDataResult,
} from './useRosServiceCaller';
import { act, renderHook } from '@testing-library/react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import taskReducer from '../features/tasks/taskSlice';
import rosConnectionManager from '../utils/rosConnectionManager';

jest.mock('../utils/rosConnectionManager', () => ({
  __esModule: true, default: { getConnection: jest.fn() },
}));

test('completion sends the next instruction without first changing Redux or recording slots', async () => {
  const tasks = taskReducer(undefined, { type: '@@INIT' });
  const state = { tasks: { ...tasks, sharedTaskInfo: { taskInstruction: ['Record overall.'] },
    inferenceTaskInfo: { ...tasks.inferenceTaskInfo, serviceType: 'groot',
      policyPath: '/workspace/model/groot/example', checkpointTaskInstruction: ['phase 1'] } },
    training: { trainingInfo: {} }, editDataset: {}, ui: { currentPage: 'inference' },
    ros: { rosbridgeUrl: 'ws://test' } };
  const store = configureStore({ reducer: (value = state) => value });
  let callback, sent;
  rosConnectionManager.getConnection.mockResolvedValue({ isConnected: true, idCounter: 0,
    once: (_id, handler) => { callback = handler; },
    callOnConnection: (request) => { sent = request; callback({ result: true, values: { success: true } }); },
  });
  const log = jest.spyOn(console, 'log').mockImplementation(() => {});
  try {
    const { result } = renderHook(() => useRosServiceCaller(), {
      wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
    });
    await act(async () => result.current.sendRecordCommand('update_instruction', {
      taskInstruction: ['phase 2'], autofillEmptyTaskFields: false,
    }));
    expect(sent.service).toBe('/task/command');
    expect(sent.args.command).toBe(16);
    expect(sent.args.task_info.task_instruction).toEqual(['phase 2']);
    expect(sent.args.task_info.subtask_instruction).toEqual([]);
    expect(store.getState().tasks.inferenceTaskInfo.checkpointTaskInstruction).toEqual(['phase 1']);
    expect(store.getState().tasks.sharedTaskInfo.taskInstruction).toEqual(['Record overall.']);
  } finally { log.mockRestore(); }
});

describe('buildInitialPoseSyncTaskInfo', () => {
  test('converts UI settings to ROS task info fields', () => {
    expect(buildInitialPoseSyncTaskInfo({
      initialPoseSync: true,
      initialPoseSyncDurationS: 7.5,
    })).toEqual({
      initial_pose_sync: true,
      initial_pose_sync_duration_s: 7.5,
    });
  });

  test('uses safe defaults for legacy UI state', () => {
    expect(buildInitialPoseSyncTaskInfo()).toEqual({
      initial_pose_sync: false,
      initial_pose_sync_duration_s: 5.0,
    });
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
