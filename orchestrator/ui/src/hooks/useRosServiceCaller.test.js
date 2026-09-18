import React from 'react';
import { EventEmitter } from 'events';
import { renderHook, waitFor } from '@testing-library/react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import taskReducer from '../features/tasks/taskSlice';
import rosConnectionManager from '../utils/rosConnectionManager';
import {
  buildInitialPoseSyncTaskInfo,
  buildPolicySelectionTaskInfo,
  getRecordCommandServiceTimeoutMs,
  transformReplayDataResult,
  useRosServiceCaller,
} from './useRosServiceCaller';

describe('pose command connection lifetime', () => {
  let ros;
  let current;

  beforeEach(() => {
    ros = Object.assign(new EventEmitter(), {
      isConnected: true, idCounter: 0, callOnConnection: jest.fn(),
    });
    current = ros;
    jest.spyOn(rosConnectionManager, 'getConnection').mockResolvedValue(ros);
    jest.spyOn(rosConnectionManager, 'getCurrentConnection').mockImplementation(() => current);
  });
  afterEach(() => jest.restoreAllMocks());

  function setup() {
    const store = configureStore({ reducer: {
      tasks: taskReducer,
      training: () => ({}), editDataset: () => ({}), ui: () => ({}),
      ros: () => ({ rosbridgeUrl: 'ws://test' }),
    } });
    const wrapper = ({ children }) => <Provider store={store}>{children}</Provider>;
    return renderHook(() => useRosServiceCaller(), { wrapper }).result;
  }

  test('accepts a response from the original live connection', async () => {
    const hook = setup();
    const request = hook.current.sendRobotPoseCommand(1, 'test_robot');
    await waitFor(() => expect(ros.callOnConnection).toHaveBeenCalled());
    const { id } = ros.callOnConnection.mock.calls[0][0];
    ros.emit(id, { values: { success: true } });
    await expect(request).resolves.toEqual(expect.objectContaining({ success: true }));
    expect(ros.listenerCount('close')).toBe(0);
  });

  test.each([2, 4])('sends duration for pose command %i', async (command) => {
    const hook = setup();
    const request = hook.current.sendRobotPoseCommand(command, 'test_robot', 8.5);
    await waitFor(() => expect(ros.callOnConnection).toHaveBeenCalled());
    const payload = ros.callOnConnection.mock.calls[0][0];
    expect(payload.args).toEqual({ command, robot_type: 'test_robot', duration_s: 8.5 });
    ros.emit(payload.id, { values: { success: true } });
    await expect(request).resolves.toEqual(expect.objectContaining({ success: true }));
  });

  test('cancels immediately on close and ignores a delayed response', async () => {
    const hook = setup();
    const request = hook.current.sendRobotPoseCommand(3, 'test_robot');
    await waitFor(() => expect(ros.callOnConnection).toHaveBeenCalled());
    const { id } = ros.callOnConnection.mock.calls[0][0];
    current = null;
    ros.emit('close');
    await expect(request).resolves.toBeNull();
    current = { isConnected: true };
    ros.emit(id, { values: { success: true } });
    expect(ros.listenerCount(id)).toBe(0);
    expect(ros.listenerCount('close')).toBe(0);
  });

  test.each([true, false])('ignores a replaced connection response, success=%s', async (success) => {
    const hook = setup();
    const request = hook.current.sendRobotPoseCommand(3, 'test_robot');
    await waitFor(() => expect(ros.callOnConnection).toHaveBeenCalled());
    const { id } = ros.callOnConnection.mock.calls[0][0];
    current = { isConnected: true };
    ros.emit(id, { result: success, values: { success, message: 'old result' } });
    await expect(request).resolves.toBeNull();
  });
});

describe('buildPolicySelectionTaskInfo', () => {
  test('serializes policy id and stable parameter JSON for TaskInfo', () => {
    expect(buildPolicySelectionTaskInfo({
      policyId: 'sample:base',
      policyParameters: { z: true, gain: 0.5 },
    })).toEqual({
      policy_id: 'sample:base',
      policy_parameters_json: '{"gain":0.5,"z":true}',
    });
  });

  test('uses legacy-compatible empty defaults', () => {
    expect(buildPolicySelectionTaskInfo()).toEqual({
      policy_id: '',
      policy_parameters_json: '{}',
    });
  });
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
