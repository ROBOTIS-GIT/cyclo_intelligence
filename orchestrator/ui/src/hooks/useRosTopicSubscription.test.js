import React from 'react';
import { act, renderHook } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import ROSLIB from 'roslib';
import rosConnectionManager from '../utils/rosConnectionManager';
import taskReducer from '../features/tasks/taskSlice';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import { useRosTopicSubscription } from './useRosTopicSubscription';

jest.mock('roslib', () => ({ __esModule: true, default: { Topic: jest.fn() } }));
jest.mock('../utils/rosConnectionManager', () => ({
  __esModule: true, default: { getConnection: jest.fn().mockResolvedValue({}) },
}));
jest.mock('./useRosServiceCaller', () => ({
  useRosServiceCaller: () => ({ getRobotInfo: jest.fn() }),
}));
jest.mock('../store/store', () => ({
  __esModule: true, default: { getState: () => ({ tasks: { inferenceStatus: {} }, ui: {} }) },
}));
jest.mock('react-hot-toast', () => ({
  __esModule: true, default: { error: jest.fn() },
}));

describe('central inference status subscription', () => {
  let callback;
  let unsubscribe;
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
    rosConnectionManager.getConnection.mockResolvedValue({});
    unsubscribe = jest.fn();
    ROSLIB.Topic.mockImplementation(() => ({
      subscribe: (handler) => { callback = handler; }, unsubscribe,
    }));
  });
  afterEach(() => {
    jest.useRealTimers();
  });

  async function subscribe() {
    const store = configureStore({
      reducer: { tasks: taskReducer, ros: () => ({ rosbridgeUrl: '' }) },
    });
    const wrapper = ({ children }) => <Provider store={store}>{children}</Provider>;
    const hook = renderHook(() => useRosTopicSubscription(), { wrapper });
    await act(async () => { await hook.result.current.subscribeToInferenceStatus(); });
    return { store, ...hook };
  }

  test('receives full model state, expires silence, and recovers from a new snapshot', async () => {
    const { store, result } = await subscribe();
    const message = {
      inference_phase: InferencePhase.INFERENCING, status_known: true,
      runtime_state: 'running', model_path: '/models/act', policy_id: 'lerobot:act',
      publish_to_robot: true, source_id: 'backend', sequence: 1,
    };
    act(() => callback(message));
    expect(store.getState().tasks.inferenceStatus).toMatchObject({
      inferencePhase: InferencePhase.INFERENCING, topicReceived: true,
      runtimeState: 'running', loadedModelPath: '/models/act',
      loadedPolicyId: 'lerobot:act', publishToRobot: true,
    });
    act(() => jest.advanceTimersByTime(8000));
    expect(store.getState().tasks.inferenceStatus).toMatchObject({
      topicReceived: false, runtimeState: 'unknown',
      inferencePhase: InferencePhase.INFERENCING, loadedModelPath: '/models/act',
    });
    act(() => callback({ ...message, sequence: 2 }));
    expect(store.getState().tasks.inferenceStatus.topicReceived).toBe(true);
    act(() => result.current.cleanup());
    expect(unsubscribe).toHaveBeenCalledTimes(1);
    expect(store.getState().tasks.inferenceStatus.topicReceived).toBe(false);
    expect(jest.getTimerCount()).toBe(0);
  });

  test('an unknown or old-format status cannot authorize inference', async () => {
    const { store } = await subscribe();
    act(() => callback({ inference_phase: InferencePhase.READY }));
    expect(store.getState().tasks.inferenceStatus.topicReceived).toBe(false);
    act(() => callback({
      inference_phase: InferencePhase.SYNCING, status_known: false,
      runtime_state: 'unknown', error: 'offline', source_id: 'backend', sequence: 1,
    }));
    expect(store.getState().tasks.inferenceStatus).toMatchObject({
      topicReceived: false, inferencePhase: InferencePhase.SYNCING, error: 'offline',
    });
  });

  test('recording status identifies its owner and disconnect invalidates it without clearing recording', async () => {
    const { store, result } = await subscribe();
    await act(async () => { await result.current.subscribeToRecordingStatus(); });
    act(() => callback({
      record_phase: RecordPhase.RECORDING, proceed_time: 10,
      task_info: { task_type: 'inference', task_name: 'inference' },
    }));
    expect(store.getState().tasks.recordStatus).toMatchObject({
      taskType: 'inference', recordPhase: RecordPhase.RECORDING,
      running: true, topicReceived: true,
    });
    act(() => result.current.cleanup());
    expect(store.getState().tasks.recordStatus).toMatchObject({
      taskType: 'inference', running: true, topicReceived: false,
    });
  });

  test('pose status expires without clearing a pending return and recovers after restart', async () => {
    const { store, result } = await subscribe();
    await act(async () => { await result.current.subscribeToPoseStatus(); });
    act(() => callback({ data: JSON.stringify({
      robot_type: 'test', device_id: 'first', returning: true, connected: true, saved: true,
    }) }));
    expect(store.getState().tasks.robotPoseStatus.available).toBe(true);
    act(() => jest.advanceTimersByTime(2100));
    expect(store.getState().tasks.robotPoseStatus).toMatchObject({
      available: false, connected: false, returning: true,
    });
    act(() => callback({ data: JSON.stringify({
      robot_type: '', device_id: 'second', returning: false, connected: false, saved: false,
    }) }));
    expect(store.getState().tasks.robotPoseStatus).toMatchObject({
      available: true, device_id: 'second', saved: false, returning: false,
    });
    act(() => result.current.cleanup());
    expect(jest.getTimerCount()).toBe(0);
  });

  test('restores editable settings independently of the loaded model and page', async () => {
    const { store } = await subscribe();
    act(() => callback({
      inference_phase: InferencePhase.INFERENCING, status_known: true,
      runtime_state: 'running', model_path: '/models/running', policy_id: 'lerobot:act',
      source_id: 'backend', sequence: 1, has_task_info: true, task_info_revision: 2,
      task_info: {
        task_type: 'inference', policy_path: '/models/next', policy_id: 'lerobot:groot',
        inference_hz: 25, control_hz: 80, task_instruction: ['Pick the ball'],
      },
    }));
    expect(store.getState().tasks.inferenceTaskInfo).toMatchObject({
      policyPath: '/models/next', policyId: 'lerobot:groot', inferenceHz: 25, controlHz: 80,
    });
    expect(store.getState().tasks.inferenceStatus.loadedModelPath).toBe('/models/running');
    expect(store.getState().tasks.inferenceTaskInfoSync.dirty).toBe(false);
  });

  test('folder changes do not require an editable settings revision', async () => {
    const { store } = await subscribe();
    const status = { inference_phase: InferencePhase.PAUSED, status_known: true,
      runtime_state: 'paused', source_id: 'backend', has_task_info: true, task_info_revision: 1 };
    act(() => callback({ ...status, sequence: 1, task_info: { task_type: 'inference', task_num: 'first' } }));
    expect(store.getState().tasks.inferenceStatus.recordingSessionId).toBe('first');
    act(() => callback({ ...status, sequence: 2, task_info: { task_type: 'inference', task_num: 'second' } }));
    expect(store.getState().tasks.inferenceStatus.recordingSessionId).toBe('second');
    act(() => callback({ ...status, sequence: 1, task_info: { task_type: 'inference', task_num: 'stale' } }));
    expect(store.getState().tasks.inferenceStatus.recordingSessionId).toBe('second');
    act(() => callback({ ...status, source_id: 'restarted', sequence: 1, task_info: { task_type: 'inference', task_num: '' } }));
    expect(store.getState().tasks.inferenceStatus.recordingSessionId).toBe('');
  });
});
