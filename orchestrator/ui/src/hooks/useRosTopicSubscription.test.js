import React from 'react';
import { act, renderHook } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import ROSLIB from 'roslib';
import rosConnectionManager from '../utils/rosConnectionManager';
import taskReducer from '../features/tasks/taskSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosTopicSubscription } from './useRosTopicSubscription';

jest.mock('roslib', () => ({ __esModule: true, default: { Topic: jest.fn() } }));
jest.mock('../utils/rosConnectionManager', () => ({
  __esModule: true, default: { getConnection: jest.fn().mockResolvedValue({}) },
}));
jest.mock('./useRosServiceCaller', () => ({
  useRosServiceCaller: () => ({ getRobotInfo: jest.fn() }),
}));
jest.mock('../store/store', () => ({
  __esModule: true, default: { getState: () => ({ tasks: {} }) },
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
});
