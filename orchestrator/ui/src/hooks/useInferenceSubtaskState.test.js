import { act, renderHook } from '@testing-library/react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import ROSLIB from 'roslib';
import rosConnectionManager from '../utils/rosConnectionManager';
import { useInferenceSubtaskState } from './useInferenceSubtaskState';

jest.mock('roslib', () => ({ __esModule: true, default: { Topic: jest.fn() } }));
jest.mock('../utils/rosConnectionManager', () => ({
  __esModule: true, default: { getConnection: jest.fn().mockResolvedValue({}) },
}));

test('automatic runtime updates synchronize the UI and ignore old phases or wrong checkpoints', async () => {
  rosConnectionManager.getConnection.mockResolvedValue({});
  let receive;
  const unsubscribe = jest.fn();
  ROSLIB.Topic.mockImplementation(() => ({ subscribe: (callback) => { receive = callback; }, unsubscribe }));
  const store = configureStore({ reducer: () => ({ ros: { rosbridgeUrl: 'ws://test' } }) });
  const onInstruction = jest.fn();
  const props = { enabled: true, modelPath: '/workspace/model/groot/example', instruction: 'phase 1',
    stages: [['phase 1'], ['phase 2'], ['phase 3']], onInstruction };
  const now = jest.spyOn(Date, 'now').mockReturnValue(100000);
  try {
    const { rerender, unmount } = renderHook((value) => useInferenceSubtaskState(value), {
      initialProps: props, wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
    });
    await act(async () => {});
    const emit = (value) => receive({ data: JSON.stringify({ available: true, model_path: props.modelPath,
      instruction: 'phase 2', subtask_index: 1, timestamp: 100, ...value }) });
    emit({ model_path: '/workspace/model/groot/other' });
    emit({ timestamp: 20 });
    receive({ data: '{' });
    expect(onInstruction).not.toHaveBeenCalled();
    emit({});
    expect(onInstruction).toHaveBeenCalledWith('phase 2');
    rerender({ ...props, instruction: 'phase 2' });
    emit({ instruction: 'phase 1', subtask_index: 0, timestamp: 100.001 });
    expect(onInstruction).toHaveBeenCalledTimes(1);
    unmount();
    expect(unsubscribe).toHaveBeenCalled();
  } finally { now.mockRestore(); }
});
