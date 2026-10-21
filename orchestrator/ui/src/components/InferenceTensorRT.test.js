import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferenceModelSelector from './InferenceModelSelector';
import TrtEngineControl from './TrtEngineControl';
import taskReducer from '../features/tasks/taskSlice';

jest.mock('react-hot-toast', () => ({ __esModule: true, default: { error: jest.fn(), success: jest.fn() } }));
jest.mock('./Tooltip', () => ({ children }) => <>{children}</>);

test('switching GR00T to ACT chooses the ACT adapter and drops the old engine path', () => {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({ reducer: { tasks: taskReducer }, preloadedState: { tasks: {
    ...initial, inferenceTaskInfo: { ...initial.inferenceTaskInfo, serviceType: 'groot', policyType: 'n17',
      accelerationMode: 'tensorrt_dit', accelerationEnginePath: '/old/dit_model_bf16.trt' },
  } } });
  render(<Provider store={store}><InferenceModelSelector /></Provider>);
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'lerobot:act' } });
  expect(store.getState().tasks.inferenceTaskInfo.accelerationMode).toBe('tensorrt_act');
  expect(store.getState().tasks.inferenceTaskInfo.accelerationEnginePath).toBe('');
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'lerobot:pi0' } });
  expect(store.getState().tasks.inferenceTaskInfo.accelerationMode).toBe('pytorch');
});

test.each(['lerobot', 'groot'])('build and status calls stay on the %s backend', async (serviceType) => {
  const previous = global.fetch;
  global.fetch = jest.fn().mockResolvedValue({ ok: true, text: async () => JSON.stringify({ status: 'missing' }) });
  try {
    render(<TrtEngineControl serviceType={serviceType} modelPath={`/workspace/model/${serviceType}/test`} robotType="ffw_sg2_rev1" />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining(`/backends/${serviceType}/trt/status?`)));
    fireEvent.click(screen.getByRole('button', { name: /^Build$/ }));
    await waitFor(() => expect(global.fetch).toHaveBeenCalledWith(`/api/backends/${serviceType}/trt/build`, expect.objectContaining({ method: 'POST' })));
  } finally {
    global.fetch = previous;
  }
});
