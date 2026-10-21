import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferenceChunkControl from './InferenceChunkControl';
import InferenceModelSelector from './InferenceModelSelector';
import taskReducer, { selectInferenceTaskInfo, setInferenceTaskInfo } from '../features/tasks/taskSlice';
import { getInferenceTaskInfoKey, rosTaskInfoToUiTaskInfo } from '../utils/taskInfoSync';
import { buildInferenceChunkTaskInfo } from '../hooks/useRosServiceCaller';

test('switching model families chooses GR00T 24 and ACT 100', () => {
  const store = configureStore({ reducer: { tasks: taskReducer } });
  render(<Provider store={store}><InferenceModelSelector /></Provider>);
  expect(selectInferenceTaskInfo(store.getState()).actionChunkSize).toBe(100);
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'groot:n17' } });
  expect(selectInferenceTaskInfo(store.getState()).actionChunkSize).toBe(24);
  act(() => store.dispatch(setInferenceTaskInfo({ actionChunkSize: 8 })));
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'lerobot:act' } });
  expect(selectInferenceTaskInfo(store.getState()).actionChunkSize).toBe(100);
});

test('displays ACT trained horizon and selected ensemble horizon without changing requested 100', async () => {
  const previous = global.fetch;
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ max_chunk_size: 30, temporal_ensemble: true }) });
  try {
    const info = { serviceType: 'lerobot', policyType: 'act', policyPath: '/workspace/model/lerobot/act', actionChunkSize: 100 };
    const onChange = jest.fn();
    const { rerender } = render(<InferenceChunkControl info={info} onChange={onChange} />);
    await waitFor(() => expect(screen.getByText(/Checkpoint horizon: 30. Used: 30/)).toBeInTheDocument());
    expect(screen.getByRole('spinbutton')).toHaveValue(100);
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value: '8' } });
    expect(onChange).toHaveBeenCalledWith('actionChunkSize', 8);
    rerender(<InferenceChunkControl info={{ ...info, actionChunkSize: 8 }} onChange={onChange} readonly />);
    expect(screen.getByText(/using 8 prediction steps/)).toBeInTheDocument();
    expect(screen.getByRole('spinbutton')).toBeDisabled();
  } finally { global.fetch = previous; }
});

test('chunk changes participate in task sync and round-trip through ROS, preserving ACT TensorRT', () => {
  const info = { serviceType: 'lerobot', policyType: 'act', actionChunkSize: 8,
    accelerationMode: 'tensorrt_act', accelerationEnginePath: '/model/act_model_fp32.trt' };
  expect(getInferenceTaskInfoKey(info)).not.toBe(getInferenceTaskInfoKey({ ...info, actionChunkSize: 30 }));
  const payload = buildInferenceChunkTaskInfo(info);
  expect(payload).toEqual({ action_chunk_size: 8, acceleration_mode: 'tensorrt_act', acceleration_engine_path: '/model/act_model_fp32.trt' });
  expect(rosTaskInfoToUiTaskInfo(payload).actionChunkSize).toBe(8);
  expect(rosTaskInfoToUiTaskInfo({ service_type: 'groot' }).actionChunkSize).toBe(24);
});
