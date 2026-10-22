import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferenceModelSelector from './InferenceModelSelector';
import taskReducer, { selectInferenceTaskInfo } from '../features/tasks/taskSlice';

test('selects ViTacFormer with its own backend and restores existing policy defaults', () => {
  const store = configureStore({ reducer: { tasks: taskReducer } });
  render(<Provider store={store}><InferenceModelSelector /></Provider>);
  const selector = screen.getByRole('combobox');
  expect(selectInferenceTaskInfo(store.getState())).toMatchObject({
    serviceType: 'lerobot', policyType: 'act', inferenceHz: 15, actionRequestMode: 'async',
  });
  fireEvent.change(selector, { target: { value: 'vitacformer:vitacformer' } });
  expect(selectInferenceTaskInfo(store.getState())).toMatchObject({
    serviceType: 'vitacformer', policyType: 'vitacformer', inferenceHz: 30,
    actionRequestMode: 'async', accelerationMode: 'pytorch',
  });
  fireEvent.change(selector, { target: { value: 'lerobot:act' } });
  expect(selectInferenceTaskInfo(store.getState())).toMatchObject({
    serviceType: 'lerobot', policyType: 'act', inferenceHz: 15, actionRequestMode: 'async',
  });
});
