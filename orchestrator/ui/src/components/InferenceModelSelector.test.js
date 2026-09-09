import React from 'react';
import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferenceModelSelector from './InferenceModelSelector';
import taskReducer from '../features/tasks/taskSlice';
import { PolicyCatalogProvider } from '../contexts/PolicyCatalogContext';
import { testPolicyCatalog } from '../testUtils/policyCatalog';

const renderSelector = (inferenceOverrides = {}) => {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: {
      tasks: {
        ...initial,
        inferenceTaskInfo: {
          ...initial.inferenceTaskInfo,
          policyId: 'lerobot:act',
          policyParameters: { stale: true },
          ...inferenceOverrides,
        },
      },
    },
  });
  render(
    <PolicyCatalogProvider initialCatalog={testPolicyCatalog}>
      <Provider store={store}><InferenceModelSelector /></Provider>
    </PolicyCatalogProvider>
  );
  return store;
};

test('model selection comes from catalog and resets model-specific values', () => {
  const store = renderSelector();

  fireEvent.change(screen.getByRole('combobox', { name: 'Policy model' }), {
    target: { value: 'groot:n17' },
  });

  const info = store.getState().tasks.inferenceTaskInfo;
  expect(info.policyId).toBe('groot:n17');
  expect(info.serviceType).toBe('groot');
  expect(info.policyType).toBe('n17');
  expect(info.policyParameters).toEqual({});
  expect(info.accelerationMode).toBe('pytorch');
  expect(info.accelerationEnginePath).toBe('');
});

test.each(['eo1', 'evo1', 'wall_x', 'pi0_fast', 'groot'])(
  'selects LeRobot %s without using the independent GR00T Worker', (model) => {
    const store = renderSelector({
      policyId: 'groot:n17', serviceType: 'groot', policyType: 'n17',
      accelerationMode: 'tensorrt_dit', accelerationEnginePath: '/old.trt',
    });
    fireEvent.change(screen.getByRole('combobox', { name: 'Policy model' }), {
      target: { value: `lerobot:${model}` },
    });
    expect(store.getState().tasks.inferenceTaskInfo).toMatchObject({
      policyId: `lerobot:${model}`, serviceType: 'lerobot', policyType: model,
      policyParameters: {}, accelerationMode: '', accelerationEnginePath: '',
    });
  }
);

test.each(['pi0', 'pi0_fast', 'groot'])(
  'restores the saved LeRobot %s selection', async (model) => {
    const store = renderSelector({ policyId: '', serviceType: 'lerobot', policyType: model });
    await waitFor(() => {
      expect(store.getState().tasks.inferenceTaskInfo.policyId).toBe(`lerobot:${model}`);
    });
  }
);

test('legacy service and policy fields are upgraded to a namespaced policy id', async () => {
  const store = renderSelector({
    policyId: '',
    serviceType: 'groot',
    policyType: 'n17',
  });

  await waitFor(() => {
    expect(store.getState().tasks.inferenceTaskInfo.policyId).toBe('groot:n17');
  });
  expect(store.getState().tasks.inferenceTaskInfoSync.dirty).toBe(false);
});

test('initial ACT normalization does not submit defaults before backend restoration', async () => {
  const store = renderSelector({ policyId: '' });
  await waitFor(() => {
    expect(store.getState().tasks.inferenceTaskInfo.policyId).toBe('lerobot:act');
  });
  expect(store.getState().tasks.inferenceTaskInfoSync.dirty).toBe(false);
});

test('a runtime with one model resolves legacy runtime-only selection', async () => {
  const store = renderSelector({
    policyId: '',
    serviceType: 'groot',
    policyType: '',
  });

  await waitFor(() => {
    expect(store.getState().tasks.inferenceTaskInfo.policyId).toBe('groot:n17');
  });
});

test('an explicit unknown policy is not silently replaced', () => {
  const store = renderSelector({
    policyId: 'future:unknown',
    serviceType: 'future',
    policyType: 'unknown',
  });

  expect(screen.getByRole('option', { name: 'Selected policy is unavailable' }))
    .toBeInTheDocument();
  expect(store.getState().tasks.inferenceTaskInfo.policyId).toBe('future:unknown');
});

test('switching away removes task fields owned only by the previous model', () => {
  const store = renderSelector({
    policyId: 'groot:n17',
    serviceType: 'groot',
    policyType: 'n17',
    accelerationMode: 'tensorrt_dit',
    accelerationEnginePath: '/workspace/model/groot/engine.trt',
  });

  fireEvent.change(screen.getByRole('combobox', { name: 'Policy model' }), {
    target: { value: 'lerobot:act' },
  });

  const info = store.getState().tasks.inferenceTaskInfo;
  expect(info.policyId).toBe('lerobot:act');
  expect(info.accelerationMode).toBe('');
  expect(info.accelerationEnginePath).toBe('');
});
