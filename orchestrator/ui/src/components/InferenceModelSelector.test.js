import React from 'react';
import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferenceModelSelector from './InferenceModelSelector';
import taskReducer from '../features/tasks/taskSlice';
import { PolicyCatalogProvider } from '../contexts/PolicyCatalogContext';
import { testPolicyCatalog } from '../testUtils/policyCatalog';

const renderSelector = (inferenceOverrides = {}, catalog = testPolicyCatalog) => {
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
    <PolicyCatalogProvider initialCatalog={catalog}>
      <Provider store={store}><InferenceModelSelector /></Provider>
    </PolicyCatalogProvider>
  );
  return store;
};

test.each([true, false])('prioritizes LeRobot when present (%s) without changing selection or catalog', (includeLeRobot) => {
  const [lerobot, groot] = testPolicyCatalog.runtimes;
  const standalone = (id, label) => ({
    ...groot, id, label,
    models: [{ ...groot.models[0], policy_id: `${id}:test`, id: 'test' }],
  });
  const runtimes = [
    standalone('abot', 'ABot'), groot,
    ...(includeLeRobot ? [lerobot] : []),
    standalone('lingbot_vla', 'LingBot-VLA'), standalone('rldx', 'RLDX'),
  ];
  const originalOrder = runtimes.map((runtime) => runtime.id);
  const store = renderSelector(
    { policyId: 'groot:n17', serviceType: 'groot', policyType: 'n17' },
    { ...testPolicyCatalog, runtimes },
  );

  const selector = screen.getByRole('combobox', { name: 'Policy model' });
  expect(Array.from(selector.querySelectorAll('optgroup'), (group) => group.label)).toEqual([
    ...(includeLeRobot ? ['LeRobot'] : []), 'ABot', 'GR00T', 'LingBot-VLA', 'RLDX',
  ]);
  expect(selector).toHaveValue('groot:n17');
  expect(store.getState().tasks.inferenceTaskInfo.policyId).toBe('groot:n17');
  expect(runtimes.map((runtime) => runtime.id)).toEqual(originalOrder);
  if (includeLeRobot) {
    expect(Array.from(selector.querySelector('optgroup').children, (option) => option.value))
      .toEqual(lerobot.models.map((model) => model.policy_id));
  }
});

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

test('switching to a model-owned queue resets Action Steps to All', () => {
  const store = renderSelector({ actionSteps: 5 });
  fireEvent.change(screen.getByRole('combobox', { name: 'Policy model' }), {
    target: { value: 'lerobot:diffusion' },
  });
  expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(0);
});

test('switching between chunk policies preserves the selected count', () => {
  const store = renderSelector({ actionSteps: 5 });
  fireEvent.change(screen.getByRole('combobox', { name: 'Policy model' }), {
    target: { value: 'groot:n17' },
  });
  expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(5);
});

test.each(['wall_x', 'groot'])(
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

test.each(['pi0', 'pi05', 'groot'])(
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

test.each(['future:unknown', 'lerobot:eo1', 'lerobot:evo1', 'lerobot:pi0_fast', 'lerobot:multi_task_dit'])(
  'unavailable policy %s is not silently replaced', (policyId) => {
  const [serviceType, policyType] = policyId.split(':');
  const store = renderSelector({
    policyId, serviceType, policyType,
  });

  expect(screen.getByRole('option', { name: 'Selected policy is unavailable' }))
    .toBeInTheDocument();
  expect(store.getState().tasks.inferenceTaskInfo.policyId).toBe(policyId);
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
