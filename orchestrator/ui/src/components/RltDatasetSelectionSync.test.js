import React from 'react';
import { act, render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import ConnectedRltDatasetSelectionSync, { RltDatasetSelectionSync } from './RltDatasetSelectionSync';
import tasks, { setInferenceTaskInfo } from '../features/tasks/taskSlice';
import offlineRL, { setOfflineRLDatasetSelections } from '../features/offlineRL/offlineRLSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));

test('selection changes while ON, including empty, never toggle training', async () => {
  jest.useFakeTimers();
  let state = { async_enabled: true, selected_paths: [], replay_source: { transitions: 0 } };
  const callService = jest.fn(async (_service, _type, request) => {
    if (request.command === 14) state = { ...state, selected_paths: request.rlt_dataset_paths };
    return { success: true, message: JSON.stringify(state) };
  });
  const props = { callService, bundlePath: '/bundle', enabled: true };
  const view = render(<RltDatasetSelectionSync {...props} paths={['/a']} />);
  try {
    await act(async () => {});
    expect(state.selected_paths).toEqual(['/a']);
    view.rerender(<RltDatasetSelectionSync {...props} paths={['/a', '/b']} />);
    await act(async () => jest.advanceTimersByTime(1000));
    expect(state.selected_paths).toEqual(['/a', '/b']);
    view.rerender(<RltDatasetSelectionSync {...props} paths={[]} />);
    await act(async () => jest.advanceTimersByTime(1000));
    expect(state.selected_paths).toEqual([]);
    expect(state.async_enabled).toBe(true);
    expect(callService.mock.calls.every((call) => [8, 14].includes(call[2].command))).toBe(true);
    const before = callService.mock.calls.filter((call) => call[2].command === 14).length;
    await act(async () => jest.advanceTimersByTime(1000));
    expect(callService.mock.calls.filter((call) => call[2].command === 14)).toHaveLength(before);
  } finally { view.unmount(); jest.useRealTimers(); }
});

test('preparation failure is visible and does not claim replacement succeeded', async () => {
  const callService = jest.fn().mockResolvedValue({ success: true, message: JSON.stringify({
    selected_paths: ['/bad'], replay_error: 'missing success labels', replay_source: { transitions: 38 },
  }) });
  render(<RltDatasetSelectionSync callService={callService} bundlePath="/bundle" enabled paths={['/bad']} />);
  expect(await screen.findByText(/missing success labels.*previous replay retained/)).toBeVisible();
  expect(callService).toHaveBeenCalledTimes(1);
});

test('connected checkbox selections sync without toggling Async RL, after explicit UI prerequisites', async () => {
  jest.useFakeTimers();
  let state = { async_enabled: false, selected_paths: [] };
  const callService = jest.fn(async (_service, _type, request) => {
    if (request.command === 14) state = { ...state, selected_paths: request.rlt_dataset_paths };
    return { success: true, message: JSON.stringify(state) };
  });
  useRosServiceCaller.mockReturnValue({ callService });
  const store = configureStore({ reducer: { tasks, offlineRL } });
  store.dispatch(setOfflineRLDatasetSelections([{ path: '/a' }]));
  const view = render(<Provider store={store}><ConnectedRltDatasetSelectionSync /></Provider>);
  try {
    expect(screen.getByRole('status')).toHaveTextContent('select GR00T N1.7');
    expect(callService).not.toHaveBeenCalled();
    act(() => store.dispatch(setInferenceTaskInfo({ serviceType: 'groot', policyType: 'n17' })));
    expect(screen.getByRole('status')).toHaveTextContent('enable RLT');
    act(() => store.dispatch(setInferenceTaskInfo({ rltEnabled: true })));
    expect(screen.getByRole('status')).toHaveTextContent('select an RLT Bundle Path');
    expect(callService).not.toHaveBeenCalled();
    await act(async () => store.dispatch(setInferenceTaskInfo({ rltBundlePath: '/bundle' })));
    expect(state.selected_paths).toEqual(['/a']);
    await act(async () => {
      store.dispatch(setOfflineRLDatasetSelections([{ path: '/a' }, { path: '/b' }]));
    });
    await act(async () => jest.advanceTimersByTime(1000));
    expect(state.selected_paths).toEqual(['/a', '/b']);
    await act(async () => store.dispatch(setOfflineRLDatasetSelections([])));
    await act(async () => jest.advanceTimersByTime(1000));
    expect(state.selected_paths).toEqual([]);
    expect(state.async_enabled).toBe(false);
    expect(callService.mock.calls.every((call) => [8, 14].includes(call[2].command))).toBe(true);
    act(() => store.dispatch(setInferenceTaskInfo({ rltEnabled: false })));
    const count = callService.mock.calls.length;
    await act(async () => jest.advanceTimersByTime(2000));
    expect(callService).toHaveBeenCalledTimes(count);
  } finally { view.unmount(); jest.useRealTimers(); }
});

test('LOAD-first errors recover by polling, without starting training or loading a model', async () => {
  jest.useFakeTimers();
  const callService = jest.fn()
    .mockResolvedValueOnce({ success: false, message: 'Unavailable: LOAD first' })
    .mockResolvedValue({ success: true, message: JSON.stringify({ selected_paths: ['/a'] }) });
  const view = render(<RltDatasetSelectionSync callService={callService} bundlePath="/bundle" enabled paths={['/a']} />);
  try {
    await act(async () => {});
    expect(screen.getByRole('status')).toHaveTextContent('Unavailable: LOAD first');
    await act(async () => jest.advanceTimersByTime(1000));
    expect(screen.getByRole('status')).toHaveTextContent('1 datasets selected');
    expect(callService.mock.calls.map((call) => call[2].command)).toEqual([8, 8]);
  } finally { view.unmount(); jest.useRealTimers(); }
});
