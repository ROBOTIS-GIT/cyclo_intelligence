import React from 'react';
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
import InferenceTryResults from './InferenceTryResults';
let mockState;
jest.mock('react-redux', () => ({ useSelector: fn => fn(mockState) }));
jest.mock('../features/tasks/taskSlice', () => ({ selectInferenceTaskInfo: s => s.info }));
beforeEach(() => {
  mockState = { info: { policyPath: '/models/selected' }, tasks: { inferenceStatus: { loadedModelPath: '/models/loaded' } } };
  global.fetch = jest.fn();
});
afterEach(() => { jest.restoreAllMocks(); jest.resetAllMocks(); });
test('uses loaded model, saves a try, and corrects it without adding another', async () => {
  const row = { try: '1', result: 'success', created_at: '2026-09-18T00:00:00Z' };
  fetch.mockResolvedValueOnce({ ok: true, json: async () => ({ rows: [], file: '/records/model.csv', revision: 'v1' }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ rows: [row], file: '/records/model.csv', revision: 'v1' }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ rows: [{ ...row, result: 'fail' }], file: '/records/model.csv', revision: 'v1' }) });
  render(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByText('Success')).toBeEnabled());
  expect(fetch.mock.calls[0][0]).toContain(encodeURIComponent('/models/loaded'));
  expect(screen.queryByRole('button', { name: 'Refresh try results' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByText('Success'));
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  expect(screen.queryByLabelText('Try 1 result')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Toggle try results' }));
  await screen.findByLabelText('Try 1 result');
  fireEvent.change(screen.getByLabelText('Try 1 result'), { target: { value: 'fail' } });
  await waitFor(() => expect(screen.getByText('1 trials · 0 success / 1 fail · 0%')).toBeInTheDocument());
  expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({ model: '/models/loaded', result: 'success' });
  expect(fetch.mock.calls[2][1].method).toBe('PATCH');
  expect(JSON.parse(fetch.mock.calls[2][1].body).revision).toBe('v1');
  fireEvent.click(screen.getByRole('button', { name: 'Toggle try results' }));
  expect(screen.queryByLabelText('Try 1 result')).not.toBeInTheDocument();
  expect(screen.queryByText(/Next try/)).not.toBeInTheDocument();
});
test('shows save failure without incrementing the try count', async () => {
  fetch.mockResolvedValueOnce({ ok: true, json: async () => ({ rows: [], file: '' }) })
    .mockResolvedValueOnce({ ok: false, json: async () => ({ detail: 'Disk full' }) });
  render(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByText('Fail')).toBeEnabled());
  fireEvent.click(screen.getByText('Fail'));
  expect(await screen.findByRole('alert')).toHaveTextContent('Disk full');
  expect(screen.queryByText(/Next try/)).not.toBeInTheDocument();
});

test('deletes only selected trials after confirmation and retains other rows', async () => {
  const row = { try: '1', result: 'success', created_at: '2026-09-18T00:00:00Z' };
  const other = { ...row, try: '2', result: 'fail' };
  fetch.mockResolvedValueOnce({ ok: true, json: async () => ({ rows: [row, other], file: '/records/model.csv', revision: 'v1' }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ rows: [{ ...other, try: '1' }], file: '/records/model.csv', revision: 'v1' }) });
  const confirm = jest.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true);
  render(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByText('Success')).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Toggle try results' }));
  expect(screen.getByRole('button', { name: 'Delete selected trials' })).toBeDisabled();
  fireEvent.click(screen.getByLabelText('Select try 1'));
  fireEvent.click(screen.getByRole('button', { name: 'Delete selected trials' }));
  expect(fetch).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button', { name: 'Delete selected trials' }));
  await waitFor(() => expect(screen.getByText('1 trials · 0 success / 1 fail · 0%')).toBeInTheDocument());
  expect(fetch.mock.calls[1][1].method).toBe('DELETE');
  expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({ model: '/models/loaded', tries: ['1'], revision: 'v1' });
  expect(screen.getByLabelText('Try 1 result')).toHaveValue('fail');
  expect(screen.queryByLabelText('Try 2 result')).not.toBeInTheDocument();
  expect(screen.getByLabelText('Select try 1')).not.toBeChecked();
  expect(screen.getByRole('button', { name: 'Delete selected trials' })).toBeDisabled();
  fireEvent.click(screen.getByLabelText('Select all trials'));
  expect(screen.getByLabelText('Select try 1')).toBeChecked();
  fireEvent.click(screen.getByLabelText('Select all trials'));
  expect(screen.getByLabelText('Select try 1')).not.toBeChecked();
  confirm.mockRestore();
});

const response = (rows = [], revision = 'v1') => ({
  ok: true, json: async () => ({ rows, revision, file: '/records/model.csv' }),
});

test('does not request or enable logging without a selected model', () => {
  mockState.info.policyPath = '';
  mockState.tasks.inferenceStatus.loadedModelPath = '';
  render(<InferenceTryResults />);
  expect(fetch).not.toHaveBeenCalled();
  expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Fail' })).toBeDisabled();
});

test('reloads persisted history on remount and falls back to the selected model', async () => {
  mockState.tasks.inferenceStatus.loadedModelPath = '';
  const row = { try: '1', result: 'fail', created_at: '2026-09-18T00:00:00Z' };
  fetch.mockResolvedValue(response([row]));
  const { unmount } = render(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled());
  expect(fetch.mock.calls[0][0]).toContain(encodeURIComponent('/models/selected'));
  unmount();
  render(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Toggle try results' }));
  expect(screen.getByLabelText('Try 1 result')).toHaveValue('fail');
  expect(fetch).toHaveBeenCalledTimes(2);
});

test('ignores an old model history arriving after the new model history', async () => {
  let resolveOld;
  fetch.mockReturnValueOnce(new Promise(resolve => { resolveOld = resolve; }))
    .mockResolvedValueOnce(response());
  const { rerender } = render(<InferenceTryResults />);
  mockState.tasks.inferenceStatus.loadedModelPath = '/models/new';
  rerender(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled());
  await act(async () => resolveOld(response([{ try: '99', result: 'fail', created_at: '2026-09-18T00:00:00Z' }])));
  fireEvent.click(screen.getByRole('button', { name: 'Toggle try results' }));
  expect(screen.queryByLabelText('Try 99 result')).not.toBeInTheDocument();
  expect(screen.getByText('0 trials · 0 success / 0 fail · 0%')).toBeInTheDocument();
});

test('old model write completion cannot release a newer pending write', async () => {
  let finishOld, finishNew;
  fetch.mockResolvedValueOnce(response())
    .mockReturnValueOnce(new Promise(resolve => { finishOld = resolve; }))
    .mockResolvedValueOnce(response())
    .mockReturnValueOnce(new Promise(resolve => { finishNew = resolve; }));
  const { rerender } = render(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Success' }));
  fireEvent.click(screen.getByRole('button', { name: 'Success' }));
  expect(fetch).toHaveBeenCalledTimes(2);
  mockState.tasks.inferenceStatus.loadedModelPath = '/models/new';
  rerender(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Fail' }));
  expect(JSON.parse(fetch.mock.calls[3][1].body).model).toBe('/models/new');
  await act(async () => finishOld(response()));
  expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();
  await act(async () => finishNew(response()));
  expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled();
});

test('reports stale edit conflict and reloads on remount without retrying the edit', async () => {
  const row = { try: '1', result: 'success', created_at: '2026-09-18T00:00:00Z' };
  fetch.mockResolvedValueOnce(response([row]))
    .mockResolvedValueOnce({ ok: false, status: 409, json: async () => ({ detail: 'Trial history changed. Refresh and try again.' }) })
    .mockResolvedValueOnce(response([{ ...row, result: 'fail' }], 'v2'));
  const { unmount } = render(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Toggle try results' }));
  fireEvent.change(screen.getByLabelText('Try 1 result'), { target: { value: 'fail' } });
  expect(await screen.findByRole('alert')).toHaveTextContent('Trial history changed');
  expect(screen.getByLabelText('Try 1 result')).toHaveValue('success');
  unmount();
  render(<InferenceTryResults />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Success' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Toggle try results' }));
  await waitFor(() => expect(screen.getByLabelText('Try 1 result')).toHaveValue('fail'));
  expect(fetch).toHaveBeenCalledTimes(3);
  expect(fetch.mock.calls[2][1].method).toBeUndefined();
});
