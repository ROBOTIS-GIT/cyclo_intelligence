import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import RltPolicyUpdateControl from './RltPolicyUpdateControl';

const initial = { auto_apply: false, async_enabled: false, can_apply: false,
  max_updates: 1000, updates_this_run: 0, limit_reached: false,
  training_version: 0, inference_version: 0, pending_version: null };
const response = (value) => ({ success: true, message: JSON.stringify(value) });

test('sends a budget on ON and displays the backend stop without issuing OFF', async () => {
  jest.useFakeTimers();
  let server = { ...initial };
  const callService = jest.fn(async (_name, _type, request) => {
    if (request.command === 12) server = { ...server, async_enabled: true, max_updates: request.rlt_max_updates };
    return response(server);
  });
  const view = render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  try {
    await act(async () => {});
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Async RL max updates' }), { target: { value: '3' } });
    await act(async () => fireEvent.click(screen.getByRole('switch', { name: 'Async RL' })));
    expect(callService).toHaveBeenLastCalledWith('/groot/inference_command', 'interfaces/srv/InferenceCommand',
      { command: 12, rlt_bundle_path: '/bundle', rlt_max_updates: 3 });
    server = { ...server, async_enabled: false, updates_this_run: 3, limit_reached: true,
      total_critic_updates: 193, total_actor_updates: 96 };
    await act(async () => jest.advanceTimersByTime(2000));
    expect(screen.getByText(/Run: 3 \/ 3/)).toHaveTextContent('Total Critic 193 / Actor 96 · Limit reached — paused');
    expect(screen.getByRole('switch', { name: 'Async RL' })).toHaveAttribute('aria-checked', 'false');
    expect(callService.mock.calls.every(call => [8, 12].includes(call[2].command))).toBe(true);
  } finally { view.unmount(); jest.useRealTimers(); }
});

test('invalid limits and older runtimes cannot start unbounded training', async () => {
  const callService = jest.fn().mockResolvedValue(response(initial));
  const view = render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  await waitFor(() => expect(screen.getByRole('switch', { name: 'Async RL' })).toBeEnabled());
  for (const value of ['', '0', '-1', '1.5']) {
    fireEvent.change(screen.getByRole('spinbutton'), { target: { value } });
    expect(screen.getByRole('switch', { name: 'Async RL' })).toBeDisabled();
  }
  view.unmount();
  callService.mockResolvedValue(response({ ...initial, max_updates: undefined }));
  render(<RltPolicyUpdateControl callService={callService} available bundlePath="/other" />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Update the GR00T runtime and interfaces');
  expect(screen.getByRole('switch', { name: 'Async RL' })).toBeDisabled();
});

test('shows the selected replay source and the RLT-only apply boundary', async () => {
  const callService = jest.fn().mockResolvedValue(response({
    ...initial, auto_apply: true, can_apply: true, training_version: 619,
    replay_source: { kind: 'selected_datasets', paths: ['/selected/lerobot'],
      episodes: 2, transitions: 38, batch_size: 32 },
  }));
  render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  const source = await screen.findByText(/Data: selected LeRobot datasets/);
  expect(source).toHaveTextContent('2 episodes · 38 transitions · batch 32');
  expect(source).toHaveAttribute('title', '/selected/lerobot');
  expect(screen.getByText(/Use Save Bundle to keep training/)).toBeVisible();
  expect(screen.getByText(/Waiting for the next RLT request/)).toBeVisible();
  expect(screen.getByRole('status')).toHaveTextContent('Staged v619 · Inference v0');
});

test('Save Bundle is separate from Apply and shows the completed output path', async () => {
  jest.useFakeTimers();
  let server = { ...initial, last_update: { completed_critic_updates: 4 } };
  const callService = jest.fn(async (_name, _type, request) => {
    if (request.command === 15) server = { ...server, saving: true };
    return response(server);
  });
  const view = render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  try {
    await act(async () => {});
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Save Bundle' })));
    expect(screen.getByRole('button', { name: 'Saving…' })).toBeDisabled();
    server = { ...server, saving: false, saved_bundle_path: '/checkpoint/async_rlt_c4_new',
      saved_critic_updates: 4, saved_actor_updates: 2 };
    await act(async () => jest.advanceTimersByTime(2000));
    expect(screen.getByText(/Saved · Critic 4 \/ Actor 2/)).toHaveTextContent('/checkpoint/async_rlt_c4_new');
    expect(callService.mock.calls.map(call => call[2].command)).toEqual([8, 15, 8]);
    expect(screen.getByRole('status')).toHaveTextContent('Inference v0');
  } finally {
    view.unmount();
    jest.useRealTimers();
  }
});

test('unloaded policy has no polling or enabled controls', () => {
  const callService = jest.fn();
  render(<RltPolicyUpdateControl callService={callService} available={false} bundlePath="/bundle" />);
  expect(screen.getByRole('switch', { name: 'Async RL' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Apply Policy' })).toBeDisabled();
  expect(callService).not.toHaveBeenCalled();
});

test('manual apply queues a candidate without claiming it is active', async () => {
  const staged = { ...initial, can_apply: true, training_version: 2 };
  const callService = jest.fn().mockResolvedValueOnce(response(staged))
    .mockResolvedValueOnce(response({ ...staged, pending_version: 2 }));
  render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Apply Policy' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: 'Apply Policy' }));
  await screen.findByText(/Inference v0 · Pending v2/);
  expect(callService).toHaveBeenLastCalledWith('/groot/inference_command',
    'interfaces/srv/InferenceCommand', { command: 11, rlt_bundle_path: '/bundle' });
  expect(screen.getByRole('button', { name: 'Apply Policy' })).toBeDisabled();
});

test('auto apply and training are independent, and server errors are visible', async () => {
  const callService = jest.fn().mockResolvedValueOnce(response(initial))
    .mockResolvedValueOnce(response({ ...initial, auto_apply: true }))
    .mockResolvedValueOnce({ success: false, message: 'Replay missing' });
  render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  const auto = screen.getByRole('switch', { name: 'Auto Apply' });
  await waitFor(() => expect(auto).toBeEnabled());
  fireEvent.click(auto);
  await waitFor(() => expect(auto).toHaveAttribute('aria-checked', 'true'));
  const training = screen.getByRole('switch', { name: 'Async RL' });
  expect(training).toHaveAttribute('aria-checked', 'false');
  fireEvent.click(training);
  await screen.findByText(/Replay missing/);
  expect(training).toHaveAttribute('aria-checked', 'false');
});

test('late status from the previous bundle cannot overwrite the current one', async () => {
  let resolveOld;
  const callService = jest.fn().mockReturnValueOnce(new Promise((resolve) => { resolveOld = resolve; }))
    .mockResolvedValueOnce(response({ ...initial, training_version: 3 }));
  const { rerender } = render(<RltPolicyUpdateControl callService={callService} available bundlePath="/old" />);
  rerender(<RltPolicyUpdateControl callService={callService} available bundlePath="/new" />);
  await screen.findByText(/Staged v3/);
  await act(async () => resolveOld(response({ ...initial, training_version: 99 })));
  expect(screen.queryByText(/Staged v99/)).not.toBeInTheDocument();
  expect(screen.getByText(/Staged v3/)).toBeInTheDocument();
});

test('polling follows prepare, candidate, pending apply, applied and paused states', async () => {
  jest.useFakeTimers();
  let server = { ...initial };
  const callService = jest.fn(async (_name, _type, request) => {
    if (request.command === 12) server = { ...server, async_enabled: true, preparing: true };
    if (request.command === 11) server = { ...server, pending_version: server.training_version };
    if (request.command === 13) server = { ...server, async_enabled: false };
    return response(server);
  });
  const view = render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  try {
    await act(async () => {});
    const training = screen.getByRole('switch', { name: 'Async RL' });
    const apply = screen.getByRole('button', { name: 'Apply Policy' });
    expect(apply).toBeDisabled();
    await act(async () => fireEvent.click(training));
    expect(training).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByRole('status')).toHaveTextContent('Preparing Async RL');
    server = { ...server, preparing: false, training_version: 1, can_apply: true };
    await act(async () => jest.advanceTimersByTime(2000));
    expect(apply).toBeEnabled();
    await act(async () => fireEvent.click(apply));
    expect(apply).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('Inference v0 · Pending v1');
    server = { ...server, inference_version: 1, pending_version: null, can_apply: false };
    await act(async () => jest.advanceTimersByTime(2000));
    expect(screen.getByRole('status')).toHaveTextContent('Staged v1 · Inference v1');
    expect(screen.getByRole('status')).not.toHaveTextContent('Pending');
    await act(async () => fireEvent.click(training));
    expect(training).toHaveAttribute('aria-checked', 'false');
    expect(callService.mock.calls.map((call) => call[2].command)).toEqual([8, 12, 8, 11, 8, 13]);
    expect(callService.mock.calls.every((call) => call[2].rlt_bundle_path === '/bundle')).toBe(true);
  } finally {
    view.unmount();
    jest.useRealTimers();
  }
});

test('a stale poll cannot turn Async RL OFF after an ON command succeeds', async () => {
  jest.useFakeTimers();
  let resolvePoll;
  const callService = jest.fn().mockResolvedValueOnce(response(initial))
    .mockImplementationOnce(() => new Promise((resolve) => { resolvePoll = resolve; }))
    .mockResolvedValueOnce(response({ ...initial, async_enabled: true }));
  const view = render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  try {
    await act(async () => {});
    await act(async () => jest.advanceTimersByTime(2000));
    const training = screen.getByRole('switch', { name: 'Async RL' });
    await act(async () => fireEvent.click(training));
    await act(async () => resolvePoll(response(initial)));
    expect(training).toHaveAttribute('aria-checked', 'true');
    expect(training).toBeEnabled();
  } finally {
    view.unmount();
    jest.useRealTimers();
  }
});

test('changing bundles during a command ignores its late reply and releases controls', async () => {
  let resolveCommand;
  const callService = jest.fn().mockResolvedValueOnce(response(initial))
    .mockImplementationOnce(() => new Promise((resolve) => { resolveCommand = resolve; }))
    .mockResolvedValueOnce(response({ ...initial, training_version: 7 }));
  const view = render(<RltPolicyUpdateControl callService={callService} available bundlePath="/old" />);
  const training = screen.getByRole('switch', { name: 'Async RL' });
  await waitFor(() => expect(training).toBeEnabled());
  fireEvent.click(training);
  expect(training).toBeDisabled();
  view.rerender(<RltPolicyUpdateControl callService={callService} available bundlePath="/new" />);
  await screen.findByText(/Staged v7/);
  await act(async () => resolveCommand(response({ ...initial, async_enabled: true, training_version: 99 })));
  expect(training).toBeEnabled();
  expect(training).toHaveAttribute('aria-checked', 'false');
  expect(screen.queryByText(/Staged v99/)).not.toBeInTheDocument();
});

test('an unloaded backend disables controls and polling can recover after load', async () => {
  jest.useFakeTimers();
  const callService = jest.fn()
    .mockResolvedValueOnce({ success: false, message: 'LOAD first' })
    .mockResolvedValueOnce(response({ ...initial, training_version: 1, can_apply: true }));
  const view = render(<RltPolicyUpdateControl callService={callService} available bundlePath="/bundle" />);
  try {
    await act(async () => {});
    expect(screen.getByRole('status')).toHaveTextContent('Unavailable: LOAD first');
    expect(screen.getByRole('switch', { name: 'Async RL' })).toBeDisabled();
    await act(async () => jest.advanceTimersByTime(2000));
    expect(screen.getByRole('switch', { name: 'Async RL' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Apply Policy' })).toBeEnabled();
    expect(screen.getByRole('status')).not.toHaveTextContent('Unavailable');
  } finally {
    view.unmount();
    jest.useRealTimers();
  }
});
