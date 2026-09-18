import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import SavedPoseControlPanel from './SavedPoseControlPanel';
import tasks, { selectRobotType, setRobotPoseStatus } from '../features/tasks/taskSlice';
import ros, { setRosbridgeUrl } from '../features/ros/rosSlice';
import toast from 'react-hot-toast';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));
jest.mock('react-hot-toast', () => ({ __esModule: true, default: { success: jest.fn(), error: jest.fn() } }));

const status = {
  robot_type: 'test', device_id: 'runtime-1', available: true, connected: true,
  saved: true, returning: false, duration_s: 5, joint_names: ['joint'], positions: [0.1], units: ['rad'],
};

async function setup(phase = InferencePhase.READY, options = {}) {
  const initial = tasks(undefined, { type: '@@INIT' });
  const store = configureStore({ reducer: { tasks, ros }, preloadedState: { tasks: {
    ...initial, robotType: 'test', robotPoseStatus: { ...status, ...options.status },
    inferenceTaskInfo: { ...initial.inferenceTaskInfo, inferenceMode: 'robot' },
    inferenceStatus: { ...initial.inferenceStatus, inferencePhase: phase },
  }, ros: { ...ros(undefined, { type: '@@INIT' }), connected: true, rosbridgeUrl: 'ws://first' } } });
  const command = options.command || jest.fn().mockResolvedValue({ ...status, success: true, message: 'OK' });
  useRosServiceCaller.mockReturnValue({ sendRobotPoseCommand: command });
  const { unmount } = render(<Provider store={store}><SavedPoseControlPanel /></Provider>);
  await act(async () => { await Promise.resolve(); });
  await waitFor(() => expect(command).toHaveBeenCalledWith(0, 'test'));
  return { store, command, unmount };
}

beforeEach(() => jest.clearAllMocks());
afterEach(() => { jest.restoreAllMocks(); jest.useRealTimers(); });

test('keeps save and return controls without routine status text or refresh', async () => {
  const { store } = await setup(InferencePhase.READY, { status: { saved: false } });
  expect(screen.queryByText('Not saved')).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Refresh pose status' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save Initial Pose' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Return to Saved Pose' })).toBeDisabled();
  act(() => store.dispatch(setRobotPoseStatus({ saved: true })));
  expect(screen.queryByText('Saved')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Return to Saved Pose' })).toBeEnabled();
  act(() => store.dispatch(setRobotPoseStatus({ returning: true })));
  expect(screen.queryByText('Returning...')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Stop pose return' })).toBeEnabled();
});

test('still displays backend errors and keeps stop available', async () => {
  await setup(InferencePhase.READY, {
    status: { returning: true, error: 'Current-pose hold failed' },
  });
  expect(screen.getByText('Current-pose hold failed')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Stop pose return' })).toBeEnabled();
});

test('queries once and uses backend status instead of browser polling', async () => {
  jest.useFakeTimers();
  const { store, command } = await setup();
  expect(command).toHaveBeenCalledTimes(1);
  act(() => jest.advanceTimersByTime(10000));
  expect(command).toHaveBeenCalledTimes(1);
  act(() => store.dispatch(setRobotPoseStatus({ returning: true })));
  expect(screen.queryByRole('button', { name: 'Return to Saved Pose' })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Stop pose return' })).toBeEnabled();
});

test('returns immediately without confirmation and sends the selected robot type', async () => {
  const { command } = await setup();
  jest.spyOn(window, 'confirm').mockReturnValue(false);
  fireEvent.click(screen.getByRole('button', { name: 'Return to Saved Pose' }));
  await waitFor(() => expect(command).toHaveBeenCalledWith(2, 'test', 5));
  expect(window.confirm).not.toHaveBeenCalled();
});

test('edits duration and synchronizes from backend topics, not service responses', async () => {
  const { store, command } = await setup();
  const input = screen.getByRole('spinbutton', { name: 'Return duration (seconds)' });
  expect(input).toHaveValue(5);
  fireEvent.focus(input);
  fireEvent.change(input, { target: { value: '8.5' } });
  fireEvent.blur(input);
  await waitFor(() => expect(command).toHaveBeenCalledWith(4, 'test', 8.5));
  expect(store.getState().tasks.robotPoseStatus.duration_s).toBe(5);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Return to Saved Pose' })).toBeEnabled());
  expect(toast.success).not.toHaveBeenCalled();
  act(() => store.dispatch(setRobotPoseStatus({ duration_s: 8.5 })));
  fireEvent.click(screen.getByRole('button', { name: 'Return to Saved Pose' }));
  await waitFor(() => expect(command).toHaveBeenCalledWith(2, 'test', 8.5));
  act(() => store.dispatch(setRobotPoseStatus({ returning: true })));
  expect(input).toBeDisabled();
});

test('restores backend duration on mount and follows other clients', async () => {
  const { store } = await setup(InferencePhase.READY, { status: { duration_s: 12 } });
  const input = screen.getByRole('spinbutton', { name: 'Return duration (seconds)' });
  expect(input).toHaveValue(12);
  act(() => store.dispatch(setRobotPoseStatus({ duration_s: 9 })));
  expect(input).toHaveValue(9);
});

test('return accepts a valid draft and saves it before motion without a topic echo', async () => {
  let finishSave;
  const command = jest.fn((code) => code === 4
    ? new Promise((resolve) => { finishSave = resolve; }) : Promise.resolve({ success: true }));
  const { store } = await setup(InferencePhase.READY, { command });
  const input = screen.getByRole('spinbutton', { name: 'Return duration (seconds)' });
  const button = screen.getByRole('button', { name: 'Return to Saved Pose' });
  fireEvent.focus(input);
  fireEvent.change(input, { target: { value: '9' } });
  expect(button).toBeEnabled();
  fireEvent.blur(input, { relatedTarget: button });
  expect(command).not.toHaveBeenCalledWith(4, 'test', 9);
  fireEvent.click(button);
  expect(command).toHaveBeenCalledWith(4, 'test', 9);
  expect(command).not.toHaveBeenCalledWith(2, 'test', 9);
  await act(async () => { finishSave({ success: true, message: 'Return duration updated.' }); });
  expect(command).toHaveBeenLastCalledWith(2, 'test', 9);
  expect(store.getState().tasks.robotPoseStatus.duration_s).toBe(5);
  expect(toast.success).not.toHaveBeenCalledWith('Return duration updated.');
});

test.each(['failed', 'disconnected', 'robot_changed'])(
  'does not return after duration save is %s', async (outcome) => {
    let finishSave;
    const command = jest.fn((code) => code === 4
      ? new Promise((resolve) => { finishSave = resolve; }) : Promise.resolve({ success: true }));
    const { store } = await setup(InferencePhase.READY, { command });
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Return duration (seconds)' }), {
      target: { value: '9' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Return to Saved Pose' }));
    if (outcome === 'robot_changed') act(() => store.dispatch(selectRobotType('other')));
    await act(async () => {
      finishSave(outcome === 'disconnected' ? null : { success: outcome !== 'failed', message: 'save result' });
    });
    expect(command.mock.calls.some(([code]) => code === 2)).toBe(false);
  }
);

test.each(['', '0', '61'])('rejects invalid duration %s without sending a setting command', async (value) => {
  const { command } = await setup();
  const input = screen.getByRole('spinbutton', { name: 'Return duration (seconds)' });
  fireEvent.change(input, { target: { value } });
  fireEvent.blur(input);
  expect(screen.getByText('Return duration must be between 1 and 60 seconds.')).toBeInTheDocument();
  expect(command).toHaveBeenCalledTimes(1);
  expect(input).toHaveValue(5);
});

test('running inference prevents save and return', async () => {
  await setup(InferencePhase.INFERENCING);
  expect(screen.getByRole('button', { name: 'Save Initial Pose' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Return to Saved Pose' })).toBeDisabled();
});

test('stale status prevents motion but keeps stop available', async () => {
  const { store, command } = await setup();
  act(() => store.dispatch(setRobotPoseStatus({ returning: true, available: false, connected: false })));
  expect(screen.queryByRole('button', { name: 'Return to Saved Pose' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Stop pose return' }));
  await waitFor(() => expect(command).toHaveBeenCalledWith(3, 'test'));
});

test('uses the same button for return and stop, changing back only on backend status', async () => {
  const { store, command } = await setup();
  const button = screen.getByRole('button', { name: 'Return to Saved Pose' });
  const row = button.parentElement;
  act(() => store.dispatch(setRobotPoseStatus({ returning: true })));
  expect(screen.getByRole('button', { name: 'Stop pose return' })).toBe(button);
  expect(button.parentElement).toBe(row);
  expect(button).toHaveTextContent('Stop');
  expect(screen.getAllByRole('button')).toHaveLength(2);
  fireEvent.click(button);
  await waitFor(() => expect(command).toHaveBeenCalledWith(3, 'test'));
  await waitFor(() => expect(button).toBeEnabled());
  // A service acknowledgement cannot claim that the return has finished.
  expect(button).toHaveTextContent('Stop');
  act(() => store.dispatch(setRobotPoseStatus({ returning: false })));
  expect(screen.getByRole('button', { name: 'Return to Saved Pose' })).toBe(button);
  expect(button).toBeEnabled();
});

test('failed stop keeps the shared button available for retry', async () => {
  const command = jest.fn((code) => Promise.resolve(code === 3
    ? { success: false, message: 'Hold failed' } : { success: true }));
  await setup(InferencePhase.READY, { command, status: { returning: true } });
  const button = screen.getByRole('button', { name: 'Stop pose return' });
  fireEvent.click(button);
  await screen.findByText('Hold failed');
  expect(button).toBeEnabled();
  expect(button).toHaveTextContent('Stop');
  expect(screen.queryByRole('button', { name: 'Return to Saved Pose' })).not.toBeInTheDocument();
});

test.each([
  [0, null], [1, 'Save Initial Pose'], [2, 'Return to Saved Pose'], [3, 'Stop pose return'],
])('late command %i response cannot overwrite newer topic state', async (code, label) => {
  let resolve;
  const delayed = new Promise((done) => { resolve = done; });
  const command = jest.fn((value) => value === code ? delayed : Promise.resolve({ success: true }));
  const { store } = await setup(InferencePhase.READY, { command, status: { returning: code === 3 } });
  if (label) fireEvent.click(screen.getByRole('button', { name: label }));
  const newer = { ...status, returning: true, saved: true, positions: [0.9] };
  act(() => store.dispatch(setRobotPoseStatus(newer)));
  await act(async () => { resolve({ ...status, success: true, saved: false, returning: false }); });
  expect(store.getState().tasks.robotPoseStatus).toEqual(newer);
});

test.each(['robot', 'runtime', 'target', 'unmount'])(
  'ignores a command result after %s changes', async (change) => {
    let resolve;
    const command = jest.fn((code) => code === 1
      ? new Promise((done) => { resolve = done; }) : Promise.resolve({ success: true }));
    const { store, unmount } = await setup(InferencePhase.READY, { command });
    fireEvent.click(screen.getByRole('button', { name: 'Save Initial Pose' }));
    if (change === 'robot') act(() => store.dispatch(selectRobotType('other')));
    if (change === 'runtime') act(() => store.dispatch(setRobotPoseStatus({ device_id: 'new-runtime' })));
    if (change === 'target') act(() => store.dispatch(setRosbridgeUrl('ws://second')));
    if (change === 'unmount') unmount();
    await act(async () => { resolve({ success: false, message: 'Old request failed' }); });
    expect(toast.error).not.toHaveBeenCalled();
    expect(screen.queryByText('Old request failed')).not.toBeInTheDocument();
  }
);

test('a disconnected command releases pending without changing pose status', async () => {
  const command = jest.fn((code) => Promise.resolve(code === 1 ? null : { success: true }));
  const { store } = await setup(InferencePhase.READY, { command });
  fireEvent.click(screen.getByRole('button', { name: 'Save Initial Pose' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save Initial Pose' })).toBeEnabled());
  expect(store.getState().tasks.robotPoseStatus).toEqual(status);
  expect(toast.success).not.toHaveBeenCalled();
  expect(toast.error).not.toHaveBeenCalled();
});

test('old request completion does not clear a newer pending request', async () => {
  const replies = [];
  const command = jest.fn((code) => code === 1
    ? new Promise((done) => { replies.push(done); }) : Promise.resolve({ success: true }));
  const { store } = await setup(InferencePhase.READY, { command });
  fireEvent.click(screen.getByRole('button', { name: 'Save Initial Pose' }));
  await act(async () => { store.dispatch(setRobotPoseStatus({ device_id: 'new-runtime' })); });
  fireEvent.click(screen.getByRole('button', { name: 'Save Initial Pose' }));
  await act(async () => { replies[0]({ success: true }); });
  expect(screen.getByRole('button', { name: 'Save Initial Pose' })).toBeDisabled();
  await act(async () => { replies[1]({ success: true }); });
  expect(screen.getByRole('button', { name: 'Save Initial Pose' })).toBeEnabled();
});
