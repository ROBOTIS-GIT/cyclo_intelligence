import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import toast from 'react-hot-toast';
import InferenceRecordingControls from './InferenceRecordingControls';
import taskReducer, { setInferenceStatus, setRecordStatus } from '../features/tasks/taskSlice';
import rosReducer from '../features/ros/rosSlice';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));
jest.mock('react-hot-toast', () => ({ __esModule: true, default: { error: jest.fn() } }));
jest.mock('./FileBrowserModal', () => ({ isOpen, onFileSelect }) => isOpen ? (
  <div>
    <button onClick={() => onFileSelect({ full_path: '/workspace/rosbag2/Task_previous_inference_MCAP' })}>Choose previous</button>
    <button onClick={() => onFileSelect({ full_path: '/workspace/rosbag2/Task_previous_inference_MCAP/0' })}>Choose episode</button>
  </div>
) : null);

function setup({ record = {}, inference = {}, command } = {}) {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: {
      tasks: taskReducer,
      ros: rosReducer,
    },
    preloadedState: {
      tasks: {
        ...initial,
        recordStatus: { ...initial.recordStatus, topicReceived: true, ...record },
        inferenceStatus: {
          ...initial.inferenceStatus,
          inferencePhase: InferencePhase.INFERENCING,
          runtimeState: 'running', publishToRobot: true, topicReceived: true, sourceId: 'runtime-1', ...inference,
        },
      },
    },
  });
  const sendRecordCommand = command || jest.fn().mockResolvedValue({ success: true });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const view = render(<Provider store={store}><InferenceRecordingControls /></Provider>);
  return { ...view, store, sendRecordCommand };
}

const activeRecording = { taskType: 'inference', recordPhase: RecordPhase.RECORDING, running: true };

beforeEach(() => jest.clearAllMocks());

test('folder selection while inference runs only changes the displayed path through the status topic', async () => {
  const { store, sendRecordCommand } = setup({ inference: {
    recordingSessionId: 'current',
  } });
  fireEvent.click(screen.getByRole('button', { name: 'Recording folder' }));
  expect(screen.getByText('/workspace/rosbag2/Task_current_inference_MCAP')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Use existing' }));
  fireEvent.click(screen.getByText('Choose episode'));
  expect(sendRecordCommand).not.toHaveBeenCalled();
  fireEvent.click(screen.getByText('Choose previous'));
  await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith('set_inference_record_folder', { recordingSessionId: 'previous' }));
  await waitFor(() => expect(screen.getByRole('group')).toHaveAttribute('aria-busy', 'false'));
  expect(store.getState().tasks.inferenceStatus.recordingSessionId).toBe('current');
  act(() => store.dispatch(setInferenceStatus({ recordingSessionId: 'previous' })));
  expect(screen.getByText('/workspace/rosbag2/Task_previous_inference_MCAP')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Use new' }));
  await waitFor(() => expect(sendRecordCommand).toHaveBeenLastCalledWith('set_inference_record_folder', { recordingSessionId: '' }));
  await waitFor(() => expect(screen.getByRole('group')).toHaveAttribute('aria-busy', 'false'));
  expect(store.getState().tasks.inferenceStatus.recordingSessionId).toBe('previous');
  expect(store.getState().tasks.inferenceStatus.runtimeState).toBe('running');
  expect(sendRecordCommand.mock.calls.map(([command]) => command)).toEqual([
    'set_inference_record_folder', 'set_inference_record_folder',
  ]);
  act(() => store.dispatch(setInferenceStatus({ recordingSessionId: '' })));
  expect(screen.getByText('New on next Record')).toBeInTheDocument();
});

test.each([
  { record: activeRecording },
  { record: activeRecording, inference: { runtimeState: 'paused', inferencePhase: InferencePhase.PAUSED } },
  { record: { running: true, recordPhase: RecordPhase.SAVING } },
  { record: { running: false, recordPhase: RecordPhase.SAVING } },
  { record: { topicReceived: false } },
  { inference: { topicReceived: false } },
])('folder changes are unavailable while busy or disconnected: %j', options => {
  setup(options);
  fireEvent.click(screen.getByRole('button', { name: 'Recording folder' }));
  expect(screen.getByRole('button', { name: 'Use existing' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Use new' })).toBeDisabled();
});

test('a late folder reply cannot replace a newer topic selection', async () => {
  let resolve;
  const command = jest.fn(() => new Promise(done => { resolve = done; }));
  const { store } = setup({ command, inference: { runtimeState: 'paused', inferencePhase: InferencePhase.PAUSED } });
  fireEvent.click(screen.getByRole('button', { name: 'Recording folder' }));
  fireEvent.click(screen.getByRole('button', { name: 'Use new' }));
  expect(screen.getByRole('button', { name: 'Use new' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Use existing' })).toBeDisabled();
  act(() => store.dispatch(setInferenceStatus({ recordingSessionId: 'from-another-ui' })));
  await act(async () => resolve({ success: true, task_num: '' }));
  expect(store.getState().tasks.inferenceStatus.recordingSessionId).toBe('from-another-ui');
});

test('Record uses the existing command, without changing the authoritative recording state', async () => {
  const { store, sendRecordCommand } = setup();
  // The legacy ROS Redux flag is not maintained by the connection manager.
  expect(store.getState().ros.connected).toBe(false);
  expect(screen.getByRole('button', { name: 'Start inference recording' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Start inference recording' }));
  await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith('start_inference_record'));
  await waitFor(() => expect(screen.getByRole('group')).toHaveAttribute('aria-busy', 'false'));
  expect(store.getState().tasks.recordStatus.recordPhase).toBe(RecordPhase.READY);
  expect(screen.queryByRole('button', { name: 'Save inference recording' })).not.toBeInTheDocument();
  act(() => store.dispatch(setRecordStatus(activeRecording)));
  expect(screen.getByRole('button', { name: 'Save inference recording' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Discard inference recording' })).toBeEnabled();
});

test.each([
  ['Save', 'stop_inference_record'],
  ['Discard', 'cancel_inference_record'],
])('%s only controls recording, including while inference is paused', async (label, command) => {
  const { store, sendRecordCommand } = setup({
    record: activeRecording,
    inference: { inferencePhase: InferencePhase.PAUSED, runtimeState: 'paused' },
  });
  fireEvent.click(screen.getByRole('button', { name: `${label} inference recording` }));
  await waitFor(() => expect(screen.getByRole('group')).toHaveAttribute('aria-busy', 'false'));
  expect(sendRecordCommand.mock.calls).toEqual([[command]]);
  expect(store.getState().tasks.inferenceStatus.runtimeState).toBe('paused');
  expect(store.getState().tasks.recordStatus.recordPhase).toBe(RecordPhase.RECORDING);
  act(() => store.dispatch(setRecordStatus({ recordPhase: RecordPhase.READY, running: false })));
  expect(screen.getByRole('button', { name: 'Start inference recording' })).toBeDisabled();
});

test.each([InferencePhase.READY, InferencePhase.LOADING, InferencePhase.PAUSED, InferencePhase.SYNCING])(
  'cannot start recording outside inference (phase %s)', (inferencePhase) => {
    setup({ inference: { inferencePhase } });
    expect(screen.getByRole('button', { name: 'Start inference recording' })).toBeDisabled();
  }
);

test.each([
  { record: { topicReceived: false } },
  { inference: { topicReceived: false } },
  { record: { recordPhase: RecordPhase.SAVING, running: true } },
  { record: { ...activeRecording, taskType: 'record' } },
])('does not control unavailable or another page recording: %j', (options) => {
  setup(options);
  expect(screen.getByRole('button', { name: 'Start inference recording' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Discard inference recording' })).not.toBeInTheDocument();
});

test('repeated clicks are blocked until the request finishes; failures allow retry', async () => {
  let resolve;
  const command = jest.fn(() => new Promise((done) => { resolve = done; }));
  setup({ record: activeRecording, command });
  fireEvent.click(screen.getByRole('button', { name: 'Save inference recording' }));
  fireEvent.click(screen.getByRole('button', { name: 'Save inference recording' }));
  expect(screen.getByRole('button', { name: 'Discard inference recording' })).toBeDisabled();
  expect(command).toHaveBeenCalledTimes(1);
  await act(async () => resolve({ success: false, message: 'Save failed' }));
  expect(toast.error).toHaveBeenCalledWith('Save failed');
  expect(screen.getByRole('button', { name: 'Save inference recording' })).toBeEnabled();
});

test('late service responses cannot undo a newer topic update', async () => {
  let resolve;
  const command = jest.fn(() => new Promise((done) => { resolve = done; }));
  const { store } = setup({ command });
  fireEvent.click(screen.getByRole('button', { name: 'Start inference recording' }));
  act(() => store.dispatch(setRecordStatus(activeRecording)));
  await act(async () => resolve({ success: true }));
  expect(screen.getByRole('button', { name: 'Save inference recording' })).toBeEnabled();
});

test.each(['disconnect', 'restart', 'unmount'])('ignores a stale response after %s', async (change) => {
  let resolve;
  const command = jest.fn(() => new Promise((done) => { resolve = done; }));
  const { store, unmount } = setup({ command });
  fireEvent.click(screen.getByRole('button', { name: 'Start inference recording' }));
  act(() => {
    if (change === 'disconnect') store.dispatch(setInferenceStatus({ topicReceived: false }));
    if (change === 'restart') store.dispatch(setInferenceStatus({ sourceId: 'runtime-2' }));
    if (change === 'unmount') unmount();
  });
  await act(async () => resolve({ success: false, message: 'Old failure' }));
  expect(toast.error).not.toHaveBeenCalled();
});

test('R has no recording shortcut', () => {
  const { sendRecordCommand } = setup();
  fireEvent.keyDown(window, { key: 'r' });
  fireEvent.keyUp(window, { key: 'r' });
  fireEvent.keyDown(window, { key: 'R' });
  fireEvent.keyUp(window, { key: 'R' });
  expect(sendRecordCommand).not.toHaveBeenCalled();
});
