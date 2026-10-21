import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferenceGripperPanel from './InferenceGripperPanel';
import { GripperControlClient } from '../utils/gripperControl';
import taskReducer, { setInferenceStatus, setRecordStatus } from '../features/tasks/taskSlice';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';

const mockRos = { isConnected: true };
jest.mock('../utils/rosConnectionManager', () => ({
  url: 'ws://test', getCurrentConnection: () => mockRos,
}));
jest.mock('../utils/gripperControl', () => ({
  GRIPPER_OPEN: 0, GRIPPER_CLOSED: 1.1, GripperControlClient: jest.fn(),
}));
let client;
beforeEach(() => {
  jest.useFakeTimers();
  client = {
    ros: mockRos, enabled: false, busy: false, tick: jest.fn(), dispose: jest.fn(),
    setEnabled(value) { this.enabled = value; },
    snapshot() { return { positions: { left: 0, right: .5 }, reasons: { left: '', right: '' },
      busy: this.busy, moving: this.busy ? { side: 'right', target: 1.1 } : null }; },
    command: jest.fn(function () { if (!this.enabled) return false; this.busy = true; return true; }),
  };
  GripperControlClient.mockImplementation(() => client);
});
afterEach(() => { jest.clearAllMocks(); jest.useRealTimers(); });
const setup = (phase = InferencePhase.READY, robotType = 'ffw_sg2_rev1', externalBusy = false) => {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({ reducer: { tasks: taskReducer, ros: (state = {}) => state }, preloadedState: {
    tasks: { ...initial, robotType, inferenceStatus: { ...initial.inferenceStatus, inferencePhase: phase } },
    ros: { rosbridgeUrl: 'ws://test' },
  } });
  const onBusyChange = jest.fn();
  const result = render(<Provider store={store}><InferenceGripperPanel externalBusy={externalBusy} onBusyChange={onBusyChange} /></Provider>);
  return { ...result, store, onBusyChange };
};

test('shows live gripper positions and closes the right gripper to 1.1 without a leader', () => {
  const { onBusyChange } = setup();
  expect(screen.getByLabelText('Right gripper position')).toHaveTextContent('0.500 rad');
  fireEvent.click(screen.getByRole('button', { name: 'Close right gripper' }));
  expect(client.command).toHaveBeenCalledWith('right', 1.1);
  expect(onBusyChange).toHaveBeenLastCalledWith(true);
  expect(screen.getByText(/Closing right gripper/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Open left gripper' })).toBeDisabled();
});

test('an ongoing pose movement disables gripper commands', () => {
  setup(InferencePhase.READY, 'ffw_sg2_rev1', true);
  expect(screen.getByRole('button', { name:'Close right gripper' })).toBeDisabled();
  expect(screen.getByText(/Wait for pose movement/)).toBeInTheDocument();
  expect(client.command).not.toHaveBeenCalled();
});

test('left open commands 0.0 and gripper-button keys do not trigger the global Start shortcut', () => {
  setup();
  const shortcut = jest.fn();
  window.addEventListener('keyup', shortcut);
  const button = screen.getByRole('button', { name: 'Open left gripper' });
  fireEvent.keyUp(button, { key: ' ', code: 'Space' });
  expect(shortcut).not.toHaveBeenCalled();
  window.removeEventListener('keyup', shortcut);
  fireEvent.click(button);
  expect(client.command).toHaveBeenCalledWith('left', 0);
});

test.each([InferencePhase.LOADING, InferencePhase.INFERENCING, InferencePhase.PAUSED, InferencePhase.SYNCING])('locks manual commands in inference phase %s', (phase) => {
  setup(phase);
  expect(screen.getByRole('button', { name: 'Close right gripper' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Close right gripper' }));
  expect(client.command).not.toHaveBeenCalled();
});

test('a lifecycle transition immediately disables buttons, and recording also locks controls', () => {
  const { store } = setup();
  act(() => store.dispatch(setInferenceStatus({ inferencePhase: InferencePhase.LOADING })));
  expect(screen.getByRole('button', { name: 'Close right gripper' })).toBeDisabled();
  act(() => store.dispatch(setInferenceStatus({ inferencePhase: InferencePhase.READY })));
  act(() => store.dispatch(setRecordStatus({ recordPhase: RecordPhase.RECORDING })));
  expect(screen.getByRole('button', { name: 'Close right gripper' })).toBeDisabled();
  expect(screen.getByText(/Finish recording/)).toBeInTheDocument();
});

test('unsupported robots cannot send SG2 gripper commands', () => {
  setup(InferencePhase.READY, 'omx');
  expect(screen.getByRole('button', { name: 'Close right gripper' })).toBeDisabled();
});
