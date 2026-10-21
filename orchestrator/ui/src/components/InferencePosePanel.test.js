import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferencePosePanel from './InferencePosePanel';
import { InferencePoseClient, POSE_GROUPS } from '../utils/inferencePoseControl';
import taskReducer, { setInferenceStatus, setRecordStatus } from '../features/tasks/taskSlice';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';

let mockRos;
jest.mock('../utils/rosConnectionManager', () => ({ url: 'ws://test', getCurrentConnection: () => mockRos }));
jest.mock('../utils/inferencePoseControl', () => ({
  ...jest.requireActual('../utils/inferencePoseControl'), InferencePoseClient: jest.fn(),
}));
let client;
const makeClient = (ros) => ({
  ros, enabled: false, moving: null, tick: jest.fn(), dispose: jest.fn(), setEnabled(value) { this.enabled = value; },
  snapshot() { return { positions: { left: Array(8).fill(0) }, reasons: {}, failures: {}, holding: {},
    busy: !!this.moving, moving: this.moving }; },
  command: jest.fn(function(group, poseIndex, hold) {
    if (!this.enabled) return false;
    this.moving = { group, poseIndex, hold, target: POSE_GROUPS[group].poses[poseIndex], duration: 5, until: 5000 };
    return true;
  }),
});
beforeEach(() => {
  jest.useFakeTimers(); mockRos = { isConnected: true }; client = makeClient(mockRos);
  InferencePoseClient.mockImplementation(() => client);
});
afterEach(() => { jest.clearAllMocks(); jest.useRealTimers(); });
const setup = ({ phase = InferencePhase.READY, robotType = 'ffw_sg2_rev1', isActive = true, externalBusy = false } = {}) => {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({ reducer: { tasks: taskReducer, ros: (state = {}) => state }, preloadedState: {
    tasks: { ...initial, robotType, inferenceStatus: { ...initial.inferenceStatus, inferencePhase: phase } }, ros: { rosbridgeUrl: 'ws://test' },
  } });
  const onBusyChange = jest.fn();
  const result = render(<Provider store={store}><InferencePosePanel isActive={isActive} externalBusy={externalBusy} onBusyChange={onBusyChange} /></Provider>);
  return { ...result, store, onBusyChange };
};

test('shows all presets and moves only the selected left pose with its precise stored values', () => {
  const { onBusyChange } = setup();
  expect(screen.getByLabelText('Left arm preset').options).toHaveLength(4);
  expect(screen.getByLabelText('Right arm preset').options).toHaveLength(2);
  expect(screen.getByLabelText('Head preset').options).toHaveLength(2);
  expect(screen.getByLabelText('Lift preset').options).toHaveLength(1);
  fireEvent.change(screen.getByLabelText('Left arm preset'), { target: { value:'3' } });
  fireEvent.click(screen.getByRole('button', { name:'Move left to selected pose' }));
  expect(client.command).toHaveBeenCalledWith('left', 3, false);
  expect(onBusyChange).toHaveBeenLastCalledWith(true);
  expect(screen.getByRole('button', { name:'Move right to selected pose' })).toBeDisabled();
  expect(screen.getByRole('button', { name:'Hold current left pose' })).toBeEnabled();
  expect(screen.getByRole('button', { name:'Hold current right pose' })).toBeDisabled();
});

test.each(['head','lift'])('provides both preset movement and Hold current for %s', (group) => {
  setup();
  fireEvent.click(screen.getByRole('button', { name:`Move ${group} to selected pose` }));
  expect(client.command).toHaveBeenLastCalledWith(group, 0, false);
  fireEvent.click(screen.getByRole('button', { name:`Hold current ${group} pose` }));
  expect(client.command).toHaveBeenLastCalledWith(group, 0, true);
});

test.each([InferencePhase.LOADING, InferencePhase.SYNCING, InferencePhase.INFERENCING, InferencePhase.PAUSED])('prevents pose commands in inference phase %s', (phase) => {
  setup({ phase });
  const button = screen.getByRole('button', { name:'Move left to selected pose' });
  expect(button).toBeDisabled(); fireEvent.click(button); expect(client.command).not.toHaveBeenCalled();
});

test.each([{ isActive:false }, { externalBusy:true }, { robotType:'omx' }])('blocks inactive, competing or unsupported controls: %p', (options) => {
  setup(options); expect(screen.getByRole('button', { name:'Move right to selected pose' })).toBeDisabled();
});

test('phase and recording changes disable commands immediately without waiting for polling', () => {
  const { store } = setup();
  act(() => store.dispatch(setInferenceStatus({ inferencePhase:InferencePhase.LOADING })));
  expect(screen.getByRole('button', { name:'Move left to selected pose' })).toBeDisabled();
  act(() => store.dispatch(setInferenceStatus({ inferencePhase:InferencePhase.READY })));
  act(() => store.dispatch(setRecordStatus({ recordPhase:RecordPhase.RECORDING })));
  expect(screen.getByRole('button', { name:'Move left to selected pose' })).toBeDisabled();
  expect(screen.getAllByText(/Finish recording/).length).toBeGreaterThan(0);
});

test('Space on pose controls cannot trigger the global inference Start shortcut', () => {
  setup(); const shortcut = jest.fn(); window.addEventListener('keyup', shortcut);
  fireEvent.keyUp(screen.getByRole('button', { name:'Move right to selected pose' }), { key:' ', code:'Space' });
  expect(shortcut).not.toHaveBeenCalled(); window.removeEventListener('keyup', shortcut);
});

test('keeps the movement lock across a disconnected and replaced ROS connection', () => {
  const { onBusyChange } = setup();
  fireEvent.click(screen.getByRole('button', { name:'Move left to selected pose' }));
  const oldClient = client; const pending = client.moving;
  mockRos = null; act(() => jest.advanceTimersByTime(100));
  expect(onBusyChange).toHaveBeenLastCalledWith(true);
  mockRos = { isConnected:true }; client = makeClient(mockRos);
  act(() => jest.advanceTimersByTime(100));
  expect(oldClient.dispose).toHaveBeenCalled();
  expect(client.moving).toEqual(pending);
  expect(onBusyChange).toHaveBeenLastCalledWith(true);
  expect(client.command).not.toHaveBeenCalled();
});
