import React, { useState } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import ROSLIB from 'roslib';
import InferencePage from './InferencePage';
import taskReducer from '../features/tasks/taskSlice';
import uiReducer from '../features/ui/uiSlice';

let mockRos;
const mockRefresh = jest.fn().mockResolvedValue({ success: true });
jest.mock('../utils/rosConnectionManager', () => ({
  url: 'ws://test', getCurrentConnection: () => mockRos?.isConnected ? mockRos : null,
}));
jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: () => ({ sendRecordCommand: mockRefresh }),
}));
jest.mock('../hooks/usePolicyBackendStatus', () => ({
  __esModule: true, default: () => ({ readiness: { ready: false, state: 'stopped' } }),
}));
jest.mock('../components/InferencePanel', () => () => <div>Model settings</div>);
jest.mock('../components/InferenceGripperPanel', () => () => null);
jest.mock('../components/InferencePosePanel', () => () => null);
jest.mock('../components/InferenceControlPanel', () => () => null);
jest.mock('../components/HeartbeatStatus', () => () => null);
jest.mock('../components/InlineSystemStatus', () => () => null);
jest.mock('../components/ImageGrid', () => () => null);
jest.mock('../components/RobotViewer3D', () => () => null);
jest.mock('../components/RecordTopicMonitor', () => () => null);

function Navigation() {
  const [inference, setInference] = useState(false);
  return <>
    <button onClick={() => setInference(false)}>Home</button>
    <button onClick={() => setInference(true)}>Inference</button>
    {inference ? <InferencePage /> : <div>Home page</div>}
  </>;
}

beforeEach(() => {
  jest.useFakeTimers();
  mockRefresh.mockResolvedValue({ success: true });
  // Keep real roslib Topic/Ros methods; only replace the network transport.
  mockRos = new ROSLIB.Ros({ encoding: 'ascii' });
  mockRos.socket = { send: jest.fn() };
});
afterEach(() => jest.useRealTimers());

test.each([true, false])('inference navigation works when ROS is already connected: %s', (connected) => {
  mockRos.isConnected = connected;
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer, ui: uiReducer, ros: () => ({ rosbridgeUrl: 'ws://test' }) },
    preloadedState: { tasks: { ...initialTasks, robotType: 'ffw_sg2_rev1' } },
  });
  const view = render(<Provider store={store}><Navigation /></Provider>);
  fireEvent.click(screen.getByRole('button', { name: 'Inference' }));
  expect(screen.getByText('Model settings')).toBeVisible();
  if (!connected) {
    act(() => { mockRos.isConnected = true; jest.advanceTimersByTime(100); });
  }
  const sent = () => mockRos.socket.send.mock.calls.map(([message]) => JSON.parse(message));
  expect(sent().filter((m) => m.op === 'subscribe' && m.topic === '/dagger/status')).toHaveLength(1);
  fireEvent.click(screen.getByRole('button', { name: 'Home' }));
  expect(mockRos.listeners('/dagger/status')).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: 'Inference' }));
  expect(screen.getByRole('tab', { name: 'Settings' })).toBeVisible();
  expect(mockRos.listeners('/dagger/status')).toHaveLength(1);
  expect(sent().filter((m) => m.op === 'publish')).toHaveLength(0);
  view.unmount();
  expect(mockRos.listeners('/dagger/status')).toHaveLength(0);
});
