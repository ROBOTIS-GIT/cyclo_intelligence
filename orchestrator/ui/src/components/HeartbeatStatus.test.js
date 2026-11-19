import { act, render, screen } from '@testing-library/react';
import { configureStore } from '@reduxjs/toolkit';
import { Provider } from 'react-redux';
import taskReducer, { setHeartbeatStatus, setLastHeartbeatTime } from '../features/tasks/taskSlice';
import HeartbeatStatus from './HeartbeatStatus';

function setup(props = {}) {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({ reducer: { tasks: taskReducer }, preloadedState: { tasks: {
    ...initial, heartbeatStatus: 'connected', lastHeartbeatTime: Date.now(),
  } } });
  const view = render(<Provider store={store}><HeartbeatStatus {...props} /></Provider>);
  return { store, ...view };
}

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

test('connected banner hides despite ongoing heartbeat updates; warnings and reconnection remain visible', () => {
  const { store } = setup({ autoHideConnectedMs: 3000 });
  expect(screen.getByText('Connected')).toBeVisible();
  for (let i = 0; i < 4; i++) {
    act(() => { jest.advanceTimersByTime(1000); store.dispatch(setLastHeartbeatTime(Date.now())); });
  }
  expect(screen.queryByText('Connected')).not.toBeInTheDocument();
  act(() => jest.advanceTimersByTime(3000));
  expect(screen.getByText('Timeout')).toBeVisible();
  act(() => jest.advanceTimersByTime(10000));
  expect(screen.getByText('Disconnected')).toBeVisible();
  act(() => {
    store.dispatch(setLastHeartbeatTime(Date.now()));
    store.dispatch(setHeartbeatStatus('connected'));
  });
  expect(screen.getByText('Connected')).toBeVisible();
});

test('default callers retain the persistent connected indicator', () => {
  const { store } = setup();
  for (let i = 0; i < 5; i++) {
    act(() => { jest.advanceTimersByTime(1000); store.dispatch(setLastHeartbeatTime(Date.now())); });
  }
  expect(screen.getByText('Connected')).toBeVisible();
});

test('unmount clears connection and monitor timers', () => {
  const { unmount } = setup({ autoHideConnectedMs: 3000 });
  unmount();
  expect(jest.getTimerCount()).toBe(0);
});
