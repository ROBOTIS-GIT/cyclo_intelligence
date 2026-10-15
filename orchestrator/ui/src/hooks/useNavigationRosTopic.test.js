import React from 'react';
import { configureStore } from '@reduxjs/toolkit';
import { act, render, renderHook, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import ROSLIB from 'roslib';
import rosConnectionManager from '../utils/rosConnectionManager';
import {
  applyNavigationGridEnvelope,
  navigationGridWebSocketUrl,
  useNavigationRosTopic,
  wrapNavigationRosMessage,
} from './useNavigationRosTopic';

test('merges a costmap dirty rectangle into the cached full grid', () => {
  const previous = {
    available: true,
    data: {
      header: { frame_id: 'map', stamp: { sec: 1, nanosec: 0 } },
      info: { width: 4, height: 3, resolution: 0.05 },
      data: [
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
      ],
    },
  };
  const incoming = {
    available: true,
    update: {
      header: { frame_id: 'map', stamp: { sec: 2, nanosec: 0 } },
      x: 1,
      y: 1,
      width: 2,
      height: 2,
      data: [50, 60, 90, 100],
    },
  };

  expect(applyNavigationGridEnvelope(previous, incoming)).toEqual({
    available: true,
    data: {
      ...previous.data,
      header: incoming.update.header,
      data: [
        0, 1, 2, 3,
        4, 50, 60, 7,
        8, 90, 100, 11,
      ],
      updateRegion: { x: 1, y: 1, width: 2, height: 2 },
    },
  });
  expect(previous.data.data).toEqual([
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
  ]);
});

test('ignores a costmap update until a valid full grid is available', () => {
  const incoming = {
    available: true,
    update: { x: 0, y: 0, width: 1, height: 1, data: [100] },
  };

  expect(applyNavigationGridEnvelope(null, incoming)).toBeNull();
});

test('wraps OccupancyGrid without losing its data and metadata fields', () => {
  const map = {
    header: { frame_id: 'map' },
    info: { width: 2, height: 1, resolution: 0.05 },
    data: [0, 100],
  };

  expect(wrapNavigationRosMessage(map)).toEqual({
    available: true,
    data: map,
  });
});

test('builds a same-origin supervisor WebSocket URL for a grid topic', () => {
  expect(navigationGridWebSocketUrl('/global_costmap/costmap', {
    protocol: 'https:',
    host: 'robot.local:8443',
  })).toBe(
    'wss://robot.local:8443/api/navigation/topics/ws?topic=%2Fglobal_costmap%2Fcostmap'
  );
});

test('opens a server grid WebSocket without a rosbridge URL', () => {
  const originalWebSocket = global.WebSocket;
  const close = jest.fn();
  const WebSocketMock = jest.fn(() => ({ close }));
  global.WebSocket = WebSocketMock;
  const store = configureStore({
    reducer: () => ({ ros: { rosbridgeUrl: '' } }),
  });
  function GridSubscriber() {
    useNavigationRosTopic('/map');
    return null;
  }

  const view = render(
    <Provider store={store}>
      <GridSubscriber />
    </Provider>
  );

  expect(WebSocketMock).toHaveBeenCalledWith(
    navigationGridWebSocketUrl('/map')
  );
  view.unmount();
  expect(close).toHaveBeenCalledTimes(1);
  global.WebSocket = originalWebSocket;
});


test('delivers each BT status event before React batches topic state', async () => {
  const store = configureStore({
    reducer: () => ({ ros: { rosbridgeUrl: 'ws://localhost:7090' } }),
  });
  const onMessage = jest.fn();
  let receive;
  const connection = jest.spyOn(rosConnectionManager, 'getConnection').mockResolvedValue({ isConnected: true });
  const topic = jest.spyOn(ROSLIB, 'Topic').mockImplementation(function MockTopic() {
    this.subscribe = callback => { receive = callback; };
    this.unsubscribe = jest.fn();
  });
  try {
    const view = renderHook(() => useNavigationRosTopic('/bt/status', { onMessage }), {
      wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
    });
    await waitFor(() => expect(receive).toEqual(expect.any(Function)));
    act(() => {
      receive({ data: 'running' });
      receive({ data: 'completed' });
    });
    expect(onMessage.mock.calls.map(([message]) => message.data)).toEqual(['running', 'completed']);
    expect(view.result.current.topicData.data.data).toBe('completed');
    view.unmount();
  } finally {
    topic.mockRestore();
    connection.mockRestore();
  }
});

 test('restores active BT node updates after rosbridge disconnects', async () => {
  jest.useFakeTimers();
  const { EventEmitter } = require('events');
  const first = new EventEmitter();
  const second = new EventEmitter();
  const connections = jest.spyOn(rosConnectionManager, 'getConnection')
    .mockResolvedValueOnce(first).mockResolvedValue(second);
  const subscriptions = [];
  const topic = jest.spyOn(ROSLIB, 'Topic').mockImplementation(function MockTopic() {
    this.subscribe = callback => { this.receive = callback; };
    this.unsubscribe = jest.fn();
    subscriptions.push(this);
  });
  const store = configureStore({ reducer: () => ({ ros: { rosbridgeUrl: 'ws://localhost:7090' } }) });
  let view;
  try {
    view = renderHook(() => useNavigationRosTopic('/bt/active_nodes'), {
      wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
    });
    await act(async () => {});
    act(() => subscriptions[0].receive({ data: 'bt_1' }));
    expect(view.result.current.topicData.data.data).toBe('bt_1');
    act(() => first.emit('close'));
    expect(view.result.current.topicData).toBeNull();
    await act(async () => { jest.advanceTimersByTime(1000); });
    expect(subscriptions).toHaveLength(2);
    expect(subscriptions[0].unsubscribe).toHaveBeenCalledTimes(1);
    expect(first.listenerCount('close')).toBe(0);
    act(() => subscriptions[1].receive({ data: 'bt_2' }));
    expect(view.result.current.topicData.data.data).toBe('bt_2');
    act(() => second.emit('close'));
    view.unmount();
    await act(async () => { jest.advanceTimersByTime(2000); });
    expect(connections).toHaveBeenCalledTimes(2);
  } finally {
    view?.unmount(); connections.mockRestore(); topic.mockRestore(); jest.useRealTimers();
  }
});

 test.each([
  ['/amcl_pose', { pose: { pose: { position: { x: 1, y: 2 } } } }],
  ['/bt/active_nodes', { data: 'bt_1' }],
])('sends a real ROSLIB subscription and receives %s', async (name, message) => {
  const ros = new ROSLIB.Ros();
  ros.isConnected = true;
  ros.socket = { send: jest.fn() };
  const connection = jest.spyOn(rosConnectionManager, 'getConnection').mockResolvedValue(ros);
  const store = configureStore({ reducer: () => ({ ros: { rosbridgeUrl: 'ws://localhost:7090' } }) });
  let view;
  try {
    view = renderHook(() => useNavigationRosTopic(name), {
      wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
    });
    await waitFor(() => expect(ros.socket.send).toHaveBeenCalled());
    expect(JSON.parse(ros.socket.send.mock.calls[0][0])).toMatchObject({ op: 'subscribe', topic: name });
    act(() => ros.emit(name, message));
    expect(view.result.current.topicData.data).toEqual(message);
    expect(view.result.current.status).toBe('connected');
    view.unmount();
    expect(JSON.parse(ros.socket.send.mock.calls.at(-1)[0])).toMatchObject({ op: 'unsubscribe', topic: name });
  } finally { view?.unmount(); connection.mockRestore(); }
});
