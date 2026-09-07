import React from 'react';
import { configureStore } from '@reduxjs/toolkit';
import { render } from '@testing-library/react';
import { Provider } from 'react-redux';
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
