// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { act, renderHook, waitFor } from '@testing-library/react';
import useBtRuntimeSubscription from './useBtRuntimeSubscription';
import { setBtStatus, setActiveNodeNames } from '../actionCanvasSlice';

const mockGetConnection = jest.fn();
const mockTopics = [];
const mockTopic = jest.fn();
function createMockTopic(options) {
  const topic = {
    ...options,
    callback: null,
    subscribe: jest.fn((callback) => { topic.callback = callback; }),
    unsubscribe: jest.fn(),
  };
  mockTopics.push(topic);
  return topic;
}

jest.mock('roslib', () => ({
  __esModule: true,
  default: { Topic: function Topic(...args) { return mockTopic(...args); } },
}));

jest.mock('../../../utils/rosConnectionManager', () => ({
  __esModule: true,
  default: { getConnection: (...args) => mockGetConnection(...args) },
}));

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function mountSubscription(dispatch = jest.fn()) {
  const view = renderHook(({ isActive }) => useBtRuntimeSubscription({
    rosbridgeUrl: 'ws://robot:9090', isActive, dispatch,
  }), { initialProps: { isActive: true } });
  return { ...view, dispatch };
}

beforeEach(() => {
  mockGetConnection.mockReset();
  mockTopic.mockReset();
  mockTopic.mockImplementation(createMockTopic);
  mockTopics.length = 0;
});

test('does not continue async setup after immediate unmount', async () => {
  const view = mountSubscription();
  view.unmount();
  await import('roslib');
  expect(mockGetConnection).not.toHaveBeenCalled();
  expect(mockTopic).not.toHaveBeenCalled();
});

test.each(['unmount', 'inactive'])(
  'does not subscribe when a pending connection resolves after %s',
  async (transition) => {
    const connection = deferred();
    const ros = { close: jest.fn() };
    mockGetConnection.mockReturnValue(connection.promise);
    const view = mountSubscription();
    await waitFor(() => expect(mockGetConnection).toHaveBeenCalledTimes(1));
    if (transition === 'unmount') view.unmount();
    else view.rerender({ isActive: false });
    await act(async () => { connection.resolve(ros); });
    expect(mockTopic).not.toHaveBeenCalled();
    expect(view.dispatch).not.toHaveBeenCalled();
    expect(ros.close).not.toHaveBeenCalled();
  },
);

test('unsubscribes both topics and ignores callbacks queued during or after cleanup', async () => {
  const ros = { close: jest.fn() };
  mockGetConnection.mockResolvedValue(ros);
  const view = mountSubscription();
  await waitFor(() => expect(mockTopic).toHaveBeenCalledTimes(2));
  const status = mockTopics.find((topic) => topic.name === '/bt/status');
  const nodes = mockTopics.find((topic) => topic.name === '/bt/active_nodes');
  act(() => {
    status.callback({ data: 'running' });
    nodes.callback({ data: 'task-a,task-b' });
    status.callback({ data: 'completed' });
  });
  expect(view.dispatch.mock.calls.map(([action]) => action)).toEqual([
    setBtStatus('running'),
    setActiveNodeNames(['task-a', 'task-b']),
    setBtStatus('completed'),
    setActiveNodeNames([]),
  ]);
  view.dispatch.mockClear();
  status.unsubscribe.mockImplementation(() => status.callback({ data: 'failed' }));
  view.unmount();
  act(() => {
    status.callback({ data: 'running' });
    nodes.callback({ data: 'stale-task' });
  });
  expect(status.unsubscribe).toHaveBeenCalledTimes(1);
  expect(nodes.unsubscribe).toHaveBeenCalledTimes(1);
  expect(view.dispatch).not.toHaveBeenCalled();
  expect(ros.close).not.toHaveBeenCalled();
});

test('a retired pending setup cannot duplicate the new active subscription', async () => {
  const oldConnection = deferred();
  const currentConnection = deferred();
  mockGetConnection.mockReturnValueOnce(oldConnection.promise)
    .mockReturnValueOnce(currentConnection.promise);
  const view = mountSubscription();
  await waitFor(() => expect(mockGetConnection).toHaveBeenCalledTimes(1));
  view.rerender({ isActive: false });
  view.rerender({ isActive: true });
  await waitFor(() => expect(mockGetConnection).toHaveBeenCalledTimes(2));
  await act(async () => { currentConnection.resolve({}); });
  expect(mockTopic).toHaveBeenCalledTimes(2);
  await act(async () => { oldConnection.resolve({}); });
  expect(mockTopic).toHaveBeenCalledTimes(2);
  expect(view.dispatch).not.toHaveBeenCalled();
});
