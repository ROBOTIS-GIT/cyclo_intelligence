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

import { act, fireEvent, render, screen } from '@testing-library/react';
import PoseCaptureButton from './PoseCaptureButton';
import ROSLIB from 'roslib';
import rosConnectionManager from '../../utils/rosConnectionManager';

let mockCallback;
const mockUnsubscribe = jest.fn();
const mockSubscribe = jest.fn((callback) => { mockCallback = callback; });
jest.mock('roslib', () => ({ Topic: jest.fn().mockImplementation(() => ({
  subscribe: mockSubscribe, unsubscribe: mockUnsubscribe,
})) }));
jest.mock('../../utils/rosConnectionManager', () => ({ getConnection: jest.fn() }));
jest.mock('react-redux', () => ({ useSelector: (selector) => selector({ ros: { rosbridgeUrl: 'ws://robot' } }) }));
const props = { nodeType: 'ArmStateGate', params: { left_target_joints: 'arm_l_joint1' } };
const message = { name: ['arm_l_joint1'], position: [0.75] };
beforeEach(() => {
  jest.clearAllMocks();
  jest.useFakeTimers();
  mockSubscribe.mockImplementation((callback) => { mockCallback = callback; });
  ROSLIB.Topic.mockImplementation(() => ({ subscribe: mockSubscribe, unsubscribe: mockUnsubscribe }));
  rosConnectionManager.getConnection.mockResolvedValue({ isConnected: true });
});
afterEach(() => jest.useRealTimers());
async function clickCapture() {
  fireEvent.click(screen.getByRole('button'));
  await act(async () => { await Promise.resolve(); });
}

test('only captures fresh feedback after clicking, then unsubscribes', async () => {
  const onCapture = jest.fn();
  render(<PoseCaptureButton {...props} onCapture={onCapture} />);
  expect(mockSubscribe).not.toHaveBeenCalled();
  await clickCapture();
  expect(onCapture).not.toHaveBeenCalled();
  act(() => mockCallback(message));
  expect(onCapture).toHaveBeenCalledWith({ left_target_positions: '0.75' });
  expect(mockUnsubscribe).toHaveBeenCalledTimes(1);
  act(() => mockCallback(message));
  expect(onCapture).toHaveBeenCalledTimes(1);
});

test('timeout and missing joints preserve existing parameters', async () => {
  const onCapture = jest.fn();
  render(<PoseCaptureButton {...props} onCapture={onCapture} />);
  await clickCapture();
  act(() => jest.advanceTimersByTime(3000));
  expect(screen.getByRole('alert')).toHaveTextContent('No fresh');
  act(() => mockCallback(message));
  expect(onCapture).not.toHaveBeenCalled();
  await clickCapture();
  act(() => mockCallback({ name: [], position: [] }));
  expect(screen.getByRole('alert')).toHaveTextContent('Missing');
  expect(onCapture).not.toHaveBeenCalled();
});

test('unmount cancels pending capture and ignores late feedback', async () => {
  const onCapture = jest.fn();
  const view = render(<PoseCaptureButton {...props} onCapture={onCapture} />);
  await clickCapture();
  view.unmount();
  act(() => mockCallback(message));
  expect(onCapture).not.toHaveBeenCalled();
  expect(mockUnsubscribe).toHaveBeenCalledTimes(1);
});

test('connection resolving after timeout does not subscribe', async () => {
  let connect;
  rosConnectionManager.getConnection.mockReturnValue(new Promise((resolve) => { connect = resolve; }));
  render(<PoseCaptureButton {...props} onCapture={jest.fn()} />);
  await clickCapture();
  act(() => jest.advanceTimersByTime(3000));
  await act(async () => connect({ isConnected: true }));
  expect(mockSubscribe).not.toHaveBeenCalled();
});
