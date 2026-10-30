import { act, renderHook } from '@testing-library/react';
import { useSelector } from 'react-redux';
import ROSLIB from 'roslib';
import rosConnectionManager from '../../utils/rosConnectionManager';
import useRobotControl from './useRobotControl';
const DEFAULT_MODE_TOPIC = '/reference/joystick/mode';
const fixtureResponse = (message) => message.service === '/get_robot_info'
  ? { success: true, joystick_mode_topic: DEFAULT_MODE_TOPIC }
  : { services: [], publishers: ['/leader/joystick_controller'] };

jest.mock('react-redux', () => ({ useSelector: jest.fn() }));
jest.mock('../../utils/rosConnectionManager', () => ({
  __esModule: true, default: { getCurrentConnection: jest.fn() },
}));

test('real roslib sends both subscriptions and receives mode/source reports without a reconnect handler', async () => {
  const ros = new ROSLIB.Ros();
  const sent = [];
  ros.isConnected = true;
  ros.socket = { send: (data) => {
    const message = JSON.parse(data);
    sent.push(message);
    if (message.op === 'call_service') {
      ros.emit(message.id, { result: true, values: fixtureResponse(message) });
    }
  } };
  useSelector.mockImplementation((selector) => selector({ ros: { rosbridgeUrl: 'ws://fixture' } }));
  rosConnectionManager.getCurrentConnection.mockReturnValue(ros);
  const { result, unmount } = renderHook(() => useRobotControl('fixture_robot'));
  await act(async () => { await Promise.resolve(); });
  expect(sent.filter((msg) => msg.op === 'subscribe').map((msg) => msg.topic)).toEqual([
    '/source', DEFAULT_MODE_TOPIC,
  ]);
  act(() => {
    ros.emit('/source', { data: 'teleop' });
    ros.emit(DEFAULT_MODE_TOPIC, { data: 'head_control' });
  });
  expect(result.current.source).toBe('teleop');
  expect(result.current.mode).toBe('head_control');
  expect(ros.listeners('close')).toHaveLength(1);
  unmount();
  expect(sent.filter((msg) => msg.op === 'unsubscribe')).toHaveLength(2);
  expect(ros.listeners('close')).toHaveLength(0);
});


test('unanswered discovery requests release real roslib listeners after timeout', async () => {
  jest.useFakeTimers();
  const ros = new ROSLIB.Ros();
  const sent = [];
  ros.isConnected = true;
  ros.socket = { send: (data) => sent.push(JSON.parse(data)) };
  useSelector.mockImplementation((selector) => selector({ ros: { rosbridgeUrl: 'ws://fixture' } }));
  rosConnectionManager.getCurrentConnection.mockReturnValue(ros);
  const { result, unmount } = renderHook(() => useRobotControl('fixture_robot'));
  try {
    const calls = sent.filter((message) => message.op === 'call_service');
    expect(calls).toHaveLength(3);
    calls.forEach((call) => {
      expect(call.timeout).toBe(4);
      expect(ros.listeners(call.id)).toHaveLength(1);
    });
    await act(async () => { jest.advanceTimersByTime(4000); });
    calls.forEach((call) => expect(ros.listeners(call.id)).toHaveLength(0));
    expect(ros.listeners('close')).toHaveLength(1);
    expect(result.current.canSetSource).toBe(false);
    expect(result.current.canSetPreset).toBe(false);
  } finally {
    unmount();
    jest.clearAllTimers();
    jest.useRealTimers();
  }
});


test('mode changes update Head/Base state without issuing robot commands', async () => {
  const ros = new ROSLIB.Ros();
  const sent = [];
  ros.isConnected = true;
  ros.socket = { send: (data) => {
    const message = JSON.parse(data);
    sent.push(message);
    if (message.op === 'call_service') ros.emit(message.id, { result: true, values: fixtureResponse(message) });
  } };
  useSelector.mockImplementation((selector) => selector({ ros: { rosbridgeUrl: 'ws://fixture' } }));
  rosConnectionManager.getCurrentConnection.mockReturnValue(ros);
  const { result, unmount } = renderHook(() => useRobotControl('fixture_robot'));
  await act(async () => { await Promise.resolve(); });
  try {
    for (const mode of ['head_control', 'swerve', 'head_control']) {
      act(() => ros.emit(DEFAULT_MODE_TOPIC, { data: mode }));
      expect(result.current.mode).toBe(mode);
    }
    expect(sent.filter((message) => message.op === 'call_service')
      .every((message) => ['/get_robot_info', '/rosapi/services_for_type', '/rosapi/publishers'].includes(message.service))).toBe(true);
    act(() => { ros.isConnected = false; ros.emit('close'); });
    expect(result.current.mode).toBeNull();
    expect(ros.listeners(DEFAULT_MODE_TOPIC)).toHaveLength(0);
  } finally { unmount(); }
});
