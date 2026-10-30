import { act, renderHook } from '@testing-library/react';
import { useSelector } from 'react-redux';
import useRobotControl from './useRobotControl';
import rosConnectionManager from '../../utils/rosConnectionManager';
import { PRESET_SERVICE, SOURCE_SERVICE } from './controlProtocol';

const DEFAULT_MODE_TOPIC = '/reference/joystick/mode';
let mockModeTopic;
let mockInfoFailure;
const mockTopics = [];
const mockCalls = [];
let mockAvailable = true;
let mockLeaderAvailable = true;

jest.mock('react-redux', () => ({ useSelector: jest.fn() }));
jest.mock('../../utils/rosConnectionManager', () => ({
  __esModule: true, default: { getCurrentConnection: jest.fn() },
}));
jest.mock('roslib', () => ({
  Topic: function Topic(options) {
    const topic = { ...options, subscribe: jest.fn((cb) => { topic.callback = cb; }), unsubscribe: jest.fn() };
    mockTopics.push(topic);
    return topic;
  },

}));

let ros;
const flush = async () => { await act(async () => { await Promise.resolve(); }); };
const emit = (name, data) => act(() => mockTopics.filter((t) => t.name === name).at(-1).callback({ data }));
const command = (name) => mockCalls.filter((call) => call.name === name).at(-1);
const advance = async (ms) => { await act(async () => { jest.advanceTimersByTime(ms); }); };

beforeEach(() => {
  jest.useFakeTimers();
  mockTopics.length = 0;
  mockCalls.length = 0;
  mockAvailable = true;
  mockLeaderAvailable = true;
  mockModeTopic = DEFAULT_MODE_TOPIC;
  mockInfoFailure = false;
  const listeners = new Map();
  ros = {
    isConnected: true, idCounter: 0,
    on: jest.fn((name, callback) => { listeners.set(name, callback); }),
    off: jest.fn((name, callback) => {
      if (listeners.get(name) === callback) listeners.delete(name);
    }),
    callOnConnection: jest.fn((message) => {
      const success = (values) => listeners.get(message.id)?.({ result: true, values });
      const failure = (values) => listeners.get(message.id)?.({ result: false, values });
      mockCalls.push({ name: message.service, serviceType: message.type, args: message.args, success, failure });
      if (message.service === '/get_robot_info') {
        if (mockInfoFailure) failure({ message: 'Unavailable' });
        else success({ success: true, joystick_mode_topic: mockModeTopic });
      }
      if (message.service === '/rosapi/publishers') success({ publishers: mockLeaderAvailable ? ['/leader/joystick_controller'] : [] });
      if (message.service === '/rosapi/services_for_type') {
        success({ services: mockAvailable ? ['/set_source', '/hand_preset_controller/set_preset'] : [] });
      }
    }),
  };
  useSelector.mockImplementation((selector) => selector({ ros: { rosbridgeUrl: 'ws://robot:7090' } }));
  rosConnectionManager.getCurrentConnection.mockReturnValue(ros);
});
afterEach(() => { jest.clearAllTimers(); jest.useRealTimers(); jest.clearAllMocks(); });

test('observes status without sending commands; never assumes the initial joystick mode', async () => {
  const { result } = renderHook(() => useRobotControl('ffw_sh5_rev1'));
  await flush();
  expect(result.current.canSetSource).toBe(true);
  expect(result.current.mode).toBeNull();
  expect(command(SOURCE_SERVICE)).toBeUndefined();
  expect(command(PRESET_SERVICE)).toBeUndefined();
  emit('/source', 'model_action');
  expect(result.current.source).toBe('external');
  emit(DEFAULT_MODE_TOPIC, 'head_control');
  expect(result.current.mode).toBe('head_control');
});

test('source acceptance does not change the displayed source; waits for a new report', async () => {
  const { result } = renderHook(() => useRobotControl('ffw_sh5_rev1'));
  await flush();
  emit('/source', 'external');
  act(() => { result.current.setSource('teleop'); result.current.setSource('teleop'); });
  expect(mockCalls.filter((call) => call.name === SOURCE_SERVICE)).toHaveLength(1);
  expect(command(SOURCE_SERVICE).args).toEqual({ data: true });
  await act(async () => command(SOURCE_SERVICE).success({ success: true }));
  expect(result.current.source).toBe('external');
  expect(result.current.pending).toBe('source');
  emit('/source', 'teleop');
  await advance(100);
  expect(result.current.pending).toBeNull();
  expect(result.current.notice).toBe('Source confirmed: Teleop.');
});

test('external uses SetBool false and never calls arm activation', async () => {
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  emit('/source', 'teleop');
  act(() => { result.current.setSource('external'); });
  expect(command(SOURCE_SERVICE).args).toEqual({ data: false });
  expect(mockCalls.some((call) => call.name === '/set_enabled')).toBe(false);
  await act(async () => command(SOURCE_SERVICE).success({ success: false, message: 'Busy' }));
  expect(result.current.error).toBe('Busy');
});

test('stale source is unknown and cannot be used to issue a new source command', async () => {
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  emit('/source', 'teleop');
  emit(DEFAULT_MODE_TOPIC, 'swerve');
  await advance(3500);
  expect(result.current.source).toBeNull();
  expect(result.current.mode).toBeNull();
  await act(async () => result.current.setSource('external'));
  expect(command(SOURCE_SERVICE)).toBeUndefined();
});

test('unsupported robots disable optional controls', async () => {
  mockAvailable = false;
  const { result } = renderHook(() => useRobotControl('legacy_robot'));
  await flush();
  expect(result.current.canSetPreset).toBe(false);
  await act(async () => result.current.setPreset('both', '2', '2'));
  expect(command(PRESET_SERVICE)).toBeUndefined();
});

test('preset service uses hand controller endpoint and only records accepted selections', async () => {
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  act(() => { result.current.setPreset('left', '2', ''); });
  expect(command(PRESET_SERVICE).args).toEqual({ target_arm: 'left', left_preset_id: 2, right_preset_id: 0 });
  await act(async () => command(PRESET_SERVICE).success({ accepted: true, message: 'Selected' }));
  expect(result.current.selected).toEqual({ left: 2, right: null });
  act(() => { result.current.setPreset('both', '3', '3'); });
  await act(async () => command(PRESET_SERVICE).success({ accepted: false, message: 'Unknown ID' }));
  expect(result.current.selected).toEqual({ left: 2, right: null });
  expect(result.current.error).toBe('Unknown ID');
});

test.each(['0', '65536', '2.5', '', '-1', 'NaN'])('rejects invalid preset ID %s before sending', async (id) => {
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  await act(async () => result.current.setPreset('both', id, '2'));
  expect(command(PRESET_SERVICE)).toBeUndefined();
  expect(result.current.error).toMatch(/whole numbers/);
});

test('service timeout reports uncertainty, never retries or marks a selection accepted', async () => {
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  act(() => { result.current.setPreset('both', '2', '2'); });
  await advance(4000);
  expect(result.current.error).toMatch(/outcome is unknown/);
  expect(result.current.selected.left).toBeNull();
  await act(async () => command(PRESET_SERVICE).success({ accepted: true }));
  expect(result.current.selected.left).toBeNull();
  expect(mockCalls.filter((call) => call.name === PRESET_SERVICE)).toHaveLength(1);
});

test('source acceptance without completion times out without claiming success', async () => {
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  emit('/source', 'external');
  act(() => { result.current.setSource('teleop'); });
  await act(async () => command(SOURCE_SERVICE).success({ success: true }));
  await advance(6100);
  expect(result.current.error).toMatch(/completion was not confirmed/);
  expect(result.current.notice).not.toMatch(/Source confirmed/);
});

test('disconnect clears status and ignores a late response; reconnect subscribes anew', async () => {
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  emit('/source', 'teleop');
  act(() => { result.current.setPreset('both', '3', '3'); });
  const oldCall = command(PRESET_SERVICE);
  ros.isConnected = false;
  act(() => ros.on.mock.calls.find(([event]) => event === 'close')[1]());
  expect(result.current.connected).toBe(false);
  expect(result.current.source).toBeNull();
  await act(async () => oldCall.success({ accepted: true }));
  expect(result.current.selected.left).toBeNull();
  rosConnectionManager.getCurrentConnection.mockReturnValue({ isConnected: true, on: jest.fn(), off: jest.fn(), callOnConnection: jest.fn() });
  await advance(500);
  await flush();
  expect(result.current.connected).toBe(true);
  expect(result.current.mode).toBeNull();
  expect(mockTopics[0].unsubscribe).toHaveBeenCalledTimes(1);
  expect(mockTopics.filter((t) => t.name === '/source')).toHaveLength(2);
});

test('switching robots invalidates the previous request and subscriptions', async () => {
  const { result, rerender, unmount } = renderHook(({ robot }) => useRobotControl(robot), { initialProps: { robot: 'first' } });
  await flush();
  act(() => { result.current.setPreset('both', '2', '2'); });
  const oldCall = command(PRESET_SERVICE);
  rerender({ robot: 'second' });
  await flush();
  await act(async () => oldCall.success({ accepted: true }));
  expect(result.current.selected.left).toBeNull();
  unmount();
  expect(mockTopics.every((topic) => topic.unsubscribe.mock.calls.length === 1)).toBe(true);
});


test('clears the last joystick mode when the leader publisher disappears', async () => {
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  emit(DEFAULT_MODE_TOPIC, 'swerve');
  expect(result.current.mode).toBe('swerve');
  mockLeaderAvailable = false;
  await advance(5000);
  await flush();
  expect(result.current.modeAvailable).toBe(false);
  expect(result.current.mode).toBeNull();
  expect(result.current.canSetSource).toBe(true);
  mockLeaderAvailable = true;
  await advance(5000);
  await flush();
  expect(result.current.modeAvailable).toBe(true);
  expect(result.current.mode).toBeNull();
  emit(DEFAULT_MODE_TOPIC, 'head_control');
  expect(result.current.mode).toBe('head_control');
});


test('uses the configured topic and retires its subscription when configuration changes', async () => {
  mockModeTopic = '/custom/leader/mode';
  const { result, unmount } = renderHook(() => useRobotControl('robot'));
  await flush();
  expect(command('/rosapi/publishers').args).toEqual({ topic: mockModeTopic });
  expect(mockTopics.some((t) => t.name === DEFAULT_MODE_TOPIC)).toBe(false);
  emit(mockModeTopic, 'swerve');
  const previous = mockTopics.at(-1);
  mockModeTopic = DEFAULT_MODE_TOPIC;
  await advance(5000);
  await flush();
  expect(previous.unsubscribe).toHaveBeenCalledTimes(1);
  expect(result.current.mode).toBeNull();
  act(() => previous.callback({ data: 'swerve' }));
  expect(result.current.mode).toBeNull();
  emit(DEFAULT_MODE_TOPIC, 'head_control');
  expect(result.current.mode).toBe('head_control');
  unmount();
  expect(mockTopics.every((t) => t.unsubscribe.mock.calls.length === 1)).toBe(true);
});

test('an empty configured mode topic is not replaced with a hard-coded fallback', async () => {
  mockModeTopic = '';
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  expect(mockTopics.map((t) => t.name)).toEqual(['/source']);
  expect(command('/rosapi/publishers')).toBeUndefined();
  expect(result.current.mode).toBeNull();
  expect(result.current.modeAvailable).toBe(false);
  expect(result.current.canSetSource).toBe(true);
});

test('robot info lookup failure does not disable source or preset services and is retried', async () => {
  mockInfoFailure = true;
  const { result } = renderHook(() => useRobotControl('robot'));
  await flush();
  expect(result.current.canSetSource).toBe(true);
  expect(result.current.canSetPreset).toBe(true);
  expect(result.current.mode).toBeNull();
  mockInfoFailure = false;
  await advance(5000);
  await flush();
  emit(DEFAULT_MODE_TOPIC, 'swerve');
  expect(result.current.mode).toBe('swerve');
});
