import { buildRosbagStartWarningMessage, classifyRecordingTopic, STATUS_IDLE } from './recordingMonitorWarnings';

const now = 10000;
const topic = (name, extra = {}) => ({ name, status: 2, rateHz: 0, secondsSinceLast: 1.1, ...extra });
const left = '/command_source_arbiter/executed/arm_left';
const right = '/command_source_arbiter/executed/arm_right';
const aux = ['/leader/joystick_controller_left/joint_trajectory', '/leader/joystick_controller_right/joint_trajectory', '/cmd_vel'];
const arbiter = (mode, extra = {}) => ({ mode, fault: '', publisher_conflict: '', receivedAtMs: now, ...extra });
const five = [topic(left), topic(right), ...aux.map((name) => topic(name, { secondsSinceLast: -1 }))];

test('normal idle commands produce no start warning and retain raw status', () => {
  const monitor = { topics: five, arbiter: arbiter('NONE') };
  expect(buildRosbagStartWarningMessage(monitor, now)).toBe('');
  expect(classifyRecordingTopic(five[0], monitor, now)).toMatchObject({ status: STATUS_IDLE, reportedStatus: 2 });
  expect(five[0].status).toBe(2);
});

test('WAYPOINT still warns about both arm streams but not unused joystick/base commands', () => {
  const message = buildRosbagStartWarningMessage({ topics: five, arbiter: arbiter('WAYPOINT') }, now);
  expect(message).toContain(left);
  expect(message).toContain(right);
  aux.forEach((name) => expect(message).not.toContain(name));
  expect(message).toContain('last 1.1s ago');
});

test.each([
  null, arbiter('NONE', { receivedAtMs: now - 2501 }),
  arbiter('NONE', { receivedAtMs: now + 1 }), arbiter('NONE', { receivedAtMs: undefined }),
  arbiter('NONE', { fault: 'waypoint lease lost' }),
  arbiter('NONE', { publisher_conflict: 'duplicate' }),
  arbiter('TELEOP'), arbiter('POLICY'),
])('unknown, stale, faulty or active control does not hide missing actions: %p', (state) => {
  const message = buildRosbagStartWarningMessage({ topics: five, arbiter: state }, now);
  expect(message).toContain(left);
  expect(message).toContain('no messages yet');
  expect(message).toContain('+2 more');
});

test('sensor and camera warnings are retained while commands are idle', () => {
  const monitor = { arbiter: arbiter('NONE'), topics: [topic('/joint_states'), topic('/odom'), topic('/camera/color/image_raw')] };
  const message = buildRosbagStartWarningMessage(monitor, now);
  monitor.topics.forEach((item) => expect(message).toContain(item.name));
});

test('a low-rate but nonzero command is not reclassified as idle', () => {
  expect(buildRosbagStartWarningMessage({ topics: [topic(left, { rateHz: 1 })], arbiter: arbiter('NONE') }, now)).toContain(left);
});


test('WAYPOINT does not hide a previously publishing auxiliary stream that stopped', () => {
  const message = buildRosbagStartWarningMessage({ topics: [topic('/cmd_vel')], arbiter: arbiter('WAYPOINT') }, now);
  expect(message).toContain('/cmd_vel');
});
