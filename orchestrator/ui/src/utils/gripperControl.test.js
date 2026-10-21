import ROSLIB from 'roslib';
import { GripperControlClient, GRIPPERS, GRIPPER_MOVE_MS, gripperPosition, gripperTrajectory } from './gripperControl';

let ros;
let client;
let now;
const sent = () => ros.sendEncodedMessage.mock.calls.map(([data]) => JSON.parse(data));
const published = () => sent().filter((message) => message.op === 'publish');
const feedback = () => ros.emit('/joint_states', {
  name: ['arm_r_joint1', 'gripper_l_joint1', 'gripper_r_joint1'], position: [-0.2, 0.0, 0.7],
});
const controllers = (rightState = 'active') => {
  client.tick();
  ros.emit(client.pendingServiceId, { values: { controller: [
    { name: 'gripper_l_controller', state: 'active' },
    { name: 'gripper_r_controller', state: rightState },
  ] } });
};

beforeEach(() => {
  now = 0;
  ros = new ROSLIB.Ros({ encoding: 'ascii' });
  ros.isConnected = true;
  ros.sendEncodedMessage = jest.fn();
  client = new GripperControlClient(ros, () => now);
  client.setEnabled(true);
  feedback(); controllers();
});
afterEach(() => client.dispose());

test('open/close targets use only the selected follower gripper, regardless of state order', () => {
  const message = { name: ['gripper_r_joint1', 'arm_l_joint1'], position: [1.1, -0.5] };
  expect(gripperPosition(message, 'right')).toBe(1.1);
  expect(gripperPosition(message, 'left')).toBeNull();
  expect(gripperTrajectory('right', 1.1).joint_names).toEqual(['gripper_r_joint1']);
  expect(gripperTrajectory('left', 0).points[0]).toEqual({
    positions: [0], velocities: [0], time_from_start: { sec: 2, nanosec: 0 },
  });
  expect(() => gripperTrajectory('right', 0.5)).toThrow();
  expect(() => gripperTrajectory('both', 1.1)).toThrow();
});

test('does not advertise on mount; a click sends one command and releases the publisher after movement', () => {
  expect(sent().filter((m) => m.op === 'advertise')).toHaveLength(0);
  expect(client.command('right', 1.1)).toBe(true);
  expect(published()).toHaveLength(1);
  expect(published()[0]).toMatchObject({ topic: GRIPPERS.right.topic,
    msg: { joint_names: ['gripper_r_joint1'], points: [{ positions: [1.1] }] } });
  expect(client.command('left', 0)).toBe(false);
  ros.emit(GRIPPERS.right.topic, published()[0].msg); // Own echo is not competition.
  expect(client.snapshot().reasons.right).toBe('');
  expect(client.snapshot().busy).toBe(true);
  now = GRIPPER_MOVE_MS + 1;
  feedback(); controllers();
  expect(client.snapshot().busy).toBe(false);
  expect(client.publishers.right.isAdvertised).toBe(false);
  expect(client.command('left', 0)).toBe(true);
  expect(published().map((m) => m.topic)).toEqual([GRIPPERS.right.topic, GRIPPERS.left.topic]);
  expect(ros.listeners('close')).toHaveLength(1);
});

test.each(['missing feedback', 'stale feedback', 'inactive controller', 'stale controller', 'operation lock'])('blocks commands with %s', (condition) => {
  if (condition === 'missing feedback') ros.emit('/joint_states', { name: ['x'], position: [1] });
  if (condition === 'stale feedback') now = 800;
  if (condition === 'inactive controller') client.controllers.gripper_r_controller = 'inactive';
  if (condition === 'stale controller') { now = 2100; feedback(); }
  if (condition === 'operation lock') client.setEnabled(false);
  expect(client.command('right', 1.1)).toBe(false);
  expect(published()).toHaveLength(0);
});

test.each([GRIPPERS.right.topic, GRIPPERS.left.topic, GRIPPERS.right.leaderTopic])('blocks both grippers when another source commands %s', (topic) => {
  ros.emit(topic, { header: {}, joint_names: ['gripper_r_joint1'] });
  expect(client.command('right', 1.1)).toBe(false);
  expect(client.command('left', 0)).toBe(false);
  expect(client.snapshot().reasons.right).toMatch(/Another robot control source/);
  expect(published()).toHaveLength(0);
});

test('disconnect and disposal never queue commands or cleanup for a later connection', () => {
  const before = sent().length;
  ros.isConnected = false;
  ros.emit('close');
  expect(client.command('right', 1.1)).toBe(false);
  client.dispose();
  expect(sent()).toHaveLength(before);
  expect(ros.listeners('connection')).toHaveLength(0);
  expect(ros.listeners('/joint_states')).toHaveLength(0);
  expect(ros.listeners('close')).toHaveLength(0);
  ros.isConnected = true;
  ros.emit('connection');
  expect(published()).toHaveLength(0);
});

test('expired controller service listeners are removed and delayed replies are rejected', () => {
  now = 1100;
  client.tick();
  const previous = client.pendingServiceId;
  now = 2200;
  client.tick();
  expect(ros.listeners(previous)).toHaveLength(0);
  const pending = client.pendingServiceId;
  now = 3201;
  ros.emit(pending, { values: { controller: [{ name: 'gripper_r_controller', state: 'active' }] } });
  feedback();
  expect(client.command('right', 1.1)).toBe(false);
  client.dispose();
  expect(ros.listeners(pending)).toHaveLength(0);
});
