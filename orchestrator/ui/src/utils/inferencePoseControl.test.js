import ROSLIB from 'roslib';
import { InferencePoseClient, POSE_GROUPS, groupPositions, parseJointLimits, poseDuration, poseTrajectories } from './inferencePoseControl';

let ros; let client; let now; let positions;
const joints = Object.values(POSE_GROUPS).flatMap((config) => config.joints);
const xml = `<robot name="sg2">${joints.map((name) => `<joint name="${name}" type="revolute"><limit lower="${name === 'lift_joint' ? -.5 : -3.14}" upper="${name === 'lift_joint' ? 0 : name === 'head_joint1' ? .83 : 3.14}"/></joint>`).join('')}</robot>`;
const sent = () => ros.sendEncodedMessage.mock.calls.map(([data]) => JSON.parse(data));
const published = () => sent().filter((message) => message.op === 'publish');
const feedback = () => ros.emit('/joint_states', { name: joints, position: joints.map((joint) => positions[joint]) });
const services = () => {
  client.tick();
  if (client.pending.controllers) ros.emit(client.pending.controllers, { values: { controller:
    Object.values(POSE_GROUPS).flatMap((config) => config.routes.map(({ controller }) => ({ name: controller, state: 'active' }))),
  } });
  if (client.pending.limits) ros.emit(client.pending.limits, { values: { values: [{ string_value: xml }] } });
};
beforeEach(() => {
  now = 0; ros = new ROSLIB.Ros({ encoding: 'ascii' }); ros.isConnected = true; ros.sendEncodedMessage = jest.fn();
  client = new InferencePoseClient(ros, () => now); client.setEnabled(true);
  positions = Object.fromEntries(joints.map((joint) => [joint, joint === 'lift_joint' ? -.1 : 0]));
  feedback(); services();
});
afterEach(() => client.dispose());

test('preserves all supplied presets and uses left names for the left arm', () => {
  expect(POSE_GROUPS.left.poses).toEqual([
    [0.2127, 0.52, 0.0685, -1.792, 0.1741, 0, -1.3972, 0],
    [-0.6273861580203218, 0.5307453683835265, -0.6734175658817965, -0.7148230629298035, 1.3759567982836094, 1.267823136962577, -1.5642494028451934, 0],
    [0.539937268885935, 0, -0.05366535912618673, -2.5617359315441153, 0.023009711818284616, 0.4601702679158816, 0.057463995994551084, 0],
    [-0.644259946687064, 0.3558715585645634, -0.7086991240031663, -0.7685123905058009, 0.7040732131896986, -0.3750463184131339, -1.3299253759818837, 0],
  ]);
  expect(POSE_GROUPS.right.poses).toEqual([
    [0.2127, -0.52, -0.0685, -1.792, -0.1741, 0, 1.3972, 0],
    [0.539937268885935, 0, 0.05366535912618673, -2.5617359315441153, -0.023009711818284616, 0.4601702679158816, -0.057463995994551084, 0],
  ]);
  expect(POSE_GROUPS.head.poses).toEqual([[.83, 0], [.7286408742454724, 0]]);
  expect(POSE_GROUPS.lift.poses).toEqual([[-.12159]]);
  const reverse = [...POSE_GROUPS.left.joints].reverse();
  expect(groupPositions({ name: reverse, position: reverse.map((_, i) => i) }, 'left')).toEqual([7,6,5,4,3,2,1,0]);
  expect(groupPositions({ name: ['arm_l_joint1', 'arm_l_joint1'], position: [0, 1] }, 'left')).toBeNull();
  expect(groupPositions({ name: POSE_GROUPS.left.joints, position: Array(8).fill(NaN) }, 'left')).toBeNull();
});

test('splits arm and gripper commands, interpolates from measured state and caps cubic peak speed', () => {
  const current = Array(8).fill(0); const target = POSE_GROUPS.left.poses[1];
  const duration = poseDuration('left', current, target);
  expect(1.5 * Math.max(...target.map(Math.abs)) / duration).toBeLessThanOrEqual(.35);
  const commands = poseTrajectories('left', current, target, duration, 'owner');
  expect(commands.map(({ topic }) => topic)).toEqual(['/arm_l_controller/joint_trajectory', '/gripper_l_controller/joint_trajectory']);
  expect(commands[0].message.joint_names).toEqual(POSE_GROUPS.left.joints.slice(0, 7));
  expect(commands[0].message.points[0]).toMatchObject({ positions: current.slice(0,7), velocities: Array(7).fill(0), time_from_start: { sec:0, nanosec:0 } });
  expect(commands[0].message.points[1].positions).toEqual(target.slice(0,7));
  expect(commands[1].message.joint_names).toEqual(['gripper_l_joint1']);
  expect(commands[1].message.points[1].positions).toEqual([0]);
  expect(() => poseTrajectories('left', [0], target, 3)).toThrow();
  expect(parseJointLimits(xml).head_joint1).toEqual([-3.14, .83]);
  expect(() => parseJointLimits('<not-a-robot/>')).toThrow();
});

test.each(['left', 'right', 'head', 'lift'])('only publishes the selected %s routes after a click', (group) => {
  expect(sent().filter((message) => message.op === 'advertise')).toHaveLength(0);
  expect(client.command(group, 0)).toBe(true);
  expect(published().map(({ topic }) => topic)).toEqual(POSE_GROUPS[group].routes.map(({ topic }) => topic));
  published().forEach(({ topic, msg }) => ros.emit(topic, msg));
  expect(client.snapshot().failures[group]).toBeUndefined();
  expect(client.command(group === 'left' ? 'right' : 'left', 0)).toBe(false);
  now = client.moving.until + 1; feedback(); services();
  expect(client.isBusy()).toBe(true); // elapsed time alone never claims success
  POSE_GROUPS[group].joints.forEach((joint, i) => { positions[joint] = POSE_GROUPS[group].poses[0][i]; });
  feedback(); client.tick();
  expect(client.snapshot().holding[group]).toEqual(POSE_GROUPS[group].poses[0]);
  expect(client.isBusy()).toBe(false);
  expect(published()).toHaveLength(POSE_GROUPS[group].routes.length); // no repeated hold commands
  expect(Object.values(client.publishers).every((publisher) => !publisher.isAdvertised)).toBe(true);
});

test.each(['missing feedback', 'stale feedback', 'no limits', 'outside limits', 'inactive controller', 'stale controller', 'operation lock'])('blocks commands with %s', (condition) => {
  if (condition === 'missing feedback') ros.emit('/joint_states', { name: [], position: [] });
  if (condition === 'stale feedback') now = 800;
  if (condition === 'no limits') client.limits = {};
  if (condition === 'outside limits') { positions.arm_l_joint1 = 4; feedback(); }
  if (condition === 'inactive controller') client.controllers.arm_l_controller = 'inactive';
  if (condition === 'stale controller') { now = 2100; feedback(); }
  if (condition === 'operation lock') client.setEnabled(false);
  expect(client.command('left', 0)).toBe(false);
  expect(published()).toHaveLength(0);
});

test('rejects presets against live robot limits without silently changing their values', () => {
  client.limits.head_joint1 = [-.23, .7];
  expect(client.command('head', 0)).toBe(false);
  expect(client.snapshot().failures.head).toMatch(/exceeds/);
  expect(published()).toHaveLength(0);
});

test.each(['/arm_l_controller/joint_trajectory', '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory'])('rejects competing commands on %s', (topic) => {
  ros.emit(topic, { header: {}, joint_names: POSE_GROUPS.left.joints });
  expect(client.command('left', 0)).toBe(false);
  expect(client.command('right', 0)).toBe(true);
});

test('reports a replaced command and clears hold when a subsequent source takes ownership', () => {
  client.command('head', 0);
  ros.emit(POSE_GROUPS.head.routes[0].topic, { header: {} });
  now = client.moving.until + 1; feedback(); client.tick();
  expect(client.isBusy()).toBe(false);
  expect(client.snapshot().failures.head).toMatch(/replaced/);
  expect(client.snapshot().holding.head).toBeUndefined();
  client.holding.left = POSE_GROUPS.left.poses[0];
  ros.emit(POSE_GROUPS.left.observed[0], { header: {} });
  expect(client.snapshot().holding.left).toBeUndefined();
});

test('Hold current replaces only an ongoing movement of the same group', () => {
  client.command('left', 1);
  expect(client.command('right', 0, true)).toBe(false);
  expect(client.command('left', 0, true)).toBe(true);
  expect(client.moving.duration).toBe(1);
  expect(client.moving.target).toEqual(Array(8).fill(0));
  expect(published().at(-2).msg.points[1].positions).toEqual(Array(7).fill(0));
});

test('feedback timeout reports failure and releases movement lock', () => {
  client.command('right', 0);
  now = client.moving.until + 5000; client.tick();
  expect(client.isBusy()).toBe(false);
  expect(client.snapshot().failures.right).toMatch(/not confirmed/);
  expect(client.snapshot().holding.right).toBeUndefined();
});

test('disconnect does not replay a trajectory and keeps the finite movement lock', () => {
  client.command('lift', 0); const before = sent().length;
  ros.isConnected = false; ros.emit('close');
  expect(client.command('right', 0)).toBe(false);
  expect(client.isBusy()).toBe(true);
  now = client.moving.until + 5000; client.tick();
  expect(client.isBusy()).toBe(false);
  client.dispose();
  expect(sent()).toHaveLength(before);
  expect(ros.listeners('connection')).toHaveLength(0);
  expect(ros.listeners('/joint_states')).toHaveLength(0);
  ros.isConnected = true; ros.emit('connection');
  expect(published()).toHaveLength(1);
});

test('expired service replies cannot revive stale status and pending listeners are removed', () => {
  now = 1100; client.tick(); const previous = client.pending.controllers;
  now = 2200; client.tick();
  expect(ros.listeners(previous)).toHaveLength(0);
  const pending = client.pending.controllers;
  now = 3201; ros.emit(pending, { values: { controller: [{ name:'arm_l_controller', state:'active' }] } }); feedback();
  expect(client.command('left', 0)).toBe(false);
  client.dispose(); expect(ros.listeners(pending)).toHaveLength(0);
});
