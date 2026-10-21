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

import fs from 'fs';
import path from 'path';
import { captureSg2Pose, SG2_GROUPS } from './sg2PoseCapture';

const names = Object.values(SG2_GROUPS).flat();
const message = { name: [...names].reverse(), position: names.map((_, i) => i / 10).reverse() };

test('captures selected joints by name in requested order, preserving radians and negative lift metres', () => {
  const state = { name: ['lift_joint', 'arm_l_joint2', 'gripper_l_joint1', 'arm_r_joint1'], position: [-0.25, 0.7, 1.1, -1.2] };
  expect(captureSg2Pose('JointControl', {
    enable_arms: 'true', enable_lift: 'true', enable_head: 'false',
    left_joint_names: 'gripper_l_joint1, arm_l_joint2', right_joint_names: 'arm_r_joint1',
  }, state)).toEqual({ left_joint_names: 'gripper_l_joint1, arm_l_joint2', left_positions: '1.1, 0.7',
    right_joint_names: 'arm_r_joint1', right_positions: '-1.2', lift_position: '-0.25' });
});

test('legacy empty arm names follow the full controller order', () => {
  const patch = captureSg2Pose('JointControl', { enable_arms: true }, message);
  expect(patch.left_positions).toBe('0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7');
  expect(patch.right_joint_names).toBe(SG2_GROUPS.arm_right.join(', '));
});

test('generic groups produce semicolon-separated positions in group order', () => {
  expect(captureSg2Pose('JointControl', { groups: 'lift,head' }, message))
    .toEqual({ positions: '1.8; 1.6, 1.7' });
});

test('gate captures only selected targets without changing gripper events or thresholds', () => {
  expect(captureSg2Pose('ArmStateGate', { left_target_joints: 'arm_l_joint3, arm_l_joint1',
    right_target_joints: '', detect_left_gripper: true, gripper_closed_value: '1.0' }, message))
    .toEqual({ left_target_positions: '0.2, 0' });
});

test.each([undefined, null, NaN, Infinity, '0.2'])('rejects missing or non-numeric feedback (%s)', (value) => {
  expect(() => captureSg2Pose('ArmStateGate', { left_target_joints: 'arm_l_joint1' },
    { name: ['arm_l_joint1'], position: [value] })).toThrow('Missing or invalid');
});

test('rejects mimic joints, duplicate names and empty capture selections', () => {
  expect(() => captureSg2Pose('ArmStateGate', { left_target_joints: 'gripper_l_joint2' }, message)).toThrow('Unsupported');
  expect(() => captureSg2Pose('JointControl', {}, { name: ['x', 'x'], position: [0, 0] })).toThrow('Duplicate');
  expect(() => captureSg2Pose('ArmStateGate', {}, message)).toThrow('select target joints');
  expect(() => captureSg2Pose('JointControl', { groups: 'mobile' }, message)).toThrow('Unsupported');
});

// Check the capture model against the shipped robot configuration and URDF.
const configRoot = process.env.SG2_CONFIG_ROOT || path.resolve(process.cwd(), '../../shared/shared/robot_configs');
test('capture joints match the shipped SG2 controller config and URDF', () => {
  const yaml = fs.readFileSync(path.join(configRoot, 'ffw_sg2_rev1_config.yaml'), 'utf8');
  const urdf = new DOMParser().parseFromString(fs.readFileSync(path.join(configRoot, 'urdf/ffw_sg2_follower.urdf'), 'utf8'), 'text/xml');
  expect(urdf.querySelector('parsererror')).toBeNull();
  names.forEach((name) => {
    const joint = [...urdf.querySelectorAll('robot > joint')].find((j) => j.getAttribute('name') === name);
    expect(joint).toBeDefined();
    expect(joint.querySelector('mimic')).toBeNull();
    expect(joint.getAttribute('type')).toBe(name === 'lift_joint' ? 'prismatic' : 'revolute');
    expect(yaml).toContain(`- ${name}`);
  });
});
