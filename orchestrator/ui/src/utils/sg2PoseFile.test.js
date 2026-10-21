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

import { SG2_GROUPS } from './sg2PoseCapture';
import { applySg2PoseFile, parseSg2Pose, poseFilename, serializeSg2Pose } from './sg2PoseFile';

const name = Object.values(SG2_GROUPS).flat();
const pose = { name, position: name.map((_, i) => i === 18 ? -0.25 : i / 10) };

test('pose XML round-trips every SG2 joint and is a standard JointControl tree', () => {
  const content = serializeSg2Pose(pose, '작업_대기');
  expect(parseSg2Pose(content)).toEqual(pose);
  const doc = new DOMParser().parseFromString(content, 'text/xml');
  expect(doc.querySelector('BehaviorTree > JointControl').getAttribute('enable_arms')).toBe('true');
  expect(poseFilename('작업_대기')).toBe('sg2_pose_작업_대기.xml');
});

test('one saved pose applies to either block, preserving selections and unrelated settings', () => {
  const saved = parseSg2Pose(serializeSg2Pose(pose, 'ready'));
  expect(applySg2PoseFile('ArmStateGate', { left_target_joints: 'arm_l_joint3, arm_l_joint1', detect_left_gripper: 'true' }, saved))
    .toEqual({ left_target_positions: '0.2, 0' });
  expect(applySg2PoseFile('JointControl', { enable_head: 'true', enable_lift: 'true', duration: '9' }, saved))
    .toEqual({ head_positions: '1.6, 1.7', lift_position: '-0.25' });
});

test('empty gate gets both seven-joint arm targets without constraining the gripper gesture', () => {
  const patch = applySg2PoseFile('ArmStateGate', {}, pose);
  expect(patch.left_target_joints.split(', ')).toEqual(SG2_GROUPS.arm_left.slice(0, 7));
  expect(patch.right_target_positions.split(', ')).toHaveLength(7);
  expect(patch.detect_left_gripper).toBeUndefined();
  expect(applySg2PoseFile('JointControl', {}, pose).enable_arms).toBe('true');
});

test('rejects incompatible, malformed or incomplete files and unsafe names', () => {
  expect(() => poseFilename('../ready')).toThrow();
  expect(() => serializeSg2Pose({ name: [], position: [] }, 'ready')).toThrow('Missing');
  const xml = serializeSg2Pose(pose, 'ready');
  expect(() => parseSg2Pose(xml.replace('ffw_sg2_rev1', 'other'))).toThrow('supported');
  expect(() => parseSg2Pose(xml.replace('lift_position="-0.25"', 'lift_position="NaN"'))).toThrow('Invalid');
  expect(() => parseSg2Pose(xml.replace('head_positions="1.6, 1.7"', 'head_positions="1.6,"'))).toThrow('Invalid');
  expect(() => parseSg2Pose('<root>')).toThrow();
});
