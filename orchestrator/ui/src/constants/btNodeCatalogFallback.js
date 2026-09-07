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

export const FALLBACK_SCHEMA_VERSION = '1.0';

export const FALLBACK_CATALOG = [
  {
    tag: 'Sequence',
    category: 'control',
    ports: [],
  },
  {
    tag: 'Loop',
    category: 'control',
    ports: [
      { name: 'max_iterations', type: 'number', default: '0' },
    ],
  },
  {
    tag: 'Rotate',
    category: 'action',
    ports: [
      { name: 'angle_deg', type: 'number', default: '90.0' },
    ],
  },
  {
    tag: 'JointControl',
    category: 'action',
    ports: [
      { name: 'enable_head', type: 'bool', default: 'true' },
      { name: 'head_positions', type: 'string', default: '0.0, 0.0' },
      { name: 'enable_arms', type: 'bool', default: 'false' },
      {
        name: 'left_joint_names',
        type: 'string',
        default:
          'arm_l_joint1, arm_l_joint2, arm_l_joint3, arm_l_joint4, '
          + 'arm_l_joint5, arm_l_joint6, arm_l_joint7, gripper_l_joint1',
      },
      {
        name: 'left_positions',
        type: 'string',
        default: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
      },
      {
        name: 'right_joint_names',
        type: 'string',
        default:
          'arm_r_joint1, arm_r_joint2, arm_r_joint3, arm_r_joint4, '
          + 'arm_r_joint5, arm_r_joint6, arm_r_joint7, gripper_r_joint1',
      },
      {
        name: 'right_positions',
        type: 'string',
        default: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
      },
      { name: 'enable_lift', type: 'bool', default: 'false' },
      { name: 'lift_position', type: 'number', default: '0.0' },
      { name: 'duration', type: 'number', default: '2.0' },
    ],
  },
  {
    tag: 'ArmStateGate',
    category: 'action',
    ports: [
      { name: 'left_target_joints', type: 'string', default: '' },
      { name: 'left_target_positions', type: 'string', default: '' },
      { name: 'right_target_joints', type: 'string', default: '' },
      { name: 'right_target_positions', type: 'string', default: '' },
      { name: 'joint_threshold', type: 'number', default: '0.01' },
      { name: 'detect_left_gripper', type: 'bool', default: 'false' },
      { name: 'detect_right_gripper', type: 'bool', default: 'false' },
      {
        name: 'left_gripper_joint',
        type: 'string',
        default: 'gripper_l_joint1',
      },
      {
        name: 'right_gripper_joint',
        type: 'string',
        default: 'gripper_r_joint1',
      },
      { name: 'gripper_closed_value', type: 'number', default: '1.0' },
      { name: 'gripper_open_value', type: 'number', default: '0.0' },
      { name: 'gripper_threshold', type: 'number', default: '0.05' },
      { name: 'timeout_sec', type: 'number', default: '0.0' },
    ],
  },
  {
    tag: 'SendCommand',
    category: 'action',
    ports: [
      { name: 'target', type: 'string', default: 'INFERENCE' },
      { name: 'command', type: 'string', default: 'LOAD' },
      { name: 'model', type: 'string', default: 'lerobot:act' },
      { name: 'policy_path', type: 'string', default: '' },
      { name: 'task_instruction', type: 'string', default: '' },
      { name: 'inference_mode', type: 'string', default: 'simulation' },
      { name: 'action_request_mode', type: 'string', default: 'async' },
      { name: 'inference_hz', type: 'number', default: '15' },
      { name: 'control_hz', type: 'number', default: '100' },
      { name: 'chunk_align_window_s', type: 'number', default: '0.3' },
      { name: 'acceleration_mode', type: 'string', default: 'pytorch' },
      { name: 'acceleration_engine_path', type: 'string', default: '' },
    ],
  },
  {
    tag: 'Wait',
    category: 'action',
    ports: [
      { name: 'duration', type: 'number', default: '5.0' },
    ],
  },
];
