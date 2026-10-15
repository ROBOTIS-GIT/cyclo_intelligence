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

export const SG2_LEFT_JOINTS = [
  'arm_l_joint1',
  'arm_l_joint2',
  'arm_l_joint3',
  'arm_l_joint4',
  'arm_l_joint5',
  'arm_l_joint6',
  'arm_l_joint7',
  'gripper_l_joint1',
];

export const SG2_RIGHT_JOINTS = [
  'arm_r_joint1',
  'arm_r_joint2',
  'arm_r_joint3',
  'arm_r_joint4',
  'arm_r_joint5',
  'arm_r_joint6',
  'arm_r_joint7',
  'gripper_r_joint1',
];

export const SG2_TARGET_JOINTS = {
  // ArmStateGate observation targets
  left_target_joints: SG2_LEFT_JOINTS,
  right_target_joints: SG2_RIGHT_JOINTS,
  // JointControl trajectory joints
  left_joint_names: SG2_LEFT_JOINTS,
  right_joint_names: SG2_RIGHT_JOINTS,
};

// joint-selection param → its paired positions param. Toggling a joint chip
// keeps the two CSVs aligned (new joints get 0.0, removed joints drop theirs).
export const TARGET_POSITION_PARAM = {
  left_target_joints: 'left_target_positions',
  right_target_joints: 'right_target_positions',
  left_joint_names: 'left_positions',
  right_joint_names: 'right_positions',
};

// Node types whose joint params render as the chip selector.
export const JOINT_SELECTOR_NODE_TYPES = new Set(['ArmStateGate', 'JointControl']);

// Nodes created before the per-joint params existed (or from a stale bt_node
// catalog) lack left/right_joint_names entirely, so the selector would never
// render for them. Synthesize the params with the full joint list — exactly
// what the engine does when the names are omitted — inserting each one just
// before its positions field so the selector renders above it.
export function withJointSelectionDefaults(params, nodeType) {
  if (nodeType !== 'JointControl') return params;
  const next = {};
  const insertBefore = {
    left_positions: ['left_joint_names', SG2_LEFT_JOINTS.join(', ')],
    right_positions: ['right_joint_names', SG2_RIGHT_JOINTS.join(', ')],
  };
  Object.entries(params).forEach(([key, value]) => {
    const missing = insertBefore[key];
    if (missing && !(missing[0] in params)) {
      next[missing[0]] = missing[1];
    }
    next[key] = value;
  });
  return next;
}

export const csvParts = (value) =>
  String(value || '')
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);

export function isSg2LikeRobot(robotType) {
  const normalized = String(robotType || '').toLowerCase();
  return !normalized || normalized.includes('sg2') || normalized.includes('bg2');
}

export function getTargetJointOptions(robotType, key) {
  if (!isSg2LikeRobot(robotType)) return [];
  return SG2_TARGET_JOINTS[key] || [];
}

export const truthy = (v) => v === true || v === 'true';

export function isJointControlFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'JointControl') return false;
  if (key === 'head_positions') return !truthy(params.enable_head);
  if (
    key === 'left_positions' || key === 'right_positions'
    || key === 'left_joint_names' || key === 'right_joint_names'
  ) {
    return !truthy(params.enable_arms);
  }
  if (key === 'lift_position') return !truthy(params.enable_lift);
  return false;  // enable_*, duration stay editable
}

export function isArmStateGateFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'ArmStateGate') return false;
  if (key === 'left_gripper_joint') {
    return !truthy(params.detect_left_gripper);
  }
  if (key === 'right_gripper_joint') {
    return !truthy(params.detect_right_gripper);
  }
  return false;
}
