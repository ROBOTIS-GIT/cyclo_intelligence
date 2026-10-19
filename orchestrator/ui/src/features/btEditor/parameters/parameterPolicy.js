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

import {
  withJointSelectionDefaults,
  isJointControlFieldDisabled,
  isArmStateGateFieldDisabled,
} from './jointParams';

export const NUMBER_PARAMS = new Set([
  'duration', 'angle_deg', 'lift_position', 'control_hz', 'inference_hz',
  'chunk_align_window_s', 'max_iterations', 'joint_threshold',
  'gripper_closed_value', 'gripper_open_value', 'gripper_threshold',
  'timeout_sec',
]);

// Per-param helper text shown beneath the input. Keep these short — they
// render directly under the field as a small gray hint.
export const HELP_TEXT = {
  image_mode: 'raw: rectify internally. rectified: use an already rectified image topic. Detection always uses rectified pixels.',
  initial_pose_sync: 'Align to the first predicted pose before RESUME. Enabled by default for robot output.',
  initial_pose_sync_duration_s: 'Pose alignment duration in seconds (1–60).',
  vla_rail_enabled: 'VLA-RAIL cubic smoothing and dual-quintic transitions for ordinary GR00T Async. Off by default.',
  max_iterations: '0 = loop forever',
};

// Boolean params render as checkboxes.
export const BOOL_PARAMS = new Set([
  'initial_pose_sync',
  'vla_rail_enabled',
  'enable_head',
  'enable_arms',
  'enable_lift',
  'detect_left_gripper',
  'detect_right_gripper',
]);

// Drop unsupported flags when opening nodes saved by earlier Canvas versions.
export const RETIRED_INFERENCE_PARAMS = new Set([
  'ttrtc_enabled', 'ttrtc_rate_limit_enabled', 'chunk_spline_smoothing',
  'chunk_endpoint_interpolation', 'remac_enabled',
]);

export const SEND_COMMAND_TARGETS = ['INFERENCE', 'DOCKER'];

export const SEND_COMMAND_COMMANDS_BY_TARGET = {
  INFERENCE: ['LOAD', 'RESUME', 'STOP', 'CLEAR'],
  DOCKER: ['START', 'STOP', 'RESTART'],
};

export function normalizeSendCommandTarget(target) {
  return String(target || 'INFERENCE').trim().toUpperCase() === 'DOCKER'
    ? 'DOCKER'
    : 'INFERENCE';
}

export function normalizeSendCommandParams(params, nodeType) {
  const next = withJointSelectionDefaults(params, nodeType);
  if (nodeType !== 'SendCommand') return next;

  const target = normalizeSendCommandTarget(next.target);
  const commands = SEND_COMMAND_COMMANDS_BY_TARGET[target];
  const requestedCommand = String(next.command || commands[0]).trim().toUpperCase();
  const command = commands.includes(requestedCommand) ? requestedCommand : commands[0];
  const remaining = Object.fromEntries(
    Object.entries(next).filter(([key]) => key !== 'target' && key !== 'command' && !RETIRED_INFERENCE_PARAMS.has(key)),
  );
  return {
    target, command, vla_rail_enabled: 'false', initial_pose_sync: 'true',
    initial_pose_sync_duration_s: '5', ...remaining,
  };
}

// Enum params surface as <select> dropdowns. Keep value lists in sync with
// the Python action definitions (send_command.COMMAND_MAP).
export const ENUM_PARAMS = {
  model: [
    'lerobot:act',
    'lerobot:diffusion',
    'lerobot:smolvla',
    'lerobot:xvla',
    'lerobot:pi0',
    'lerobot:pi05',
    'lerobot:molmoact2',
    'lerobot:vla_jepa',
    'lerobot:fastwam',
    'groot:n17',
    'groot',
    'lerobot',
  ],
  inference_mode: ['simulation', 'robot'],
  action_request_mode: ['async', 'sync'],
  acceleration_mode: ['pytorch', 'tensorrt_dit'],
};

export function enumOptionsFor(nodeType, key, params) {
  if (['ArucoDock', 'DockRetreat'].includes(nodeType) && key === 'footprint_profile') return ['circle', 'sg2_autonomy_base'];
  if (nodeType === 'ArucoDock' && key === 'image_mode') return ['raw', 'rectified'];
  if (nodeType === 'SendCommand') {
    if (key === 'target') return SEND_COMMAND_TARGETS;
    if (key === 'command') {
      return SEND_COMMAND_COMMANDS_BY_TARGET[
        normalizeSendCommandTarget(params.target)
      ];
    }
  }
  return ENUM_PARAMS[key];
}

// SendCommand inputs that are meaningful per command. Anything outside
// the set for the current command is rendered disabled — the value stays
// in params so flipping back to LOAD restores the user's earlier entries.
// 'command' itself is always editable.
export const SEND_COMMAND_ACTIVE_FIELDS = {
  LOAD: new Set([
    'target', 'command', 'model', 'policy_path', 'task_instruction',
    'inference_mode', 'action_request_mode', 'vla_rail_enabled', 'inference_hz', 'control_hz',
    'chunk_align_window_s', 'acceleration_mode', 'acceleration_engine_path',
  ]),
  // Resume can re-condition language mid-run; output mode is fixed by LOAD.
  RESUME: new Set(['target', 'command', 'task_instruction',
    'initial_pose_sync', 'initial_pose_sync_duration_s']),
  STOP: new Set(['target', 'command']),
  CLEAR: new Set(['target', 'command']),
};

export const SEND_COMMAND_DOCKER_ACTIVE_FIELDS = new Set(['target', 'command', 'model']);

// JointControl: each group's positions input is gated on its enable_*
// flag. enable_* toggles themselves + duration are always editable.
export const truthy = (v) => v === true || v === 'true';

export function isSendCommandFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'SendCommand') return false;
  if (normalizeSendCommandTarget(params.target) === 'DOCKER') {
    return !SEND_COMMAND_DOCKER_ACTIVE_FIELDS.has(key);
  }
  if (key === 'vla_rail_enabled' && (!String(params.model || '').startsWith('groot')
    || String(params.action_request_mode || 'async') !== 'async')) return true;
  const cmd = String(params.command || 'LOAD').toUpperCase();
  const active = SEND_COMMAND_ACTIVE_FIELDS[cmd];
  if (!active) return false;
  return !active.has(key);
}

export function isFieldDisabled(nodeType, key, params) {
  return (
    isSendCommandFieldDisabled(nodeType, key, params) ||
    isJointControlFieldDisabled(nodeType, key, params) ||
    isArmStateGateFieldDisabled(nodeType, key, params)
  );
}
