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

// Catalog of node types the BT Manager UI can spawn.
//
// Keep parameter defaults in sync with the Python action __init__ defaults
// (orchestrator/bt/actions/*.py). XML attributes are always strings, so we
// store them as strings here too — BTParamPanel handles per-key type coercion
// via its NUMBER_PARAMS / BOOL_PARAMS / ENUM_PARAMS maps.

export const BT_NODE_CATALOG = [
  // Controls
  { tag: 'Sequence', category: 'control', params: {} },
  { tag: 'Loop',     category: 'control', params: {} },

  // Actions — defaults mirror Python __init__ values
  { tag: 'Rotate', category: 'action', params: { angle_deg: '90.0' } },
  {
    tag: 'MoveHead',
    category: 'action',
    params: { head_positions: '0.0, 0.0', duration: '5.0' },
  },
  {
    tag: 'MoveArms',
    category: 'action',
    params: {
      left_positions: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
      right_positions: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
      duration: '2.0',
    },
  },
  {
    tag: 'MoveLift',
    category: 'action',
    params: { lift_position: '0.0', duration: '5.0' },
  },
  {
    tag: 'SendCommand',
    category: 'action',
    params: {
      command: 'LOAD',
      model: 'groot',
      policy_path: '',
      task_instruction: '',
      inference_hz: '15',
      control_hz: '100',
      chunk_align_window_s: '0.3',
    },
  },
  { tag: 'Wait', category: 'action', params: { duration: '5.0' } },
  { tag: 'GripperClosed', category: 'action', params: {} },
  { tag: 'GripperOpened', category: 'action', params: {} },
  { tag: 'ArmsStatic',    category: 'action', params: {} },
  {
    tag: 'PoseGripperChange',
    category: 'action',
    params: {
      left_positions: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
      right_positions: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
      tolerance: '0.1',
      check_delay: '5.0',
      gripper_check: 'none',
    },
  },
];

export const findNodeMeta = (tag) =>
  BT_NODE_CATALOG.find((n) => n.tag === tag);

export const isControlTag = (tag) =>
  findNodeMeta(tag)?.category === 'control';

export const CATALOG_BY_CATEGORY = {
  control: BT_NODE_CATALOG.filter((n) => n.category === 'control'),
  action: BT_NODE_CATALOG.filter((n) => n.category === 'action'),
};
