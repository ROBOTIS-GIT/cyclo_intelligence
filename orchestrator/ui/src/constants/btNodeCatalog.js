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
  // Loop.max_iterations: 0 (or any non-positive) means "loop forever";
  // a positive integer caps the run to that many child-success cycles.
  { tag: 'Loop',     category: 'control', params: { max_iterations: '0' } },

  // Actions — defaults mirror Python __init__ values
  { tag: 'Rotate', category: 'action', params: { angle_deg: '90.0' } },
  // JointControl runs one or more sub-groups (head / arms / lift) at
  // once. Each enable_* toggle decides whether its positions get
  // published; BTParamPanel greys out the matching positions input when
  // its toggle is off. position_threshold sticks to the backend default
  // (0.01) — no UI knob.
  {
    tag: 'JointControl',
    category: 'action',
    params: {
      enable_head: 'true',
      head_positions: '0.0, 0.0',
      enable_arms: 'false',
      left_positions: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
      right_positions: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
      enable_lift: 'false',
      lift_position: '0.0',
      duration: '2.0',
    },
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
];

export const findNodeMeta = (tag) =>
  BT_NODE_CATALOG.find((n) => n.tag === tag);

export const isControlTag = (tag) =>
  findNodeMeta(tag)?.category === 'control';

export const CATALOG_BY_CATEGORY = {
  control: BT_NODE_CATALOG.filter((n) => n.category === 'control'),
  action: BT_NODE_CATALOG.filter((n) => n.category === 'action'),
};
