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

export const FALLBACK_SCHEMA_VERSION = '1.1';

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
        name: 'left_positions',
        type: 'string',
        default: '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
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
    tag: 'SendCommand',
    category: 'action',
    ports: [
      { name: 'command', type: 'string', default: 'LOAD' },
      { name: 'model', type: 'string', default: 'lerobot:act' },
      { name: 'policy_path', type: 'string', default: '' },
      { name: 'task_instruction', type: 'string', default: '' },
      { name: 'inference_mode', type: 'string', default: 'simulation' },
      { name: 'action_request_mode', type: 'string', default: 'async' },
      { name: 'inference_hz', type: 'number', default: '15' },
      { name: 'control_hz', type: 'number', default: '100' },
      { name: 'chunk_align_window_s', type: 'number', default: '0.3' },
      { name: 'remote_host', type: 'string', default: '' },
      { name: 'remote_port', type: 'number', default: '0' },
      { name: 'remote_timeout_ms', type: 'number', default: '0' },
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
