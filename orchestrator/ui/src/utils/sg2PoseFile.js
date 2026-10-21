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

import { captureSg2Pose, SG2_GROUPS } from './sg2PoseCapture';

export const POSE_FILE_PREFIX = 'sg2_pose_';
const fullPoseParams = { enable_head: 'true', enable_arms: 'true', enable_lift: 'true' };

export function poseFilename(name) {
  const clean = name.trim().replace(/\.xml$/i, '').replace(/^sg2_pose_/, '');
  if (!/^[\p{L}\p{N}_-]+$/u.test(clean)) {
    throw new Error('Use letters, numbers, hyphens or underscores for the pose name.');
  }
  return `${POSE_FILE_PREFIX}${clean}.xml`;
}

// A standard, executable single-node BT also serves as the portable pose file.
export function serializeSg2Pose(message, name) {
  const filename = poseFilename(name);
  const patch = captureSg2Pose('JointControl', fullPoseParams, message);
  const doc = new DOMParser().parseFromString('<root/>', 'text/xml');
  const root = doc.documentElement;
  root.setAttribute('BTCPP_format', '4');
  root.setAttribute('main_tree_to_execute', 'MainTree');
  root.setAttribute('cyclo_pose_version', '1');
  root.setAttribute('robot_type', 'ffw_sg2_rev1');
  const tree = doc.createElement('BehaviorTree');
  tree.setAttribute('ID', 'MainTree');
  const node = doc.createElement('JointControl');
  Object.entries({ name: filename.slice(0, -4), ...fullPoseParams, ...patch, duration: '2.0' })
    .forEach(([key, value]) => node.setAttribute(key, value));
  tree.appendChild(node);
  root.appendChild(tree);
  return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(doc) + '\n';
}

export function parseSg2Pose(content) {
  const doc = new DOMParser().parseFromString(content, 'text/xml');
  const root = doc.documentElement;
  const nodes = doc.querySelectorAll('JointControl');
  if (doc.querySelector('parsererror') || root.tagName !== 'root'
      || root.getAttribute('cyclo_pose_version') !== '1'
      || root.getAttribute('robot_type') !== 'ffw_sg2_rev1' || nodes.length !== 1) {
    throw new Error('This file is not a supported SG2 captured pose.');
  }
  const node = nodes[0];
  const message = { name: [], position: [] };
  Object.entries({ arm_left: 'left_positions', arm_right: 'right_positions', head: 'head_positions', lift: 'lift_position' })
    .forEach(([group, field]) => {
      const names = SG2_GROUPS[group];
      const side = group.replace('arm_', '');
      if (group.startsWith('arm_') && node.getAttribute(`${side}_joint_names`) !== names.join(', ')) {
        throw new Error('Invalid SG2 pose joint order.');
      }
      const values = (node.getAttribute(field) || '').split(',').map((v) => v.trim());
      if (values.length !== names.length || values.some((v) => !v || !Number.isFinite(Number(v)))) {
        throw new Error(`Invalid pose positions: ${field}`);
      }
      message.name.push(...names);
      message.position.push(...values.map(Number));
    });
  return message;
}

export function applySg2PoseFile(nodeType, params, message) {
  const defaults = {};
  if (nodeType === 'ArmStateGate'
      && !String(params.left_target_joints || '').trim() && !String(params.right_target_joints || '').trim()) {
    // With no selection, observe the seven arm joints per side. Gripper event
    // detection remains independent, so a close/open gesture can still finish.
    defaults.left_target_joints = SG2_GROUPS.arm_left.slice(0, 7).join(', ');
    defaults.right_target_joints = SG2_GROUPS.arm_right.slice(0, 7).join(', ');
  }
  if (nodeType === 'JointControl' && !String(params.groups || '').trim()
      && !['enable_head', 'enable_arms', 'enable_lift'].some((key) =>
        ['true', '1', 'yes', 'on'].includes(String(params[key]).toLowerCase()))) {
    defaults.enable_arms = 'true';
  }
  return { ...defaults, ...captureSg2Pose(nodeType, { ...params, ...defaults }, message) };
}
