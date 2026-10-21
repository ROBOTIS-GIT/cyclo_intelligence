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

// SG2 controller order from shared/robot_configs/ffw_sg2_rev1_config.yaml.
// URDF: shared/robot_configs/urdf/ffw_sg2_follower.urdf, based on
// ROBOTIS-GIT/ai_worker ffw_description/urdf/ffw_sg2_rev1_follower.
// Capture measured joint coordinates unchanged: revolute joints (including
// gripper *_joint1) are radians; lift_joint is prismatic, in metres.
// Mimic fingers, wheels and fixed joints are not controller targets.
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


export const SG2_GROUPS = {
  arm_left: SG2_LEFT_JOINTS,
  arm_right: SG2_RIGHT_JOINTS,
  head: ['head_joint1', 'head_joint2'],
  lift: ['lift_joint'],
};
const csv = (value) => String(value || '').split(',').map((s) => s.trim()).filter(Boolean);
const enabled = (value) => value === true || ['true', '1', 'yes', 'on'].includes(String(value).toLowerCase());

export function captureSg2Pose(nodeType, params, message) {
  if (!Array.isArray(message?.name) || !Array.isArray(message?.position)) {
    throw new Error('Invalid /joint_states message.');
  }
  const values = new Map();
  message.name.forEach((name, i) => {
    if (values.has(name)) throw new Error(`Duplicate joint in /joint_states: ${name}`);
    values.set(name, message.position[i]);
  });
  const positions = (joints, allowed) => joints.map((name) => {
    if (!allowed.includes(name)) throw new Error(`Unsupported SG2 joint: ${name}`);
    if (!Number.isFinite(values.get(name))) throw new Error(`Missing or invalid joint state: ${name}`);
    return String(values.get(name));
  }).join(', ');
  const patch = {};
  if (nodeType === 'JointControl') {
    const groups = csv(params.groups);
    if (groups.length) {
      patch.positions = groups.map((group) => {
        if (!SG2_GROUPS[group]) throw new Error(`Unsupported SG2 group: ${group}`);
        return positions(SG2_GROUPS[group], SG2_GROUPS[group]);
      }).join('; ');
    } else {
      if (enabled(params.enable_head)) patch.head_positions = positions(SG2_GROUPS.head, SG2_GROUPS.head);
      if (enabled(params.enable_lift)) patch.lift_position = positions(SG2_GROUPS.lift, SG2_GROUPS.lift);
      if (enabled(params.enable_arms)) {
        ['left', 'right'].forEach((side) => {
          const allowed = SG2_GROUPS[`arm_${side}`];
          // Empty names mean full configured group in JointControl.from_xml_params.
          const names = csv(params[`${side}_joint_names`]);
          const selected = names.length ? names : allowed;
          patch[`${side}_joint_names`] = selected.join(', ');
          patch[`${side}_positions`] = positions(selected, allowed);
        });
      }
    }
  } else if (nodeType === 'ArmStateGate') {
    ['left', 'right'].forEach((side) => {
      const names = csv(params[`${side}_target_joints`]);
      if (names.length) patch[`${side}_target_positions`] = positions(names, SG2_GROUPS[`arm_${side}`]);
    });
  }
  if (!Object.keys(patch).length) {
    throw new Error('Enable a joint group or select target joints before capturing.');
  }
  return patch;
}
