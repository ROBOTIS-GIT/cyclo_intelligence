// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

export const SOURCE_TOPIC = '/source';
export const SOURCE_SERVICE = '/set_source';
export const PRESET_SERVICE = '/hand_preset_controller/set_preset';
export const SOURCE_TYPE = 'std_srvs/srv/SetBool';
export const PRESET_TYPE = 'robotis_interfaces/srv/SetPreset';
export const SOURCE_STALE_MS = 3000;

export function normalizeSource(value) {
  if (value === 'teleop') return 'teleop';
  if (value === 'external' || value === 'model_action') return 'external';
  return null;
}

export function modeLabel(value) {
  if (value === 'head_control') return 'Head';
  if (value === 'swerve') return 'Base';
  if (value === 'arm_control') return 'Arm';
  return value || 'Unknown';
}

export function presetRequest(target, left, right) {
  if (!['left', 'right', 'both'].includes(target)) throw new Error('Choose a target hand.');
  const valid = (value) => /^\d+$/.test(String(value)) && Number(value) >= 1 && Number(value) <= 65535;
  if ((target !== 'right' && !valid(left)) || (target !== 'left' && !valid(right))) {
    throw new Error('Preset IDs must be whole numbers from 1 to 65535.');
  }
  return {
    target_arm: target,
    left_preset_id: target === 'right' ? 0 : Number(left),
    right_preset_id: target === 'left' ? 0 : Number(right),
  };
}
