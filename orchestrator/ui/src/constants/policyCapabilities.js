// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Whitelist of (service_type, policy_type) pairs that require a
// task_instruction. Missing entries fall back to
// DEFAULT_REQUIRES_INSTRUCTION (false → hide the field).
//
// Key shape: '<service_type>:<policy_type>' or '<service_type>:*'
// Exact key beats wildcard.
//
// Grow this map one line at a time as policies are validated end-to-end.
// Examples for follow-up PRs:
//   'lerobot:smolvla': true,
//   'lerobot:pi0':     true,
//   'lerobot:pi05':    true,
//   'lerobot:xvla':    true,
//   'lerobot:diffusion': false,  // already implicit but explicit is fine
export const POLICY_REQUIRES_INSTRUCTION = {
  'groot:n16': true,
  'lerobot:act': false,
};

export const DEFAULT_REQUIRES_INSTRUCTION = false;

export function requiresInstruction(serviceType, policyType) {
  const exact = `${serviceType}:${policyType}`;
  if (exact in POLICY_REQUIRES_INSTRUCTION) {
    return POLICY_REQUIRES_INSTRUCTION[exact];
  }
  const wild = `${serviceType}:*`;
  if (wild in POLICY_REQUIRES_INSTRUCTION) {
    return POLICY_REQUIRES_INSTRUCTION[wild];
  }
  return DEFAULT_REQUIRES_INSTRUCTION;
}
