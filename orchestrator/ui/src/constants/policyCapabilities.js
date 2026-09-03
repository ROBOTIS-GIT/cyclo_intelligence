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
export const POLICY_REQUIRES_INSTRUCTION = {
  'groot:n17': true,
  'lerobot:act': false,
  'lerobot:smolvla': true,
  'lerobot:xvla': true,
  'lerobot:pi0': true,
  'lerobot:pi05': true,
  'lerobot:diffusion': false,
  'lerobot:multi_task_dit': true,
};

export const DEFAULT_REQUIRES_INSTRUCTION = false;

const RLT_INFERENCE_POLICIES = new Set([
  'groot:n17',
]);

const TENSORRT_INFERENCE_POLICIES = new Set([
  'groot:n17',
]);

const TT_RTC_INFERENCE_POLICIES = new Set([
  'groot:n17',
]);

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

/**
 * Policies whose frozen feature boundary is compatible with the staged RLT
 * bundle contract. Runtime routing is connected independently from this UI
 * capability so unsupported policies never expose a misleading RLT control.
 */
export function supportsRltInference(serviceType, policyType) {
  return RLT_INFERENCE_POLICIES.has(`${serviceType}:${policyType}`);
}

/**
 * Policies whose runtime implements the GR00T DiT TensorRT contract.
 * Keep this pair-based rather than backend-wide so a future GR00T policy
 * cannot accidentally inherit an engine built for N1.7.
 */
export function supportsTensorRtInference(serviceType, policyType) {
  return TENSORRT_INFERENCE_POLICIES.has(`${serviceType}:${policyType}`);
}

/**
 * Policies trained with the prefix-conditioned Training-Time RTC contract.
 * Keep this explicit so an ordinary async policy cannot expose TT-RTC by
 * inheriting a backend-wide capability.
 */
export function supportsTtRtcInference(serviceType, policyType) {
  return TT_RTC_INFERENCE_POLICIES.has(`${serviceType}:${policyType}`);
}
