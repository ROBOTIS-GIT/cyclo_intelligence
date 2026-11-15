// Each architecture owns its TensorRT graph adapter and plan name.
export function tensorRtMode(serviceType, policyType) {
  if (serviceType === 'groot') return 'tensorrt_dit';
  if (serviceType === 'lerobot' && policyType === 'act') return 'tensorrt_act';
  return null;
}

export function isTensorRtMode(mode) {
  return ['tensorrt_dit', 'tensorrt_act'].includes(mode);
}

export function trtStatusUrl(serviceType, modelPath, enginePath = '', actionRequestMode = 'async') {
  const params = new URLSearchParams({ model_path: modelPath });
  if (enginePath) params.set('engine_path', enginePath);
  params.set('action_request_mode', actionRequestMode);
  return `/api/backends/${serviceType}/trt/status?${params.toString()}`;
}

// Engine artifacts are specific to their architecture; never reuse a DiT
// engine after changing the policy family to ACT (or the reverse).
export function normalizeInferenceAcceleration(taskInfo = {}) {
  const expected = taskInfo.serviceType === 'groot' && taskInfo.policyType !== 'n17'
    ? null : tensorRtMode(taskInfo.serviceType, taskInfo.policyType);
  const requested = String(taskInfo.accelerationMode || 'pytorch').trim();
  const valid = requested === expected;
  return {
    accelerationMode: valid ? requested : 'pytorch',
    accelerationEnginePath: valid ? String(taskInfo.accelerationEnginePath || '').trim() : '',
  };
}
