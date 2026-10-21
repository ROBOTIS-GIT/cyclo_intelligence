// Each architecture owns its TensorRT graph adapter and plan name.
export function tensorRtMode(serviceType, policyType) {
  if (serviceType === 'groot') return 'tensorrt_dit';
  if (serviceType === 'lerobot' && policyType === 'act') return 'tensorrt_act';
  return null;
}

export function isTensorRtMode(mode) {
  return ['tensorrt_dit', 'tensorrt_act'].includes(mode);
}

export function trtStatusUrl(serviceType, modelPath, enginePath = '') {
  const params = new URLSearchParams({ model_path: modelPath });
  if (enginePath) params.set('engine_path', enginePath);
  return `/api/backends/${serviceType}/trt/status?${params.toString()}`;
}
