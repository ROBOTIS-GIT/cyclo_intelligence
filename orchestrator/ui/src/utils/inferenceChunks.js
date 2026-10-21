export function defaultInferenceChunkSize(serviceType, policyType) {
  if (serviceType === 'groot') return 24;
  if (serviceType === 'lerobot' && policyType === 'act') return 100;
  return 0;
}

export function inferenceChunkSizeError(info) {
  if (!defaultInferenceChunkSize(info.serviceType, info.policyType)) return '';
  const value = info.actionChunkSize;
  if (value === '' || value == null) return 'Enter Chunk Size.';
  const n = Number(value);
  return Number.isInteger(n) && n >= 1 && n <= 65535
    ? '' : 'Chunk Size must be an integer from 1 to 65535.';
}
