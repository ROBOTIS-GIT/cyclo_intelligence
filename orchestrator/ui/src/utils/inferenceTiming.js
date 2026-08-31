export const UNUSUAL_DATASET_FPS = 120;
export const UNUSUAL_CONTROL_HZ = 1000;

const positiveFiniteNumber = (value) => {
  if (value === '' || value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
};

export const hasIncompleteInferenceTiming = (taskInfo = {}) => (
  positiveFiniteNumber(taskInfo.inferenceHz) == null ||
  positiveFiniteNumber(taskInfo.controlHz) == null
);

export const getInferenceTimingWarnings = (taskInfo = {}) => {
  const datasetFps = positiveFiniteNumber(taskInfo.inferenceHz);
  const controlHz = positiveFiniteNumber(taskInfo.controlHz);
  if (datasetFps == null || controlHz == null) return [];

  const warnings = [];
  if (datasetFps > UNUSUAL_DATASET_FPS) {
    warnings.push(
      `Dataset FPS is unusually high (${datasetFps}). Confirm it matches the training dataset.`
    );
  }
  if (controlHz > UNUSUAL_CONTROL_HZ) {
    warnings.push(
      `Control Hz is unusually high (${controlHz} Hz). Confirm the robot controller supports it.`
    );
  }
  if (datasetFps > controlHz) {
    warnings.push(
      'Dataset FPS is higher than Control Hz, so action waypoints will be downsampled.'
    );
  }
  return warnings;
};
