// InferenceCommand policy-update wire IDs; 7/8 are supervision commands.
// Keep aligned with policy/common/runtime/engine_process/protocol.py.
export const RLT_COMMAND = Object.freeze({
  STATUS: 21, AUTO_ON: 22, AUTO_OFF: 23, APPLY: 24,
  TRAIN_ON: 25, TRAIN_OFF: 26, DATASETS: 27, SAVE: 28,
});
