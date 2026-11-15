export const RLT_TRAINING_PHASES = [
  { value: 'actor_warmup', label: 'Actor warmup' },
  { value: 'critic_warmup', label: 'Critic warmup' },
  { value: 'rl', label: 'RL training' },
];

export function rltPhaseBlockedReason(status, phase, initializationMode) {
  if (status?.phase_contract_version !== 2) {
    return 'The training API does not support explicit RLT phases. Update the API before starting.';
  }
  if (phase !== 'actor_warmup' && initializationMode !== 'resume') {
    return 'Critic warmup and RL training require Resume from an MLP bundle.';
  }
  if (!status?.supported_training_phases?.includes(phase)) {
    return status?.unavailable_training_phases?.[phase] || 'This training phase is not available yet.';
  }
  return '';
}
