import { requiresInstruction } from './policyCapabilities';
import { MODEL_OPTIONS } from '../components/InferenceModelSelector';

describe('policy capabilities', () => {
  test('MultiTaskDiT Flow inference requires a task instruction', () => {
    expect(requiresInstruction('lerobot', 'multi_task_dit')).toBe(true);
  });

  test('registers the Flow checkpoint under its explicit policy type', () => {
    expect(MODEL_OPTIONS).toContainEqual(expect.objectContaining({
      value: 'lerobot:multi_task_dit',
      label: 'Diffusion Transformer (Flow)',
      serviceType: 'lerobot',
      policyType: 'multi_task_dit',
    }));
  });
});
