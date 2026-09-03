import {
  requiresInstruction,
  supportsRltInference,
  supportsTensorRtInference,
  supportsTtRtcInference,
} from './policyCapabilities';
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

  test('exposes RLT inference settings only for GR00T N1.7', () => {
    expect(supportsRltInference('groot', 'n17')).toBe(true);
    expect(supportsRltInference('groot', 'future')).toBe(false);
    expect(supportsRltInference('lerobot', 'pi05')).toBe(false);
    expect(supportsRltInference('lerobot', 'act')).toBe(false);
  });

  test('exposes TensorRT inference only for GR00T N1.7', () => {
    expect(supportsTensorRtInference('groot', 'n17')).toBe(true);
    expect(supportsTensorRtInference('groot', 'future')).toBe(false);
    expect(supportsTensorRtInference('lerobot', 'act')).toBe(false);
    expect(supportsTensorRtInference('lerobot', 'multi_task_dit')).toBe(false);
  });

  test('exposes TT-RTC inference only for GR00T N1.7', () => {
    expect(supportsTtRtcInference('groot', 'n17')).toBe(true);
    expect(supportsTtRtcInference('groot', 'future')).toBe(false);
    expect(supportsTtRtcInference('lerobot', 'act')).toBe(false);
    expect(supportsTtRtcInference('lerobot', 'multi_task_dit')).toBe(false);
  });
});
