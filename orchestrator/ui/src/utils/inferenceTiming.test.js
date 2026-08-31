import {
  getInferenceTimingWarnings,
  hasIncompleteInferenceTiming,
} from './inferenceTiming';

describe('inference timing helpers', () => {
  test('treats blank and invalid timing as incomplete', () => {
    expect(hasIncompleteInferenceTiming({ inferenceHz: '', controlHz: 100 })).toBe(true);
    expect(hasIncompleteInferenceTiming({ inferenceHz: 15, controlHz: 0 })).toBe(true);
    expect(hasIncompleteInferenceTiming({ inferenceHz: 15, controlHz: 100 })).toBe(false);
  });

  test('warns without rejecting unusual timing', () => {
    expect(getInferenceTimingWarnings({ inferenceHz: 1515, controlHz: 200 }))
      .toEqual([
        'Dataset FPS is unusually high (1515). Confirm it matches the training dataset.',
        'Dataset FPS is higher than Control Hz, so action waypoints will be downsampled.',
      ]);
    expect(getInferenceTimingWarnings({ inferenceHz: 15, controlHz: 100 }))
      .toEqual([]);
  });
});
