import {
  buildTopicRotationMap,
  normalizeColumnWeights,
  normalizeRotationDeg,
} from './ImageGrid';
import { getQuarterTurnCoverSize } from './ImageGridCell';

describe('ImageGrid rotation helpers', () => {
  test('normalizes camera rotations to css-friendly degrees', () => {
    expect(normalizeRotationDeg(0)).toBe(0);
    expect(normalizeRotationDeg(270)).toBe(270);
    expect(normalizeRotationDeg(-90)).toBe(270);
    expect(normalizeRotationDeg(450)).toBe(90);
    expect(normalizeRotationDeg(undefined)).toBe(0);
  });

  test('maps rotation_deg_list by matching image_topic_list index', () => {
    expect(buildTopicRotationMap(
      ['/zed/image/compressed', '/camera_left/image/compressed'],
      [0, 270]
    )).toEqual({
      '/zed/image/compressed': 0,
      '/camera_left/image/compressed': 270,
    });
  });

  test('defaults missing rotation entries to zero', () => {
    expect(buildTopicRotationMap(
      ['/zed/image/compressed', '/camera_left/image/compressed'],
      [270]
    )).toEqual({
      '/zed/image/compressed': 270,
      '/camera_left/image/compressed': 0,
    });
  });

  test('accepts complete positive camera weights and rejects unsafe partial layouts', () => {
    expect(normalizeColumnWeights([4, 5, 4])).toEqual([4, 5, 4]);
    expect(normalizeColumnWeights([4, 5])).toBeNull();
    expect(normalizeColumnWeights([4, 0, 4])).toBeNull();
  });

  test('sizes a quarter-turn wrapper from the real camera cell dimensions', () => {
    expect(getQuarterTurnCoverSize(480, 320)).toEqual({ width: 320, height: 480 });
    expect(getQuarterTurnCoverSize(0, 320)).toBeNull();
  });
});
