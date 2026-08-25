import {
  buildInferenceRecordingFolderPath,
  createInferenceRecordingFolder,
  getInferenceRecordingFolderName,
  getInferenceRecordingSessionId,
} from './inferenceRecordingFolder';

describe('inference recording folder', () => {
  test('extracts a session ID only from a root inference dataset folder', () => {
    const folder =
      '/workspace/rosbag2/Task_20260812_082903_inference_MCAP';

    expect(getInferenceRecordingSessionId(folder))
      .toBe('20260812_082903');
    expect(getInferenceRecordingFolderName(folder))
      .toBe('Task_20260812_082903_inference_MCAP');
  });

  test('rejects non-inference, nested, and traversal-like folders', () => {
    expect(getInferenceRecordingSessionId(
      '/workspace/rosbag2/Task_1_record_MCAP'
    )).toBe('');
    expect(getInferenceRecordingSessionId(
      '/workspace/rosbag2/archive/Task_1_inference_MCAP'
    )).toBe('');
    expect(getInferenceRecordingSessionId(
      '/workspace/rosbag2/Task_bad..id_inference_MCAP'
    )).toBe('');
  });

  test('builds a deterministic collision-safe recording folder', () => {
    const folder = createInferenceRecordingFolder({
      now: new Date('2026-08-21T05:45:12.123Z'),
      nonce: 'a1b2c3d4',
    });

    expect(folder).toBe(
      '/workspace/rosbag2/Task_20260821T054512_123Z_a1b2c3d4_inference_MCAP'
    );
    expect(getInferenceRecordingSessionId(folder))
      .toBe('20260821T054512_123Z_a1b2c3d4');
    expect(buildInferenceRecordingFolderPath('round_01'))
      .toBe('/workspace/rosbag2/Task_round_01_inference_MCAP');
  });

  test('rejects an empty or traversal-like generated session suffix', () => {
    expect(() => createInferenceRecordingFolder({ nonce: '../' })).toThrow(
      'Invalid RL recording nonce'
    );
    expect(() => buildInferenceRecordingFolderPath('bad..session')).toThrow(
      'Invalid RL recording session ID'
    );
  });
});
