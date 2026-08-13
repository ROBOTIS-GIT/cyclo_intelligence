import {
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
});
