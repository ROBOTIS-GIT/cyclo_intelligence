import { getInferenceRecordingSessionId, inferenceRecordingPath } from './inferenceRecordingFolder';

test('only a direct inference recording folder is selectable', () => {
  expect(getInferenceRecordingSessionId('/workspace/rosbag2/Task_20260921_120000_inference_MCAP/')).toBe('20260921_120000');
  for (const path of ['/workspace/rosbag2', '/tmp/Task_a_inference_MCAP',
    '/workspace/rosbag2/Task_a_inference_MCAP/0', '/workspace/rosbag2/Task_.._inference_MCAP',
    '/workspace/rosbag2/Task_a_record_MCAP']) {
    expect(getInferenceRecordingSessionId(path)).toBe('');
  }
  expect(inferenceRecordingPath('')).toBe('');
});
