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

test('model-named folders round trip and legacy names are preserved', () => {
  for (const name of ['260928_1630_peanut', '260928_1630_peanut_02']) {
    const path = `/workspace/rosbag2/${name}`;
    expect(getInferenceRecordingSessionId(`${path}/`)).toBe(name);
    expect(inferenceRecordingPath(name)).toBe(path);
  }
  expect(inferenceRecordingPath('20260921_120000')).toBe(
    '/workspace/rosbag2/Task_20260921_120000_inference_MCAP');
  for (const name of ['260928_1630_peanut/0', '260928_1630_../escape',
    '260928_1630_', `260928_1630_${'a'.repeat(150)}`]) {
    expect(getInferenceRecordingSessionId(`/workspace/rosbag2/${name}`)).toBe('');
  }
});
