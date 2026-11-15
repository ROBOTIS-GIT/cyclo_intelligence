import {
  buildInferenceRecordingFolderPath,
  createRecordRecordingFolder,
  createNamedRecordingFolder,
  createInferenceRecordingFolder,
  getInferenceRecordingFolderName,
  getInferenceRecordingSessionId,
  getRecordingDatasetFolderInfo,
  getRecordingDatasetName,
  resolveRecordingDatasetFolder,
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

  test('accepts standard Record datasets directly under rosbag2', () => {
    const folder =
      '/workspace/rosbag2/Task_700009_OMX_Insert_F2_Intern_MCAP';

    expect(getRecordingDatasetFolderInfo(folder)).toEqual({
      folderPath: folder,
      folderName: 'Task_700009_OMX_Insert_F2_Intern_MCAP',
      kind: 'record',
      sessionId: '',
    });
    expect(resolveRecordingDatasetFolder(
      'Task_700009_OMX_Insert_F2_Intern_MCAP'
    )).toBe(folder);
  });

  test('turns a user-entered name into a standard Record dataset path', () => {
    expect(createRecordRecordingFolder('F2 조립 작업', {
      now: new Date('2026-09-11T08:30:45.123Z'),
    })).toBe(
      '/workspace/rosbag2/Task_20260911T083045_123Z_F2_조립_작업_MCAP'
    );
  });

  test('maps an inference dataset name to one stable rosbag folder', () => {
    const folder = createNamedRecordingFolder('F2 조립 작업');

    expect(folder).toBe('/workspace/rosbag2/Task_F2_조립_작업_MCAP');
    expect(createNamedRecordingFolder('F2 조립 작업')).toBe(folder);
    expect(getRecordingDatasetName(folder)).toBe('F2_조립_작업');
    expect(createNamedRecordingFolder('Task_existing_MCAP')).toBe(
      '/workspace/rosbag2/Task_existing_MCAP'
    );
  });

  test('rejects nested and traversal-like Record targets', () => {
    expect(getRecordingDatasetFolderInfo(
      '/workspace/rosbag2/archive/Task_700009_record_MCAP'
    )).toBeNull();
    expect(() => resolveRecordingDatasetFolder('../escape')).toThrow();
  });
});
