import { isFieldDisabled, normalizeSendCommandParams } from './parameterPolicy';

test.each(['true', 'false'])('LOAD pose sync is editable and preserves %s', (value) => {
  const params = normalizeSendCommandParams({ target: 'INFERENCE', command: 'LOAD', initial_pose_sync: value }, 'SendCommand');
  expect(isFieldDisabled('SendCommand', 'initial_pose_sync', params)).toBe(false);
  expect(isFieldDisabled('SendCommand', 'initial_pose_sync_duration_s', params)).toBe(false);
  expect(params.initial_pose_sync).toBe(value);
});
