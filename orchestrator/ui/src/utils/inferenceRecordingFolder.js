import { DEFAULT_PATHS } from '../constants/paths';

const isModelFolder = (name) => /^\d{6}_\d{4}_[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(name) &&
  name.length <= 160 && !name.includes('..');

export function getInferenceRecordingSessionId(path) {
  const root = DEFAULT_PATHS.ROSBAG2_PATH.replace(/\/+$/, '');
  const value = String(path || '').replace(/\/+$/, '');
  if (!value.startsWith(`${root}/`)) return '';
  const name = value.slice(root.length + 1);
  if (isModelFolder(name)) return name;
  const match = /^Task_([A-Za-z0-9][A-Za-z0-9_.-]{0,159})_inference_MCAP$/.exec(name);
  return match && !match[1].includes('..') ? match[1] : '';
}

export function inferenceRecordingPath(sessionId) {
  if (!sessionId) return '';
  const name = isModelFolder(sessionId) ? sessionId : `Task_${sessionId}_inference_MCAP`;
  return `${DEFAULT_PATHS.ROSBAG2_PATH.replace(/\/+$/, '')}/${name}`;
}
