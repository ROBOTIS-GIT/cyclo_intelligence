import { DEFAULT_PATHS } from '../constants/paths';

export function getInferenceRecordingSessionId(path) {
  const root = DEFAULT_PATHS.ROSBAG2_PATH.replace(/\/+$/, '');
  const value = String(path || '').replace(/\/+$/, '');
  if (!value.startsWith(`${root}/`)) return '';
  const match = /^Task_([A-Za-z0-9][A-Za-z0-9_.-]{0,159})_inference_MCAP$/.exec(value.slice(root.length + 1));
  return match && !match[1].includes('..') ? match[1] : '';
}

export function inferenceRecordingPath(sessionId) {
  return sessionId ? `${DEFAULT_PATHS.ROSBAG2_PATH.replace(/\/+$/, '')}/Task_${sessionId}_inference_MCAP` : '';
}
