import { DEFAULT_PATHS } from '../constants/paths';

const INFERENCE_FOLDER_PATTERN =
  /^Task_([A-Za-z0-9][A-Za-z0-9_.-]{0,159})_inference_MCAP$/;

const normalizePath = (value) => String(value || '').trim().replace(/\/+$/, '');

export function buildInferenceRecordingFolderPath(
  sessionId,
  recordingRoot = DEFAULT_PATHS.ROSBAG2_PATH
) {
  const normalizedSessionId = String(sessionId || '').trim();
  const normalizedRoot = normalizePath(recordingRoot);
  const folder = `${normalizedRoot}/Task_${normalizedSessionId}_inference_MCAP`;
  if (!getInferenceRecordingSessionId(folder, normalizedRoot)) {
    throw new Error('Invalid RL recording session ID');
  }
  return folder;
}

export function createInferenceRecordingFolder({
  now = new Date(),
  nonce = Math.floor(Math.random() * 0xFFFFFFFF)
    .toString(36)
    .padStart(7, '0'),
  recordingRoot = DEFAULT_PATHS.ROSBAG2_PATH,
} = {}) {
  const date = now instanceof Date ? now : new Date(now);
  if (Number.isNaN(date.getTime())) {
    throw new Error('Invalid RL recording timestamp');
  }
  const safeNonce = String(nonce || '');
  if (!/^[A-Za-z0-9-]{1,32}$/.test(safeNonce)) {
    throw new Error('Invalid RL recording nonce');
  }
  const timestamp = date.toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.(\d{3})Z$/, '_$1Z');
  return buildInferenceRecordingFolderPath(
    `${timestamp}_${safeNonce}`,
    recordingRoot
  );
}

export function getInferenceRecordingSessionId(
  folderPath,
  recordingRoot = DEFAULT_PATHS.ROSBAG2_PATH
) {
  const normalizedPath = normalizePath(folderPath);
  if (!normalizedPath) return '';

  const normalizedRoot = normalizePath(recordingRoot);
  const separatorIndex = normalizedPath.lastIndexOf('/');
  const parentPath = separatorIndex >= 0
    ? normalizedPath.slice(0, separatorIndex)
    : '';
  const folderName = separatorIndex >= 0
    ? normalizedPath.slice(separatorIndex + 1)
    : normalizedPath;

  if (parentPath !== normalizedRoot) return '';

  const match = INFERENCE_FOLDER_PATTERN.exec(folderName);
  if (!match || match[1].includes('..')) return '';
  return match[1];
}

export function getInferenceRecordingFolderName(folderPath) {
  const normalizedPath = normalizePath(folderPath);
  if (!normalizedPath) return '';
  return normalizedPath.slice(normalizedPath.lastIndexOf('/') + 1);
}
