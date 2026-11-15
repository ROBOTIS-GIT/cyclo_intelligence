import { DEFAULT_PATHS } from '../constants/paths';

const INFERENCE_FOLDER_PATTERN =
  /^Task_([A-Za-z0-9][A-Za-z0-9_.-]{0,159})_inference_MCAP$/;
const RECORDING_FOLDER_PATTERN = /^Task_(.+)_MCAP$/u;

const normalizePath = (value) => String(value || '').trim().replace(/\/+$/, '');
const hasUnsafePathCharacters = (value) => (
  String(value || '').includes('/') ||
  String(value || '').includes('\\') ||
  [...String(value || '')].some((char) => {
    const code = char.codePointAt(0);
    return code < 32 || code === 127;
  })
);

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

export function getRecordingDatasetFolderInfo(
  folderPath,
  recordingRoot = DEFAULT_PATHS.ROSBAG2_PATH
) {
  const normalizedPath = normalizePath(folderPath);
  if (!normalizedPath) return null;

  const normalizedRoot = normalizePath(recordingRoot);
  const separatorIndex = normalizedPath.lastIndexOf('/');
  const parentPath = separatorIndex >= 0
    ? normalizedPath.slice(0, separatorIndex)
    : '';
  const folderName = separatorIndex >= 0
    ? normalizedPath.slice(separatorIndex + 1)
    : normalizedPath;

  if (parentPath !== normalizedRoot) return null;
  const match = RECORDING_FOLDER_PATTERN.exec(folderName);
  if (
    !match ||
    folderName.length > 240 ||
    match[1].includes('..') ||
    hasUnsafePathCharacters(folderName)
  ) {
    return null;
  }

  const inferenceMatch = INFERENCE_FOLDER_PATTERN.exec(folderName);
  const isInference = Boolean(
    inferenceMatch && !inferenceMatch[1].includes('..')
  );
  return {
    folderPath: normalizedPath,
    folderName,
    kind: isInference ? 'inference' : 'record',
    sessionId: isInference ? inferenceMatch[1] : '',
  };
}

export function createRecordRecordingFolder(
  datasetName,
  {
    now = new Date(),
    recordingRoot = DEFAULT_PATHS.ROSBAG2_PATH,
  } = {}
) {
  const date = now instanceof Date ? now : new Date(now);
  if (Number.isNaN(date.getTime())) {
    throw new Error('Invalid recording timestamp');
  }
  const safeName = String(datasetName || '')
    .normalize('NFKC')
    .trim()
    .replace(/\s+/gu, '_');
  if (
    !safeName ||
    safeName.length > 120 ||
    safeName.includes('..') ||
    hasUnsafePathCharacters(safeName)
  ) {
    throw new Error('Dataset name must not contain path separators or ".."');
  }
  const timestamp = date.toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.(\d{3})Z$/, '_$1Z');
  const folderPath = `${normalizePath(recordingRoot)}/Task_${timestamp}_${safeName}_MCAP`;
  if (!getRecordingDatasetFolderInfo(folderPath, recordingRoot)) {
    throw new Error('Invalid Record dataset name');
  }
  return folderPath;
}

export function createNamedRecordingFolder(
  datasetName,
  recordingRoot = DEFAULT_PATHS.ROSBAG2_PATH
) {
  const normalizedName = String(datasetName || '')
    .normalize('NFKC')
    .trim()
    .replace(/\s+/gu, '_');
  if (
    !normalizedName ||
    normalizedName.length > 200 ||
    normalizedName.includes('..') ||
    hasUnsafePathCharacters(normalizedName)
  ) {
    throw new Error('Dataset name must not contain path separators or ".."');
  }

  const folderName = normalizedName.startsWith('Task_') &&
    normalizedName.endsWith('_MCAP')
    ? normalizedName
    : `Task_${normalizedName}_MCAP`;
  const folderPath = `${normalizePath(recordingRoot)}/${folderName}`;
  if (!getRecordingDatasetFolderInfo(folderPath, recordingRoot)) {
    throw new Error('Invalid recording dataset name');
  }
  return folderPath;
}

export function getRecordingDatasetName(
  folderPath,
  recordingRoot = DEFAULT_PATHS.ROSBAG2_PATH
) {
  const info = getRecordingDatasetFolderInfo(folderPath, recordingRoot);
  if (!info) return '';
  return info.folderName.slice('Task_'.length, -'_MCAP'.length);
}

export function resolveRecordingDatasetFolder(value, options = {}) {
  const normalized = normalizePath(value);
  if (!normalized) return '';
  const recordingRoot = options.recordingRoot || DEFAULT_PATHS.ROSBAG2_PATH;
  const directPath = normalized.includes('/')
    ? normalized
    : `${normalizePath(recordingRoot)}/${normalized}`;
  const existingFormat = getRecordingDatasetFolderInfo(
    directPath,
    recordingRoot
  );
  if (existingFormat) return existingFormat.folderPath;
  if (normalized.includes('/')) {
    throw new Error(
      'Dataset must be a Task_*_MCAP folder directly under /workspace/rosbag2'
    );
  }
  return createRecordRecordingFolder(normalized, options);
}

export function getInferenceRecordingSessionId(
  folderPath,
  recordingRoot = DEFAULT_PATHS.ROSBAG2_PATH
) {
  const info = getRecordingDatasetFolderInfo(folderPath, recordingRoot);
  return info?.kind === 'inference' ? info.sessionId : '';
}

export function getInferenceRecordingFolderName(folderPath) {
  const normalizedPath = normalizePath(folderPath);
  if (!normalizedPath) return '';
  return normalizedPath.slice(normalizedPath.lastIndexOf('/') + 1);
}
