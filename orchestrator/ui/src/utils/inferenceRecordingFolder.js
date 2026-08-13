import { DEFAULT_PATHS } from '../constants/paths';

const INFERENCE_FOLDER_PATTERN =
  /^Task_([A-Za-z0-9][A-Za-z0-9_.-]{0,159})_inference_MCAP$/;

const normalizePath = (value) => String(value || '').trim().replace(/\/+$/, '');

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
