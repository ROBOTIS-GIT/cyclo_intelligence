import { DEFAULT_PATHS } from '../../constants/paths';

export const DOWNLOAD_MODEL_BACKENDS = [
  {
    value: 'lerobot',
    label: 'LeRobot',
    path: DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH,
  },
  {
    value: 'groot',
    label: 'GR00T',
    path: DEFAULT_PATHS.GROOT_CHECKPOINTS_PATH,
  },
  {
    value: 'rldx',
    label: 'RLDX',
    path: DEFAULT_PATHS.RLDX_CHECKPOINTS_PATH,
  },
];

const MANAGED_DOWNLOAD_PATHS = new Set([
  DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH,
  DEFAULT_PATHS.ROSBAG2_PATH,
  DEFAULT_PATHS.POLICY_CHECKPOINTS_PATH,
  DEFAULT_PATHS.HF_MODEL_DOWNLOAD_PATH,
  ...DOWNLOAD_MODEL_BACKENDS.map((option) => option.path),
]
  .map((path) => (path || '').trim().replace(/\/+$/, ''))
  .filter(Boolean));

export const getDefaultDownloadPath = (downloadType, modelBackend = 'lerobot') => {
  if (downloadType !== 'model') {
    return DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH;
  }

  return (
    DOWNLOAD_MODEL_BACKENDS.find((option) => option.value === modelBackend)?.path ||
    DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH
  );
};

export const isManagedDownloadPath = (path) => {
  const normalizedPath = (path || '').trim().replace(/\/+$/, '');
  if (!normalizedPath) return false;

  return Array.from(MANAGED_DOWNLOAD_PATHS).some(
    (managedPath) =>
      normalizedPath === managedPath || normalizedPath.startsWith(`${managedPath}/`)
  );
};
