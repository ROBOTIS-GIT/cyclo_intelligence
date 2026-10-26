import { DEFAULT_PATHS } from '../../constants/paths';
import {
  DOWNLOAD_MODEL_BACKENDS,
  getDefaultDownloadPath,
  isManagedDownloadPath,
} from './hfDownloadPaths';

describe('hfDownloadPaths', () => {
  test('routes model downloads to backend model folders', () => {
    expect(getDefaultDownloadPath('model', 'lerobot')).toBe(
      DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH
    );
    expect(getDefaultDownloadPath('model', 'groot')).toBe(
      DEFAULT_PATHS.GROOT_CHECKPOINTS_PATH
    );
    expect(getDefaultDownloadPath('model', 'vitacformer')).toBe(
      DEFAULT_PATHS.VITACFORMER_CHECKPOINTS_PATH
    );
  });

  test('keeps dataset downloads under the rosbag2 root', () => {
    expect(getDefaultDownloadPath('dataset')).toBe(
      DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH
    );
  });

  test('tracks all automatic download destinations but leaves custom paths alone', () => {
    for (const option of DOWNLOAD_MODEL_BACKENDS) {
      expect(isManagedDownloadPath(option.path)).toBe(true);
    }
    expect(isManagedDownloadPath(DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH)).toBe(true);
    expect(isManagedDownloadPath(DEFAULT_PATHS.ROSBAG2_PATH)).toBe(true);
    expect(isManagedDownloadPath(`${DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH}/task_a`)).toBe(
      true
    );
    expect(isManagedDownloadPath(`${DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH}/repo_a`)).toBe(
      true
    );
    expect(isManagedDownloadPath(`${DEFAULT_PATHS.GROOT_CHECKPOINTS_PATH}/repo_a/`)).toBe(
      true
    );
    expect(isManagedDownloadPath(`${DEFAULT_PATHS.VITACFORMER_CHECKPOINTS_PATH}/repo_a`)).toBe(
      true
    );
    expect(isManagedDownloadPath(' /workspace/model ')).toBe(true);
    expect(isManagedDownloadPath('/workspace/custom_models')).toBe(false);
  });
});
