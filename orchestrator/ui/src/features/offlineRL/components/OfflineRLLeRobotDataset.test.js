import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import editDatasetReducer, {
  setConversionStatus,
} from '../../editDataset/editDatasetSlice';
import {
  deleteOfflineRLDatasetEpisodes,
  getOfflineRLDatasetEpisodeData,
  getOfflineRLDatasetInfo,
  getOfflineRLDatasets,
  getOfflineRLStatus,
} from '../../../utils/offlineRlApi';
import offlineRLReducer, {
  setOfflineRLConversionDestinationPath,
  selectOfflineRLDatasetPaths,
  setOfflineRLConvertedDatasetPaths,
  setOfflineRLDatasetPath,
  setOfflineRLDatasetSelection,
} from '../offlineRLSlice';
import OfflineRLLeRobotDataset, {
  buildSelectedTrainingComposition,
  buildLeRobotEpisodeMedia,
  compareDatasetSelections,
  normalizeLeRobotEpisodes,
} from './OfflineRLLeRobotDataset';

jest.mock('../../../utils/offlineRlApi', () => ({
  deleteOfflineRLDatasetEpisodes: jest.fn(),
  getOfflineRLDatasetEpisodeData: jest.fn(),
  getOfflineRLDatasetInfo: jest.fn(),
  getOfflineRLDatasets: jest.fn(),
  getOfflineRLStatus: jest.fn(),
}));

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return { __esModule: true, default: toast };
});

const datasetPath = '/workspace/lerobot/Task_test_lerobot_v30';

const renderDataset = () => {
  const store = configureStore({
    reducer: {
      editDataset: editDatasetReducer,
      offlineRL: offlineRLReducer,
    },
  });
  store.dispatch(setOfflineRLDatasetPath(datasetPath));
  render(
    <Provider store={store}>
      <OfflineRLLeRobotDataset />
    </Provider>
  );
  return store;
};

beforeEach(() => {
  jest.clearAllMocks();
  getOfflineRLDatasetEpisodeData.mockResolvedValue({
    joint_timestamps: [],
    joint_names: [],
    joint_positions: [],
    action_timestamps: [],
    action_names: [],
    action_values: [],
    duration: 0,
  });
  getOfflineRLStatus.mockResolvedValue({ status: 'idle' });
  getOfflineRLDatasets.mockResolvedValue({
    root_path: '/workspace/lerobot',
    datasets: [{
      dataset_path: datasetPath,
      name: 'Task_test_lerobot_v30',
      version: 'v3.0',
    }],
  });
});

test('ignores a stale inventory response after the collection root changes', async () => {
  const oldPath = '/workspace/lerobot/old/Task_old_lerobot_v30';
  const newPath = '/workspace/lerobot/new/Task_new_lerobot_v30';
  let resolveOldInventory;
  getOfflineRLDatasets
    .mockImplementationOnce(() => new Promise((resolve) => {
      resolveOldInventory = resolve;
    }))
    .mockResolvedValueOnce({
      root_path: '/workspace/lerobot/new',
      datasets: [{ dataset_path: newPath, name: 'Task_new_lerobot_v30', version: 'v3.0' }],
    });
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v3.0',
    total_episodes: 1,
    fps: 15,
    episodes: [{ index: 0, outcome: 'success' }],
  });
  const store = configureStore({
    reducer: {
      editDataset: editDatasetReducer,
      offlineRL: offlineRLReducer,
    },
  });
  render(
    <Provider store={store}>
      <OfflineRLLeRobotDataset />
    </Provider>
  );

  await waitFor(() => expect(getOfflineRLDatasets).toHaveBeenCalledTimes(1));
  act(() => {
    store.dispatch(setOfflineRLConversionDestinationPath('/workspace/lerobot/new'));
  });
  await waitFor(() => {
    expect(store.getState().offlineRL.datasetPath).toBe(newPath);
    expect(screen.getByRole('button', {
      name: 'Preview Task_new_lerobot_v30 v3.0',
    })).toHaveAttribute('aria-pressed', 'true');
  });

  await act(async () => {
    resolveOldInventory({
      root_path: '/workspace/lerobot',
      datasets: [{ dataset_path: oldPath, name: 'Task_old_lerobot_v30', version: 'v3.0' }],
    });
  });
  expect(store.getState().offlineRL.datasetPath).toBe(newPath);
  expect(screen.getByRole('button', {
    name: 'Preview Task_new_lerobot_v30 v3.0',
  })).toHaveAttribute('aria-pressed', 'true');
});

test('normalizes API episode outcomes and indices', () => {
  expect(normalizeLeRobotEpisodes({
    episodes: [
      { episode_index: 2, episode_success: false },
      { index: 0, outcome: 'success' },
      { index: 1, outcome: 'unlabeled' },
    ],
  })).toEqual([
    { index: 0, outcome: 'success' },
    { index: 1, outcome: 'unlabeled' },
    { index: 2, outcome: 'failure' },
  ]);
});

test('builds safe ordered LeRobot camera segments for one episode', () => {
  const [episode] = normalizeLeRobotEpisodes({
    episodes: [{
      index: 0,
      outcome: 'success',
      frames: 459,
      tasks: ['Pick up the pack'],
      media: [
        {
          camera_key: 'observation.images.rgb.cam_right_wrist',
          relative_path: 'videos/right/chunk-000/file-000.mp4',
          from_s: 30.6,
          to_s: 64.6,
        },
        {
          camera_key: 'observation.images.rgb.cam_left_wrist',
          relative_path: 'videos/left/chunk-000/file-000.mp4',
          from_s: 30.6,
          to_s: 64.6,
        },
        {
          camera_key: 'observation.images.rgb.cam_left_head',
          relative_path: 'videos/head/chunk-000/file-000.mp4',
          from_s: 30.6,
          to_s: 64.6,
        },
      ],
    }],
  });

  expect(episode.frames).toBe(459);
  expect(episode.tasks).toEqual(['Pick up the pack']);
  expect(buildLeRobotEpisodeMedia(datasetPath, episode, 15)).toEqual([
    expect.objectContaining({
      label: 'Left wrist',
      url: `/files${datasetPath}/videos/left/chunk-000/file-000.mp4`,
      fromS: 30.6,
      toS: 64.6,
      fps: 15,
    }),
    expect.objectContaining({ label: 'Head' }),
    expect.objectContaining({ label: 'Right wrist' }),
  ]);
});

test('rejects unsafe LeRobot media paths in the browser URL builder', () => {
  expect(buildLeRobotEpisodeMedia(datasetPath, {
    media: [{
      camera_key: 'cam_left_head',
      relative_path: '../outside.mp4',
    }],
  }, 15)).toEqual([]);
});

test('keeps a legacy checkpoint root before newly numbered Data Epochs', () => {
  const legacy = { path: '/workspace/lerobot/legacy_v30', dataEpoch: null };
  const epoch = {
    path: '/workspace/lerobot/data_epoch_0000/task_v30',
    dataEpoch: 0,
  };

  expect([epoch, legacy].sort(compareDatasetSelections)).toEqual([legacy, epoch]);
});

test('aggregates composition from checked Training Data Epochs only', () => {
  const epoch1 = '/workspace/lerobot/data_epoch_0001/Task_one_lerobot_v30';
  const epoch2 = '/workspace/lerobot/data_epoch_0002/Task_two_lerobot_v30';
  const result = buildSelectedTrainingComposition(
    [{ path: epoch2, version: 'v3.0' }],
    [
      {
        dataset_path: epoch1,
        total_episodes: 2,
        episodes: [
          { index: 0, outcome: 'success' },
          { index: 1, outcome: 'success' },
        ],
      },
      {
        dataset_path: epoch2,
        total_episodes: 3,
        episodes: [
          { index: 0, outcome: 'failure' },
          { index: 1, outcome: 'failure' },
          { index: 2, outcome: 'unlabeled' },
        ],
      },
    ]
  );

  expect(result.totalCount).toBe(3);
  expect(result.episodes.map((episode) => episode.outcome)).toEqual([
    'failure',
    'failure',
    'unlabeled',
  ]);
  expect(result.episodes.every((episode) => episode.sourcePath === epoch2)).toBe(true);
});

test('renders converted episodes with an episode-weighted success percentage', async () => {
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v3.0',
    total_episodes: 3,
    fps: 15,
    episodes: [
      { index: 0, outcome: 'success' },
      { index: 1, outcome: 'failure' },
      { index: 2, outcome: 'success' },
    ],
  });

  renderDataset();

  expect(await screen.findByText('episode_002')).toBeInTheDocument();
  expect(screen.getByText('Success rate 67%')).toBeInTheDocument();
  const episodeList = screen.getByRole('list', { name: 'LeRobot Dataset episodes' });
  const successBar = screen.getByRole('progressbar', {
    name: 'LeRobot episodes success rate',
  });
  expect(successBar.compareDocumentPosition(episodeList))
    .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  expect(screen.getByRole('button', {
    name: 'Preview Task_test_lerobot_v30 v3.0',
  })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.queryByText('Dataset preview')).not.toBeInTheDocument();
  expect(screen.getByTestId('offline-rl-lerobot-dataset'))
    .toHaveClass('min-h-0', 'flex-col', 'gap-2');
  expect(screen.getByTestId('offline-rl-lerobot-episode-region'))
    .toHaveClass('min-h-[232px]', 'shrink-0');
  expect(screen.getByTestId('replay-buffer-composition-layout'))
    .toHaveClass('md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]');
  expect(screen.getByText('Training composition')).toBeInTheDocument();
  expect(screen.getByRole('group', { name: 'Training Data Epochs' }))
    .toHaveClass('max-h-[108px]', 'overflow-y-auto');
  expect(episodeList).toHaveClass(
    'h-[156px]',
    'min-h-[156px]',
    'max-h-[156px]',
    'flex-none',
    'overflow-y-auto',
    'overscroll-contain'
  );
});

test('opens the selected LeRobot episode camera segment in the media dialog', async () => {
  getOfflineRLDatasetEpisodeData.mockResolvedValue({
    joint_timestamps: [0, 1 / 15],
    joint_names: ['arm_l_joint1'],
    joint_positions: [0.1, 0.2],
    action_timestamps: [0, 1 / 15],
    action_names: ['arm_l_joint1'],
    action_values: [0.15, 0.25],
    duration: 1 / 15,
  });
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v3.0',
    total_episodes: 1,
    fps: 15,
    episodes: [{
      index: 0,
      outcome: 'success',
      frames: 459,
      tasks: ['Pick up the pack'],
      media: [
        {
          camera_key: 'observation.images.rgb.cam_left_head',
          relative_path: 'videos/head/chunk-000/file-000.mp4',
          from_s: 30.6,
          to_s: 64.6,
        },
        {
          camera_key: 'observation.images.rgb.cam_left_wrist',
          relative_path: 'videos/left/chunk-000/file-000.mp4',
          from_s: 30.6,
          to_s: 64.6,
        },
        {
          camera_key: 'observation.images.rgb.cam_right_wrist',
          relative_path: 'videos/right/chunk-000/file-000.mp4',
          from_s: 30.6,
          to_s: 64.6,
        },
      ],
    }],
  });

  renderDataset();
  fireEvent.click(await screen.findByLabelText('Open episode 0 video'));

  expect(await screen.findByRole('dialog', { name: 'episode_000' }))
    .toBeInTheDocument();
  const videos = screen.getAllByLabelText(/episode video$/);
  expect(videos).toHaveLength(3);
  expect(videos[0]).toHaveAttribute(
    'src',
    `/files${datasetPath}/videos/left/chunk-000/file-000.mp4`
  );
  expect(videos[1]).toHaveAttribute(
    'src',
    `/files${datasetPath}/videos/head/chunk-000/file-000.mp4`
  );
  expect(videos[2]).toHaveAttribute(
    'src',
    `/files${datasetPath}/videos/right/chunk-000/file-000.mp4`
  );
  expect(screen.getByText('Pick up the pack')).toBeInTheDocument();
  await waitFor(() => {
    expect(getOfflineRLDatasetEpisodeData).toHaveBeenCalledWith(datasetPath, 0);
  });
  expect(screen.getByText('Joint Data')).toBeInTheDocument();
  expect(screen.getByText('arm_l_joint1')).toBeInTheDocument();
});

test('keeps a fixed episode viewport and scrolls a large dataset internally', async () => {
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v3.0',
    total_episodes: 40,
    fps: 15,
    episodes: Array.from({ length: 40 }, (_, index) => ({
      index,
      outcome: index % 2 === 0 ? 'success' : 'failure',
    })),
  });

  renderDataset();

  expect(await screen.findByText('episode_039')).toBeInTheDocument();
  const viewport = screen.getByTestId('offline-rl-lerobot-episode-region');
  const episodeList = screen.getByRole('list', { name: 'LeRobot Dataset episodes' });
  expect(viewport).toHaveClass(
    'min-h-[232px]',
    'shrink-0'
  );
  expect(episodeList).toHaveClass(
    'h-[156px]',
    'min-h-[156px]',
    'max-h-[156px]',
    'flex-none',
    'overflow-y-auto',
    'overscroll-contain'
  );
  expect(screen.getByText('Success rate 50%')).toBeInTheDocument();
});

test('renders a converted v2.1 dataset with transactional episode deletion enabled', async () => {
  getOfflineRLDatasets.mockResolvedValue({
    root_path: '/workspace/lerobot',
    datasets: [{
      dataset_path: datasetPath,
      name: 'Task_test_lerobot_v21',
      version: 'v2.1',
    }],
  });
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v2.1',
    total_episodes: 2,
    fps: 15,
    episodes: [
      { index: 0, outcome: 'success' },
      { index: 1, outcome: 'success' },
    ],
  });

  renderDataset();

  expect(await screen.findByText('episode_001')).toBeInTheDocument();
  expect(screen.getByText('Success rate 100%')).toBeInTheDocument();
  expect(screen.getByRole('button', {
    name: 'Preview Task_test_lerobot_v21 v2.1',
  })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByRole('checkbox', {
    name: 'Include Task_test_lerobot_v21 v2.1 in training',
  })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Delete episode 1' }))
    .toBeInTheDocument();
});

test('discovers nested converted datasets and restores a selection after reload', async () => {
  const nestedPath = '/workspace/lerobot/RLTEST/Task_new_lerobot_v21';
  getOfflineRLDatasets.mockResolvedValue({
    root_path: '/workspace/lerobot',
    datasets: [{
      dataset_path: nestedPath,
      name: 'Task_new_lerobot_v21',
      version: 'v2.1',
    }],
  });
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v2.1',
    total_episodes: 1,
    fps: 15,
    episodes: [{ index: 0, outcome: 'success' }],
  });

  const store = configureStore({
    reducer: {
      editDataset: editDatasetReducer,
      offlineRL: offlineRLReducer,
    },
  });
  render(
    <Provider store={store}>
      <OfflineRLLeRobotDataset />
    </Provider>
  );

  await waitFor(() => {
    expect(store.getState().offlineRL.datasetPath).toBe(nestedPath);
    expect(store.getState().offlineRL.datasetVersion).toBe('v2.1');
  });
  expect(await screen.findByText('episode_000')).toBeInTheDocument();
  expect(screen.getByRole('button', {
    name: 'Preview Task_new_lerobot_v21 v2.1',
  })).toHaveAttribute('aria-pressed', 'true');
});

test('keeps preview and checked Training Data Epochs independent in deterministic order', async () => {
  const epoch1 = '/workspace/lerobot/data_epoch_0001/Task_one_lerobot_v30';
  const epoch2 = '/workspace/lerobot/data_epoch_0002/Task_two_lerobot_v30';
  getOfflineRLDatasets.mockResolvedValue({
    root_path: '/workspace/lerobot',
    // Inventory is newest-first; the training selection must not inherit
    // that presentation order.
    datasets: [
      {
        dataset_path: epoch2,
        name: 'Task_two_lerobot_v30',
        version: 'v3.0',
        total_episodes: 2,
        episodes: [
          { index: 0, outcome: 'failure' },
          { index: 1, outcome: 'failure' },
        ],
        data_epoch_provenance: { data_epoch: 2, epoch_name: 'data_epoch_0002' },
      },
      {
        dataset_path: epoch1,
        name: 'Task_one_lerobot_v30',
        version: 'v3.0',
        total_episodes: 3,
        episodes: [
          { index: 0, outcome: 'success' },
          { index: 1, outcome: 'success' },
          { index: 2, outcome: 'success' },
        ],
        data_epoch_provenance: { data_epoch: 1, epoch_name: 'data_epoch_0001' },
      },
    ],
  });
  getOfflineRLDatasetInfo.mockImplementation(async (path) => (
    path === epoch1
      ? {
        codebase_version: 'v3.0',
        total_episodes: 3,
        fps: 15,
        episodes: [
          { index: 0, outcome: 'success' },
          { index: 1, outcome: 'success' },
          { index: 2, outcome: 'success' },
        ],
      }
      : {
        codebase_version: 'v3.0',
        total_episodes: 2,
        fps: 15,
        episodes: [
          { index: 0, outcome: 'failure' },
          { index: 1, outcome: 'failure' },
        ],
      }
  ));
  const store = configureStore({
    reducer: {
      editDataset: editDatasetReducer,
      offlineRL: offlineRLReducer,
    },
  });
  render(
    <Provider store={store}>
      <OfflineRLLeRobotDataset />
    </Provider>
  );

  // The initially previewed newest dataset remains the only implicit choice.
  expect(await screen.findByLabelText('Include data_epoch_0002 v3.0 in training'))
    .toBeChecked();
  expect(screen.getByLabelText('Include data_epoch_0001 v3.0 in training'))
    .not.toBeChecked();
  await waitFor(() => {
    expect(screen.getByRole('img', {
      name: 'LeRobot episodes buffer composition',
    })).toHaveAttribute('data-capacity-used', '2');
  });
  fireEvent.click(screen.getByLabelText('Include data_epoch_0001 v3.0 in training'));

  await waitFor(() => {
    expect(store.getState().offlineRL.datasetSelections.map((item) => item.path))
      .toEqual([epoch1, epoch2]);
  });
  expect(store.getState().offlineRL.datasetPath).toBe(epoch2);
  expect(screen.getByRole('button', {
    name: 'Preview data_epoch_0002 v3.0',
  })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByText('2 included')).toBeInTheDocument();
  await waitFor(() => {
    expect(screen.getByRole('img', {
      name: 'LeRobot episodes buffer composition',
    })).toHaveAttribute('data-capacity-used', '5');
  });
  expect(screen.getByText('Success rate 60%')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', {
    name: 'Preview data_epoch_0001 v3.0',
  }));
  await waitFor(() => {
    expect(store.getState().offlineRL.datasetPath).toBe(epoch1);
  });
  expect(store.getState().offlineRL.datasetSelections.map((item) => item.path))
    .toEqual([epoch1, epoch2]);
  expect(screen.getByLabelText('Include data_epoch_0001 v3.0 in training'))
    .toBeChecked();
  expect(screen.getByLabelText('Include data_epoch_0002 v3.0 in training'))
    .toBeChecked();
});

test('keeps checked v3 roots when a new conversion is added to dataset_paths', async () => {
  const epoch0 = '/workspace/lerobot/data_epoch_0000/Task_zero_lerobot_v30';
  const epoch1 = '/workspace/lerobot/data_epoch_0001/Task_one_lerobot_v30';
  const epoch1V21 = '/workspace/lerobot/data_epoch_0001/Task_one_lerobot_v21';
  const oldDataset = {
    dataset_path: epoch0,
    name: 'Task_zero_lerobot_v30',
    version: 'v3.0',
    data_epoch_provenance: { data_epoch: 0, epoch_name: 'data_epoch_0000' },
  };
  const newDataset = {
    dataset_path: epoch1,
    name: 'Task_one_lerobot_v30',
    version: 'v3.0',
    data_epoch_provenance: { data_epoch: 1, epoch_name: 'data_epoch_0001' },
  };
  getOfflineRLDatasets
    .mockResolvedValueOnce({
      root_path: '/workspace/lerobot',
      datasets: [oldDataset],
    })
    .mockResolvedValue({
      root_path: '/workspace/lerobot',
      datasets: [
        newDataset,
        oldDataset,
        {
          dataset_path: epoch1V21,
          name: 'Task_one_lerobot_v21',
          version: 'v2.1',
          data_epoch_provenance: { data_epoch: 1, epoch_name: 'data_epoch_0001' },
        },
      ],
    });
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v3.0',
    total_episodes: 1,
    fps: 15,
    episodes: [{ index: 0, outcome: 'success' }],
  });
  const store = configureStore({
    reducer: {
      editDataset: editDatasetReducer,
      offlineRL: offlineRLReducer,
    },
  });
  store.dispatch(setOfflineRLDatasetSelection({
    path: epoch0,
    version: 'v3.0',
    dataEpoch: 0,
  }));
  render(
    <Provider store={store}>
      <OfflineRLLeRobotDataset />
    </Provider>
  );

  expect(await screen.findByLabelText('Include data_epoch_0000 v3.0 in training'))
    .toBeChecked();

  act(() => {
    // This is the same pair of Redux updates emitted by Step 2 after a
    // verified v3 conversion. setOfflineRLDatasetSelection must append,
    // never replace, the user's existing checked replay roots.
    store.dispatch(setOfflineRLConvertedDatasetPaths({ v21: epoch1V21, v30: epoch1 }));
    store.dispatch(setOfflineRLDatasetSelection({ path: epoch1, version: 'v3.0' }));
    store.dispatch(setConversionStatus({
      status: 'completed',
      jobId: 'conversion-job-1',
      progress: 100,
    }));
  });

  const epoch1Checkboxes = await screen.findAllByLabelText(
    /Include data_epoch_0001 v(?:2\.1|3\.0) in training/
  );
  const epoch1V30Checkbox = epoch1Checkboxes.find((checkbox) => !checkbox.disabled);
  const epoch1V21Checkbox = epoch1Checkboxes.find((checkbox) => checkbox.disabled);
  expect(epoch1V30Checkbox).toBeChecked();
  expect(epoch1V21Checkbox).not.toBeChecked();
  expect(screen.getByLabelText('Include data_epoch_0000 v3.0 in training')).toBeChecked();
  await waitFor(() => {
    expect(selectOfflineRLDatasetPaths(store.getState())).toEqual([epoch0, epoch1]);
  });
  expect(screen.getByText('2 included')).toBeInTheDocument();
});

test('deletes through the transactional dataset API and refreshes compacted indices', async () => {
  getOfflineRLDatasetInfo
    .mockResolvedValueOnce({
      codebase_version: 'v3.0',
      total_episodes: 2,
      fps: 15,
      episodes: [
        { index: 0, outcome: 'success' },
        { index: 1, outcome: 'failure' },
      ],
    })
    .mockResolvedValueOnce({
      codebase_version: 'v3.0',
      total_episodes: 1,
      fps: 15,
      episodes: [{ index: 0, outcome: 'success' }],
    });
  deleteOfflineRLDatasetEpisodes.mockResolvedValue({ total_episodes: 1 });
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

  renderDataset();
  fireEvent.click(await screen.findByRole('button', { name: 'Delete episode 1' }));

  await waitFor(() => {
    expect(deleteOfflineRLDatasetEpisodes).toHaveBeenCalledWith(datasetPath, [1]);
  });
  await waitFor(() => {
    expect(screen.queryByText('episode_001')).not.toBeInTheDocument();
  });
  expect(screen.getByText('Success rate 100%')).toBeInTheDocument();
  confirmSpy.mockRestore();
});

test('keeps a completed checkpoint Data Epoch immutable', async () => {
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v3.0',
    total_episodes: 1,
    fps: 15,
    episodes: [{ index: 0, outcome: 'success' }],
  });
  getOfflineRLStatus.mockResolvedValue({
    status: 'complete',
    dataset_path: datasetPath,
    dataset_paths: [datasetPath],
  });
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

  renderDataset();
  fireEvent.click(await screen.findByRole('button', { name: 'Delete episode 0' }));

  await waitFor(() => expect(getOfflineRLStatus).toHaveBeenCalled());
  expect(deleteOfflineRLDatasetEpisodes).not.toHaveBeenCalled();
  expect(confirmSpy).not.toHaveBeenCalled();
  confirmSpy.mockRestore();
});

test('deleting the final episode removes the empty dataset and refreshes inventory', async () => {
  getOfflineRLDatasetInfo.mockResolvedValue({
    codebase_version: 'v3.0',
    total_episodes: 1,
    fps: 15,
    episodes: [{ index: 0, outcome: 'success' }],
  });
  deleteOfflineRLDatasetEpisodes.mockResolvedValue({
    ok: true,
    dataset: null,
    dataset_deleted: true,
  });
  getOfflineRLDatasets
    .mockResolvedValueOnce({
      root_path: '/workspace/lerobot',
      datasets: [{
        dataset_path: datasetPath,
        name: 'Task_test_lerobot_v30',
        version: 'v3.0',
      }],
    })
    .mockResolvedValueOnce({
      root_path: '/workspace/lerobot',
      datasets: [],
    });
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

  const store = renderDataset();
  fireEvent.click(await screen.findByRole('button', { name: 'Delete episode 0' }));

  await waitFor(() => {
    expect(deleteOfflineRLDatasetEpisodes).toHaveBeenCalledWith(datasetPath, [0]);
    expect(store.getState().offlineRL.datasetPath).toBe('');
  });
  expect(confirmSpy.mock.calls[0][0]).toContain('entire selected LeRobot dataset folder');
  expect(await screen.findByText('No saved episodes')).toBeInTheDocument();
  expect(getOfflineRLDatasetInfo).toHaveBeenCalledTimes(1);
  confirmSpy.mockRestore();
});
