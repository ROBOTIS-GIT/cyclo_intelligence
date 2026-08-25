import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import editDatasetReducer, {
  setConversionStatus,
} from '../../editDataset/editDatasetSlice';
import {
  deleteOfflineRLDatasetEpisodes,
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
  compareDatasetSelections,
  normalizeLeRobotEpisodes,
} from './OfflineRLLeRobotDataset';

jest.mock('../../../utils/offlineRlApi', () => ({
  deleteOfflineRLDatasetEpisodes: jest.fn(),
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
    expect(screen.getByRole('combobox', { name: 'LeRobot dataset' })).toHaveValue(newPath);
  });

  await act(async () => {
    resolveOldInventory({
      root_path: '/workspace/lerobot',
      datasets: [{ dataset_path: oldPath, name: 'Task_old_lerobot_v30', version: 'v3.0' }],
    });
  });
  expect(screen.getByRole('combobox', { name: 'LeRobot dataset' })).toHaveValue(newPath);
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

test('keeps a legacy checkpoint root before newly numbered Data Epochs', () => {
  const legacy = { path: '/workspace/lerobot/legacy_v30', dataEpoch: null };
  const epoch = {
    path: '/workspace/lerobot/data_epoch_0000/task_v30',
    dataEpoch: 0,
  };

  expect([epoch, legacy].sort(compareDatasetSelections)).toEqual([legacy, epoch]);
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
  const datasetDetail = screen.getByText('v3.0 · 15 FPS · ready for training');
  const episodeList = screen.getByRole('list', { name: 'LeRobot Dataset episodes' });
  const successBar = screen.getByRole('progressbar', {
    name: 'LeRobot episodes success rate',
  });
  expect(datasetDetail.compareDocumentPosition(episodeList))
    .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  expect(episodeList.compareDocumentPosition(successBar))
    .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  expect(screen.getByTestId('offline-rl-lerobot-dataset'))
    .toHaveClass('h-full', 'min-h-0', 'overflow-hidden');
  expect(screen.getByTestId('offline-rl-lerobot-episode-region'))
    .toHaveClass('h-[160px]', 'min-h-0', 'shrink-0', 'overflow-hidden');
  expect(episodeList).toHaveClass('flex-1', 'overflow-y-auto');
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
    'h-[160px]',
    'min-h-0',
    'shrink-0',
    'overflow-hidden'
  );
  expect(episodeList).toHaveClass('flex-1', 'overflow-y-auto');
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
  expect(screen.getByText('v2.1 · 15 FPS · editable; select v3.0 for training'))
    .toBeInTheDocument();
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
  expect(screen.getByRole('combobox', { name: 'LeRobot dataset' })).toHaveValue(nestedPath);
});

test('explicitly includes multiple Data Epochs in deterministic epoch order', async () => {
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
        data_epoch_provenance: { data_epoch: 2, epoch_name: 'data_epoch_0002' },
      },
      {
        dataset_path: epoch1,
        name: 'Task_one_lerobot_v30',
        version: 'v3.0',
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
  render(
    <Provider store={store}>
      <OfflineRLLeRobotDataset />
    </Provider>
  );

  // The initially previewed newest dataset remains the only implicit choice.
  expect(await screen.findByLabelText('Include data_epoch_0002 in training'))
    .toBeChecked();
  expect(screen.getByLabelText('Include data_epoch_0001 in training'))
    .not.toBeChecked();
  fireEvent.click(screen.getByLabelText('Include data_epoch_0001 in training'));

  await waitFor(() => {
    expect(store.getState().offlineRL.datasetSelections.map((item) => item.path))
      .toEqual([epoch1, epoch2]);
  });
  expect(screen.getByRole('combobox', { name: 'LeRobot dataset' })).toHaveValue(epoch1);
  expect(screen.getByText('2 Data Epochs included')).toBeInTheDocument();
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

  expect(await screen.findByLabelText('Include data_epoch_0000 in training'))
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
    'Include data_epoch_0001 in training'
  );
  const epoch1V30Checkbox = epoch1Checkboxes.find((checkbox) => !checkbox.disabled);
  const epoch1V21Checkbox = epoch1Checkboxes.find((checkbox) => checkbox.disabled);
  expect(epoch1V30Checkbox).toBeChecked();
  expect(epoch1V21Checkbox).not.toBeChecked();
  expect(screen.getByLabelText('Include data_epoch_0000 in training')).toBeChecked();
  await waitFor(() => {
    expect(selectOfflineRLDatasetPaths(store.getState())).toEqual([epoch0, epoch1]);
  });
  expect(screen.getByText('2 Data Epochs included')).toBeInTheDocument();
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
