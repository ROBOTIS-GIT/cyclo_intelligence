import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { RecordPhase } from '../../../constants/taskPhases';
import taskReducer, {
  InferenceRecordingUiPhase,
  setInferenceRecordingUiPhase,
} from '../../tasks/taskSlice';
import offlineRLReducer from '../offlineRLSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import OfflineRLReplayBuffer, {
  normalizeReplayEpisodes,
  ReplayBufferStack,
  resolveReplayBufferPath,
} from './OfflineRLReplayBuffer';

jest.mock('../../../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return { __esModule: true, default: toast };
});

test('normalizes and sorts outcome-aware episode indices', () => {
  expect(normalizeReplayEpisodes({
    success_episode_indices: [3, 0],
    failure_episode_indices: [2],
    unlabeled_episode_indices: [1],
  })).toEqual([
    { index: 0, outcome: 'success' },
    { index: 1, outcome: 'unlabeled' },
    { index: 2, outcome: 'failure' },
    { index: 3, outcome: 'success' },
  ]);
});

test('resolves explicit and automatic inference recording folders', () => {
  expect(resolveReplayBufferPath('/workspace/rosbag2/custom/', 'ignored'))
    .toBe('/workspace/rosbag2/custom');
  expect(resolveReplayBufferPath('', '20260821_100000'))
    .toBe('/workspace/rosbag2/Task_20260821_100000_inference_MCAP');
});

test('renders saved episodes as stacked outcome boxes', () => {
  render(<ReplayBufferStack
    episodes={[
      { index: 0, outcome: 'success' },
      { index: 1, outcome: 'failure' },
    ]}
    totalCount={2}
  />);

  expect(screen.getAllByRole('listitem')).toHaveLength(2);
  expect(screen.getByText('episode_000')).toBeInTheDocument();
  expect(screen.getByText('episode_001')).toBeInTheDocument();
  expect(screen.getByText('Success 1 · Fail 1')).toBeInTheDocument();
  expect(screen.getByText('Success rate 50%')).toBeInTheDocument();
  const episodeList = screen.getByRole('list', { name: 'Replay Buffer episodes' });
  const successBar = screen.getByRole('progressbar', {
    name: 'MCAP episodes success rate',
  });
  expect(episodeList).toHaveClass('min-h-[104px]', 'flex-1', 'overflow-y-auto');
  expect(episodeList.parentElement).toHaveClass('flex-1', 'flex-col');
  expect(successBar).toHaveAttribute('aria-valuenow', '50');
});

test('loads the current recording session from the existing dataset service', async () => {
  const getDatasetInfo = jest.fn().mockResolvedValue({
    success: true,
    dataset_info: {
      episode_count: 3,
      success_episode_indices: [0, 2],
      failure_episode_indices: [1],
      unlabeled_episode_indices: [],
    },
  });
  useRosServiceCaller.mockReturnValue({
    getDatasetInfo,
    sendEditDatasetCommand: jest.fn(),
  });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer, offlineRL: offlineRLReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        recordStatus: {
          ...initialTasks.recordStatus,
          taskType: 'inference',
          recordInferenceMode: true,
          taskNum: 'session_01',
          recordPhase: RecordPhase.READY,
          currentEpisodeNumber: 3,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <OfflineRLReplayBuffer />
    </Provider>
  );

  await waitFor(() => {
    expect(getDatasetInfo).toHaveBeenCalledWith(
      '/workspace/rosbag2/Task_session_01_inference_MCAP'
    );
  });
  await waitFor(() => {
    expect(screen.getAllByRole('listitem')).toHaveLength(3);
  });
  expect(screen.getByText('Success 2 · Fail 1')).toBeInTheDocument();
});

test('retains the replay path after inference Clear drops the live session id', async () => {
  const getDatasetInfo = jest.fn().mockResolvedValue({
    success: true,
    dataset_info: {
      episode_count: 1,
      success_episode_indices: [0],
      failure_episode_indices: [],
      unlabeled_episode_indices: [],
    },
  });
  useRosServiceCaller.mockReturnValue({
    getDatasetInfo,
    sendEditDatasetCommand: jest.fn(),
  });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer, offlineRL: offlineRLReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        recordStatus: {
          ...initialTasks.recordStatus,
          taskType: 'inference',
          recordInferenceMode: true,
          taskNum: 'keep_me',
          recordPhase: RecordPhase.READY,
          currentEpisodeNumber: 1,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <OfflineRLReplayBuffer />
    </Provider>
  );
  await screen.findByText('episode_000');

  act(() => {
    store.dispatch({
      type: 'tasks/setRecordStatus',
      payload: {
        taskType: '',
        recordInferenceMode: false,
        taskNum: '',
        currentEpisodeNumber: 0,
        recordPhase: RecordPhase.READY,
      },
    });
  });

  await waitFor(() => {
    expect(screen.getByText('episode_000')).toBeInTheDocument();
  });
  expect(store.getState().offlineRL.replayBufferPath)
    .toBe('/workspace/rosbag2/Task_keep_me_inference_MCAP');
});

test('waits for recording lifecycle to finish before refreshing a new folder', async () => {
  const getDatasetInfo = jest.fn().mockResolvedValue({
    success: true,
    dataset_info: {
      episode_count: 1,
      success_episode_indices: [0],
      failure_episode_indices: [],
      unlabeled_episode_indices: [],
    },
  });
  useRosServiceCaller.mockReturnValue({
    getDatasetInfo,
    sendEditDatasetCommand: jest.fn(),
  });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const recordingFolder =
    '/workspace/rosbag2/Task_pending_inference_MCAP';
  const store = configureStore({
    reducer: { tasks: taskReducer, offlineRL: offlineRLReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          recordingFolder,
        },
        inferenceRecordingUi: {
          phase: InferenceRecordingUiPhase.STARTING,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <OfflineRLReplayBuffer />
    </Provider>
  );
  await act(async () => {});
  expect(getDatasetInfo).not.toHaveBeenCalled();

  act(() => {
    store.dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.IDLE
    ));
  });

  await waitFor(() => {
    expect(getDatasetInfo).toHaveBeenCalledWith(recordingFolder);
  });
});

test('deletes one episode only from its explicit red minus button', async () => {
  const getDatasetInfo = jest.fn()
    .mockResolvedValueOnce({
      success: true,
      dataset_info: {
        episode_count: 2,
        success_episode_indices: [0],
        failure_episode_indices: [1],
        unlabeled_episode_indices: [],
      },
    })
    .mockResolvedValueOnce({
      success: true,
      dataset_info: {
        episode_count: 1,
        success_episode_indices: [0],
        failure_episode_indices: [],
        unlabeled_episode_indices: [],
      },
    });
  const sendEditDatasetCommand = jest.fn().mockResolvedValue({
    success: true,
    affected_count: 1,
  });
  useRosServiceCaller.mockReturnValue({ getDatasetInfo, sendEditDatasetCommand });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer, offlineRL: offlineRLReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        recordStatus: {
          ...initialTasks.recordStatus,
          taskType: 'inference',
          recordInferenceMode: true,
          taskNum: 'delete_test',
          recordPhase: RecordPhase.READY,
          currentEpisodeNumber: 2,
        },
      },
    },
  });
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

  render(
    <Provider store={store}>
      <OfflineRLReplayBuffer />
    </Provider>
  );
  await screen.findByRole('button', { name: 'Delete episode 1' });
  fireEvent.click(screen.getByRole('button', { name: 'Delete episode 1' }));

  await waitFor(() => {
    expect(sendEditDatasetCommand).toHaveBeenCalledWith('delete', {
      deleteTaskDir: '/workspace/rosbag2/Task_delete_test_inference_MCAP',
      deleteEpisodeNums: [1],
      deleteCompact: false,
    });
  });
  await waitFor(() => {
    expect(screen.queryByText('episode_001')).not.toBeInTheDocument();
  });
  expect(confirmSpy).toHaveBeenCalled();
  confirmSpy.mockRestore();
});
