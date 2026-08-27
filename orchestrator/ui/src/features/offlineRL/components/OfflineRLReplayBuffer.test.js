import { configureStore } from '@reduxjs/toolkit';
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
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
  resolveMcapEpisodeMedia,
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

jest.mock('./OfflineRLEpisodeMediaModal', () => {
  return function MockOfflineRLEpisodeMediaModal({
    open,
    episode,
    media,
    jointData,
    loading,
    error,
    onBack,
    onDelete,
  }) {
    if (!open) return null;
    return (
      <div role="dialog" aria-label="Episode video">
        <span>{`Selected episode ${episode?.index}`}</span>
        {loading && <span>Media loading</span>}
        {error && <span>{error}</span>}
        {media.map((item) => (
          <a key={item.key} href={item.url}>{item.label}</a>
        ))}
        <span>{`Joint samples ${jointData?.joint_timestamps?.length || 0}`}</span>
        <button type="button" onClick={onBack}>Back</button>
        <button type="button" onClick={onDelete}>Delete from modal</button>
      </div>
    );
  };
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

test('resolves segmented MCAP replay descriptors into playable media', () => {
  expect(resolveMcapEpisodeMedia({
    video_segments: [
      {
        name: '0_0',
        replay_start_s: 0,
        replay_end_s: 2.5,
        video_files: [
          'videos/0_0/cam_left_wrist.mp4',
          'videos/0_0/cam_left_head.mp4',
        ],
        video_names: ['Left wrist', 'Head'],
        video_fps: [15, 14.9],
      },
    ],
  }, '/workspace/rosbag2/session/4')).toEqual([
    {
      key: '0-videos/0_0/cam_left_wrist.mp4',
      label: 'Left wrist',
      url: '/files/workspace/rosbag2/session/4/videos/0_0/cam_left_wrist.mp4',
      fromS: 0,
      toS: 2.5,
      fps: 15,
    },
    {
      key: '0-videos/0_0/cam_left_head.mp4',
      label: 'Head',
      url: '/files/workspace/rosbag2/session/4/videos/0_0/cam_left_head.mp4',
      fromS: 0,
      toS: 2.5,
      fps: 14.9,
    },
  ]);
});

test('orders the standard MCAP cameras and rejects an unsafe video path', () => {
  const media = resolveMcapEpisodeMedia({
    video_files: [
      'videos/0_0/cam_left_head.mp4',
      'videos/0_0/cam_left_wrist.mp4',
      'videos/0_0/cam_right_wrist.mp4',
      '../outside.mp4',
    ],
    video_names: [
      'cam_left_head',
      'cam_left_wrist',
      'cam_right_wrist',
      'outside',
    ],
    duration: 2,
  }, '/workspace/rosbag2/session/0');

  expect(media.map((item) => item.label)).toEqual([
    'Left wrist',
    'Head',
    'Right wrist',
  ]);
});

test('renders a 40/60 composition and episode manager with internal scrolling', () => {
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
  expect(screen.getByTestId('replay-buffer-composition-layout'))
    .toHaveClass('md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]');
  expect(screen.getByTestId('replay-buffer-composition'))
    .toHaveClass('max-w-full', 'overflow-hidden');
  expect(screen.getByTestId('replay-buffer-episode-manager')).toBeInTheDocument();
  expect(screen.getByTestId('replay-composition-stats'))
    .toHaveClass('grid', 'grid-cols-2');
  expect(screen.getByTestId('replay-outcome-summary')).toBeInTheDocument();
  expect(screen.getByTestId('replay-outcome-legend'))
    .toHaveClass('text-[10px]', 'gap-1.5');
  expect(screen.getByRole('img', { name: 'MCAP episodes buffer composition' }))
    .toBeInTheDocument();
  expect(screen.getByRole('textbox', { name: 'Search MCAP episodes' }))
    .toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'All 2' }))
    .toHaveClass('text-[10px]');
  expect(screen.getByRole('button', { name: 'Success 1' }))
    .toHaveClass('text-[10px]');
  expect(screen.getByText('episode_000').parentElement)
    .toHaveClass('text-[10px]');
  expect(within(episodeList).getByText('Success'))
    .toHaveClass('text-[9px]');
  expect(episodeList).toHaveClass(
    'h-[156px]',
    'min-h-[156px]',
    'max-h-[156px]',
    'flex-none',
    'overflow-y-auto',
    'overscroll-contain'
  );
  expect(episodeList.parentElement)
    .toHaveAttribute('data-testid', 'replay-buffer-episode-manager');
  expect(successBar).toHaveAttribute('aria-valuenow', '50');
  expect(screen.queryByText('Lists stay light; no video is decoded here.'))
    .not.toBeInTheDocument();
  expect(screen.queryByText('View opens video + joints'))
    .not.toBeInTheDocument();
});

test('keeps unused capacity visible independently from outcome composition', () => {
  render(<ReplayBufferStack
    episodes={[
      { index: 0, outcome: 'success' },
      { index: 1, outcome: 'success' },
      { index: 2, outcome: 'success' },
      { index: 3, outcome: 'failure' },
    ]}
    totalCount={4}
  />);

  const cylinder = screen.getByRole('img', {
    name: 'MCAP episodes buffer composition',
  });
  expect(cylinder).toHaveAttribute('data-capacity-used', '4');
  expect(cylinder).toHaveAttribute('data-capacity-empty', '196');
  expect(cylinder).toHaveAttribute('data-capacity-percent', '2');
  expect(cylinder).toHaveAttribute('data-visible-disc-count', '4');
  expect(cylinder).toHaveClass(
    'h-[168px]',
    'max-h-[168px]',
    'min-w-0',
    'max-w-full'
  );
  expect(screen.getAllByTestId('replay-cylinder-disc')).toHaveLength(4);
  expect(screen.getAllByTestId('replay-cylinder-disc-face')).toHaveLength(4);
  expect(screen.getAllByTestId('replay-cylinder-disc-edge')).toHaveLength(4);
  expect(screen.getByTestId('replay-cylinder-occupied-capacity'))
    .toHaveAttribute('data-rendered-discs', '4');
  expect(screen.getByTestId('replay-cylinder-occupied-capacity'))
    .toHaveAttribute('data-full-plate-stack', 'true');
  expect(screen.getByTestId('replay-cylinder-occupied-capacity'))
    .not.toHaveAttribute('clip-path');
  expect(cylinder.querySelector('clipPath')).toBeNull();
  expect(screen.getByTestId('replay-cylinder-base')).toHaveAttribute('cx', '90');
  expect(screen.getByTestId('replay-cylinder-base')).toHaveAttribute('cy', '164');
  expect(screen.getByTestId('replay-cylinder-base')).toHaveAttribute('rx', '70');
  expect(screen.getAllByTestId('replay-cylinder-disc')[0])
    .toHaveAttribute('data-base-aligned', 'true');
  expect(screen.getAllByTestId('replay-cylinder-disc')[0])
    .toHaveAttribute('data-bottom-y', '164');
  expect(screen.getAllByTestId('replay-cylinder-disc-face')[0])
    .toHaveAttribute('cx', '90');
  expect(screen.getAllByTestId('replay-cylinder-disc-face')[0])
    .toHaveAttribute('rx', '70');
  expect(screen.getByRole('progressbar', {
    name: 'MCAP episodes success rate',
  })).toHaveAttribute('aria-valuenow', '75');
  expect(screen.getByText('4 stored')).toBeInTheDocument();
});

test('bounds SVG plate rendering when many episodes are stored', () => {
  const episodes = Array.from({ length: 80 }, (_, index) => ({
    index,
    outcome: index < 60 ? 'success' : 'failure',
  }));
  render(<ReplayBufferStack episodes={episodes} totalCount={80} />);

  expect(screen.getByRole('img', {
    name: 'MCAP episodes buffer composition',
  })).toHaveAttribute('data-visible-disc-count', '36');
  expect(screen.getAllByTestId('replay-cylinder-disc')).toHaveLength(36);
  expect(screen.getAllByTestId('replay-cylinder-disc-face')).toHaveLength(36);
});

test('filters the episode manager without changing the composition totals', () => {
  render(<ReplayBufferStack
    episodes={[
      { index: 0, outcome: 'success' },
      { index: 1, outcome: 'failure' },
      { index: 2, outcome: 'unlabeled' },
    ]}
    totalCount={3}
  />);

  fireEvent.click(screen.getByRole('button', { name: 'Success 1' }));
  expect(screen.getAllByRole('listitem')).toHaveLength(1);
  expect(screen.getByText('episode_000')).toBeInTheDocument();
  expect(screen.queryByText('episode_001')).not.toBeInTheDocument();
  expect(screen.getByText('Success 1 · Fail 1')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'All 3' }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Search MCAP episodes' }), {
    target: { value: '002' },
  });
  expect(screen.getAllByRole('listitem')).toHaveLength(1);
  expect(screen.getByText('episode_002')).toBeInTheDocument();
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
  const getReplayData = jest.fn();
  useRosServiceCaller.mockReturnValue({
    getDatasetInfo,
    getReplayData,
    sendEditDatasetCommand,
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
  expect(getReplayData).not.toHaveBeenCalled();
  confirmSpy.mockRestore();
});

test('opens an MCAP episode video and deletes it through the existing safe handler', async () => {
  const getDatasetInfo = jest.fn()
    .mockResolvedValueOnce({
      success: true,
      dataset_info: {
        episode_count: 1,
        success_episode_indices: [0],
        failure_episode_indices: [],
        unlabeled_episode_indices: [],
      },
    })
    .mockResolvedValueOnce({
      success: true,
      dataset_info: {
        episode_count: 0,
        success_episode_indices: [],
        failure_episode_indices: [],
        unlabeled_episode_indices: [],
      },
    });
  const getReplayData = jest.fn().mockResolvedValue({
    success: true,
    video_files: ['videos/0_0/cam_left_head.mp4'],
    video_names: ['Head'],
    video_fps: [15],
    video_segments: [],
    joint_timestamps: [0, 0.1],
    joint_names: ['arm_l_joint1'],
    joint_positions: [0.2, 0.3],
    duration: 2,
  });
  const sendEditDatasetCommand = jest.fn().mockResolvedValue({
    success: true,
    affected_count: 1,
  });
  useRosServiceCaller.mockReturnValue({
    getDatasetInfo,
    getReplayData,
    sendEditDatasetCommand,
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
          taskNum: 'media_test',
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

  const episodeRow = await screen.findByRole('listitem');
  fireEvent.click(episodeRow);
  await waitFor(() => {
    expect(getReplayData).toHaveBeenCalledWith(
      '/workspace/rosbag2/Task_media_test_inference_MCAP/0'
    );
  });
  expect(await screen.findByRole('link', { name: 'Head' })).toHaveAttribute(
    'href',
    '/files/workspace/rosbag2/Task_media_test_inference_MCAP/0/videos/0_0/cam_left_head.mp4'
  );
  expect(screen.getByText('Joint samples 2')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Delete from modal' }));
  await waitFor(() => {
    expect(sendEditDatasetCommand).toHaveBeenCalledWith('delete', {
      deleteTaskDir: '/workspace/rosbag2/Task_media_test_inference_MCAP',
      deleteEpisodeNums: [0],
      deleteCompact: false,
    });
  });
  await waitFor(() => {
    expect(screen.queryByRole('dialog', { name: 'Episode video' }))
      .not.toBeInTheDocument();
  });
  expect(getDatasetInfo).toHaveBeenCalledTimes(2);
});
