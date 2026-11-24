import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Provider } from 'react-redux';
import RecordTopicMonitor from './RecordTopicMonitor';
import taskReducer, { selectRobotType, setRecordingMonitor } from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { RecordPhase } from '../constants/taskPhases';
import { requestSupervisorApi } from '../features/missionCanvas/lib/btNodeApi';

jest.mock('../features/missionCanvas/lib/btNodeApi', () => ({
  requestSupervisorApi: jest.fn(),
}));

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

const renderMonitor = (recordingMonitor, recordStatus = {}, props = {}, robotType = '') => {
  const sendRecordCommand = jest.fn();
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: {
      tasks: taskReducer,
    },
    preloadedState: {
      tasks: {
        ...initialTasks,
        recordingMonitor,
        robotType,
        recordStatus: { ...initialTasks.recordStatus, ...recordStatus },
      },
    },
  });

  const view = render(
    <Provider store={store}>
      <RecordTopicMonitor {...props} />
    </Provider>
  );
  return { sendRecordCommand, store, ...view };
};

describe('RecordTopicMonitor', () => {
  beforeEach(() => {
    jest.resetAllMocks();
    requestSupervisorApi.mockResolvedValue({ topic_keys: {} });
  });

  test.each([RecordPhase.RECORDING, RecordPhase.SAVING])('blocks topic refresh while recording phase is %s', (recordPhase) => {
    const { sendRecordCommand } = renderMonitor({ topics: [{ name: '/joint_states', status: 0 }], cameraTopics: [] }, { recordPhase });
    const refresh = screen.getByTitle('Reset topic subscriptions');
    expect(refresh).toBeDisabled();
    fireEvent.click(refresh);
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('renders camera monitor rows even without rosbag monitor rows', () => {
    renderMonitor({
      topics: [],
      cameraTopics: [
        {
          name: '/zed/zed_node/left/image_rect_color/compressed',
          cameraName: 'cam_left_head',
          rateHz: 29.8,
          baselineHz: 30,
          secondsSinceLast: 0.1,
          status: 0,
          timestampStatus: 0,
          timestampSkewS: -0.01,
          source: 'camera',
        },
      ],
      totalReceived: 0,
      totalWritten: 0,
    });

    expect(screen.getByText('Topic Monitor')).toBeInTheDocument();
    expect(screen.getByText('cam_left_head')).toHaveAttribute('title', '/zed/zed_node/left/image_rect_color/compressed (cam_left_head)');
    expect(screen.getByText('29.8')).toBeInTheDocument();
    expect(screen.getByText('All 1 OK')).toBeInTheDocument();
  });

  test('uses selected config aliases deterministically, including cameras, and keeps full extra topics', async () => {
    requestSupervisorApi.mockResolvedValue({
      topic_keys: {
        '/robot/state': ['upper_body', 'arm_left', 'upper_body'],
        '/robot/command': ['arm_left'],
        '/camera/image': ['camera_key', 'camera_alias'],
      },
    });
    const { store } = renderMonitor({ topics: [
      { name: '/robot/state', status: 0 },
      { name: '/robot/command', status: 0 },
      { name: '/camera/image', status: 0 },
      { name: '/robot/extra/camera_info', status: 0 },
    ] }, {}, {}, 'custom_robot');

    expect(await screen.findByText('arm_left, upper_body')).toHaveAttribute('title', '/robot/state');
    expect(screen.getByText('arm_left')).toHaveAttribute('title', '/robot/command');
    expect(screen.getByText('camera_alias, camera_key')).toHaveAttribute('title', '/camera/image');
    expect(screen.getByText('/robot/extra/camera_info')).toHaveAttribute('title', '/robot/extra/camera_info');
    expect(requestSupervisorApi).toHaveBeenCalledWith('/robots/custom_robot/topic-keys', expect.any(Object));
    act(() => store.dispatch(setRecordingMonitor({ totalReceived: 42 })));
    expect(requestSupervisorApi).toHaveBeenCalledTimes(1);
  });

  test('clears old labels immediately on switch and ignores a stale response', async () => {
    let resolveOld;
    requestSupervisorApi
      .mockResolvedValueOnce({ topic_keys: { '/state': ['first_key'] } })
      .mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }))
      .mockResolvedValueOnce({ topic_keys: { '/state': ['new_key'] } });
    const { store } = renderMonitor({ topics: [{ name: '/state', status: 0 }] }, {}, {}, 'first');
    await screen.findByText('first_key');
    act(() => store.dispatch(selectRobotType('old_pending')));
    expect(screen.queryByText('first_key')).not.toBeInTheDocument();
    expect(screen.getByText('/state')).toBeInTheDocument();
    act(() => store.dispatch(selectRobotType('new')));
    await screen.findByText('new_key');
    await act(async () => resolveOld({ topic_keys: { '/state': ['stale_key'] } }));
    expect(screen.getByText('new_key')).toBeInTheDocument();
    expect(screen.queryByText('stale_key')).not.toBeInTheDocument();
    expect(requestSupervisorApi.mock.calls[1][1].signal.aborted).toBe(true);
  });

  test('distinguishes observation and action keys on the same topic without counting issues twice', async () => {
    requestSupervisorApi.mockResolvedValue({ topic_roles: {
      '/shared': { observation: ['upper_body'], action: ['arm_left'] },
      '/camera/image': { observation: ['eye'] },
    } });
    renderMonitor({ topics: [
      { name: '/shared', status: 2, rateHz: 0 },
      { name: '/extra', status: 0, rateHz: 10 },
    ], cameraTopics: [{ name: '/camera/image', status: 0, rateHz: 30 }] }, {}, {}, 'robot');
    await screen.findByText('upper_body');
    const rows = screen.getAllByRole('row');
    expect(within(rows[1]).getByText('Observation')).toBeVisible();
    expect(within(rows[1]).getByText('upper_body')).toHaveAttribute('title', '/shared');
    expect(within(rows[2]).getByText('Action')).toBeVisible();
    expect(within(rows[2]).getByText('arm_left')).toHaveAttribute('title', '/shared');
    expect(within(rows[3]).getByText('Unknown')).toBeVisible();
    expect(within(rows[4]).getByText('Observation')).toBeVisible();
    expect(within(rows[4]).getByText('eye')).toBeVisible();
    expect(screen.getByText('1 issue(s)')).toBeVisible();
  });

  test('does not retain another robot\'s role while metadata is loading', async () => {
    requestSupervisorApi.mockResolvedValueOnce({ topic_roles: { '/state': { action: ['arm'] } } })
      .mockReturnValueOnce(new Promise(() => {}));
    const { store } = renderMonitor({ topics: [{ name: '/state', status: 0 }] }, {}, {}, 'first');
    await screen.findByText('Action');
    act(() => store.dispatch(selectRobotType('second')));
    expect(screen.queryByText('Action')).not.toBeInTheDocument();
    expect(screen.getByText('Unknown')).toBeVisible();
    expect(screen.getByText('/state')).toBeVisible();
  });

  test('metadata failure leaves the original topic visible', async () => {
    requestSupervisorApi.mockRejectedValue(new Error('Unavailable'));
    renderMonitor({ topics: [{ name: '/full/original/topic', status: 0 }] }, {}, {}, 'robot');
    await waitFor(() => expect(requestSupervisorApi).toHaveBeenCalledTimes(1));
    expect(screen.getByText('/full/original/topic')).toHaveAttribute('title', '/full/original/topic');
  });

  test('embedded collapse keeps header/actions mounted and hides the body without refetching', async () => {
    requestSupervisorApi.mockResolvedValue({ topic_keys: { '/state': ['upper_body'] } });
    const fold = jest.fn();
    const actions = <button onClick={fold}>Fold</button>;
    const { container, rerender, store } = renderMonitor(
      { topics: [{ name: '/state', status: 0 }] }, {}, { embedded: true, actions }, 'robot'
    );
    await screen.findByText('upper_body');
    const table = screen.getByRole('table');
    expect(screen.getByRole('columnheader', { name: 'Key' })).toBeVisible();
    // Structural assertions cover the parent framing and native hidden contract.
    // eslint-disable-next-line testing-library/no-node-access
    expect(container.firstChild).not.toHaveClass('shadow-md', 'rounded-2xl', 'border');
    // eslint-disable-next-line testing-library/no-node-access
    expect(container.firstChild).toHaveClass('h-full', 'min-h-0', 'flex', 'flex-col');
    rerender(<Provider store={store}><RecordTopicMonitor embedded actions={actions} collapsed /></Provider>);
    expect(table).toBeInTheDocument();
    expect(table).not.toBeVisible();
    // eslint-disable-next-line testing-library/no-node-access
    const body = table.parentElement;
    expect(body).toHaveAttribute('hidden');
    expect(body).toHaveClass('flex-1', 'min-h-0', 'overflow-y-auto');
    expect(screen.getByText('Topic Monitor')).toBeVisible();
    // eslint-disable-next-line testing-library/no-node-access
    expect(screen.getByText('Topic Monitor').parentElement).toHaveClass('min-h-[46px]', 'shrink-0', 'border-b');
    fireEvent.click(screen.getByRole('button', { name: 'Fold' }));
    expect(fold).toHaveBeenCalledTimes(1);
    expect(requestSupervisorApi).toHaveBeenCalledTimes(1);
    rerender(<Provider store={store}><RecordTopicMonitor embedded actions={actions} /></Provider>);
    expect(table).toBeVisible();
  });

  test('empty standalone monitor retains its frame, header, actions, and minimal body', () => {
    const { container, rerender, store } = renderMonitor(null, {}, { actions: <button>Fold</button> });
    // eslint-disable-next-line testing-library/no-node-access
    expect(container.firstChild).toHaveClass('shadow-md', 'rounded-2xl', 'border');
    expect(screen.getByText('Topic Monitor')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Fold' })).toBeVisible();
    expect(screen.getByText('No topics')).toBeVisible();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(requestSupervisorApi).not.toHaveBeenCalled();
    rerender(<Provider store={store}><RecordTopicMonitor collapsed /></Provider>);
    expect(screen.getByText('No topics')).not.toBeVisible();
    expect(screen.getByText('Topic Monitor')).toBeVisible();
  });

  test('refresh remains independent of inference and disables until completion', async () => {
    jest.useFakeTimers();
    try {
      const { sendRecordCommand, store } = renderMonitor({ topics: [{ name: '/state', status: 0 }] });
      act(() => store.dispatch({ type: 'tasks/setInferenceStatus', payload: { running: true } }));
      const refresh = screen.getByTitle('Reset topic subscriptions');
      expect(refresh).toBeEnabled();
      fireEvent.click(refresh);
      await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith('refresh_topics'));
      expect(refresh).toBeDisabled();
      fireEvent.click(refresh);
      expect(sendRecordCommand).toHaveBeenCalledTimes(1);
      act(() => jest.advanceTimersByTime(400));
      expect(refresh).toBeEnabled();
    } finally {
      jest.useRealTimers();
    }
  });

  test('running recording blocks refresh even with a ready phase', () => {
    renderMonitor(null, { running: true, recordPhase: RecordPhase.READY });
    expect(screen.getByTitle('Reset topic subscriptions')).toBeDisabled();
  });

  test('labels camera timestamp skew in the status tooltip', () => {
    renderMonitor({
      topics: [],
      cameraTopics: [
        {
          name: '/camera/image/compressed',
          cameraName: 'cam_right_wrist',
          rateHz: 30,
          baselineHz: 30,
          secondsSinceLast: 0,
          status: 2,
          timestampStatus: 2,
          timestampSkewS: 1.25,
          source: 'camera',
        },
      ],
      totalReceived: 0,
      totalWritten: 0,
    });

    expect(screen.getByTitle(/Timestamp skew/)).toBeInTheDocument();
    expect(screen.getByTitle(/skew 1.250s/)).toBeInTheDocument();
  });
});
