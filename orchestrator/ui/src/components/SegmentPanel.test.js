import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import toast from 'react-hot-toast';
import SegmentPanel from './SegmentPanel';
import taskReducer, { setRecordStatus, setRecordTaskInfo } from '../features/tasks/taskSlice';
import { RecordPhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return {
    __esModule: true,
    default: toast,
  };
});

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('./InfoPanel', () => function MockInfoPanel() {
  return <div data-testid="info-panel" />;
});

const renderPanel = ({ taskOverrides = {}, sendRecordCommand, panelProps = {} } = {}) => {
  const sendCommand = sendRecordCommand || jest.fn().mockResolvedValue({
    success: true,
    message: 'ok',
  });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand: sendCommand });

  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const tasks = {
    ...initialTasks,
    recordTaskInfo: {
      ...initialTasks.recordTaskInfo,
      taskNum: '1',
      taskName: 'discard-target-test',
    },
    taskInfo: {
      ...initialTasks.taskInfo,
      taskNum: '1',
      taskName: 'discard-target-test',
    },
    plannedCount: 2,
    plannedSubTasks: ['pick', 'place'],
    slotToServerIdx: [0, -1],
    activeSlotIndex: 1,
    ...taskOverrides,
    recordStatus: {
      ...initialTasks.recordStatus,
      taskNum: '1',
      taskName: 'discard-target-test',
      recordPhase: RecordPhase.READY,
      currentEpisodeNumber: 7,
      currentSubtaskIndex: 1,
      subtaskCount: 2,
      topicReceived: true,
      ...taskOverrides.recordStatus,
    },
  };
  const store = configureStore({
    reducer: {
      tasks: taskReducer,
    },
    preloadedState: {
      tasks,
    },
  });

  render(
    <Provider store={store}>
      <SegmentPanel {...panelProps} />
    </Provider>
  );

  return { sendRecordCommand: sendCommand, store };
};

describe('SegmentPanel discard episode target', () => {
  test('inactive record shortcuts do not discard an active segment', () => {
    const { sendRecordCommand } = renderPanel({
      panelProps: { shortcutsEnabled: false },
      taskOverrides: { recordStatus: { recordPhase: RecordPhase.RECORDING, running: true, topicReceived: true } },
    });
    fireEvent.keyUp(window, { key: 'Escape' });
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('shared toolbar saves using the existing final-segment sequence with inactive shortcuts', async () => {
    const target = document.createElement('div');
    document.body.appendChild(target);
    const { sendRecordCommand } = renderPanel({
      panelProps: { shortcutsEnabled: false, actionsTarget: target },
      taskOverrides: { recordStatus: {
        recordPhase: RecordPhase.RECORDING, running: true, topicReceived: true,
        currentEpisodeNumber: 7, currentSubtaskIndex: 1, subtaskCount: 2,
      } },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save recording' }));
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith('finish_episode', expect.any(Object)));
    expect(sendRecordCommand.mock.calls.map(([command]) => command)).toEqual(['stop_segment', 'finish_episode']);
    expect(screen.getByRole('button', { name: 'Discard recording episode' })).toBeDisabled();
    target.remove();
  });
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test.each([false, true])('remount during saving keeps all recording mutations blocked (running=%s)', (running) => {
    const { sendRecordCommand } = renderPanel({ taskOverrides: {
      recordStatus: {
        taskType: 'record', recordPhase: RecordPhase.SAVING, running,
        currentEpisodeNumber: 7, subtaskCount: 2, currentSubtaskIndex: 1, topicReceived: true,
      },
    } });
    expect(screen.getByRole('button', { name: /discard episode/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /record start/i })).toBeDisabled();
    for (const button of screen.getAllByTitle(/Save subtask|Cancel current recording|Discard saved subtask/)) {
      expect(button).toBeDisabled();
    }
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('sends the captured full episode index when discarding an episode', async () => {
    const { sendRecordCommand } = renderPanel();

    const discardButton = await screen.findByRole('button', {
      name: /discard episode/i,
    });
    await waitFor(() => expect(discardButton).toBeEnabled());

    fireEvent.click(discardButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'discard_episode',
        expect.objectContaining({
          segmentIndex: 8,
          subtaskInstruction: ['pick', 'place'],
        })
      );
    });
    expect(toast.success).toHaveBeenCalledWith('Discard episode: ok');
  });

  test('discard after a task switch and segment cancel targets the new episode and allows recording again', async () => {
    const { store, sendRecordCommand } = renderPanel();
    const discard = screen.getByRole('button', { name: /discard episode/i });
    act(() => store.dispatch(setRecordTaskInfo({ taskNum: '12', taskName: 'discard-target-test' })));
    expect(discard).toBeDisabled(); // The latest status still belongs to task 1.
    act(() => store.dispatch(setRecordStatus({
      taskNum: '12', currentEpisodeNumber: 0, currentSubtaskIndex: 1, savedSubtaskIndices: [0],
      running: true, recordPhase: RecordPhase.RECORDING,
    })));
    act(() => store.dispatch(setRecordStatus({ running: false, recordPhase: RecordPhase.READY })));
    fireEvent.click(discard);
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith(
      'discard_episode', expect.objectContaining({ segmentIndex: 1 })
    ));
    const record = screen.getByRole('button', { name: /record start/i });
    expect(record).toBeDisabled(); // Wait for the empty server status, not just the service ack.
    act(() => store.dispatch(setRecordStatus({ currentSubtaskIndex: 0, savedSubtaskIndices: [] })));
    await waitFor(() => expect(record).toBeEnabled());
    expect(discard).toBeDisabled();
    fireEvent.click(record);
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith(
      'start_segment', expect.objectContaining({ segmentIndex: 0 })
    ));
  });

  test('drops duplicate save clicks while a save sequence is in flight', async () => {
    const sendRecordCommand = jest.fn(() => new Promise(() => {}));
    renderPanel({
      sendRecordCommand,
      taskOverrides: {
        recordStatus: {
          recordPhase: RecordPhase.RECORDING,
          running: true,
          currentEpisodeNumber: 7,
          currentSubtaskIndex: 0,
          subtaskCount: 2,
          topicReceived: true,
        },
        slotToServerIdx: [-1, -1],
        activeSlotIndex: 0,
      },
    });

    const saveButton = await screen.findByRole('button', {
      name: /save subtask 1/i,
    });

    fireEvent.click(saveButton);
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledTimes(1);
    });
    expect(sendRecordCommand).toHaveBeenCalledWith(
      'stop_segment',
      expect.objectContaining({
        segmentIndex: 0,
      })
    );
  });

  test('warns about stalled rosbag topics when starting a record segment', async () => {
    const { sendRecordCommand } = renderPanel({
      taskOverrides: {
        recordingMonitor: {
          topics: [{
            name: '/leader/joint_trajectory',
            secondsSinceLast: -1,
            status: 2,
          }],
          cameraTopics: [],
          totalReceived: 0,
          totalWritten: 0,
        },
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /record start/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        expect.stringContaining('/leader/joint_trajectory'),
        expect.objectContaining({ duration: 9000 })
      );
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_segment',
        expect.objectContaining({
          segmentIndex: 1,
        })
      );
    });
  });

  test('does not restore saved checkmarks from stale status after discarding episode', async () => {
    const { sendRecordCommand, store } = renderPanel({
      taskOverrides: {
        recordStatus: {
          recordPhase: RecordPhase.RECORDING,
          running: true,
          currentEpisodeNumber: 0,
          currentSubtaskIndex: 2,
          subtaskCount: 3,
          topicReceived: true,
        },
        plannedCount: 3,
        plannedSubTasks: ['pick', 'place', 'release'],
        slotToServerIdx: [0, 1, -1],
        activeSlotIndex: 2,
      },
    });

    const discardButton = await screen.findByRole('button', {
      name: /discard episode/i,
    });
    await waitFor(() => expect(discardButton).toBeEnabled());

    fireEvent.click(discardButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'discard_episode',
        expect.objectContaining({ segmentIndex: 1 })
      );
      expect(store.getState().tasks.slotToServerIdx).toEqual([-1, -1, -1]);
    });

    await act(async () => {
      store.dispatch(setRecordStatus({
        recordPhase: RecordPhase.READY,
        running: false,
        currentEpisodeNumber: 0,
        currentSubtaskIndex: 2,
        subtaskCount: 3,
        savedSubtaskIndices: [],
        topicReceived: true,
      }));
      await Promise.resolve();
    });

    expect(store.getState().tasks.slotToServerIdx).toEqual([-1, -1, -1]);
    expect(store.getState().tasks.activeSlotIndex).toBe(0);
  });

  test('uses saved subtask indices as authoritative status instead of cursor inference', async () => {
    const { store } = renderPanel({
      taskOverrides: {
        recordStatus: {
          recordPhase: RecordPhase.READY,
          running: false,
          currentEpisodeNumber: 0,
          currentSubtaskIndex: 2,
          subtaskCount: 3,
          savedSubtaskIndices: [0, 2],
          topicReceived: true,
        },
        plannedCount: 3,
        plannedSubTasks: ['pick', 'place', 'release'],
        slotToServerIdx: [0, 1, 2],
        activeSlotIndex: 2,
      },
    });

    await waitFor(() => {
      expect(store.getState().tasks.slotToServerIdx).toEqual([0, -1, 2]);
      expect(store.getState().tasks.activeSlotIndex).toBe(1);
    });
  });

  test('finishes episode after re-recording a middle missing subtask', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({
      success: true,
      message: 'ok',
    });
    renderPanel({
      sendRecordCommand,
      taskOverrides: {
        recordStatus: {
          recordPhase: RecordPhase.RECORDING,
          running: true,
          currentEpisodeNumber: 0,
          currentSubtaskIndex: 1,
          subtaskCount: 3,
          savedSubtaskIndices: [0, 2],
          topicReceived: true,
        },
        plannedCount: 3,
        plannedSubTasks: ['pick', 'place', 'release'],
        slotToServerIdx: [0, -1, 2],
        activeSlotIndex: 1,
      },
    });

    const saveButton = await screen.findByRole('button', {
      name: /save subtask 2/i,
    });
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_segment',
        expect.objectContaining({ segmentIndex: 1 })
      );
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'finish_episode',
        expect.objectContaining({
          subtaskInstruction: ['pick', 'place', 'release'],
        })
      );
    });
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'start_segment',
      expect.anything()
    );
  });

  test('does not restore saved checkmarks from stale status while finish is pending', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({
      success: true,
      message: 'ok',
    });
    const { store } = renderPanel({
      sendRecordCommand,
      taskOverrides: {
        recordStatus: {
          recordPhase: RecordPhase.RECORDING,
          running: true,
          currentEpisodeNumber: 0,
          currentSubtaskIndex: 2,
          subtaskCount: 3,
          savedSubtaskIndices: [0, 1],
          topicReceived: true,
        },
        plannedCount: 3,
        plannedSubTasks: ['pick', 'place', 'release'],
        slotToServerIdx: [0, 1, -1],
        activeSlotIndex: 2,
      },
    });

    const saveButton = await screen.findByRole('button', {
      name: /save subtask 3/i,
    });
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'finish_episode',
        expect.objectContaining({
          subtaskInstruction: ['pick', 'place', 'release'],
        })
      );
      expect(store.getState().tasks.slotToServerIdx).toEqual([-1, -1, -1]);
    });

    await act(async () => {
      store.dispatch(setRecordStatus({
        recordPhase: RecordPhase.READY,
        running: false,
        currentEpisodeNumber: 0,
        currentSubtaskIndex: 2,
        subtaskCount: 3,
        savedSubtaskIndices: [0, 1, 2],
        topicReceived: true,
      }));
      await Promise.resolve();
    });

    expect(store.getState().tasks.slotToServerIdx).toEqual([-1, -1, -1]);
    expect(store.getState().tasks.activeSlotIndex).toBe(0);

    await act(async () => {
      store.dispatch(setRecordStatus({
        recordPhase: RecordPhase.READY,
        running: false,
        currentEpisodeNumber: 1,
        currentSubtaskIndex: 0,
        subtaskCount: 3,
        savedSubtaskIndices: [],
        topicReceived: true,
      }));
      await Promise.resolve();
    });

    expect(store.getState().tasks.slotToServerIdx).toEqual([-1, -1, -1]);
    expect(store.getState().tasks.activeSlotIndex).toBe(0);
  });

  test('releases finish pending lock when background archive fails', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({
      success: true,
      message: 'ok',
    });
    const { store } = renderPanel({
      sendRecordCommand,
      taskOverrides: {
        recordStatus: {
          recordPhase: RecordPhase.RECORDING,
          running: true,
          currentEpisodeNumber: 0,
          currentSubtaskIndex: 2,
          subtaskCount: 3,
          savedSubtaskIndices: [0, 1],
          topicReceived: true,
        },
        plannedCount: 3,
        plannedSubTasks: ['pick', 'place', 'release'],
        slotToServerIdx: [0, 1, -1],
        activeSlotIndex: 2,
      },
    });

    const saveButton = await screen.findByRole('button', {
      name: /save subtask 3/i,
    });
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'finish_episode',
        expect.anything()
      );
      expect(store.getState().tasks.slotToServerIdx).toEqual([-1, -1, -1]);
    });

    await act(async () => {
      store.dispatch(setRecordStatus({
        recordPhase: RecordPhase.READY,
        running: false,
        currentEpisodeNumber: 0,
        currentSubtaskIndex: 2,
        subtaskCount: 3,
        savedSubtaskIndices: [0, 1, 2],
        recordingOperationStatus: 'failed',
        recordingOperationStage: 'FINISH_EPISODE',
        recordingOperationMessage: 'Episode finish failed: disk full',
        topicReceived: true,
      }));
      await Promise.resolve();
    });

    expect(store.getState().tasks.slotToServerIdx).toEqual([0, 1, 2]);
    expect(store.getState().tasks.activeSlotIndex).toBe(2);
    expect(toast.error).toHaveBeenCalledWith('Episode finish failed: disk full');
  });

});
