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
import InferenceRecordingControls from './InferenceRecordingControls';
import { EpisodeOutcome } from '../constants/taskCommand';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import taskReducer, {
  InferenceRecordingUiPhase,
  setRecordStatus,
} from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return { __esModule: true, default: toast };
});

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

const renderControls = ({
  inferenceMode = 'robot',
  recordInferenceMode = true,
  currentEpisodeNumber = 0,
  sendRecordCommand = jest.fn().mockResolvedValue({ success: true }),
  variant = 'default',
  inferenceActions = null,
  guideMessage = '',
  policyEpoch = null,
  mode = 'inference',
  inferencePhase = mode === 'recording'
    ? InferencePhase.READY
    : InferencePhase.INFERENCING,
  robotType = 'ffw_sg2_rev1',
  prepareRecording = null,
  startBlocked = false,
  segmentedRecording = false,
  segmentIndex = 0,
  canFinalize = true,
  discardEpisodeOnCancel = false,
  onEpisodeSaved = null,
  onEpisodeCancelled = null,
} = {}) => {
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        robotType,
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          inferenceMode,
          recordInferenceMode,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
        recordStatus: {
          ...initialTasks.recordStatus,
          taskType: 'inference',
          recordInferenceMode: true,
          currentEpisodeNumber,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <InferenceRecordingControls
        variant={variant}
        inferenceActions={inferenceActions}
        guideMessage={guideMessage}
        policyEpoch={policyEpoch}
        mode={mode}
        prepareRecording={prepareRecording}
        startBlocked={startBlocked}
        segmentedRecording={segmentedRecording}
        segmentIndex={segmentIndex}
        canFinalize={canFinalize}
        discardEpisodeOnCancel={discardEpisodeOnCancel}
        onEpisodeSaved={onEpisodeSaved}
        onEpisodeCancelled={onEpisodeCancelled}
      />
    </Provider>
  );
  return { store, sendRecordCommand };
};

describe('InferenceRecordingControls', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('is hidden for simulation even when the stale toggle is true', () => {
    renderControls({ inferenceMode: 'simulation' });

    expect(screen.queryByRole('button', {
      name: /record inference rollout/i,
    })).not.toBeInTheDocument();
  });

  test.each([
    ['simulation deploy', { inferenceMode: 'simulation' }],
    ['disabled recording mode', { recordInferenceMode: false }],
  ])('workspace stays visible but disables Record for %s', (_label, overrides) => {
    renderControls({ variant: 'workspace', ...overrides });

    expect(screen.getByRole('group', {
      name: /^inference recording controls$/i,
    })).toBeInTheDocument();
    expect(screen.getByText('Inference Recording')).toBeInTheDocument();
    expect(screen.getByRole('button', {
      name: /record inference rollout/i,
    })).toBeDisabled();
  });

  test('separates the policy RL Epoch from the recording episode cursor', () => {
    renderControls({
      variant: 'workspace',
      policyEpoch: 3,
      currentEpisodeNumber: 12,
    });

    expect(screen.getByLabelText('Current policy RL Epoch 3'))
      .toHaveTextContent('RL Epoch E0003');
    expect(screen.getByText('Episodes')).toBeInTheDocument();
    expect(screen.getByLabelText('Saved inference episodes')).toHaveTextContent('12');
  });

  test('workspace enables the three outcome actions only while recording', async () => {
    renderControls({ variant: 'workspace' });

    const success = screen.getByRole('button', {
      name: /save inference rollout as success/i,
    });
    const failed = screen.getByRole('button', {
      name: /save inference rollout as fail/i,
    });
    const cancel = screen.getByRole('button', {
      name: /cancel and discard inference rollout/i,
    });
    expect(success).toBeDisabled();
    expect(failed).toBeDisabled();
    expect(cancel).toBeDisabled();

    fireEvent.click(screen.getByRole('button', {
      name: /record inference rollout/i,
    }));

    await waitFor(() => {
      expect(success).toBeEnabled();
      expect(failed).toBeEnabled();
      expect(cancel).toBeEnabled();
    });
  });

  test('records a standalone rollout into one stable Replay Buffer session', async () => {
    const recordingOptions = {
      recordingFolder: '/workspace/rosbag2/Task_manual_inference_MCAP',
      taskInstruction: ['ACT_dataset'],
      includeRobotisLicense: true,
    };
    const prepareRecording = jest.fn(() => recordingOptions);
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({
      variant: 'workspace',
      mode: 'recording',
      prepareRecording,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }));

    await waitFor(() => {
      expect(prepareRecording).toHaveBeenCalledTimes(1);
      expect(sendRecordCommand).toHaveBeenCalledWith('start_record', {
        ...recordingOptions,
        taskSource: 'inference',
      });
    });
    expect(screen.getByRole('group', { name: 'Recording controls' }))
      .toBeInTheDocument();

    fireEvent.click(await screen.findByRole('button', {
      name: 'Save recording as Success',
    }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenLastCalledWith(
        'stop_inference_record',
        {
          ...recordingOptions,
          episodeOutcome: EpisodeOutcome.SUCCESS,
          taskSource: 'inference',
        }
      );
    });
  });

  test('cancels standalone recording with the same stable session target', async () => {
    const recordingOptions = {
      recordingFolder: '/workspace/rosbag2/Task_cancel_inference_MCAP',
    };
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({
      variant: 'workspace',
      mode: 'recording',
      prepareRecording: () => recordingOptions,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }));
    fireEvent.click(await screen.findByRole('button', {
      name: 'Cancel and discard recording',
    }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenLastCalledWith(
        'cancel_inference_record',
        {
          ...recordingOptions,
          taskSource: 'inference',
        }
      );
    });
  });

  test('saves a standalone recording with a Failure outcome', async () => {
    const recordingOptions = {
      recordingFolder: '/workspace/rosbag2/Task_failed_inference_MCAP',
    };
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({
      variant: 'workspace',
      mode: 'recording',
      prepareRecording: () => recordingOptions,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }));
    fireEvent.click(await screen.findByRole('button', {
      name: 'Save recording as Fail',
    }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenLastCalledWith(
        'stop_inference_record',
        {
          ...recordingOptions,
          episodeOutcome: EpisodeOutcome.FAILURE,
          taskSource: 'inference',
        }
      );
    });
  });

  test('starts a segmented recording and keeps Cancel available before the final subtask', async () => {
    const recordingOptions = {
      recordingFolder: '/workspace/rosbag2/Task_segments_inference_MCAP',
      subtaskInstruction: ['Approach', 'Grasp'],
      segmentIndex: 0,
    };
    const onEpisodeCancelled = jest.fn();
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({
      variant: 'workspace',
      mode: 'recording',
      prepareRecording: () => recordingOptions,
      segmentedRecording: true,
      segmentIndex: 0,
      canFinalize: false,
      discardEpisodeOnCancel: true,
      onEpisodeCancelled,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_segment', {
        ...recordingOptions,
        taskSource: 'inference',
      });
    });
    expect(screen.getByRole('button', {
      name: 'Save recording as Success',
    })).toBeDisabled();
    expect(screen.getByRole('button', {
      name: 'Save recording as Fail',
    })).toBeDisabled();
    expect(screen.getByRole('button', {
      name: 'Cancel and discard recording',
    })).toBeEnabled();

    fireEvent.click(screen.getByRole('button', {
      name: 'Cancel and discard recording',
    }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenNthCalledWith(
        2,
        'cancel_inference_record',
        { ...recordingOptions, taskSource: 'inference' }
      );
      expect(sendRecordCommand).toHaveBeenNthCalledWith(
        3,
        'discard_episode',
        { ...recordingOptions, segmentIndex: 0, taskSource: 'inference' }
      );
      expect(onEpisodeCancelled).toHaveBeenCalledTimes(1);
    });
  });

  test.each([
    ['without a robot', { robotType: '' }],
    ['while inference is active', {
      inferencePhase: InferencePhase.INFERENCING,
    }],
  ])('blocks standalone recording %s', (_label, overrides) => {
    renderControls({
      variant: 'workspace',
      mode: 'recording',
      ...overrides,
    });
    expect(screen.getByRole('button', { name: 'Start recording' })).toBeDisabled();
  });

  test('workspace places inference status and actions beside Record', () => {
    renderControls({
      variant: 'workspace',
      guideMessage: 'Ready to start',
      inferenceActions: [
        <button key="start" type="button">Start</button>,
        <button key="stop" type="button">Stop</button>,
        <button key="clear" type="button">Clear</button>,
      ],
    });

    const group = screen.getByRole('group', {
      name: /^inference recording controls$/i,
    });
    expect(within(group).getByRole('status')).toHaveTextContent('Ready to start');
    expect(within(group).getAllByRole('button').slice(0, 4)
      .map((button) => button.textContent.trim()))
      .toEqual(['Record', 'Start', 'Stop', 'Clear']);
  });

  test('sends one start request and labels the rollout with an outcome', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({ sendRecordCommand });

    fireEvent.click(screen.getByRole('button', {
      name: /record inference rollout/i,
    }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        { taskSource: 'inference' }
      );
    });

    fireEvent.click(await screen.findByRole('button', {
      name: /save inference rollout as success/i,
    }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenLastCalledWith(
        'stop_inference_record',
        {
          episodeOutcome: EpisodeOutcome.SUCCESS,
          taskSource: 'inference',
        }
      );
    });
  });

  test('shows the saved episode count from recording status', () => {
    renderControls({ currentEpisodeNumber: 7 });

    expect(screen.getByLabelText(/saved rl episodes/i)).toHaveTextContent('7');
  });

  test('cancels the active rollout without applying an outcome', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      currentEpisodeNumber: 4,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', {
      name: /record inference rollout/i,
    }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        { taskSource: 'inference' }
      );
    });

    fireEvent.click(screen.getByRole('button', {
      name: /cancel and discard inference rollout/i,
    }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenLastCalledWith(
        'cancel_inference_record',
        { taskSource: 'inference' }
      );
    });
    expect(screen.getByLabelText(/saved rl episodes/i)).toHaveTextContent('4');
    expect(store.getState().tasks.inferenceRecordingUi.phase)
      .toBe(InferenceRecordingUiPhase.CANCELLING);
  });

  test('blocks duplicate Record clicks while the RPC is pending', () => {
    const sendRecordCommand = jest.fn(() => new Promise(() => {}));
    renderControls({ sendRecordCommand });
    const record = screen.getByRole('button', {
      name: /record inference rollout/i,
    });

    fireEvent.click(record);
    fireEvent.click(record);

    expect(sendRecordCommand).toHaveBeenCalledTimes(1);
  });

  test('recovers from an RPC error using the latest server status', async () => {
    let rejectCommand;
    const sendRecordCommand = jest.fn(() => new Promise((_, reject) => {
      rejectCommand = reject;
    }));
    const { store } = renderControls({ sendRecordCommand });

    fireEvent.click(screen.getByRole('button', {
      name: /record inference rollout/i,
    }));
    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
      }));
      rejectCommand(new Error('RPC timeout'));
    });

    await waitFor(() => {
      expect(store.getState().tasks.inferenceRecordingUi.phase)
        .toBe(InferenceRecordingUiPhase.RECORDING);
    });
  });
});
