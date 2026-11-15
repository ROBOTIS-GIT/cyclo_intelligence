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
  setInferenceRecordingUiPhase,
  setInferenceStatus,
  setInferenceTaskInfo,
  setRecordStatus,
} from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import {
  getInterventionState,
  sendInterventionAction,
  sendOperatorPose,
} from '../utils/interventionApi';

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return { __esModule: true, default: toast };
});

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('../utils/interventionApi', () => ({
  getInterventionState: jest.fn(),
  sendInterventionAction: jest.fn(),
  sendOperatorPose: jest.fn(),
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
  recordingTags = [],
  partialRecordMode = false,
  showIndependentInterventionRecord = true,
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
          tags: recordingTags,
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
        partialRecordMode={partialRecordMode}
        showIndependentInterventionRecord={showIndependentInterventionRecord}
      />
    </Provider>
  );
  return { store, sendRecordCommand };
};

describe('InferenceRecordingControls', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    window.sessionStorage.clear();
    getInterventionState.mockResolvedValue({
      active: false,
      current: null,
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
      standby: {
        active: false,
        current: { model: 'mini', phase: 'READY' },
        ready: true,
      },
    });
    sendInterventionAction.mockResolvedValue({ ok: true });
    sendOperatorPose.mockResolvedValue({ ok: true });
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

  test('workspace continues refreshing pose safety state after task mode resets', async () => {
    renderControls({
      variant: 'workspace',
      inferenceMode: 'simulation',
      recordInferenceMode: false,
    });

    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());
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
    expect(within(group).getByRole('button', {
      name: /start total recording/i,
    })).toHaveTextContent('Total Record');
    expect(within(group).getByRole('button', {
      name: /record inference rollout/i,
    })).toHaveTextContent('Inference Record');
    expect(within(group).getByRole('button', {
      name: /start intervention recording/i,
    })).toHaveTextContent('Intervention Record');
    expect(['Start', 'Stop', 'Clear'].map((label) => (
      within(group).getByRole('button', { name: label }).textContent.trim()
    ))).toEqual(['Start', 'Stop', 'Clear']);
  });

  test('playground exposes Partial Record without a standalone Intervention Record', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      variant: 'workspace',
      partialRecordMode: true,
      showIndependentInterventionRecord: false,
      sendRecordCommand,
    });

    expect(screen.queryByRole('button', {
      name: /start intervention recording/i,
    })).not.toBeInTheDocument();
    expect(screen.getByRole('group', {
      name: /inference control mode/i,
    })).toBeInTheDocument();
    const autoTriggers = screen.getByRole('group', {
      name: /automatic partial record triggers/i,
    });
    const interventionAuto = within(autoTriggers).getByRole('button', {
      name: /on intervention on/i,
    });
    const mlpAuto = within(autoTriggers).getByRole('button', {
      name: /when mlp becomes active/i,
    });
    expect(interventionAuto).toHaveAttribute('aria-pressed', 'true');
    expect(mlpAuto).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(interventionAuto);
    fireEvent.click(mlpAuto);
    expect(interventionAuto).toHaveAttribute('aria-pressed', 'false');
    expect(mlpAuto).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(interventionAuto);
    fireEvent.click(mlpAuto);
    expect(interventionAuto).toHaveAttribute('aria-pressed', 'true');
    expect(mlpAuto).toHaveAttribute('aria-pressed', 'true');
    const partial = screen.getByRole('button', {
      name: /start partial recording/i,
    });
    expect(partial).toHaveTextContent('Partial Record');

    fireEvent.click(partial);
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith(
      'start_inference_record',
      { recordingSource: 'inference', taskSource: 'inference' }
    ));
    expect(screen.getByRole('group', {
      name: /partial record control mode/i,
    })).toBeInTheDocument();
    expect(screen.getByRole('button', {
      name: /switch partial recording to intervention mode/i,
    })).toBeInTheDocument();
    expect(screen.getByRole('button', {
      name: /save partial recording as success/i,
    })).toBeEnabled();
    expect(screen.getByRole('button', {
      name: /save partial recording as fail/i,
    })).toBeEnabled();
    expect(screen.getByRole('button', {
      name: /cancel and discard partial recording/i,
    })).toBeEnabled();

    // A delayed READY echo can arrive before the recorder publishes its first
    // RECORDING status. The Total-style mode switch must remain visible.
    act(() => {
      store.dispatch(setRecordStatus({ recordPhase: RecordPhase.READY }));
    });
    expect(screen.getByRole('group', {
      name: /partial record control mode/i,
    })).toBeInTheDocument();
  });

  test('starts Partial Record when the confirmed MLP route becomes active', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      variant: 'workspace',
      partialRecordMode: true,
      showIndependentInterventionRecord: false,
      sendRecordCommand,
    });

    act(() => {
      store.dispatch(setInferenceTaskInfo({
        serviceType: 'groot',
        policyType: 'n17',
        actionRequestMode: 'tt_rtc',
        actionPolicyMode: 'rlt',
      }));
    });

    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith(
      'start_inference_record',
      {
        forceInferenceRecording: true,
        recordingSource: 'inference',
        taskSource: 'inference',
      }
    ));
    expect(screen.getByRole('button', {
      name: /start partial recording/i,
    })).toHaveTextContent('Recording');
  });

  test('records intervention as Partial and exposes Success Fail Cancel outcomes', async () => {
    let serverState = {
      active: false,
      current: null,
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
      standby: { active: false, current: null, ready: false },
    };
    getInterventionState.mockImplementation(async () => serverState);
    sendInterventionAction.mockImplementation(async (action) => {
      if (action === 'start') {
        serverState = {
          ...serverState,
          active: true,
          current: { phase: 'READY', leader_model: 'a3' },
          inference_phase: 'PAUSED',
          arbiter: { ...serverState.arbiter, mode: 'TELEOP' },
        };
      } else if (action === 'recording') {
        serverState = {
          ...serverState,
          current: { ...serverState.current, phase: 'RECORDING' },
        };
      }
      return { ok: true, ...serverState };
    });
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({
      variant: 'workspace',
      partialRecordMode: true,
      showIndependentInterventionRecord: false,
      sendRecordCommand,
    });

    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button', {
      name: /use a3 leader for intervention/i,
    }));
    const intervention = screen.getByRole('button', {
      name: /switch inference to intervention mode/i,
    });
    await waitFor(() => expect(intervention).toBeEnabled());
    fireEvent.click(intervention);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('stop_inference', {
        taskSource: 'inference',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('start', {
        model: 'a3',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'inference',
          taskSource: 'inference',
        }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('recording', {
        model: 'a3',
      });
    });
    const success = screen.getByRole('button', {
      name: /save partial intervention episode as success/i,
    });
    expect(success).toBeEnabled();
    expect(screen.getByRole('button', {
      name: /save partial intervention episode as fail/i,
    })).toBeEnabled();
    expect(screen.getByRole('button', {
      name: /cancel and discard partial intervention episode/i,
    })).toBeEnabled();

    fireEvent.click(success);
    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('end', {
        model: 'a3',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_inference_record',
        {
          episodeOutcome: EpisodeOutcome.SUCCESS,
          recordingSource: 'inference',
          taskSource: 'inference',
        }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('reset', {
        model: 'a3',
      });
    });
  });

  test('keeps one Partial Record open while handing control to a leader', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      variant: 'workspace',
      partialRecordMode: true,
      showIndependentInterventionRecord: false,
      sendRecordCommand,
    });

    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button', {
      name: /use a3 leader for intervention/i,
    }));
    fireEvent.click(screen.getByRole('button', {
      name: /start partial recording/i,
    }));
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith(
      'start_inference_record',
      { recordingSource: 'inference', taskSource: 'inference' }
    ));

    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
        tags: ['recording_source:inference'],
      }));
    });

    const intervention = await screen.findByRole('button', {
      name: /switch partial recording to intervention mode/i,
    });
    await waitFor(() => expect(intervention).toBeEnabled());
    fireEvent.click(intervention);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('stop_inference', {
        taskSource: 'inference',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('start', {
        model: 'a3',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('recording', {
        model: 'a3',
      });
    });
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'stop_inference_record',
      expect.anything()
    );
  });

  test('returns leader control to inference without closing the Partial Record', async () => {
    let serverState = {
      active: true,
      current: {
        phase: 'RECORDING',
        leader_model: 'a3',
        recording_continued: true,
      },
      inference_status_ready: true,
      inference_phase: 'PAUSED',
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: {
        available: true,
        mode: 'TELEOP',
        fault: '',
        publisher_conflict: '',
        intervention_enabled: true,
      },
      standby: { active: false, current: null, ready: false },
    };
    getInterventionState.mockImplementation(async () => serverState);
    sendInterventionAction.mockImplementation(async (action) => {
      if (action === 'end') {
        serverState = {
          ...serverState,
          current: { ...serverState.current, phase: 'READY_TO_RESUME' },
          arbiter: {
            ...serverState.arbiter,
            mode: 'NONE',
            intervention_enabled: false,
          },
        };
      } else if (action === 'complete') {
        serverState = {
          ...serverState,
          active: false,
          current: null,
          inference_phase: 'INFERENCING',
          arbiter: { ...serverState.arbiter, mode: 'POLICY' },
        };
      }
      return { ok: true, ...serverState };
    });
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      variant: 'workspace',
      partialRecordMode: true,
      showIndependentInterventionRecord: false,
      inferencePhase: InferencePhase.PAUSED,
      sendRecordCommand,
    });
    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
        tags: ['recording_source:inference'],
      }));
      store.dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.RECORDING
      ));
    });

    const controlMode = await screen.findByRole('group', {
      name: /partial record control mode/i,
    });
    const returnToInference = within(controlMode).getByRole('button', {
      name: /switch partial recording to inference mode/i,
    });
    await waitFor(() => expect(returnToInference).toBeEnabled());
    fireEvent.click(returnToInference);

    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('end', {
        model: 'a3',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith('resume_inference', {
        taskSource: 'inference',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('complete', {
        model: 'a3',
      });
    });
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'stop_inference_record',
      expect.anything()
    );
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'cancel_inference_record',
      expect.anything()
    );
  });

  test('requires selecting Mini or A3 before Intervention Record starts', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({ variant: 'workspace', sendRecordCommand });

    const interventionRecord = screen.getByRole('button', {
      name: /start intervention recording/i,
    });
    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());
    expect(interventionRecord).toBeDisabled();

    fireEvent.click(screen.getByRole('button', {
      name: /use mini leader for intervention/i,
    }));
    await waitFor(() => expect(interventionRecord).toBeEnabled());
    fireEvent.click(interventionRecord);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenNthCalledWith(1, 'stop_inference', {
        taskSource: 'inference',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('start', {
        model: 'mini',
      });
      expect(sendRecordCommand).toHaveBeenNthCalledWith(
        2,
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'intervention',
          taskSource: 'inference',
        }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('recording', {
        model: 'mini',
      });
    });
  });

  test('warms Mini in isolation before enabling Intervention Record', async () => {
    let resolveStandby;
    const standbyRequest = new Promise((resolve) => {
      resolveStandby = resolve;
    });
    getInterventionState
      .mockResolvedValueOnce({
        active: false,
        current: null,
        preflight: { ok: true, failures: [] },
        fault_reset: { ok: false, failures: [] },
        arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
        standby: { active: false, current: null, ready: false },
      })
      .mockResolvedValue({
        active: false,
        current: null,
        preflight: { ok: true, failures: [] },
        fault_reset: { ok: false, failures: [] },
        arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
        standby: {
          active: true,
          current: { model: 'mini', phase: 'READY' },
          ready: true,
        },
      });
    sendInterventionAction.mockImplementationOnce(() => standbyRequest);
    renderControls({ variant: 'workspace' });
    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());

    fireEvent.click(screen.getByRole('button', {
      name: /use mini leader for intervention/i,
    }));

    expect(sendInterventionAction).toHaveBeenCalledWith('standby', {
      model: 'mini',
    });
    expect(screen.getByRole('button', {
      name: /start intervention recording/i,
    })).toHaveTextContent('Connecting Mini');
    expect(screen.getByRole('button', {
      name: /start intervention recording/i,
    })).toBeDisabled();

    await act(async () => {
      resolveStandby({ ok: true });
      await standbyRequest;
    });

    await waitFor(() => expect(screen.getByRole('button', {
      name: /start intervention recording/i,
    })).toBeEnabled());
  });

  test('disconnects an active Mini standby from the leader selector', async () => {
    getInterventionState
      .mockResolvedValueOnce({
        active: false,
        current: null,
        preflight: { ok: true, failures: [] },
        fault_reset: { ok: false, failures: [] },
        arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
        standby: {
          active: true,
          current: { model: 'mini', phase: 'READY' },
          ready: true,
        },
      })
      .mockResolvedValue({
        active: false,
        current: null,
        preflight: { ok: true, failures: [] },
        fault_reset: { ok: false, failures: [] },
        arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
        standby: { active: false, current: null, ready: false },
      });
    renderControls({ variant: 'workspace' });

    const disconnect = await screen.findByRole('button', {
      name: /disconnect mini leader/i,
    });
    fireEvent.click(disconnect);

    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('standby_stop', {
        model: 'mini',
      });
      expect(screen.queryByRole('button', {
        name: /disconnect mini leader/i,
      })).not.toBeInTheDocument();
      expect(screen.getByRole('status', {
        name: /mini leader standby status/i,
      })).toHaveTextContent('Offline');
    });
  });

  test('uses an already-ready server Mini without another selection click', async () => {
    getInterventionState.mockResolvedValue({
      active: false,
      current: null,
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
      standby: {
        active: true,
        current: { model: 'mini', phase: 'READY' },
        ready: true,
      },
    });
    renderControls({ variant: 'workspace' });

    await waitFor(() => expect(screen.getByRole('button', {
      name: /start intervention recording/i,
    })).toBeEnabled());
    expect(sendInterventionAction).not.toHaveBeenCalled();
  });

  test('does not overwrite an explicit A3 choice with another Mini standby', async () => {
    window.sessionStorage.setItem('cyclo-intervention-leader', 'a3');
    getInterventionState.mockResolvedValue({
      active: false,
      current: null,
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
      standby: {
        active: true,
        current: { model: 'mini', phase: 'READY' },
        ready: true,
      },
    });
    renderControls({ variant: 'workspace' });

    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());
    expect(screen.getByRole('button', {
      name: /use a3 leader for intervention/i,
    })).toHaveAttribute('aria-pressed', 'true');
  });

  test('does not let a stale client recording phase lock Intervention ON', async () => {
    const { store } = renderControls({ variant: 'workspace' });
    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button', {
      name: /use mini leader for intervention/i,
    }));
    await waitFor(() => expect(screen.getByRole('button', {
      name: /start intervention recording/i,
    })).toBeEnabled());
    act(() => {
      store.dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.STARTING
      ));
    });

    expect(screen.getByRole('button', {
      name: /start intervention recording/i,
    })).toBeEnabled();
  });

  test('shows and starts Intervention Record without rollout recording opt-in', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({ recordInferenceMode: false, sendRecordCommand });

    const interventionRecord = screen.getByRole('button', {
      name: /start intervention recording/i,
    });
    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());
    fireEvent.click(screen.getByRole('button', {
      name: /use mini leader for intervention/i,
    }));
    await waitFor(() => expect(interventionRecord).toBeEnabled());
    fireEvent.click(interventionRecord);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'intervention',
          taskSource: 'inference',
        }
      );
    });
  });

  test('closes a standalone inference part before a new intervention part', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({ variant: 'workspace', sendRecordCommand });
    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
        tags: ['recording_source:inference'],
      }));
    });

    fireEvent.click(screen.getByRole('button', {
      name: /use a3 leader for intervention/i,
    }));
    const interventionRecord = screen.getByRole('button', {
      name: /start intervention recording/i,
    });
    await waitFor(() => expect(interventionRecord).toBeEnabled());
    fireEvent.click(interventionRecord);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenNthCalledWith(
        1,
        'stop_inference',
        { taskSource: 'inference' }
      );
      expect(sendRecordCommand).toHaveBeenNthCalledWith(
        2,
        'stop_inference_record',
        {
          episodeOutcome: EpisodeOutcome.FAILURE,
          recordingSource: 'inference',
          taskSource: 'inference',
        }
      );
      expect(sendRecordCommand).toHaveBeenNthCalledWith(
        3,
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'intervention',
          taskSource: 'inference',
        }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('start', {
        model: 'a3',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('recording', {
        model: 'a3',
      });
    });
  });

  test('starts Total Record with an explicit continuous-session tag', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({
      variant: 'workspace',
      sendRecordCommand,
      // A READY status may briefly retain tags from the prior episode.
      recordingTags: ['recording_source:inference'],
    });

    fireEvent.click(screen.getByRole('button', {
      name: /start total recording/i,
    }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'total',
          taskSource: 'inference',
        }
      );
    });
    expect(screen.getByRole('button', {
      name: /total recording active/i,
    })).toHaveTextContent('Total Recording');
  });

  test('pre-arms Total Record in READY and starts it with inference', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.READY,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', {
      name: /pre-arm total recording/i,
    }));

    expect(sendRecordCommand).not.toHaveBeenCalled();
    expect(screen.getByRole('button', {
      name: /cancel total recording pre-arm/i,
    })).toHaveTextContent('Total Armed');
    expect(screen.getByRole('button', {
      name: /switch total recording to intervention mode/i,
    })).toHaveTextContent('Intervention OFF');
    expect(screen.getByRole('button', {
      name: /switch total recording to intervention mode/i,
    })).toBeDisabled();
    expect(screen.getByRole('group', {
      name: /total record control mode/i,
    })).toBeInTheDocument();

    act(() => {
      store.dispatch(setInferenceStatus({
        inferencePhase: InferencePhase.INFERENCING,
      }));
    });

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'total',
          taskSource: 'inference',
        }
      );
    });
    expect(screen.getByRole('button', {
      name: /total recording active/i,
    })).toHaveTextContent('Total Recording');
  });

  test('shows an explicit Total Record intervention switch and turns it on', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.READY,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', {
      name: /pre-arm total recording/i,
    }));
    fireEvent.click(screen.getByRole('button', {
      name: /use mini leader for intervention/i,
    }));
    act(() => {
      store.dispatch(setInferenceStatus({
        inferencePhase: InferencePhase.INFERENCING,
      }));
    });

    const interventionSwitch = await screen.findByRole('button', {
      name: /switch total recording to intervention mode/i,
    });
    await waitFor(() => expect(interventionSwitch).toBeEnabled());
    expect(interventionSwitch).toHaveTextContent('Intervention OFF');
    fireEvent.click(screen.getByRole('button', {
      name: /switch total recording to intervention mode/i,
    }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('stop_inference', {
        taskSource: 'inference',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('start', {
        model: 'mini',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('recording', {
        model: 'mini',
      });
    });
    expect(sendRecordCommand).toHaveBeenCalledTimes(2);
  });

  test('starts an armed Total Record directly in intervention mode from PAUSED', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.READY,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', {
      name: /pre-arm total recording/i,
    }));
    fireEvent.click(screen.getByRole('button', {
      name: /use mini leader for intervention/i,
    }));
    act(() => {
      store.dispatch(setInferenceStatus({
        inferencePhase: InferencePhase.PAUSED,
      }));
    });

    await waitFor(() => expect(screen.getByRole('button', {
      name: /switch total recording to intervention mode/i,
    })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', {
      name: /switch total recording to intervention mode/i,
    }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'total',
          taskSource: 'inference',
        }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('start', {
        model: 'mini',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('recording', {
        model: 'mini',
      });
    });
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'stop_inference',
      expect.anything()
    );
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'start_inference_record',
      expect.objectContaining({ recordingSource: 'intervention' })
    );
  });

  test('closes a completed intervention session without resuming an idle inference', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    getInterventionState.mockResolvedValue({
      active: true,
      record_phase: RecordPhase.SAVING,
      current: {
        phase: 'READY_TO_RESUME',
        leader_model: 'mini',
        prewarmed: true,
      },
      command_recent: false,
      preflight: { ok: false, failures: [] },
      fault_reset: { ok: true, failures: [] },
      arbiter: {
        available: true,
        mode: 'NONE',
        fault: '',
        publisher_conflict: '',
      },
      standby: {
        active: true,
        current: { model: 'mini', phase: 'READY' },
        ready: true,
      },
    });
    renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.READY,
      sendRecordCommand,
    });

    const closeSession = await screen.findByRole('button', {
      name: /close completed intervention session/i,
    });
    expect(closeSession).toBeEnabled();
    fireEvent.click(closeSession);

    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('reset', {
        model: 'mini',
      });
    });
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'resume_inference',
      expect.anything()
    );
  });

  test('returns Total Record to policy without closing the episode', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    getInterventionState.mockResolvedValue({
      active: true,
      current: {
        phase: 'RECORDING',
        leader_model: 'a3',
        recording_continued: true,
      },
      preflight: { ok: false, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: { mode: 'TELEOP', fault: '', publisher_conflict: '' },
      standby: { active: false, current: null, ready: false },
    });
    const { store } = renderControls({ variant: 'workspace', sendRecordCommand });
    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
        tags: ['recording_source:total'],
      }));
    });

    fireEvent.click(await screen.findByRole('button', {
      name: /switch total recording to inference mode/i,
    }));

    await waitFor(() => {
      expect(sendRecordCommand).not.toHaveBeenCalledWith(
        'stop_inference_record',
        expect.anything()
      );
      expect(sendRecordCommand).not.toHaveBeenCalledWith(
        'cancel_inference_record',
        expect.anything()
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('end', { model: 'a3' });
      expect(sendRecordCommand).toHaveBeenCalledWith('resume_inference', {
        taskSource: 'inference',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('complete', {
        model: 'a3',
      });
    });
  });

  test('saves a continuous Total episode directly from intervention mode', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    getInterventionState.mockResolvedValue({
      active: true,
      record_phase: RecordPhase.SAVING,
      current: {
        phase: 'RECORDING',
        leader_model: 'mini',
        recording_continued: true,
      },
      inference_phase: 'PAUSED',
      preflight: { ok: false, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: {
        mode: 'TELEOP',
        fault: '',
        publisher_conflict: '',
        intervention_enabled: true,
      },
      standby: {
        active: true,
        current: { model: 'mini', phase: 'IN_USE' },
        ready: false,
      },
    });
    const onEpisodeSaved = jest.fn();
    const { store } = renderControls({
      variant: 'workspace',
      sendRecordCommand,
      onEpisodeSaved,
    });
    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
        tags: ['recording_source:total'],
      }));
    });

    const save = await screen.findByRole('button', {
      name: /save total intervention episode as success/i,
    });
    expect(save).toBeEnabled();
    fireEvent.click(save);

    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('end', {
        model: 'mini',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_inference_record',
        {
          episodeOutcome: EpisodeOutcome.SUCCESS,
          recordingSource: 'total',
          taskSource: 'inference',
        }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('reset', {
        model: 'mini',
      });
      expect(sendRecordCommand).not.toHaveBeenCalledWith(
        'resume_inference',
        expect.anything()
      );
      expect(sendInterventionAction).not.toHaveBeenCalledWith(
        'complete',
        expect.anything()
      );
      expect(onEpisodeSaved).toHaveBeenCalledWith({
        episodeOutcome: EpisodeOutcome.SUCCESS,
        commandOptions: {},
      });
    });

    const endOrder = sendInterventionAction.mock.invocationCallOrder.find((_, index) => (
      sendInterventionAction.mock.calls[index][0] === 'end'
    ));
    const saveOrder = sendRecordCommand.mock.invocationCallOrder.find((_, index) => (
      sendRecordCommand.mock.calls[index][0] === 'stop_inference_record'
    ));
    const resetOrder = sendInterventionAction.mock.invocationCallOrder.find((_, index) => (
      sendInterventionAction.mock.calls[index][0] === 'reset'
    ));
    expect(endOrder).toBeLessThan(saveOrder);
    expect(saveOrder).toBeLessThan(resetOrder);
  });

  test('keeps policy paused after saving a standalone intervention record', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    getInterventionState.mockResolvedValue({
      active: true,
      current: {
        phase: 'RECORDING',
        leader_model: 'mini',
        recording_continued: false,
      },
      inference_phase: 'PAUSED',
      preflight: { ok: false, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: { mode: 'TELEOP', fault: '', publisher_conflict: '' },
      standby: {
        active: true,
        current: { model: 'mini', phase: 'IN_USE' },
        ready: false,
      },
    });
    const { store } = renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.PAUSED,
      sendRecordCommand,
    });
    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
        tags: ['recording_source:intervention'],
      }));
    });

    const save = await screen.findByRole('button', {
      name: /save intervention record/i,
    });
    fireEvent.click(save);

    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('end', {
        model: 'mini',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_inference_record',
        {
          episodeOutcome: EpisodeOutcome.SUCCESS,
          recordingSource: 'intervention',
          taskSource: 'inference',
        }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('reset', {
        model: 'mini',
      });
    });
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'resume_inference',
      expect.anything()
    );
  });

  test('moves to operator-entered lists only from the guarded paused state', async () => {
    getInterventionState.mockResolvedValue({
      active: false,
      current: null,
      inference_phase: 'PAUSED',
      record_phase: 'READY',
      pose_move: { ok: true, failures: [], warnings: [] },
      operation: { running: false, kind: null },
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: { mode: 'NONE', fault: '', publisher_conflict: '' },
      standby: {
        active: true,
        current: { model: 'mini', phase: 'READY' },
        ready: true,
      },
    });
    renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.PAUSED,
    });

    const move = await screen.findByRole('button', {
      name: /move to operator-entered initial pose/i,
    });
    await waitFor(() => expect(move).toBeEnabled());
    fireEvent.click(move);
    expect(sendOperatorPose).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText('Left arm initial position list'), {
      target: { value: '[0, 0, 0, -1, 0, 0, 0, 0]' },
    });
    fireEvent.change(screen.getByLabelText('Right arm initial position list'), {
      target: { value: '[0, 0, 0, -1, 0, 0, 0, 0]' },
    });
    fireEvent.change(screen.getByLabelText('Head initial position list'), {
      target: { value: '[0, 0]' },
    });
    fireEvent.change(screen.getByLabelText('Lift initial position list'), {
      target: { value: '[-0.2]' },
    });

    fireEvent.click(move);

    await waitFor(() => {
      expect(sendOperatorPose).toHaveBeenCalledWith({
        model: 'sg2',
        left_arm: [0, 0, 0, -1, 0, 0, 0, 0],
        right_arm: [0, 0, 0, -1, 0, 0, 0, 0],
        head: [0, 0],
        lift: [-0.2],
        duration: 8,
      });
    });
  });

  test('recovers a transient paused return and permits another Total intervention', async () => {
    let resumeCalls = 0;
    let completeCalls = 0;
    let serverState = {
      active: true,
      current: {
        phase: 'RECORDING',
        leader_model: 'a3',
        recording_continued: true,
      },
      inference_phase: 'PAUSED',
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: {
        available: true,
        mode: 'TELEOP',
        fault: '',
        publisher_conflict: '',
        intervention_enabled: true,
      },
      standby: { active: false, current: null, ready: false },
    };
    getInterventionState.mockImplementation(async () => serverState);
    sendInterventionAction.mockImplementation(async (action) => {
      if (action === 'end') {
        serverState = {
          ...serverState,
          current: { ...serverState.current, phase: 'READY_TO_RESUME' },
          inference_phase: 'PAUSED',
          arbiter: {
            ...serverState.arbiter,
            mode: 'NONE',
            intervention_enabled: false,
          },
        };
      } else if (action === 'complete') {
        completeCalls += 1;
        if (completeCalls === 1) {
          throw new Error('Inference return was not observed yet');
        }
        serverState = {
          ...serverState,
          active: false,
          current: null,
          inference_phase: 'INFERENCING',
          arbiter: { ...serverState.arbiter, mode: 'POLICY' },
        };
      } else if (action === 'start') {
        serverState = {
          ...serverState,
          active: true,
          current: { phase: 'READY', leader_model: 'a3' },
          inference_phase: 'PAUSED',
          arbiter: {
            ...serverState.arbiter,
            mode: 'TELEOP',
            intervention_enabled: true,
          },
        };
      } else if (action === 'recording') {
        serverState = {
          ...serverState,
          current: { ...serverState.current, phase: 'RECORDING' },
        };
      }
      return { ok: true, ...serverState };
    });
    const sendRecordCommand = jest.fn(async (command) => {
      if (command === 'resume_inference') {
        resumeCalls += 1;
        if (resumeCalls > 1) {
          serverState = {
            ...serverState,
            inference_phase: 'INFERENCING',
            arbiter: { ...serverState.arbiter, mode: 'POLICY' },
          };
        }
      } else if (command === 'stop_inference') {
        serverState = {
          ...serverState,
          inference_phase: 'PAUSED',
          arbiter: { ...serverState.arbiter, mode: 'NONE' },
        };
      }
      return { success: true };
    });
    const { store } = renderControls({
      variant: 'workspace',
      sendRecordCommand,
    });
    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
        tags: ['recording_source:total'],
      }));
    });

    fireEvent.click(await screen.findByRole('button', {
      name: /switch total recording to inference mode/i,
    }));

    await waitFor(() => {
      expect(resumeCalls).toBe(2);
      expect(completeCalls).toBe(2);
      expect(screen.getByRole('button', {
        name: /switch total recording to intervention mode/i,
      })).toBeEnabled();
    });

    fireEvent.click(screen.getByRole('button', {
      name: /switch total recording to intervention mode/i,
    }));

    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('start', {
        model: 'a3',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('recording', {
        model: 'a3',
      });
    });
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'start_inference_record',
      expect.anything()
    );
  });

  test('cancels a partially started intervention recording when state confirmation fails', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    getInterventionState
      .mockResolvedValueOnce({
        active: false,
        current: null,
        preflight: { ok: true, failures: [] },
        fault_reset: { ok: false, failures: [] },
        arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
        standby: { active: false, current: null, ready: false },
      })
      .mockResolvedValue({
        active: false,
        current: null,
        preflight: { ok: true, failures: [] },
        fault_reset: { ok: false, failures: [] },
        arbiter: { mode: 'POLICY', fault: '', publisher_conflict: '' },
        standby: {
          active: true,
          current: { model: 'mini', phase: 'READY' },
          ready: true,
        },
      });
    sendInterventionAction
      .mockResolvedValueOnce({ ok: true })
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(new Error('recording state unavailable'))
      .mockResolvedValue({ ok: true });
    renderControls({ variant: 'workspace', sendRecordCommand });

    await waitFor(() => expect(getInterventionState).toHaveBeenCalled());

    fireEvent.click(screen.getByRole('button', {
      name: /use mini leader for intervention/i,
    }));
    await waitFor(() => expect(sendInterventionAction).toHaveBeenCalledWith(
      'standby',
      { model: 'mini' }
    ));
    const interventionRecord = screen.getByRole('button', {
      name: /start intervention recording/i,
    });
    await waitFor(() => expect(interventionRecord).toBeEnabled());
    fireEvent.click(interventionRecord);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'cancel_inference_record',
        {
          recordingSource: 'intervention',
          taskSource: 'inference',
        }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('abort', {
        model: 'mini',
      });
    });
  });

  test('offers a guarded routing fault reset', async () => {
    getInterventionState.mockResolvedValue({
      active: false,
      current: null,
      preflight: { ok: false, failures: ['fault'] },
      fault_reset: { ok: true, failures: [] },
      arbiter: { mode: 'NONE', fault: 'policy lease lost' },
    });
    renderControls({ variant: 'workspace' });

    const reset = await screen.findByRole('button', {
      name: /reset command routing fault/i,
    });
    expect(reset).toBeEnabled();
    fireEvent.click(reset);

    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('reset_fault', {
        model: 'a3',
      });
    });
  });

  test('recovers a paused-safe intervention and resumes inference explicitly', async () => {
    getInterventionState.mockResolvedValue({
      active: true,
      current: { phase: 'PAUSED_SAFE', leader_model: 'a3' },
      preflight: { ok: false, failures: [] },
      fault_reset: { ok: true, failures: [] },
      arbiter: { mode: 'NONE', fault: 'teleop lease lost' },
    });
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.PAUSED,
      sendRecordCommand,
    });

    const resume = await screen.findByRole('button', {
      name: /reset aborted intervention and acquire new data/i,
    });
    fireEvent.click(resume);

    await waitFor(() => {
      expect(sendInterventionAction).toHaveBeenCalledWith('reset', {
        model: 'a3',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith('resume_inference', {
        taskSource: 'inference',
      });
    });
  });

  test('resets an aborted Total Record and arms a fresh episode', async () => {
    let sessionReset = false;
    const pausedSafeState = {
      active: true,
      current: {
        phase: 'PAUSED_SAFE',
        leader_model: 'mini',
        recording_continued: true,
      },
      record_phase: RecordPhase.READY,
      command_recent: false,
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: false, failures: [] },
      arbiter: {
        available: true,
        mode: 'NONE',
        fault: '',
        publisher_conflict: '',
      },
      standby: { active: false, current: null, ready: false },
    };
    getInterventionState.mockImplementation(async () => (
      sessionReset
        ? {
          ...pausedSafeState,
          active: false,
          current: null,
          standby: {
            active: true,
            current: { model: 'mini', phase: 'READY' },
            ready: true,
          },
        }
        : pausedSafeState
    ));
    sendInterventionAction.mockImplementation(async (action) => {
      if (action === 'reset') sessionReset = true;
      return { ok: true };
    });
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    const { store } = renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.PAUSED,
      sendRecordCommand,
    });
    act(() => {
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
        tags: ['recording_source:total'],
      }));
    });

    const reset = await screen.findByRole('button', {
      name: /reset aborted intervention and acquire new data/i,
    });
    expect(reset).toHaveTextContent('Reset & Record Again');
    fireEvent.click(reset);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'cancel_inference_record',
        { recordingSource: 'total', taskSource: 'inference' }
      );
      expect(sendInterventionAction).toHaveBeenCalledWith('reset', {
        model: 'mini',
      });
      expect(sendInterventionAction).toHaveBeenCalledWith('standby', {
        model: 'mini',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith('resume_inference', {
        taskSource: 'inference',
      });
    });

    act(() => {
      store.dispatch(setRecordStatus({ recordPhase: RecordPhase.READY }));
      store.dispatch(setInferenceStatus({
        inferencePhase: InferencePhase.INFERENCING,
      }));
    });
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'total',
          taskSource: 'inference',
        }
      );
    });
  });

  test('shows aborted Total reset from authoritative server state after recording is ready', async () => {
    getInterventionState.mockResolvedValue({
      active: true,
      current: {
        phase: 'PAUSED_SAFE',
        leader_model: 'mini',
        recording_continued: true,
      },
      inference_status_ready: true,
      inference_phase: 'PAUSED',
      record_phase: RecordPhase.READY,
      command_recent: false,
      preflight: { ok: true, failures: [] },
      fault_reset: { ok: true, failures: [] },
      arbiter: {
        available: true,
        mode: 'NONE',
        fault: '',
        publisher_conflict: '',
      },
      standby: { active: false, current: null, ready: false },
    });
    renderControls({
      variant: 'workspace',
      inferencePhase: InferencePhase.READY,
    });

    const reset = await screen.findByRole('button', {
      name: /reset aborted intervention and acquire new data/i,
    });
    expect(reset).toBeEnabled();
    expect(reset).toHaveTextContent('Reset & Record Again');
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
        { recordingSource: 'inference', taskSource: 'inference' }
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
        { recordingSource: 'inference', taskSource: 'inference' }
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
