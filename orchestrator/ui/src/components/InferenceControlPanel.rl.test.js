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
import toast from 'react-hot-toast';
import InferenceControlPanel from './InferenceControlPanel';
import taskReducer, {
  InferenceRecordingUiPhase,
  setInferenceRecordingUiPhase,
  setInferenceStatus,
  setRecordStatus,
  receiveServerRecordTaskInfo,
} from '../features/tasks/taskSlice';
import rosReducer from '../features/ros/rosSlice';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { getInterventionState } from '../utils/interventionApi';

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  toast.dismiss = jest.fn();
  return {
    __esModule: true,
    default: toast,
    useToasterStore: () => ({ toasts: [] }),
  };
});

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('../hooks/usePolicyBackendStatus', () => ({
  __esModule: true,
  default: () => ({
    readiness: {
      ready: true,
      state: 'ready',
      message: 'Backend ready',
    },
    refreshStatus: jest.fn(),
  }),
  getPolicyBackendReadiness: (status) => status,
}));

jest.mock('../utils/interventionApi', () => ({
  getInterventionState: jest.fn(),
  sendInterventionAction: jest.fn().mockResolvedValue({ ok: true }),
}));

const renderPanel = ({
  robotType = 'ffw_sg2_rev1',
  inferenceMode = 'robot',
  inferencePhase = InferencePhase.READY,
  taskOverrides = {},
  sendRecordCommand: sendOverride = null,
  showRecordingControls = true,
  variant = 'default',
  policyEpoch = null,
} = {}) => {
  const sendRecordCommand = sendOverride || jest.fn().mockResolvedValue({
    success: true,
    message: 'ok',
  });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });

  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const initialRos = rosReducer(undefined, { type: '@@INIT' });
  const { taskInstruction, ...inferenceOverrides } = taskOverrides;
  const sharedTaskInstruction =
    taskInstruction ?? initialTasks.sharedTaskInfo.taskInstruction;
  const store = configureStore({
    reducer: {
      tasks: taskReducer,
      ros: rosReducer,
    },
    preloadedState: {
      tasks: {
        ...initialTasks,
        robotType,
        sharedTaskInfo: {
          ...initialTasks.sharedTaskInfo,
          taskInstruction: sharedTaskInstruction,
        },
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          policyPath: '/policy_checkpoints/lerobot/model',
          inferenceMode,
          ...inferenceOverrides,
        },
        taskInfo: {
          ...initialTasks.taskInfo,
          policyPath: '/policy_checkpoints/lerobot/model',
          inferenceMode,
          ...taskOverrides,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
      },
      ros: {
        ...initialRos,
        rosHost: 'localhost',
      },
    },
  });

  render(
    <Provider store={store}>
      <InferenceControlPanel
        showRecordingControls={showRecordingControls}
        variant={variant}
        policyEpoch={policyEpoch}
      />
    </Provider>
  );

  return { store, sendRecordCommand };
};

describe('InferenceControlPanel deploy safety', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    getInterventionState.mockResolvedValue({ active: false, current: null });
  });

  test('shows a warning instead of starting immediately for Real Robot Deploy', async () => {
    const { sendRecordCommand } = renderPanel({ inferenceMode: 'robot' });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    expect(await screen.findByRole('dialog', { name: /real robot deploy/i }))
      .toBeInTheDocument();
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('checks the TT-RTC TensorRT engine with the TT-RTC request mode', async () => {
    const originalFetch = global.fetch;
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      text: async () => JSON.stringify({ status: 'ready' }),
    });
    try {
      const { sendRecordCommand } = renderPanel({
        inferenceMode: 'simulation',
        taskOverrides: {
          serviceType: 'groot',
          policyType: 'n17',
          policyPath: '/workspace/model/groot/example',
          taskInstruction: ['Drill into screw.'],
          accelerationMode: 'tensorrt_dit',
          accelerationEnginePath: '',
          actionRequestMode: 'tt_rtc',
        },
      });

      fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining(
          'action_request_mode=tt_rtc'
        ));
        expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
          inferenceMode: 'simulation',
          taskSource: 'inference',
        });
      });
    } finally {
      global.fetch = originalFetch;
    }
  });

  test('builds a missing TensorRT engine from Start and continues inference', async () => {
    const originalFetch = global.fetch;
    global.fetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        text: async () => JSON.stringify({ status: 'missing' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        text: async () => JSON.stringify({ status: 'building' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        text: async () => JSON.stringify({ status: 'ready' }),
      });
    try {
      const { sendRecordCommand } = renderPanel({
        inferenceMode: 'simulation',
        taskOverrides: {
          serviceType: 'groot',
          policyType: 'n17',
          policyPath: '/workspace/model/groot/example',
          taskInstruction: ['Drill into screw.'],
          accelerationMode: 'tensorrt_dit',
          accelerationEnginePath: '',
          actionRequestMode: 'tt_rtc',
        },
      });

      fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

      await waitFor(() => {
        expect(global.fetch).toHaveBeenNthCalledWith(
          2,
          '/api/backends/groot/trt/build',
          expect.objectContaining({ method: 'POST' })
        );
        const buildRequest = JSON.parse(global.fetch.mock.calls[1][1].body);
        expect(buildRequest).toEqual(expect.objectContaining({
          model_path: '/workspace/model/groot/example',
          robot_type: 'ffw_sg2_rev1',
          action_request_mode: 'tt_rtc',
        }));
        expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
          inferenceMode: 'simulation',
          taskSource: 'inference',
        });
      });
    } finally {
      global.fetch = originalFetch;
    }
  });

  test('explains the approved RLT override and base-policy start in the robot warning', async () => {
    renderPanel({
      inferenceMode: 'robot',
      taskOverrides: {
        rltEnabled: true,
        rltRobotOverride: true,
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    const dialog = await screen.findByRole('dialog', { name: /real robot deploy/i });
    expect(within(dialog).getByText('RLT experimental override approved.'))
      .toBeInTheDocument();
    expect(within(dialog).getByText(
      'This session starts with GR00T. RLT activates only after you press Use RLT Action.'
    )).toBeInTheDocument();
  });

  test('starts robot deploy only after explicit confirmation', async () => {
    const { sendRecordCommand } = renderPanel({ inferenceMode: 'robot' });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));
    fireEvent.click(await screen.findByRole('button', { name: /^Real Robot Deploy$/i }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'robot',
        taskSource: 'inference',
      });
    });
  });

  test('can switch the pending start to 3D Sim Deploy from the warning', async () => {
    const { store, sendRecordCommand } = renderPanel({ inferenceMode: 'robot' });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));
    fireEvent.click(await screen.findByRole('button', { name: /^3D Sim Deploy$/i }));

    await waitFor(() => {
      expect(store.getState().tasks.taskInfo.inferenceMode).toBe('simulation');
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'simulation',
        taskSource: 'inference',
      });
    });
  });

  test('keeps loading state when start command times out after LOADING begins', async () => {
    let rejectStart;
    const sendRecordCommand = jest.fn(() => new Promise((_, reject) => {
      rejectStart = reject;
    }));
    const { store } = renderPanel({
      inferenceMode: 'simulation',
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'simulation',
        taskSource: 'inference',
      });
    });

    act(() => {
      store.dispatch(setInferenceStatus({ inferencePhase: InferencePhase.LOADING }));
      rejectStart(new Error('Service call timeout for /task/command'));
    });

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        'Model loading is still running. Large downloads can take several minutes.'
      );
    });
    expect(toast.error).not.toHaveBeenCalledWith(
      expect.stringContaining('Command timeout')
    );
    expect(store.getState().tasks.inferenceStatus.inferencePhase)
      .toBe(InferencePhase.LOADING);
  });

  test('does not expose recording controls on the inference panel', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation',
      inferencePhase: InferencePhase.INFERENCING,
    });

    expect(screen.queryByRole('button', { name: /start recording/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /save recording/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /discard recording/i })).not.toBeInTheDocument();

    fireEvent.keyDown(window, { key: 'r', code: 'KeyR' });
    fireEvent.keyUp(window, { key: 'r', code: 'KeyR' });

    await waitFor(() => {
      expect(sendRecordCommand).not.toHaveBeenCalled();
    });
  });

  test('keeps the official inference toolbar free of Playground recording controls', () => {
    renderPanel({ inferenceMode: 'robot', inferencePhase: InferencePhase.INFERENCING, taskOverrides: { recordInferenceMode: true, recordingFolder: '/workspace/rosbag2/Task_playground_MCAP' } });
    expect(screen.getByRole('button', { name: 'Pause inference (model stays loaded)' })).toBeVisible();
    expect(screen.queryByRole('group', { name: /rl recording controls/i })).not.toBeInTheDocument();
    expect(screen.queryByRole('group', { name: 'Action policy routing' })).not.toBeInTheDocument();
  });

  test('can hide recording controls when they are rendered by the workspace', () => {
    renderPanel({
      variant: 'offlineRL',
      showRecordingControls: false,
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.INFERENCING,
      taskOverrides: { recordInferenceMode: true },
      showRecordingControls: false,
    });

    expect(screen.getByRole('group', {
      name: /^inference controls$/i,
    })).toBeInTheDocument();
    expect(screen.queryByRole('group', {
      name: /^rl recording controls$/i,
    })).not.toBeInTheDocument();
  });

  test('combines Offline RL inference and recording controls at the bottom', () => {
    renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.READY,
      taskOverrides: {
        recordInferenceMode: true,
        recordingFolder: '/workspace/rosbag2/Task_playground_MCAP',
      },
      variant: 'offlineRL',
      policyEpoch: 2,
    });

    const combined = screen.getByRole('group', {
      name: /^inference recording controls$/i,
    });
    expect(within(combined).getByText('Inference Recording')).toBeInTheDocument();
    expect(within(combined).getByLabelText('Current policy RL Epoch 2'))
      .toHaveTextContent('RL Epoch E0002');
    expect(within(combined).getByRole('status')).toHaveTextContent('Ready to start');

    expect(within(combined).getByRole('button', {
      name: /start partial recording/i,
    })).toBeDisabled();
    expect(within(combined).queryByRole('button', {
      name: /start intervention recording/i,
    })).not.toBeInTheDocument();
    expect(within(combined).getByRole('button', {
      name: /start inference/i,
    })).toBeEnabled();
    expect(within(combined).getByRole('button', {
      name: /pause inference/i,
    })).toBeDisabled();
    expect(within(combined).getByRole('button', {
      name: /unload model/i,
    })).toBeDisabled();
  });

  test('blocks Clear but keeps Pause available during RL recording', async () => {
    const { store, sendRecordCommand } = renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.INFERENCING,
      taskOverrides: { recordInferenceMode: true },
    });

    act(() => {
      store.dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.RECORDING
      ));
      store.dispatch(setRecordStatus({
        taskType: 'inference',
        recordInferenceMode: true,
        recordPhase: RecordPhase.RECORDING,
      }));
    });

    expect(screen.getByRole('button', {
      name: /stop inference and unload model/i,
    })).toBeDisabled();
    const pause = screen.getByRole('button', {
      name: /pause inference/i,
    });
    expect(pause).toBeEnabled();
    fireEvent.click(pause);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('stop_inference', {
        taskSource: 'inference',
      });
    });
  });

  test('requires shared task instruction for language-conditioned inference', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation',
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        taskInstruction: [],
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Missing required fields: Task Instruction');
      expect(sendRecordCommand).not.toHaveBeenCalled();
    });
  });

  test('allows language-conditioned inference when shared task instruction is set', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation',
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        taskInstruction: ['pick up the red cup'],
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'simulation',
        taskSource: 'inference',
      });
    });
  });

  test('starts with a preloaded RLT bundle and resets routing to base GR00T', async () => {
    const { store, sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation',
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        taskInstruction: ['pick up the jelly bag'],
        rltEnabled: true,
        rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
        actionPolicyMode: 'rlt',
      },
    });

    fireEvent.click(screen.getByRole('button', {
      name: /start inference/i,
    }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'simulation',
        taskSource: 'inference',
      });
      expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode)
        .toBe('base');
    });
  });

  test('falls back to the base policy when RLT is enabled without a bundle', async () => {
    const { store, sendRecordCommand } = renderPanel({
      inferenceMode: 'simulation',
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        taskInstruction: ['pick up the jelly bag'],
        rltEnabled: true,
        rltBundlePath: '',
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(toast).toHaveBeenCalledWith(
        'RLT Bundle Path is empty; starting the base policy without RLT'
      );
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'simulation',
        taskSource: 'inference',
      });
      expect(store.getState().tasks.inferenceTaskInfo.rltEnabled).toBe(false);
      expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode).toBe('base');
    });
  });

  test('hot-switches between preloaded GR00T and RLT actions without stopping', async () => {
    const { store, sendRecordCommand } = renderPanel({
      variant: 'offlineRL',
      showRecordingControls: false,
      inferenceMode: 'simulation',
      inferencePhase: InferencePhase.INFERENCING,
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        taskInstruction: ['pick up the jelly bag'],
        rltEnabled: true,
        rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
        actionPolicyMode: 'base',
      },
    });

    const baseButton = screen.getByRole('button', { name: 'Use GR00T Action' });
    const rltButton = screen.getByRole('button', { name: 'Use RLT Action' });
    expect(baseButton).toHaveAttribute('aria-pressed', 'true');
    expect(baseButton).toBeDisabled();
    expect(rltButton).toBeEnabled();

    fireEvent.click(rltButton);
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('set_action_policy', {
        actionPolicyMode: 'rlt',
        taskSource: 'inference',
      });
      expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode)
        .toBe('rlt');
    });

    // /data/recording/status still carries the TaskInfo snapshot captured at
    // inference start. It must not revert the live RLT route back to base.
    act(() => {
      store.dispatch(receiveServerRecordTaskInfo({
        taskType: 'inference',
        actionPolicyMode: 'base',
      }));
    });
    expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode)
      .toBe('rlt');

    const updatedBaseButton = screen.getByRole('button', { name: 'Use GR00T Action' });
    expect(updatedBaseButton).toBeEnabled();
    fireEvent.click(updatedBaseButton);
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('set_action_policy', {
        actionPolicyMode: 'base',
        taskSource: 'inference',
      });
      expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode)
        .toBe('base');
    });
    expect(sendRecordCommand).not.toHaveBeenCalledWith(
      'stop_inference',
      expect.anything()
    );
  });

  test.each([
    [false, '', InferencePhase.READY, 'Enable RLT in Settings'],
    [true, '', InferencePhase.READY, 'Set RLT Bundle Path in Settings'],
    [true, '/workspace/checkpoint/rlt/bundle', InferencePhase.READY, 'Start inference with the RLT bundle loaded'],
  ])('shows disabled routing with missing prerequisites', (enabled, bundle, phase, message) => {
    const { sendRecordCommand } = renderPanel({
      variant: 'offlineRL', showRecordingControls: true,
      inferencePhase: phase,
      taskOverrides: { serviceType: 'groot', policyType: 'n17',
        actionRequestMode: 'tt_rtc', rltEnabled: enabled, rltBundlePath: bundle },
    });
    const button = screen.getByRole('button', { name: 'Use MLP Action' });
    expect(button).toBeDisabled();
    expect(screen.getByText(message)).toBeInTheDocument();
    fireEvent.click(button);
    expect(sendRecordCommand).not.toHaveBeenCalledWith('set_action_policy', expect.anything());
  });

  test('labels the preloaded TT-RTC routes as VLA and MLP actions', async () => {
    const { store, sendRecordCommand } = renderPanel({
      variant: 'offlineRL',
      showRecordingControls: false,
      inferenceMode: 'simulation',
      inferencePhase: InferencePhase.INFERENCING,
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        actionRequestMode: 'tt_rtc',
        rltEnabled: true,
        rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
        actionPolicyMode: 'base',
      },
    });

    const vlaButton = screen.getByRole('button', { name: 'Use VLA Action' });
    const mlpButton = screen.getByRole('button', { name: 'Use MLP Action' });
    expect(vlaButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByRole('button', { name: 'Use GR00T Action' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Use RLT Action' }))
      .not.toBeInTheDocument();

    fireEvent.click(mlpButton);
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('set_action_policy', {
        actionPolicyMode: 'rlt',
        taskSource: 'inference',
      });
      expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode)
        .toBe('rlt');
    });
  });

  test('starts Partial Record after the MLP action route is confirmed active', async () => {
    const { store, sendRecordCommand } = renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.INFERENCING,
      variant: 'offlineRL',
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        actionRequestMode: 'tt_rtc',
        rltEnabled: true,
        rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
        rltRobotOverride: true,
        actionPolicyMode: 'base',
        recordInferenceMode: false,
        recordingFolder: '/workspace/rosbag2/Task_mlp_partial_MCAP',
      },
    });

    const autoTriggers = screen.getByRole('group', {
      name: /automatic partial record triggers/i,
    });
    expect(within(autoTriggers).getAllByRole('button')).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Use MLP Action' }));

    await waitFor(() => {
      expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode)
        .toBe('rlt');
      expect(sendRecordCommand).toHaveBeenCalledWith('set_action_policy', {
        actionPolicyMode: 'rlt',
        taskSource: 'inference',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_inference_record',
        {
          forceInferenceRecording: true,
          recordingSource: 'inference',
          taskSource: 'inference',
        }
      );
    });
  });

  test('asks for explicit confirmation before an unapproved Real Robot RLT switch', async () => {
    const { sendRecordCommand } = renderPanel({
      variant: 'offlineRL',
      showRecordingControls: false,
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.INFERENCING,
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        rltEnabled: true,
        rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
      },
    });

    const rltButton = screen.getByRole('button', { name: 'Use RLT Action' });
    expect(rltButton).toBeEnabled();
    expect(rltButton).toHaveAttribute(
      'title',
      expect.stringContaining('requires explicit safety confirmation')
    );
    expect(screen.getByTestId('real-robot-rlt-approval-status'))
      .toHaveTextContent('requires safety confirmation');
    fireEvent.click(rltButton);

    expect(await screen.findByRole('dialog', { name: 'Enable RLT on Real Robot' }))
      .toBeInTheDocument();
    expect(sendRecordCommand).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog', { name: 'Enable RLT on Real Robot' }))
      .not.toBeInTheDocument();
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('approves and applies an unapproved Real Robot RLT switch', async () => {
    const { store, sendRecordCommand } = renderPanel({
      variant: 'offlineRL',
      showRecordingControls: false,
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.INFERENCING,
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        rltEnabled: true,
        rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
        actionPolicyMode: 'base',
      },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Use RLT Action' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Confirm RLT Action' }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('set_action_policy', {
        actionPolicyMode: 'rlt',
        rltRobotOverride: true,
        taskSource: 'inference',
      });
      expect(store.getState().tasks.inferenceTaskInfo.rltRobotOverride).toBe(true);
      expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode).toBe('rlt');
    });
  });

  test('allows Real Robot RLT hot-switch after explicit opt-in', async () => {
    const { store, sendRecordCommand } = renderPanel({
      variant: 'offlineRL',
      showRecordingControls: false,
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.INFERENCING,
      taskOverrides: {
        serviceType: 'groot',
        policyType: 'n17',
        rltEnabled: true,
        rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
        rltRobotOverride: true,
        actionPolicyMode: 'base',
      },
    });

    const rltButton = screen.getByRole('button', { name: 'Use RLT Action' });
    expect(rltButton).toBeEnabled();
    expect(rltButton).toHaveAttribute(
      'title',
      expect.stringContaining('safety override approved')
    );
    expect(screen.getByTestId('real-robot-rlt-approval-status'))
      .toHaveTextContent('safety override approved');

    fireEvent.click(rltButton);

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('set_action_policy', {
        actionPolicyMode: 'rlt',
        taskSource: 'inference',
      });
      expect(store.getState().tasks.inferenceTaskInfo.actionPolicyMode)
        .toBe('rlt');
    });
  });
});
