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
} from '../features/tasks/taskSlice';
import rosReducer from '../features/ros/rosSlice';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

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

const renderPanel = ({
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
  });

  test('shows a warning instead of starting immediately for Real Robot Deploy', async () => {
    const { sendRecordCommand } = renderPanel({ inferenceMode: 'robot' });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    expect(await screen.findByRole('dialog', { name: /real robot deploy/i }))
      .toBeInTheDocument();
    expect(sendRecordCommand).not.toHaveBeenCalled();
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

  test('places RL recording actions in a separate block below inference controls', () => {
    renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.INFERENCING,
      taskOverrides: { recordInferenceMode: true },
    });

    const inferenceControls = screen.getByRole('group', {
      name: /^inference controls$/i,
    });
    const recordingControls = screen.getByRole('group', {
      name: /^rl recording controls$/i,
    });

    expect(inferenceControls).not.toContainElement(screen.getByRole('button', {
      name: /record inference rollout/i,
    }));
    expect(inferenceControls.compareDocumentPosition(recordingControls))
      .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  test('can hide recording controls when they are rendered by the workspace', () => {
    renderPanel({
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
      taskOverrides: { recordInferenceMode: true },
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

    const buttons = within(combined).getAllByRole('button');
    expect(buttons.slice(0, 4).map((button) => button.textContent.trim()))
      .toEqual(['Record', 'Start', 'Stop', 'Clear']);
    expect(buttons[0]).toBeDisabled();
    expect(buttons[1]).toBeEnabled();
    expect(buttons[2]).toBeDisabled();
    expect(buttons[3]).toBeDisabled();
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
});
