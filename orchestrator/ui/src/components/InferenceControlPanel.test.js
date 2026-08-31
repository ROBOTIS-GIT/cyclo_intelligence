import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import toast from 'react-hot-toast';
import InferenceControlPanel from './InferenceControlPanel';
import taskReducer, { setInferenceStatus } from '../features/tasks/taskSlice';
import rosReducer from '../features/ros/rosSlice';
import { InferencePhase } from '../constants/taskPhases';
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
      <InferenceControlPanel />
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
      });
    });
  });

  test('shows the configured initial pose sync duration in the robot warning', async () => {
    renderPanel({
      inferenceMode: 'robot',
      taskOverrides: {
        initialPoseSync: true,
        initialPoseSyncDurationS: 7.5,
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    expect(await screen.findByText('Initial Pose Sync: 7.5 s')).toBeInTheDocument();
  });

  test('shows unusual Dataset FPS in the robot confirmation without blocking deploy', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'robot',
      taskOverrides: { inferenceHz: 1515, controlHz: 200 },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    expect(await screen.findByText(
      'Dataset FPS is unusually high (1515). Confirm it matches the training dataset.'
    )).toBeInTheDocument();
    expect(screen.getByText(
      'Dataset FPS is higher than Control Hz, so action waypoints will be downsampled.'
    )).toBeInTheDocument();
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('does not start while Dataset FPS is blank', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'robot',
      taskOverrides: { inferenceHz: '' },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Missing required fields: Dataset FPS');
    });
    expect(screen.queryByRole('dialog', { name: /real robot deploy/i }))
      .not.toBeInTheDocument();
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('resumes a regular robot session without another deploy warning', async () => {
    const { store, sendRecordCommand } = renderPanel({
      inferenceMode: 'robot',
      taskOverrides: {
        initialPoseSync: true,
        initialPoseSyncDurationS: 10.0,
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));
    fireEvent.click(await screen.findByRole('button', { name: /^Real Robot Deploy$/i }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'robot',
      });
    });

    act(() => {
      store.dispatch(setInferenceStatus({ inferencePhase: InferencePhase.INFERENCING }));
    });
    act(() => {
      store.dispatch(setInferenceStatus({ inferencePhase: InferencePhase.PAUSED }));
    });
    sendRecordCommand.mockClear();

    fireEvent.click(screen.getByRole('button', { name: /resume inference/i }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('resume_inference', {
        inferenceMode: 'robot',
      });
    });
    expect(screen.queryByRole('dialog', { name: /real robot deploy/i }))
      .not.toBeInTheDocument();
  });

  test('resumes an interrupted sync without another deploy warning', async () => {
    const { store, sendRecordCommand } = renderPanel({
      inferenceMode: 'robot',
      taskOverrides: {
        initialPoseSync: true,
        initialPoseSyncDurationS: 10.0,
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));
    fireEvent.click(await screen.findByRole('button', { name: /^Real Robot Deploy$/i }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'robot',
      });
    });

    act(() => {
      store.dispatch(setInferenceStatus({ inferencePhase: InferencePhase.SYNCING }));
    });
    act(() => {
      store.dispatch(setInferenceStatus({ inferencePhase: InferencePhase.PAUSED }));
    });
    sendRecordCommand.mockClear();

    fireEvent.click(screen.getByRole('button', { name: /resume inference/i }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('resume_inference', {
        inferenceMode: 'robot',
      });
    });
    expect(screen.queryByRole('dialog', { name: /real robot deploy/i }))
      .not.toBeInTheDocument();
  });

  test('rejects an invalid initial pose sync duration before robot confirmation', async () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'robot',
      taskOverrides: {
        initialPoseSync: true,
        initialPoseSyncDurationS: 0.5,
      },
    });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        'Initial Pose Sync duration must be between 1 and 60 seconds'
      );
    });
    expect(screen.queryByRole('dialog', { name: /real robot deploy/i }))
      .not.toBeInTheDocument();
    expect(sendRecordCommand).not.toHaveBeenCalled();
  });

  test('keeps Stop and Clear enabled while initial pose sync is active', () => {
    renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.SYNCING,
    });

    expect(screen.getByRole('button', { name: /start inference/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /pause inference/i })).toBeEnabled();
    expect(screen.getByRole('button', { name: /unload model/i })).toBeEnabled();
    expect(screen.getByText('Synchronizing initial robot pose...')).toBeInTheDocument();
  });

  test.each([
    ['Stop', /pause inference/i, 'stop_inference'],
    ['Clear', /unload model/i, 'finish'],
  ])('keeps SYNCING after a failed %s hold', async (
    _label,
    buttonName,
    command,
  ) => {
    const sendRecordCommand = jest.fn().mockResolvedValue({
      success: false,
      message: 'current-pose hold failed; retry',
    });
    const { store } = renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.SYNCING,
      sendRecordCommand,
    });

    fireEvent.click(screen.getByRole('button', { name: buttonName }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(command, {});
      expect(toast.error).toHaveBeenCalledWith(
        'Command failed: current-pose hold failed; retry'
      );
    });
    expect(store.getState().tasks.inferenceStatus.inferencePhase)
      .toBe(InferencePhase.SYNCING);
    expect(screen.getByRole('button', { name: /pause inference/i })).toBeEnabled();
    expect(screen.getByRole('button', { name: /unload model/i })).toBeEnabled();
  });

  test('can switch the pending start to 3D Sim Deploy from the warning', async () => {
    const { store, sendRecordCommand } = renderPanel({ inferenceMode: 'robot' });

    fireEvent.click(screen.getByRole('button', { name: /start inference/i }));
    fireEvent.click(await screen.findByRole('button', { name: /^3D Sim Deploy$/i }));

    await waitFor(() => {
      expect(store.getState().tasks.taskInfo.inferenceMode).toBe('simulation');
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference', {
        inferenceMode: 'simulation',
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
      });
    });
  });
});
