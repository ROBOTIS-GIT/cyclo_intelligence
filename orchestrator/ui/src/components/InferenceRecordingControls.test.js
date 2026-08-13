import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
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
} = {}) => {
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          inferenceMode,
          recordInferenceMode,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase: InferencePhase.INFERENCING,
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
      <InferenceRecordingControls />
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

  test('sends one start request and labels the rollout with an outcome', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderControls({ sendRecordCommand });

    fireEvent.click(screen.getByRole('button', {
      name: /record inference rollout/i,
    }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference_record');
    });

    fireEvent.click(await screen.findByRole('button', {
      name: /save inference rollout as success/i,
    }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenLastCalledWith(
        'stop_inference_record',
        { episodeOutcome: EpisodeOutcome.SUCCESS }
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
      expect(sendRecordCommand).toHaveBeenCalledWith('start_inference_record');
    });

    fireEvent.click(screen.getByRole('button', {
      name: /cancel and discard inference rollout/i,
    }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenLastCalledWith(
        'cancel_inference_record'
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
