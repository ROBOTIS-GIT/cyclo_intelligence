import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferencePanel from './InferencePanel';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import taskReducer from '../features/tasks/taskSlice';
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
jest.mock('./FileBrowserModal', () => function MockFileBrowserModal() {
  return null;
});
jest.mock('./InferenceModelSelector', () => function MockModelSelector() {
  return <div data-testid="model-selector" />;
});
jest.mock('./PolicyBackendControl', () => function MockBackendControl() {
  return <div data-testid="backend-control" />;
});
jest.mock('./TrtEngineControl', () => function MockTrtControl() {
  return <div data-testid="trt-control" />;
});
jest.mock('./Tooltip', () => function MockTooltip({ children }) {
  return <>{children}</>;
});

const renderPanel = ({
  inferenceMode = 'simulation',
  recordInferenceMode = false,
  inferencePhase = InferencePhase.READY,
  recordPhase = RecordPhase.READY,
  taskType = '',
} = {}) => {
  const sendRecordCommand = jest.fn().mockResolvedValue({
    success: true,
    message: 'ok',
  });
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
          inferencePhase,
        },
        recordStatus: {
          ...initialTasks.recordStatus,
          taskType,
          recordInferenceMode: taskType === 'inference',
          recordPhase,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <InferencePanel />
    </Provider>
  );
  return { sendRecordCommand, store };
};

describe('InferencePanel RL Recording', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('normalizes RL Recording off and disables it for simulation', async () => {
    const { store } = renderPanel({ recordInferenceMode: true });
    const toggle = screen.getByRole('checkbox', { name: /enable rl recording/i });

    expect(toggle).toBeDisabled();
    await waitFor(() => {
      expect(store.getState().tasks.inferenceTaskInfo.recordInferenceMode)
        .toBe(false);
    });
  });

  test('allows opting in before a Real Robot deploy', () => {
    const { store } = renderPanel({ inferenceMode: 'robot' });
    const toggle = screen.getByRole('checkbox', { name: /enable rl recording/i });

    expect(toggle).toBeEnabled();
    fireEvent.click(toggle);
    expect(store.getState().tasks.inferenceTaskInfo.recordInferenceMode)
      .toBe(true);
  });

  test('places RL Recording last with a settings divider', () => {
    renderPanel({ inferenceMode: 'robot' });

    const controlHz = screen.getByText('Control Hz');
    const rlRecording = screen.getByText('RL Recording');
    const divider = screen.getByRole('separator', {
      name: /rl recording settings/i,
    });

    expect(controlHz.compareDocumentPosition(divider))
      .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(divider.compareDocumentPosition(rlRecording))
      .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });

  test('blocks deploy target switching while an inference recording is active', () => {
    const { sendRecordCommand } = renderPanel({
      inferenceMode: 'robot',
      recordInferenceMode: true,
      inferencePhase: InferencePhase.INFERENCING,
      recordPhase: RecordPhase.RECORDING,
      taskType: 'inference',
    });

    expect(screen.getByRole('button', { name: /use 3d sim deploy/i }))
      .toBeDisabled();
    expect(screen.getByRole('button', { name: /use real robot deploy/i }))
      .toBeDisabled();
    expect(sendRecordCommand).not.toHaveBeenCalledWith('finish');
  });
});
