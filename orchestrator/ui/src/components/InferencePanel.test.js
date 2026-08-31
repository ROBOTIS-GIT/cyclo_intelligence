import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferencePanel from './InferencePanel';
import taskReducer from '../features/tasks/taskSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    error: jest.fn(),
    success: jest.fn(),
  },
}));

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('./InferenceModelSelector', () => () => <div />);
jest.mock('./PolicyBackendControl', () => () => <div />);
jest.mock('./TrtEngineControl', () => () => <div />);
jest.mock('./FileBrowserModal', () => () => null);
jest.mock('./Tooltip', () => ({ children }) => <>{children}</>);

const renderPanel = ({
  inferenceMode,
  inferencePhase = InferencePhase.READY,
  initialPoseSync = true,
  inferenceHz = 15,
  controlHz = 100,
} = {}) => {
  const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
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
          initialPoseSync,
          initialPoseSyncDurationS: 5.0,
          inferenceHz,
          controlHz,
        },
        taskInfo: {
          ...initialTasks.taskInfo,
          inferenceMode,
          initialPoseSync,
          initialPoseSyncDurationS: 5.0,
          inferenceHz,
          controlHz,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <InferencePanel />
    </Provider>
  );
  return { store, sendRecordCommand };
};

describe('InferencePanel initial pose sync settings', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('preserves but disables initial pose sync in simulation mode', () => {
    renderPanel({ inferenceMode: 'simulation' });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeDisabled();
  });

  test('allows initial pose sync editing for an idle real robot session', () => {
    renderPanel({ inferenceMode: 'robot' });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeEnabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeEnabled();
  });

  test('shows duration only after initial pose sync is enabled', () => {
    renderPanel({ inferenceMode: 'robot', initialPoseSync: false });

    expect(screen.queryByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }));

    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeEnabled();
  });

  test('makes initial pose sync settings read-only while synchronizing', () => {
    renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.SYNCING,
    });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeDisabled();
  });

  test('keeps Dataset FPS blank while the user replaces its value', () => {
    jest.useFakeTimers();
    const { sendRecordCommand } = renderPanel({ inferenceMode: 'robot' });
    const datasetFpsInput = screen.getByRole('spinbutton', { name: 'Dataset FPS' });

    fireEvent.change(datasetFpsInput, { target: { value: '' } });
    expect(datasetFpsInput).toHaveValue(null);

    act(() => {
      jest.advanceTimersByTime(1000);
    });
    expect(datasetFpsInput).toHaveValue(null);
    expect(sendRecordCommand).not.toHaveBeenCalled();
    jest.useRealTimers();
  });

  test('shows a non-blocking warning for unusual Dataset FPS', () => {
    renderPanel({ inferenceMode: 'robot', inferenceHz: 1515, controlHz: 200 });

    expect(screen.getByRole('status', { name: 'Timing warnings' }))
      .toHaveTextContent('Dataset FPS is unusually high (1515)');
  });
});
