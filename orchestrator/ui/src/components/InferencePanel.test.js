import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferencePanel from './InferencePanel';
import taskReducer, { receiveServerInferenceTaskInfo, setInferenceStatus, setInferenceTaskInfo } from '../features/tasks/taskSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { PolicyCatalogProvider } from '../contexts/PolicyCatalogContext';
import { testPolicyCatalog } from '../testUtils/policyCatalog';

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
jest.mock('./InferenceTryResults', () => () => <section aria-label="Try Results" />);
jest.mock('./PolicyBackendControl', () => () => <div />);
jest.mock('./TrtEngineControl', () => () => <div />);
jest.mock('./FileBrowserModal', () => () => null);
jest.mock('./Tooltip', () => ({ children, content }) => <span title={content}>{children}</span>);

const renderPanel = ({
  inferenceMode,
  inferencePhase = InferencePhase.READY,
  initialPoseSync = true,
  inferenceHz = 15,
  controlHz = 100,
  policyId = 'lerobot:act',
  catalog = testPolicyCatalog,
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
          policyId,
        },
        taskInfo: {
          ...initialTasks.taskInfo,
          inferenceMode,
          initialPoseSync,
          initialPoseSyncDurationS: 5.0,
          inferenceHz,
          controlHz,
          policyId,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
      },
    },
  });

  render(
    <PolicyCatalogProvider initialCatalog={catalog}>
      <Provider store={store}>
        <InferencePanel />
      </Provider>
    </PolicyCatalogProvider>
  );
  return { store, sendRecordCommand };
};

describe('InferencePanel initial pose sync settings', () => {
  test('model-owned queues display All and reject count edits', () => {
    const { store } = renderPanel({ policyId: 'lerobot:diffusion' });
    const input = screen.getByRole('spinbutton', { name: 'Action Steps' });
    expect(input).toBeDisabled();
    expect(input).toHaveAttribute('placeholder', 'All');
    fireEvent.change(input, { target: { value: '5' } });
    expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(0);
    expect(screen.getByLabelText('About Action Steps').parentElement)
      .toHaveAttribute('title', expect.stringContaining('manages its own action queue'));
  });

  test('an old catalog without execution metadata does not enable numeric selection', () => {
    const catalog = JSON.parse(JSON.stringify(testPolicyCatalog));
    catalog.runtimes.forEach(runtime => runtime.models.forEach(model => delete model.execution_mode));
    renderPanel({ catalog });
    const input = screen.getByRole('spinbutton', { name: 'Action Steps' });
    expect(input).toBeDisabled();
    expect(input).toHaveAttribute('placeholder', 'Unknown');
  });
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('explains Action Steps beside its label without changing the input label', () => {
    renderPanel();
    const help = screen.getByLabelText('About Action Steps');
    expect(help.parentElement).toHaveAttribute('title',
      'Choose how many actions to use from each prediction after alignment. Leave empty for All. Max shows the latest model chunk length after Start, not the selected count. Changes apply on Start; alignment may leave fewer actions.');
    expect(screen.getByRole('spinbutton', { name: 'Action Steps' })).toBeInTheDocument();
  });

  test('shows the latest chunk without changing All or the selected execution limit', () => {
    const { store } = renderPanel({ inferencePhase: InferencePhase.PAUSED });
    act(() => {
      store.dispatch(setInferenceTaskInfo({ policyPath: '/models/act/', actionSteps: 0 }));
      store.dispatch(setInferenceStatus({
        topicReceived: true, loadedPolicyId: 'lerobot:act', loadedModelPath: '/models/act',
        observedChunkSize: 15, sourceId: 'runtime', sequence: 1,
      }));
    });
    const input = screen.getByRole('spinbutton', { name: 'Action Steps' });
    expect(input).toHaveAttribute('placeholder', 'All');
    expect(screen.getByText('Max 15')).toBeInTheDocument();
    expect(input).toHaveValue(null);
    expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(0);
    fireEvent.change(input, { target: { value: '2' } });
    expect(input).toHaveValue(2);
    expect(screen.getByText('Max 15')).toBeInTheDocument();
    fireEvent.change(input, { target: { value: '5' } });
    expect(screen.getByText('Max 15')).toBeInTheDocument();
    act(() => store.dispatch(setInferenceStatus({ observedChunkSize: 10, sourceId: 'runtime', sequence: 2 })));
    expect(input).toHaveValue(5);
    expect(screen.getByText('Max 10')).toBeInTheDocument();
    act(() => store.dispatch(setInferenceStatus({ observedChunkSize: 99, sourceId: 'runtime', sequence: 1 })));
    expect(screen.getByText('Max 10')).toBeInTheDocument();
    fireEvent.change(input, { target: { value: '123213' } });
    expect(input).toHaveValue(123213);
    expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(123213);
    expect(screen.getByText('Max 10')).toBeInTheDocument();
    act(() => store.dispatch(setInferenceStatus({ observedChunkSize: 3, sourceId: 'runtime', sequence: 3 })));
    expect(screen.getByText('Max 3')).toBeInTheDocument();
    expect(input).toHaveValue(123213);
    fireEvent.change(input, { target: { value: '' } });
    expect(input).toHaveValue(null);
    expect(screen.getByText('Max 3')).toBeInTheDocument();
    expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(0);
    act(() => store.dispatch(setInferenceStatus({ topicReceived: false })));
    expect(screen.queryByText(/^Max /)).not.toBeInTheDocument();
    expect(input).not.toHaveAttribute('aria-describedby');
  });

  test.each([
    { topicReceived: false },
    { loadedModelPath: '/models/other' },
    { loadedPolicyId: 'lerobot/groot' },
    { inferencePhase: InferencePhase.READY },
    { inferencePhase: InferencePhase.LOADING },
    { observedChunkSize: 0 },
    { observedChunkSize: -1 },
    { observedChunkSize: 1.5 },
  ])('does not show stale or unrelated chunk metadata: %j', (override) => {
    const { store } = renderPanel({ inferencePhase: InferencePhase.PAUSED });
    act(() => {
      store.dispatch(setInferenceTaskInfo({ policyPath: '/models/act', actionSteps: 0 }));
      store.dispatch(setInferenceStatus({
        topicReceived: true, loadedPolicyId: 'lerobot:act', loadedModelPath: '/models/act',
        observedChunkSize: 15, ...override,
      }));
    });
    expect(screen.getByRole('spinbutton', { name: 'Action Steps' })).toHaveAttribute('placeholder', 'All');
    expect(screen.queryByText(/^Max /)).not.toBeInTheDocument();
    expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(0);
  });

  test.each([InferencePhase.READY, InferencePhase.PAUSED])(
    'edits Action Steps in phase %s and clearing restores All', (inferencePhase) => {
      const { store } = renderPanel({ inferencePhase });
      const input = screen.getByRole('spinbutton', { name: 'Action Steps' });
      expect(input).toBeEnabled();
      expect(input).toHaveAttribute('placeholder', 'All');
      fireEvent.change(input, { target: { value: '10' } });
      expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(10);
      fireEvent.change(input, { target: { value: '-2' } });
      expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(10);
      fireEvent.change(input, { target: { value: '2.5' } });
      expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(10);
      fireEvent.change(input, { target: { value: '' } });
      expect(store.getState().tasks.inferenceTaskInfo.actionSteps).toBe(0);
    }
  );

  test.each([InferencePhase.LOADING, InferencePhase.INFERENCING, InferencePhase.SYNCING])(
    'locks Action Steps in phase %s', (inferencePhase) => {
      renderPanel({ inferencePhase });
      expect(screen.getByRole('spinbutton', { name: 'Action Steps' })).toBeDisabled();
    }
  );

  test('groups model settings, execution settings, saved pose, and recording tools in order', () => {
    renderPanel({ inferenceMode: 'robot', policyId: 'lerobot:groot' });
    const separators = screen.getAllByRole('separator');
    expect(separators).toHaveLength(3);
    const ordered = [
      screen.getByPlaceholderText('Enter Policy Path or Repo ID'),
      screen.getByPlaceholderText('Enter Task Instruction'),
      separators[0],
      screen.getByText('Action Request'),
      screen.getByRole('spinbutton', { name: 'Dataset FPS' }),
      screen.getByRole('spinbutton', { name: 'Control Hz' }),
      screen.getByRole('checkbox', { name: 'Slow Start' }),
      screen.getByRole('spinbutton', { name: 'Slow Start duration' }),
      separators[1],
      screen.getByRole('region', { name: 'Saved Initial Pose' }),
      separators[2],
      screen.getByRole('group', { name: 'Inference recording' }),
      screen.getByRole('region', { name: 'Try Results' }),
    ];
    ordered.slice(1).forEach((element, index) => {
      expect(ordered[index].compareDocumentPosition(element) & Node.DOCUMENT_POSITION_FOLLOWING)
        .toBeTruthy();
    });
  });

  test('does not leave separators for hidden robot-only tools in simulation', () => {
    renderPanel({ inferenceMode: 'simulation' });
    expect(screen.getAllByRole('separator')).toHaveLength(1);
    expect(screen.queryByRole('region', { name: 'Saved Initial Pose' })).not.toBeInTheDocument();
  });

  test('restores a fresh panel from the backend without submitting its initial defaults', () => {
    jest.useFakeTimers();
    try {
      const { store, sendRecordCommand } = renderPanel();
      act(() => store.dispatch(receiveServerInferenceTaskInfo({
        sourceId: 'backend', revision: 4, hasTaskInfo: true,
        taskInfo: {
          taskType: 'inference', policyPath: '/models/saved',
          policyId: 'lerobot:groot', serviceType: 'lerobot', policyType: 'groot',
          inferenceHz: 30, controlHz: 80, inferenceMode: 'robot',
          actionSteps: 12,
          taskInstruction: ['Saved instruction'], initialPoseSync: true,
          initialPoseSyncDurationS: 7.0,
        },
      })));
      expect(screen.getByPlaceholderText('Enter Policy Path or Repo ID')).toHaveValue('/models/saved');
      expect(screen.getByRole('spinbutton', { name: 'Dataset FPS' })).toHaveValue(30);
      expect(screen.getByRole('spinbutton', { name: 'Control Hz' })).toHaveValue(80);
      expect(screen.getByRole('spinbutton', { name: 'Action Steps' })).toHaveValue(12);
      expect(screen.getByPlaceholderText('Enter Task Instruction')).toHaveValue('Saved instruction');
      expect(screen.getByRole('spinbutton', { name: 'Slow Start duration' })).toHaveValue(7);
      act(() => jest.advanceTimersByTime(1000));
      expect(sendRecordCommand).not.toHaveBeenCalled();
    } finally {
      jest.useRealTimers();
    }
  });

  test('preserves but disables initial pose sync in simulation mode', () => {
    renderPanel({ inferenceMode: 'simulation' });

    expect(screen.queryByRole('region', { name: 'Try Results' })).not.toBeInTheDocument();
    expect(screen.queryByRole('group', { name: 'Inference recording' })).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Slow Start' }))
      .toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Slow Start' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Slow Start duration' }))
      .toBeDisabled();
  });

  test('allows initial pose sync editing for an idle real robot session', () => {
    renderPanel({ inferenceMode: 'robot' });

    expect(screen.getByRole('region', { name: 'Try Results' })).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Inference recording' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Slow Start' }))
      .toBeEnabled();
    expect(screen.getByRole('spinbutton', { name: 'Slow Start duration' }))
      .toBeEnabled();
  });

  test('shows duration only after initial pose sync is enabled', () => {
    renderPanel({ inferenceMode: 'robot', initialPoseSync: false });

    expect(screen.queryByRole('spinbutton', { name: 'Slow Start duration' }))
      .not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: 'Slow Start' }));

    expect(screen.getByRole('spinbutton', { name: 'Slow Start duration' }))
      .toBeEnabled();
  });

  test('makes initial pose sync settings read-only while synchronizing', () => {
    renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.SYNCING,
    });

    expect(screen.getByRole('checkbox', { name: 'Slow Start' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Slow Start duration' }))
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
