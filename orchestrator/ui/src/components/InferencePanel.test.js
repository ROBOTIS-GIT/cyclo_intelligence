import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
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
jest.mock('./FileBrowserModal', () => function MockFileBrowserModal(props) {
  return (
    <>
      <span data-testid="file-browser-modal" data-title={props.title} />
      {props.isOpen && (
        <button
          type="button"
          onClick={() => props.onFileSelect({
            is_directory: true,
            name: 'Task_existing_inference_MCAP',
            full_path: '/workspace/rosbag2/Task_existing_inference_MCAP',
          })}
        >
          Choose {props.title}
        </button>
      )}
    </>
  );
});
jest.mock('./InferenceModelSelector', () => function MockModelSelector({ label = 'Model' }) {
  return <div data-testid="model-selector">{label}</div>;
});
jest.mock('./PolicyBackendControl', () => function MockBackendControl({ children }) {
  return <div data-testid="backend-control">{children}</div>;
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
  serviceType = 'lerobot',
  policyType = 'act',
  accelerationMode = 'pytorch',
  taskInstruction = ['Pick up the object'],
  panelProps = {},
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
        sharedTaskInfo: {
          ...initialTasks.sharedTaskInfo,
          taskInstruction,
        },
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          inferenceMode,
          recordInferenceMode,
          serviceType,
          policyType,
          accelerationMode,
          taskInstruction,
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

  const view = render(
    <Provider store={store}>
      <InferencePanel {...panelProps} />
    </Provider>
  );
  return { ...view, sendRecordCommand, store };
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

  test('selects and clears an existing RL Recording folder', () => {
    const { store } = renderPanel({
      inferenceMode: 'robot',
      recordInferenceMode: true,
    });

    expect(screen.getByText('Automatic new folder')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', {
      name: /select rl recording folder/i,
    }));
    fireEvent.click(screen.getByRole('button', {
      name: /choose select rl recording folder/i,
    }));

    expect(store.getState().tasks.inferenceTaskInfo.recordingFolder).toBe(
      '/workspace/rosbag2/Task_existing_inference_MCAP'
    );
    expect(screen.getByText('Task_existing_inference_MCAP')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', {
      name: /clear rl recording folder/i,
    }));
    expect(store.getState().tasks.inferenceTaskInfo.recordingFolder).toBe('');
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

  test('supports the embedded inference workspace composition', () => {
    renderPanel({
      panelProps: {
        title: 'Inference Settings',
        embedded: true,
        showPolicyPath: false,
        showRecordingSettings: false,
        modelLabel: 'Policy',
      },
    });

    const panel = screen.getByText('Inference Settings').parentElement;
    expect(panel).not.toHaveClass(
      'border',
      'border-gray-200',
      'shadow-md',
      'p-4',
      'max-w-[350px]'
    );
    expect(screen.getByTestId('model-selector')).toHaveTextContent('Policy');
    expect(screen.queryByText('Policy Path')).not.toBeInTheDocument();
    expect(screen.queryByText('RL Recording')).not.toBeInTheDocument();
    expect(screen.queryByRole('separator', {
      name: /rl recording settings/i,
    })).not.toBeInTheDocument();
    expect(screen.queryAllByTestId('file-browser-modal')).toHaveLength(0);
    expect(screen.getByText('Action Request')).toBeInTheDocument();
    expect(screen.getByText('Inference Hz')).toBeInTheDocument();
    expect(screen.getByText('Control Hz')).toBeInTheDocument();
    expect(screen.queryByTestId('offline-rl-runtime-row')).not.toBeInTheDocument();
    expect(screen.queryByTestId('offline-rl-action-path-row')).not.toBeInTheDocument();
  });

  test('uses the same Offline RL active green for 3D Sim Deploy', () => {
    renderPanel({
      panelProps: {
        embedded: true,
        variant: 'offlineRL',
        settingsAside: <div data-testid="workspace-paths-slot">Paths</div>,
      },
    });

    expect(screen.getByRole('button', { name: /use 3d sim deploy/i }))
      .toHaveClass('bg-[#69866f]');
    expect(screen.queryByText(/Edit mode/i)).not.toBeInTheDocument();
    expect(screen.getByTestId('offline-rl-runtime-row')).toHaveClass(
      'grid',
      'md:grid-cols-2'
    );
    expect(screen.getByTestId('offline-rl-action-path-row')).toHaveClass(
      'grid',
      'md:grid-cols-2'
    );
    expect(screen.getByTestId('workspace-paths-slot')).toBeInTheDocument();
  });

  test('places GR00T instruction inside Deploy Target and TensorRT inside GR00T Docker', () => {
    renderPanel({
      serviceType: 'groot',
      policyType: 'n17',
      accelerationMode: 'tensorrt_dit',
      panelProps: {
        embedded: true,
        variant: 'offlineRL',
        settingsAside: <div data-testid="workspace-paths-slot">Paths</div>,
      },
    });

    const deployTargetCard = screen.getByText('Deploy Target')
      .parentElement.parentElement;
    const actionTimingCard = screen.getByText('Action & timing').parentElement;
    const backendCard = screen.getByTestId('backend-control');

    expect(deployTargetCard).not.toBeNull();
    expect(actionTimingCard).not.toBeNull();
    expect(within(deployTargetCard).getByText('Task Instruction'))
      .toBeInTheDocument();
    expect(within(backendCard).getByText('TensorRT')).toBeInTheDocument();
    expect(within(backendCard).getByTestId('trt-control')).toBeInTheDocument();
    expect(within(actionTimingCard).queryByText('TensorRT'))
      .not.toBeInTheDocument();
  });

  test('places Pi instruction inside Deploy Target without TensorRT', () => {
    renderPanel({
      serviceType: 'lerobot',
      policyType: 'pi05',
      panelProps: {
        embedded: true,
        variant: 'offlineRL',
        settingsAside: <div data-testid="workspace-paths-slot">Paths</div>,
      },
    });

    const deployTargetCard = screen.getByText('Deploy Target')
      .parentElement.parentElement;

    expect(deployTargetCard).not.toBeNull();
    expect(within(deployTargetCard).getByText('Task Instruction'))
      .toBeInTheDocument();
    expect(screen.queryByText('TensorRT')).not.toBeInTheDocument();
  });

  test('hides Task Instruction and TensorRT for ACT', () => {
    renderPanel({
      serviceType: 'lerobot',
      policyType: 'act',
      panelProps: {
        embedded: true,
        variant: 'offlineRL',
        settingsAside: <div data-testid="workspace-paths-slot">Paths</div>,
      },
    });

    expect(screen.queryByText('Task Instruction')).not.toBeInTheDocument();
    expect(screen.queryByText('TensorRT')).not.toBeInTheDocument();
  });

  test.each([
    ['GR00T', 'groot', 'n17'],
    ['Pi0.5', 'lerobot', 'pi05'],
  ])('updates the running %s task instruction through the inference command', async (
    _label,
    serviceType,
    policyType
  ) => {
    const { sendRecordCommand } = renderPanel({
      serviceType,
      policyType,
      inferencePhase: InferencePhase.INFERENCING,
      taskInstruction: ['Pick up the jelly bag'],
      panelProps: {
        embedded: true,
        variant: 'offlineRL',
      },
    });

    const updateButton = screen.getByRole('button', {
      name: 'Update Task Instruction',
    });
    expect(updateButton).toBeEnabled();
    fireEvent.click(updateButton);

    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith(
      'update_instruction',
      { taskSource: 'inference' }
    ));
  });
});
