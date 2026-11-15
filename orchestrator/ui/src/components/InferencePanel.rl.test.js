import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferencePanel from './RLInferencePanel';
import { resolveRLTStage2Bundle } from '../utils/offlineRlApi';
jest.mock('../utils/offlineRlApi', () => ({ resolveRLTStage2Bundle: jest.fn() }));
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
  const isRltBundle = props.title === 'Select RLT bundle folder';
  return (
    <>
      <span data-testid="file-browser-modal" data-title={props.title} data-target-files={JSON.stringify(props.targetFileName)} data-target-folders={JSON.stringify(props.targetFolderName)} />
      {props.isOpen && (
        <button
          type="button"
          onClick={() => props.onFileSelect({
            is_directory: true,
            name: isRltBundle
              ? 'showroom_groot_bundle'
              : 'Task_900003_Drill_into_Screw_Fast_MCAP',
            full_path: isRltBundle
              ? '/workspace/checkpoint/rlt/showroom_groot_bundle'
              : '/workspace/rosbag2/Task_900003_Drill_into_Screw_Fast_MCAP',
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
  actionRequestMode = 'async',
  controlHz = 100,
  rltEnabled = false,
  rltBundlePath = '',
  rltRobotOverride = false,
  initialPoseSync = false,
  initialPoseSyncDurationS = 5.0,
  taskInstruction = ['Pick up the object'],
  panelProps = { variant: 'offlineRL' },
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
          actionRequestMode,
          controlHz,
          rltEnabled,
          rltBundlePath,
          rltRobotOverride,
          initialPoseSync,
          initialPoseSyncDurationS,
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

  test('selects and clears an existing ROS bag dataset as the RL Recording folder', () => {
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
      '/workspace/rosbag2/Task_900003_Drill_into_Screw_Fast_MCAP'
    );
    expect(screen.getByText('Task_900003_Drill_into_Screw_Fast_MCAP'))
      .toBeInTheDocument();

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

  test('offers TT-RTC only for GR00T N1.7 and stores the selection', () => {
    const { store } = renderPanel({
      serviceType: 'groot',
      policyType: 'n17',
      accelerationMode: 'tensorrt_dit',
      controlHz: 200,
    });

    const ttRtc = screen.getByRole('button', {
      name: 'Use TT-RTC action requests',
    });
    expect(ttRtc).toBeEnabled();
    fireEvent.click(ttRtc);
    expect(store.getState().tasks.inferenceTaskInfo.actionRequestMode)
      .toBe('tt_rtc');
    expect(store.getState().tasks.inferenceTaskInfo.accelerationMode)
      .toBe('tensorrt_dit');
    expect(store.getState().tasks.inferenceTaskInfo.accelerationEnginePath)
      .toBe('');
    expect(store.getState().tasks.inferenceTaskInfo.inferenceHz).toBe(15);
    expect(store.getState().tasks.inferenceTaskInfo.controlHz).toBe(200);
    expect(screen.getByRole('checkbox', { name: 'Enable TensorRT' }))
      .toBeEnabled();
    expect(screen.getByRole('checkbox', { name: 'Enable TensorRT' }))
      .toBeChecked();
    expect(screen.getByRole('spinbutton', { name: 'Inference Hz' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Control Hz' }))
      .toBeEnabled();
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Control Hz' }), {
      target: { value: '100' },
    });
    expect(store.getState().tasks.inferenceTaskInfo.controlHz).toBe(100);
  });

  test.each([
    ['ACT', 'lerobot', 'act'],
    ['unsupported GR00T', 'groot', 'future'],
  ])('does not offer TT-RTC for %s', (_label, serviceType, policyType) => {
    renderPanel({ serviceType, policyType });
    expect(screen.queryByRole('button', {
      name: 'Use TT-RTC action requests',
    })).not.toBeInTheDocument();
  });

  test('configures RLT preload without the removed robot override or async update UI', () => {
    const { store } = renderPanel({
      serviceType: 'groot',
      policyType: 'n17',
      accelerationMode: 'tensorrt_dit',
      panelProps: {
        embedded: true,
        variant: 'offlineRL',
      },
    });

    const rltToggle = screen.getByRole('checkbox', { name: 'Enable RLT' });
    const bundlePath = screen.getByRole('textbox', { name: 'RLT Bundle Path' });
    const bundleBrowser = screen.getByRole('button', {
      name: 'Browse RLT Bundle Path',
    });
    expect(screen.queryByRole('checkbox', { name: 'Allow RLT on Real Robot' })).not.toBeInTheDocument();
    expect(screen.queryByText('MLP Policy Update')).not.toBeInTheDocument();
    expect(screen.queryByRole('switch', { name: 'Async RL' })).not.toBeInTheDocument();

    expect(rltToggle).not.toBeChecked();
    expect(bundlePath).toBeDisabled();
    expect(bundleBrowser).toBeDisabled();

    fireEvent.click(rltToggle);

    expect(store.getState().tasks.inferenceTaskInfo.rltEnabled).toBe(true);
    expect(store.getState().tasks.inferenceTaskInfo.accelerationMode)
      .toBe('tensorrt_dit');
    expect(bundlePath).toBeEnabled();
    expect(bundleBrowser).toBeEnabled();
    expect(screen.getByRole('checkbox', { name: 'Enable TensorRT' }))
      .toBeEnabled();
    expect(screen.getByRole('checkbox', { name: 'Enable TensorRT' }))
      .toBeChecked();

    fireEvent.click(bundleBrowser);
    fireEvent.click(screen.getByRole('button', {
      name: 'Choose Select RLT bundle folder',
    }));
    expect(store.getState().tasks.inferenceTaskInfo.rltBundlePath).toBe(
      '/workspace/checkpoint/rlt/showroom_groot_bundle'
    );


    fireEvent.click(rltToggle);
    expect(store.getState().tasks.inferenceTaskInfo.rltEnabled).toBe(false);
    expect(store.getState().tasks.inferenceTaskInfo.rltRobotOverride).toBe(false);
  });

  test.each([
    ['ACT', 'lerobot', 'act'],
    ['Pi0.5', 'lerobot', 'pi05'],
    ['another GR00T policy', 'groot', 'future'],
  ])('hides RLT controls for %s', (_label, serviceType, policyType) => {
    renderPanel({ serviceType, policyType });

    expect(screen.queryByRole('checkbox', { name: 'Enable RLT' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'RLT Bundle Path' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'Allow RLT on Real Robot' }))
      .not.toBeInTheDocument();
  });

  test('locks RLT settings while inference is running', () => {
    renderPanel({
      serviceType: 'groot',
      policyType: 'n17',
      rltEnabled: true,
      rltBundlePath: '/workspace/checkpoint/rlt/showroom_groot_bundle',
      inferencePhase: InferencePhase.INFERENCING,
    });

    expect(screen.getByRole('checkbox', { name: 'Enable RLT' })).toBeDisabled();
    expect(screen.getByRole('textbox', { name: 'RLT Bundle Path' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Browse RLT Bundle Path' }))
      .toBeDisabled();
    expect(screen.queryByRole('checkbox', { name: 'Allow RLT on Real Robot' }))
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

  test('hides TensorRT for an unsupported policy on the GR00T backend', () => {
    renderPanel({
      serviceType: 'groot',
      policyType: 'future',
      accelerationMode: 'tensorrt_dit',
    });

    expect(screen.queryByText('TensorRT')).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'Enable TensorRT' }))
      .not.toBeInTheDocument();
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

describe('InferencePanel Initial Pose Sync', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('is opt-in and disabled for simulation', () => {
    renderPanel({ inferenceMode: 'simulation' });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeDisabled();
  });

  test('stores a real-robot minimum duration for automatic extension', () => {
    const { store } = renderPanel({
      inferenceMode: 'robot',
      initialPoseSync: true,
      initialPoseSyncDurationS: 5.0,
    });
    const duration = screen.getByRole('spinbutton', {
      name: 'Initial Pose Sync minimum duration',
    });

    expect(duration).toBeEnabled();
    fireEvent.change(duration, { target: { value: '3.5' } });

    expect(store.getState().tasks.inferenceTaskInfo.initialPoseSyncDurationS)
      .toBe(3.5);
  });

  test('locks controls while syncing', () => {
    renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.SYNCING,
      initialPoseSync: true,
    });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', {
      name: 'Initial Pose Sync minimum duration',
    })).toBeDisabled();
  });
});


describe('Official inference presentation', () => {
  test('shows upstream settings and preserves the configured TT-RTC state', () => {
    const { store } = renderPanel({ serviceType: 'groot', policyType: 'n17', actionRequestMode: 'tt_rtc', panelProps: {} });
    expect(screen.getByText('Task Information')).toBeVisible();
    expect(screen.getByLabelText('Dataset FPS')).toBeDisabled();
    expect(screen.getByLabelText('Control Hz')).toBeEnabled();
    expect(screen.getByText('TT-RTC')).toBeVisible();
    expect(screen.queryByText('RL Recording')).not.toBeInTheDocument();
    expect(screen.queryByText('MLP Policy Update')).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'Enable RLT' })).not.toBeInTheDocument();
    expect(store.getState().tasks.inferenceTaskInfo.actionRequestMode).toBe('tt_rtc');
  });
});


test('migrates a saved legacy RLT path and restricts the picker to completed bundles', async () => {
  const legacy = '/workspace/checkpoint/rlt/stage2/steps_0010000_173c3d33c1ce';
  const target = '/workspace/checkpoint/rlt/stage2/actor/latest';
  resolveRLTStage2Bundle.mockResolvedValueOnce({ bundle_path: target });
  const { store } = renderPanel({ serviceType: 'groot', policyType: 'n17', rltEnabled: true, rltBundlePath: legacy, panelProps: { variant: 'offlineRL' } });
  await waitFor(() => expect(store.getState().tasks.inferenceTaskInfo.rltBundlePath).toBe(target));
  expect(resolveRLTStage2Bundle).toHaveBeenCalledWith(legacy);
  const picker = screen.getAllByTestId('file-browser-modal').find((element) => element.dataset.title === 'Select RLT bundle folder');
  expect(JSON.parse(picker.dataset.targetFiles)).toEqual(['manifest.json']);
  expect(JSON.parse(picker.dataset.targetFolders)).toEqual(['artifacts', 'training_state', 'replay_cache']);
});
