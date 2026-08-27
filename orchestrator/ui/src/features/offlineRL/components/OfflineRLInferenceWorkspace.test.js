import { configureStore } from '@reduxjs/toolkit';
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import { Provider } from 'react-redux';
import { EpisodeOutcome } from '../../../constants/taskCommand';
import { InferencePhase } from '../../../constants/taskPhases';
import taskReducer, {
  receiveServerRecordTaskInfo,
} from '../../tasks/taskSlice';
import offlineRLReducer from '../offlineRLSlice';
import OfflineRLInferenceWorkspace from './OfflineRLInferenceWorkspace';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';

jest.mock('../../../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('react-hot-toast', () => {
  const toast = jest.fn();
  toast.error = jest.fn();
  toast.success = jest.fn();
  return { __esModule: true, default: toast };
});

jest.mock('../../../components/ImageGrid', () => function MockImageGrid({
  labels,
  preferConfiguredOrder,
  persistAssignment,
  readOnly,
  fillHeight,
  columnWeights,
  edgeToEdge,
  coverCell,
}) {
  return (
    <div
      data-testid="image-grid"
      data-prefer-configured-order={String(preferConfiguredOrder)}
      data-persist-assignment={String(persistAssignment)}
      data-read-only={String(readOnly)}
      data-fill-height={String(fillHeight)}
      data-column-weights={columnWeights?.join(',')}
      data-edge-to-edge={String(edgeToEdge)}
      data-cover-cell={String(coverCell)}
    >
      {labels.map((label) => <span key={label}>{label}</span>)}
    </div>
  );
});

jest.mock('../../../components/InferenceControlPanel', () => {
  return function MockInferenceControlPanel({ showRecordingControls, variant, policyEpoch }) {
    return (
      <div
        data-testid="inference-controls"
        data-variant={variant}
        data-policy-epoch={String(policyEpoch)}
      >
        Inference controls: {String(showRecordingControls)}
      </div>
    );
  };
});

jest.mock('../../../components/InferencePanel', () => function MockInferencePanel({
  title,
  modelLabel,
  showPolicyPath,
  showRecordingSettings,
  variant,
  settingsAside,
}) {
  return (
    <div data-testid="inference-settings" data-variant={variant}>
      {title} · {modelLabel} · policy path {String(showPolicyPath)} · recording settings{' '}
      {String(showRecordingSettings)}
      {settingsAside}
    </div>
  );
});

jest.mock('./OfflineRLWorkspaceStatusModal', () => {
  return function MockOfflineRLWorkspaceStatusModal({
    isOpen,
    onClose,
    workspaceMode,
    settingsContent,
  }) {
    if (!isOpen) return null;
    return (
      <div
        role="dialog"
        aria-label={workspaceMode === 'recording'
          ? 'Recording Workspace Status'
          : 'Inference Workspace Status'}
        data-testid="mock-workspace-status-modal"
        data-workspace-mode={workspaceMode}
      >
        <button type="button" onClick={onClose}>Back</button>
        {settingsContent}
      </div>
    );
  };
});

jest.mock('../../../components/FileBrowserModal', () => {
  return function MockFileBrowserModal({
    isOpen,
    onFileSelect,
    title,
    overlayZClass,
  }) {
    if (!isOpen) return null;
    return (
      <button
        type="button"
        onClick={() => onFileSelect({ full_path: '/workspace/selected' })}
        data-overlay-z-class={overlayZClass}
      >
        Choose {title}
      </button>
    );
  };
});

function renderWorkspace({
  inferenceMode = 'simulation',
  inferencePhase = InferencePhase.READY,
  isActive = true,
  workspaceMode = 'inference',
  policyEpoch = 0,
  workspaceStatusOpen = true,
  onCloseWorkspaceStatus = jest.fn(),
  robotType = 'ffw_sg2_rev1',
  sendRecordCommand = jest.fn().mockResolvedValue({ success: true }),
} = {}) {
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer, offlineRL: offlineRLReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        robotType,
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          inferenceMode,
          recordInferenceMode: false,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
      },
    },
  });

  const view = render(
    <Provider store={store}>
      <OfflineRLInferenceWorkspace
        isActive={isActive}
        workspaceMode={workspaceMode}
        policyEpoch={policyEpoch}
        workspaceStatusOpen={workspaceStatusOpen}
        onCloseWorkspaceStatus={onCloseWorkspaceStatus}
      />
    </Provider>
  );
  return {
    ...view,
    store,
    sendRecordCommand,
    onCloseWorkspaceStatus,
  };
}

describe('OfflineRLInferenceWorkspace', () => {
  test('reuses inference controls and shows cameras in rollout order', () => {
    const { container } = renderWorkspace();

    expect(screen.getByText('Inference controls: true')).toBeInTheDocument();
    expect(screen.getByText(/Inference Settings · Policy/)).toBeInTheDocument();
    expect(screen.getByTestId('inference-controls'))
      .toHaveAttribute('data-variant', 'offlineRL');
    expect(screen.getByTestId('inference-controls'))
      .toHaveAttribute('data-policy-epoch', '0');
    expect(screen.getByTestId('inference-settings'))
      .toHaveAttribute('data-variant', 'offlineRL');
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-prefer-configured-order',
      'true'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-persist-assignment',
      'false'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-read-only',
      'true'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-fill-height',
      'true'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-column-weights',
      '4,5,4'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-edge-to-edge',
      'true'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-cover-cell',
      'true'
    );
    expect(screen.getByTestId('offline-rl-camera-region')).toHaveClass(
      'flex-1',
      'min-h-[260px]'
    );
    expect(screen.getByTestId('offline-rl-camera-region')).not.toHaveClass('p-2');
    expect(screen.getByTestId('offline-rl-workspace-paths')).toBeInTheDocument();
    expect(screen.queryByTestId('offline-rl-settings-slot')).not.toBeInTheDocument();
    expect(screen.getByTestId('inference-settings').closest('[role="dialog"]'))
      .toBe(screen.getByRole('dialog', { name: 'Inference Workspace Status' }));
    const content = container.textContent;
    expect(content.indexOf('Left wrist')).toBeLessThan(content.indexOf('Head'));
    expect(content.indexOf('Head')).toBeLessThan(content.indexOf('Right wrist'));
    expect(content.indexOf('Right wrist')).toBeLessThan(
      content.indexOf('Inference controls: true')
    );
    expect(content.indexOf('Inference controls: true')).toBeLessThan(
      content.indexOf('Inference Settings')
    );
  });

  test('connects recording settings to one stable Replay Buffer target', async () => {
    const { store, sendRecordCommand } = renderWorkspace({
      workspaceMode: 'recording',
    });

    expect(screen.getByTestId('image-grid')).toBeInTheDocument();
    expect(screen.getByRole('group', {
      name: 'Recording controls',
    })).toBeInTheDocument();
    expect(screen.getByText('Ready to record')).toBeInTheDocument();
    expect(screen.getByLabelText('Current policy RL Epoch 0'))
      .toHaveTextContent('RL Epoch E0000');
    expect(screen.getByTestId('offline-rl-recording-settings'))
      .toHaveTextContent('Recording Settings');
    expect(screen.getByText('ffw_sg2_rev1')).toBeInTheDocument();
    expect(screen.getByLabelText('Task Instruction')).toHaveValue('');
    expect(screen.getByLabelText('Add ROBOTIS License')).not.toBeChecked();
    expect(screen.getByLabelText('Recording MCAP Dataset')).toHaveValue('');
    expect(screen.queryByTestId('inference-controls')).not.toBeInTheDocument();
    expect(screen.queryByTestId('inference-settings')).not.toBeInTheDocument();
    expect(screen.getByTestId('mock-workspace-status-modal'))
      .toHaveAttribute('data-workspace-mode', 'recording');
    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith('refresh_topics', {
        taskSource: 'inference',
      });
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_record',
        expect.objectContaining({
          taskSource: 'inference',
          taskInstruction: ['ACT_dataset'],
          includeRobotisLicense: false,
          recordingFolder: expect.stringMatching(
            /^\/workspace\/rosbag2\/Task_.+_inference_MCAP$/
          ),
        })
      );
    });
    const recordingFolder = store.getState().tasks.inferenceTaskInfo.recordingFolder;
    expect(recordingFolder).toMatch(
      /^\/workspace\/rosbag2\/Task_.+_inference_MCAP$/
    );
    expect(store.getState().offlineRL.replayBufferPath).toBe(recordingFolder);
    expect(store.getState().tasks.sharedTaskInfo.taskInstruction)
      .toEqual(['ACT_dataset']);
  });

  test('allows editing Task Instruction before recording and locks it during the episode', async () => {
    const { store, sendRecordCommand } = renderWorkspace({
      workspaceMode: 'recording',
    });
    const taskInstruction = screen.getByLabelText('Task Instruction');

    expect(taskInstruction).toBeEnabled();
    expect(taskInstruction).toHaveAttribute(
      'title',
      'Editable before recording starts'
    );

    fireEvent.change(taskInstruction, {
      target: { value: 'Pick up the jelly bag' },
    });

    expect(taskInstruction).toHaveValue('Pick up the jelly bag');
    expect(store.getState().tasks.sharedTaskInfo.taskInstruction)
      .toEqual(['Pick up the jelly bag']);
    expect(store.getState().tasks.inferenceTaskInfoSync.dirty).toBe(true);

    store.dispatch(receiveServerRecordTaskInfo({
      taskType: 'inference',
      taskInstruction: ['Old server instruction'],
    }));
    expect(taskInstruction).toHaveValue('Pick up the jelly bag');

    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }));

    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_record',
        expect.objectContaining({
          taskInstruction: ['Pick up the jelly bag'],
          taskSource: 'inference',
        })
      );
      expect(taskInstruction).toBeDisabled();
    });
    expect(taskInstruction).toHaveAttribute(
      'title',
      'Locked while recording or saving an episode'
    );
  });

  test('records ordered subtasks and labels the completed episode once', async () => {
    const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
    renderWorkspace({
      workspaceMode: 'recording',
      sendRecordCommand,
    });

    const record = screen.getByRole('button', { name: 'Start recording' });
    fireEvent.change(screen.getByLabelText('Count'), {
      target: { value: '3' },
    });
    expect(record).toBeDisabled();

    ['Approach', 'Grasp', 'Retreat'].forEach((instruction, index) => {
      fireEvent.change(screen.getByLabelText(
        `Subtask ${index + 1} instruction`
      ), { target: { value: instruction } });
    });
    expect(record).toBeEnabled();

    fireEvent.click(record);
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_segment',
        expect.objectContaining({
          segmentIndex: 0,
          subtaskInstruction: ['Approach', 'Grasp', 'Retreat'],
          taskSource: 'inference',
        })
      );
    });
    expect(screen.getByRole('button', {
      name: 'Save recording as Success',
    })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: /Save & Next/i }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_segment',
        expect.objectContaining({ segmentIndex: 0, taskSource: 'inference' })
      );
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_segment',
        expect.objectContaining({ segmentIndex: 1, taskSource: 'inference' })
      );
      expect(screen.getByText('Subtask 2 / 3')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /Save & Next/i }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_segment',
        expect.objectContaining({ segmentIndex: 1, taskSource: 'inference' })
      );
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'start_segment',
        expect.objectContaining({ segmentIndex: 2, taskSource: 'inference' })
      );
      expect(screen.getByText('Finish with Success or Fail'))
        .toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', {
      name: 'Save recording as Success',
    }));
    await waitFor(() => {
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'stop_inference_record',
        expect.objectContaining({
          episodeOutcome: EpisodeOutcome.SUCCESS,
          subtaskInstruction: ['Approach', 'Grasp', 'Retreat'],
          taskSource: 'inference',
        })
      );
    });
  });

  test('connects Model to inference and keeps the MCAP source separate from Step 3', () => {
    const { store } = renderWorkspace();
    const model = screen.getByLabelText('Model');
    const dataset = screen.getByLabelText('MCAP Dataset');

    fireEvent.change(model, { target: { value: '/workspace/model/act' } });
    fireEvent.change(dataset, {
      target: { value: '/workspace/rosbag2/Task_test_inference_MCAP' },
    });

    expect(store.getState().tasks.inferenceTaskInfo.policyPath)
      .toBe('/workspace/model/act');
    expect(store.getState().offlineRL.replayBufferPath)
      .toBe('/workspace/rosbag2/Task_test_inference_MCAP');
    expect(store.getState().offlineRL.datasetPath).toBe('');
    expect(dataset).toHaveValue('/workspace/rosbag2/Task_test_inference_MCAP');
    expect(screen.queryByLabelText('Checkpoint')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Browse Model' }));
    const modelBrowser = screen.getByRole('button', {
      name: 'Choose Select inference model',
    });
    expect(modelBrowser).toHaveAttribute('data-overlay-z-class', 'z-[80]');
    fireEvent.click(modelBrowser);
    expect(store.getState().tasks.inferenceTaskInfo.policyPath)
      .toBe('/workspace/selected');

    fireEvent.click(screen.getByRole('button', { name: 'Browse MCAP Dataset' }));
    fireEvent.click(screen.getByRole('button', {
      name: 'Choose Select source MCAP dataset',
    }));
    expect(store.getState().offlineRL.replayBufferPath)
      .toBe('/workspace/selected');
    expect(store.getState().offlineRL.datasetPath).toBe('');
  });

  test('enables the backend inference-recording contract automatically for robot deploy', async () => {
    const { store } = renderWorkspace({ inferenceMode: 'robot' });

    await waitFor(() => {
      expect(store.getState().tasks.inferenceTaskInfo.recordInferenceMode)
        .toBe(true);
    });
  });

  test('locks workspace paths while inference is running', () => {
    renderWorkspace({ inferencePhase: InferencePhase.INFERENCING });

    expect(screen.getByLabelText('Model')).toBeDisabled();
    expect(screen.getByLabelText('MCAP Dataset')).toBeDisabled();
    expect(screen.queryByLabelText('Checkpoint')).not.toBeInTheDocument();
  });

  test('moves settings into Workspace Status while cameras fill the environment canvas', () => {
    renderWorkspace();

    const cameraRegion = screen.getByTestId('offline-rl-camera-region');
    const recordingDock = screen.getByTestId('offline-rl-recording-dock');

    expect(screen.getByTestId('inference-settings')).toBeInTheDocument();
    expect(screen.getByTestId('offline-rl-workspace-paths')).toBeInTheDocument();
    expect(screen.queryByTestId('offline-rl-settings-slot')).not.toBeInTheDocument();
    expect(screen.queryByTestId('offline-rl-inference-settings-panel'))
      .not.toBeInTheDocument();
    expect(cameraRegion).toHaveClass('flex-1', 'min-h-[260px]');
    expect(recordingDock).toHaveClass('mx-auto', 'w-full');
    expect(recordingDock.className).toMatch(/(?:^|\s)max-w-/);
    expect(within(recordingDock).getByTestId('inference-controls'))
      .toBeInTheDocument();
  });

  test('closes Workspace Status through the workspace-owned modal', () => {
    const onCloseWorkspaceStatus = jest.fn();
    renderWorkspace({ onCloseWorkspaceStatus });

    fireEvent.click(screen.getByRole('button', { name: 'Back' }));

    expect(onCloseWorkspaceStatus).toHaveBeenCalledTimes(1);
  });
});
