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
  imageFit,
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
      data-image-fit={imageFit}
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
        onClick={() => onFileSelect({
          full_path: title === 'Select inference data folder'
            ? '/workspace/rosbag2/Task_700009_OMX_Insert_F2_Intern_MCAP'
            : '/workspace/selected',
        })}
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
  recordingFolder = null,
  sendRecordCommand = jest.fn().mockResolvedValue({ success: true }),
  getDatasetInfo = jest.fn().mockResolvedValue({ success: false }),
} = {}) {
  useRosServiceCaller.mockReturnValue({ sendRecordCommand, getDatasetInfo });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const initialRecordingFolder = recordingFolder ?? (
    workspaceMode === 'recording'
      ? '/workspace/rosbag2/Task_playground_recording_MCAP'
      : ''
  );
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
          recordingFolder: initialRecordingFolder,
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
    getDatasetInfo,
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
      '9,16,9'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-edge-to-edge',
      'true'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-cover-cell',
      'undefined'
    );
    expect(screen.getByTestId('image-grid')).toHaveAttribute(
      'data-image-fit',
      'contain'
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
    await waitFor(() => {
      expect(screen.getByLabelText('Inference Dataset Name'))
        .toHaveValue('playground_recording');
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'prepare_session',
        expect.objectContaining({
          taskSource: 'inference',
          recordingFolder:
            '/workspace/rosbag2/Task_playground_recording_MCAP',
        })
      );
    });
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
          recordingFolder:
            '/workspace/rosbag2/Task_playground_recording_MCAP',
        })
      );
    });
    const recordingFolder = store.getState().tasks.inferenceTaskInfo.recordingFolder;
    expect(recordingFolder).toBe(
      '/workspace/rosbag2/Task_playground_recording_MCAP'
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

  test('accepts a new dataset name and prepares a standard Record folder', async () => {
    const { sendRecordCommand } = renderWorkspace({
      workspaceMode: 'recording',
    });

    fireEvent.click(screen.getByRole('button', { name: /New Dataset/i }));
    const datasetInput = screen.getByLabelText('Inference Dataset Name');
    fireEvent.change(datasetInput, { target: { value: 'F2 조립 작업' } });
    fireEvent.keyDown(datasetInput, { key: 'Enter' });

    await waitFor(() => {
      expect(datasetInput).toHaveValue('F2_조립_작업');
      expect(sendRecordCommand).toHaveBeenCalledWith(
        'prepare_session',
        expect.objectContaining({
          taskSource: 'inference',
          recordingFolder: '/workspace/rosbag2/Task_F2_조립_작업_MCAP',
        })
      );
    });
  });

  test('loads the subtask plan from an existing standard Record dataset', async () => {
    const getDatasetInfo = jest.fn().mockResolvedValue({
      success: true,
      dataset_info: {
        task_instruction: 'Insert the OMX components',
        subtask_instruction: ['Approach', 'Insert', 'Release'],
        episode_count: 98,
        success_episode_indices: [0, 1, 99],
        failure_episode_indices: [],
        unlabeled_episode_indices: [],
      },
    });
    const { store } = renderWorkspace({
      workspaceMode: 'recording',
      getDatasetInfo,
    });

    fireEvent.click(screen.getByRole('button', {
      name: 'Browse Inference Data Folder',
    }));
    fireEvent.click(screen.getByRole('button', {
      name: 'Choose Select inference data folder',
    }));

    const folder =
      '/workspace/rosbag2/Task_700009_OMX_Insert_F2_Intern_MCAP';
    await waitFor(() => {
      expect(getDatasetInfo).toHaveBeenCalledWith(folder);
      expect(screen.getByLabelText('Count')).toHaveValue(3);
      expect(screen.getByLabelText('Subtask 1 instruction'))
        .toHaveValue('Approach');
      expect(store.getState().tasks.sharedTaskInfo.taskInstruction)
        .toEqual(['Insert the OMX components']);
      expect(screen.getByLabelText('Saved recording episodes'))
        .toHaveTextContent('100');
    });
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

  test('connects Model and a name-only inference recording target', () => {
    const { store } = renderWorkspace();
    const model = screen.getByLabelText('Model');
    const dataset = screen.getByLabelText('Inference Dataset Name');

    fireEvent.change(model, { target: { value: '/workspace/model/act' } });
    fireEvent.change(dataset, {
      target: { value: 'test_inference' },
    });

    expect(store.getState().tasks.inferenceTaskInfo.policyPath)
      .toBe('/workspace/model/act');
    expect(store.getState().offlineRL.replayBufferPath)
      .toBe('/workspace/rosbag2/Task_test_inference_MCAP');
    expect(store.getState().tasks.inferenceTaskInfo.recordingFolder)
      .toBe('/workspace/rosbag2/Task_test_inference_MCAP');
    expect(store.getState().offlineRL.datasetPath).toBe('');
    expect(dataset).toHaveValue('test_inference');
    expect(screen.queryByLabelText('Checkpoint')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Browse Model' }));
    const modelBrowser = screen.getByRole('button', {
      name: 'Choose Select inference model',
    });
    expect(modelBrowser).toHaveAttribute('data-overlay-z-class', 'z-[80]');
    fireEvent.click(modelBrowser);
    expect(store.getState().tasks.inferenceTaskInfo.policyPath)
      .toBe('/workspace/selected');

    fireEvent.click(screen.getByRole('button', {
      name: 'Browse Inference Data Folder',
    }));
    fireEvent.click(screen.getByRole('button', {
      name: 'Choose Select inference data folder',
    }));
    expect(store.getState().offlineRL.replayBufferPath)
      .toBe('/workspace/rosbag2/Task_700009_OMX_Insert_F2_Intern_MCAP');
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
    expect(screen.getByLabelText('Inference Dataset Name')).toBeDisabled();
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
