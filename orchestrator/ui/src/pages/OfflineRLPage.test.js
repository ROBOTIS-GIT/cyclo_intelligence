import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { Provider } from 'react-redux';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import taskReducer, {
  setRecordStatus,
  setInferenceStatus,
  setInferenceTaskInfo,
} from '../features/tasks/taskSlice';
import OfflineRLPage from './OfflineRLPage';
import { OFFLINE_RL_LINEAGE_STORAGE_KEY } from '../utils/offlineRlLineageState';

jest.mock('react-hot-toast', () => ({
  success: jest.fn(),
}));

jest.mock('../features/offlineRL/components/OfflineRLInferenceWorkspace', () => {
  return function MockOfflineRLInferenceWorkspace({ isActive, workspaceMode, policyEpoch }) {
    return (
      <div
        data-testid="offline-rl-workspace"
        data-active={String(isActive)}
        data-workspace-mode={workspaceMode}
        data-policy-epoch={String(policyEpoch)}
      >
        <span>Left wrist</span>
        <span>Head</span>
        <span>Right wrist</span>
        <span>Inference Settings</span>
      </div>
    );
  };
});

jest.mock('../features/offlineRL/components/OfflineRLReplayBuffer', () => {
  return function MockOfflineRLReplayBuffer() {
    return <div data-testid="replay-buffer-stack">Replay Buffer stack</div>;
  };
});

jest.mock('../features/offlineRL/components/OfflineRLDatasetConversion', () => {
  return function MockOfflineRLDatasetConversion() {
    return <div data-testid="dataset-conversion">Conversion controls</div>;
  };
});

jest.mock('../features/offlineRL/components/OfflineRLLeRobotDataset', () => {
  return function MockOfflineRLLeRobotDataset() {
    return <div data-testid="lerobot-dataset-stack">LeRobot Dataset stack</div>;
  };
});

jest.mock('../features/offlineRL/components/OfflineRLTrainingSection', () => {
  return function MockOfflineRLTrainingSection({
    isActive,
    variant,
    inferencePhase,
    onDeploymentStateChange,
    currentPolicyEpoch,
    forceFreshLineage,
    onFreshLineageConsumed,
    onRunningChange,
  }) {
    return (
      <div
        data-testid="workflow-training-controller"
        data-active={String(isActive)}
        data-variant={variant}
        data-inference-phase={String(inferencePhase)}
        data-policy-epoch={String(currentPolicyEpoch)}
        data-force-fresh-lineage={String(forceFreshLineage)}
      >
        <div role="group" aria-label="Policy model">
          <button type="button" aria-pressed="true">ACT</button>
          <button type="button" aria-pressed="false" disabled>GR00T</button>
          <button type="button" aria-pressed="false" disabled>Pi0.5</button>
        </div>
        <div role="group" aria-label="RL algorithm">
          <button type="button" aria-pressed="true">TD3</button>
          <button type="button" aria-pressed="false" disabled>SAC</button>
        </div>
        <div data-testid="offline-rl-training-architecture">ACT architecture</div>
        <div
          className="mt-auto shrink-0"
          data-testid="offline-rl-training-footer"
        >
          <span>Training progress</span>
          <span>Training action</span>
          <button type="button" disabled>Start Training</button>
          <button type="button" onClick={() => onRunningChange?.(false)}>
            Mark training status ready
          </button>
          <button type="button" onClick={() => onFreshLineageConsumed?.()}>
            Consume fresh lineage
          </button>
          <button
            type="button"
            onClick={() => onDeploymentStateChange?.({
              ready: true,
              modelPath: '/workspace/model/round1/pretrained_model',
              serviceType: 'lerobot',
              policyType: 'act',
              rlEpoch: 1,
            })}
          >
            Mark training complete
          </button>
          <button
            type="button"
            onClick={() => onDeploymentStateChange?.({
              ready: true,
              modelPath: '/workspace/model/multi_task_dit/ppo/pretrained_model',
              serviceType: 'lerobot',
              policyType: 'multi_task_dit',
              rlEpoch: currentPolicyEpoch,
            })}
          >
            Mark MultiTaskDiT training complete
          </button>
        </div>
      </div>
    );
  };
});

jest.mock('../features/offlineRL/components/OfflineRLWorkspaceStatusModal', () => {
  return function MockOfflineRLWorkspaceStatusModal({ isOpen, onClose }) {
    if (!isOpen) return null;
    return (
      <div role="dialog" aria-label="Inference Workspace Status">
        <button type="button" onClick={onClose}>Back</button>
      </div>
    );
  };
});

const renderPage = (props = {}) => {
  const testStore = configureStore({ reducer: { tasks: taskReducer } });
  const view = render(
    <Provider store={testStore}>
      <OfflineRLPage {...props} />
    </Provider>
  );
  return { ...view, testStore };
};

beforeEach(() => {
  window.sessionStorage.clear();
});

test('renders the TD3 and ACT workflow in pipeline order', () => {
  const { container } = renderPage();

  expect(screen.queryByRole('heading', {
    name: 'Offline Reinforcement Learning',
  })).not.toBeInTheDocument();
  expect(screen.getByRole('heading', {
    name: 'Inference Workspace',
  })).toBeInTheDocument();
  expect(screen.getByText('Replay Buffer')).toBeInTheDocument();
  expect(screen.getByText('Dataset Conversion')).toBeInTheDocument();
  expect(screen.getByText('LeRobot Dataset')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Training' })).toBeInTheDocument();
  expect(screen.getByLabelText('Training policy RL Epoch 0 to 1'))
    .toHaveTextContent('RL Epoch E0000 → E0001');
  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-policy-epoch', '0');
  expect(screen.getByTestId('replay-buffer-stack')).toBeInTheDocument();
  expect(screen.getByTestId('dataset-conversion')).toBeInTheDocument();
  expect(screen.getByTestId('lerobot-dataset-stack')).toBeInTheDocument();

  const content = container.textContent;
  expect(content.indexOf('Replay Buffer')).toBeLessThan(
    content.indexOf('Dataset Conversion')
  );
  expect(content.indexOf('Dataset Conversion')).toBeLessThan(
    content.indexOf('LeRobot Dataset')
  );
  expect(content.indexOf('LeRobot Dataset')).toBeLessThan(
    content.indexOf('Training')
  );
  expect(screen.getByTestId('offline-rl-workflow-grid')).toHaveClass(
    'xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]'
  );
  expect(screen.getByTestId('offline-rl-workflow-steps')).toHaveClass(
    'xl:grid-rows-[minmax(260px,2fr)_minmax(390px,3fr)_auto]'
  );
  expect(screen.getByTestId('offline-rl-dataset-pipeline')).toHaveClass(
    'min-h-[260px]'
  );
});

test('switches the environment between inference and recording workspaces', () => {
  renderPage();

  const modeGroup = screen.getByRole('group', { name: 'Workspace mode' });
  const inferenceButton = within(modeGroup).getByRole('button', {
    name: 'Inference',
  });
  const recordingButton = within(modeGroup).getByRole('button', {
    name: 'Recording',
  });

  expect(inferenceButton).toHaveAttribute('aria-pressed', 'true');
  expect(recordingButton).toHaveAttribute('aria-pressed', 'false');
  expect(inferenceButton).toHaveClass('bg-[#69866f]');
  expect(recordingButton).not.toHaveClass('bg-[#a86b68]');
  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-workspace-mode', 'inference');

  fireEvent.click(recordingButton);

  expect(screen.getByRole('heading', {
    name: 'Recording Workspace',
  })).toBeInTheDocument();
  expect(inferenceButton).toHaveAttribute('aria-pressed', 'false');
  expect(recordingButton).toHaveAttribute('aria-pressed', 'true');
  expect(recordingButton).toHaveClass('bg-[#a86b68]');
  expect(inferenceButton).not.toHaveClass('bg-[#69866f]');
  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-workspace-mode', 'recording');

  fireEvent.click(inferenceButton);

  expect(screen.getByRole('heading', {
    name: 'Inference Workspace',
  })).toBeInTheDocument();
  expect(inferenceButton).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-workspace-mode', 'inference');
});

test('locks workspace switching while inference or recording is active', () => {
  const { testStore } = renderPage();
  const recordingButton = screen.getByRole('button', { name: 'Recording' });

  act(() => {
    testStore.dispatch(setInferenceStatus({
      inferencePhase: InferencePhase.INFERENCING,
    }));
  });
  expect(recordingButton).toBeDisabled();
  expect(screen.getByTestId('workflow-training-controller'))
    .toHaveAttribute('data-inference-phase', String(InferencePhase.INFERENCING));

  act(() => {
    testStore.dispatch(setInferenceStatus({
      inferencePhase: InferencePhase.READY,
    }));
    testStore.dispatch(setRecordStatus({
      recordPhase: RecordPhase.RECORDING,
      taskType: 'record',
    }));
  });
  expect(recordingButton).toBeDisabled();
  expect(screen.getByRole('heading', {
    name: 'Inference Workspace',
  })).toBeInTheDocument();
});

test('keeps ACT and TD3 active while future model and algorithm options are disabled', () => {
  renderPage();

  expect(screen.getAllByText('TD3').length).toBeGreaterThan(0);
  expect(screen.getAllByText('ACT').length).toBeGreaterThan(0);
  expect(screen.getByRole('button', { name: 'GR00T' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Pi0.5' })).toBeDisabled();
  expect(screen.queryByText('RLT')).not.toBeInTheDocument();

  const trainingSection = screen.getByRole('heading', {
    name: 'Training',
  }).closest('section');
  const algorithmGroup = within(trainingSection).getByRole('group', {
    name: 'RL algorithm',
  });
  expect(within(algorithmGroup).getByRole('button', { name: 'TD3' }))
    .toHaveAttribute('aria-pressed', 'true');
  expect(within(algorithmGroup).getByRole('button', { name: 'SAC' }))
    .toBeDisabled();
  const policyGroup = within(trainingSection).getByRole('group', {
    name: 'Policy model',
  });
  expect(within(policyGroup).getByRole('button', { name: 'ACT' }))
    .toHaveAttribute('aria-pressed', 'true');
});

test('keeps architecture above one fixed training footer', () => {
  const { container } = renderPage();
  const architecture = screen.getByTestId('offline-rl-training-architecture');
  const footer = screen.getByTestId('offline-rl-training-footer');

  expect(footer).toHaveClass('mt-auto', 'shrink-0');
  expect(container.textContent.indexOf(architecture.textContent))
    .toBeLessThan(container.textContent.indexOf('Training progress'));
  expect(within(footer).getByText('Training progress')).toBeInTheDocument();
  expect(within(footer).getByText('Training action')).toBeInTheDocument();
});

test('shows three camera slots without enabling unfinished actions', () => {
  const { container } = renderPage();

  expect(screen.getByText('Left wrist')).toBeInTheDocument();
  expect(screen.getByText('Head')).toBeInTheDocument();
  expect(screen.getByText('Right wrist')).toBeInTheDocument();
  const content = container.textContent;
  expect(content.indexOf('Left wrist')).toBeLessThan(content.indexOf('Head'));
  expect(content.indexOf('Head')).toBeLessThan(content.indexOf('Right wrist'));
  expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Deploy Policy' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Discard Policy' })).toBeDisabled();
});

test('opens workspace status from the Environment icon and closes with Back', () => {
  renderPage();

  const statusButton = screen.getByRole('button', {
    name: 'Open inference workspace status',
  });
  expect(statusButton).toHaveAttribute(
    'title',
    'Open inference workspace status'
  );

  fireEvent.click(statusButton);
  expect(screen.getByRole('dialog', {
    name: 'Inference Workspace Status',
  })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Back' }));
  expect(screen.queryByRole('dialog', {
    name: 'Inference Workspace Status',
  })).not.toBeInTheDocument();
});

test('deploys a completed model into the shared inference Model path', () => {
  const { testStore } = renderPage();

  expect(screen.getByText('Policy Deploy')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Mark training complete' }));
  const deployButton = screen.getByRole('button', { name: 'Deploy Policy' });
  expect(deployButton).not.toBeDisabled();

  fireEvent.click(deployButton);

  expect(testStore.getState().tasks.inferenceTaskInfo.policyPath)
    .toBe('/workspace/model/round1/pretrained_model');
  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-policy-epoch', '1');
  expect(screen.getByLabelText('Training policy RL Epoch 1 to 2'))
    .toHaveTextContent('RL Epoch E0001 → E0002');
  expect(testStore.getState().tasks.inferenceTaskInfoSync).toMatchObject({
    dirty: true,
    syncStatus: 'pending',
  });
  expect(screen.getByRole('button', { name: 'Deploy Policy' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Discard Policy' })).not.toBeDisabled();
});

test('starts a new RL lineage at Epoch 0 without deleting or clearing selected paths', () => {
  const { testStore } = renderPage();
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/current_policy/pretrained_model',
    }));
  });
  fireEvent.click(screen.getByRole('button', { name: 'Mark training status ready' }));
  const newLineage = screen.getByRole('button', { name: 'New RL Lineage' });
  expect(newLineage).not.toBeDisabled();
  fireEvent.click(newLineage);

  expect(screen.getByTestId('workflow-training-controller'))
    .toHaveAttribute('data-force-fresh-lineage', 'true');
  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-policy-epoch', '0');
  expect(testStore.getState().tasks.inferenceTaskInfo.policyPath)
    .toBe('/workspace/model/current_policy/pretrained_model');
  expect(JSON.parse(
    window.sessionStorage.getItem(OFFLINE_RL_LINEAGE_STORAGE_KEY)
  )).toMatchObject({
    policyEpoch: 0,
    policyPath: '/workspace/model/current_policy/pretrained_model',
    forceFresh: true,
  });

  confirmSpy.mockRestore();
});

test('discards only the active deployment and restores its previous inference path', () => {
  const { testStore } = renderPage();

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/original/pretrained_model',
    }));
  });
  fireEvent.click(screen.getByRole('button', { name: 'Mark training complete' }));
  fireEvent.click(screen.getByRole('button', { name: 'Deploy Policy' }));
  fireEvent.click(screen.getByRole('button', { name: 'Discard Policy' }));

  expect(testStore.getState().tasks.inferenceTaskInfo.policyPath)
    .toBe('/workspace/model/original/pretrained_model');
  expect(testStore.getState().tasks.inferenceTaskInfoSync).toMatchObject({
    dirty: true,
    syncStatus: 'pending',
  });
  expect(screen.getByRole('button', { name: 'Discard Policy' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Deploy Policy' })).not.toBeDisabled();
});

test('deploys MultiTaskDiT as lerobot:multi_task_dit and restores all inference fields', () => {
  const { testStore } = renderPage();

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/groot/original',
      serviceType: 'groot',
      policyType: 'groot',
    }));
  });
  fireEvent.click(screen.getByRole('button', {
    name: 'Mark MultiTaskDiT training complete',
  }));
  fireEvent.click(screen.getByRole('button', { name: 'Deploy Policy' }));

  expect(testStore.getState().tasks.inferenceTaskInfo).toMatchObject({
    policyPath: '/workspace/model/multi_task_dit/ppo/pretrained_model',
    serviceType: 'lerobot',
    policyType: 'multi_task_dit',
  });
  expect(screen.getByRole('button', { name: 'Deploy Policy' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Discard Policy' })).not.toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Discard Policy' }));

  expect(testStore.getState().tasks.inferenceTaskInfo).toMatchObject({
    policyPath: '/workspace/model/groot/original',
    serviceType: 'groot',
    policyType: 'groot',
  });
  expect(testStore.getState().tasks.inferenceTaskInfoSync).toMatchObject({
    dirty: true,
    syncStatus: 'pending',
  });
  expect(screen.getByRole('button', { name: 'Discard Policy' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Deploy Policy' })).not.toBeDisabled();
});

test('does not let discard overwrite a manually selected policy path', () => {
  const { testStore } = renderPage();

  fireEvent.click(screen.getByRole('button', { name: 'Mark training complete' }));
  fireEvent.click(screen.getByRole('button', { name: 'Deploy Policy' }));
  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/manual/pretrained_model',
    }));
  });

  expect(screen.getByRole('button', { name: 'Discard Policy' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Discard Policy' }));
  expect(testStore.getState().tasks.inferenceTaskInfo.policyPath)
    .toBe('/workspace/model/manual/pretrained_model');
});

test('keeps deployment locked while inference is running', () => {
  const { testStore } = renderPage();

  fireEvent.click(screen.getByRole('button', { name: 'Mark training complete' }));
  expect(screen.getByRole('button', { name: 'Deploy Policy' })).not.toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Deploy Policy' }));
  expect(screen.getByRole('button', { name: 'Discard Policy' })).not.toBeDisabled();

  act(() => {
    testStore.dispatch(setInferenceStatus({
      inferencePhase: InferencePhase.INFERENCING,
    }));
  });

  expect(screen.getByRole('button', { name: 'Deploy Policy' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Discard Policy' })).toBeDisabled();
});

test('reflects inactive page state', () => {
  renderPage({ isActive: false });

  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-active', 'false');
});

test('collapses the workflow panel and lets the inference workspace use the full grid width', () => {
  renderPage();

  const workflowGrid = screen.getByTestId('offline-rl-workflow-grid');
  const workflowSteps = screen.getByTestId('offline-rl-workflow-steps');
  expect(workflowGrid).toHaveClass(
    'xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]'
  );
  expect(screen.getByTestId('offline-rl-workflow-steps')).toBeInTheDocument();
  expect(screen.getByTestId('replay-buffer-stack')).toBeInTheDocument();
  expect(screen.getByTestId('dataset-conversion')).toBeInTheDocument();
  expect(screen.getByTestId('lerobot-dataset-stack')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', {
    name: 'Hide workflow panel',
  }));

  expect(workflowSteps).toHaveClass('hidden');
  expect(workflowSteps).toHaveAttribute('aria-hidden', 'true');
  expect(screen.getByTestId('replay-buffer-stack')).toBeInTheDocument();
  expect(screen.getByTestId('dataset-conversion')).toBeInTheDocument();
  expect(screen.getByTestId('lerobot-dataset-stack')).toBeInTheDocument();
  expect(workflowGrid).toHaveClass('xl:grid-cols-[minmax(0,1fr)]');
  expect(workflowGrid).not.toHaveClass(
    'xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]'
  );
  expect(screen.getByRole('button', {
    name: 'Show workflow panel',
  })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', {
    name: 'Show workflow panel',
  }));

  expect(screen.getByRole('button', {
    name: 'Hide workflow panel',
  })).toBeInTheDocument();
  expect(screen.getByTestId('offline-rl-workflow-steps')).toBeInTheDocument();
  expect(workflowSteps).toHaveClass('grid');
  expect(workflowSteps).toHaveAttribute('aria-hidden', 'false');
  expect(screen.getByTestId('replay-buffer-stack')).toBeInTheDocument();
  expect(screen.getByTestId('dataset-conversion')).toBeInTheDocument();
  expect(screen.getByTestId('lerobot-dataset-stack')).toBeInTheDocument();
  expect(workflowGrid).toHaveClass(
    'xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]'
  );
});
