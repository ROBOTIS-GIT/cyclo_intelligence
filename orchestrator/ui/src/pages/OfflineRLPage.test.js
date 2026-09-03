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
  return function MockOfflineRLInferenceWorkspace({
    isActive,
    workspaceMode,
    policyEpoch,
    workspaceStatusOpen,
    onCloseWorkspaceStatus,
  }) {
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
        {workspaceStatusOpen && (
          <div role="dialog" aria-label="Inference Workspace Status">
            <button type="button" onClick={onCloseWorkspaceStatus}>Back</button>
            <span>Inference Settings</span>
          </div>
        )}
      </div>
    );
  };
});

jest.mock('../features/offlineRL/components/OfflineRLReplayBuffer', () => {
  return function MockOfflineRLReplayBuffer({ isActive }) {
    return (
      <div data-testid="replay-buffer-stack" data-active={String(isActive)}>
        Replay Buffer stack
      </div>
    );
  };
});

jest.mock('../features/offlineRL/components/OfflineRLDatasetConversion', () => {
  return function MockOfflineRLDatasetConversion({ isActive }) {
    return (
      <div data-testid="dataset-conversion" data-active={String(isActive)}>
        Conversion controls
      </div>
    );
  };
});

jest.mock('../features/offlineRL/components/OfflineRLLeRobotDataset', () => {
  return function MockOfflineRLLeRobotDataset({ isActive }) {
    return (
      <div data-testid="lerobot-dataset-stack" data-active={String(isActive)}>
        LeRobot Dataset stack
      </div>
    );
  };
});

jest.mock('../features/offlineRL/components/OfflineRLTrainingSection', () => {
  return function MockOfflineRLTrainingSection({
    isActive,
    variant,
    inferencePhase,
    onDeploymentStateChange,
    onTrainingMethodStateChange,
    currentPolicyEpoch,
    forceFreshLineage,
    onFreshLineageConsumed,
    onRunningChange,
    onCompactLayoutChange,
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
          <button type="button" onClick={() => onCompactLayoutChange?.(true)}>
            Compact training layout
          </button>
          <button type="button" onClick={() => onCompactLayoutChange?.(false)}>
            Restore training layout
          </button>
          <button
            type="button"
            onClick={() => onTrainingMethodStateChange?.('reinforcement')}
          >
            Select RL
          </button>
          <button
            type="button"
            onClick={() => onTrainingMethodStateChange?.('imitation')}
          >
            Select IL
          </button>
          <button
            type="button"
            onClick={() => onTrainingMethodStateChange?.('critic')}
          >
            Select Critic
          </button>
          <button
            type="button"
            onClick={() => onDeploymentStateChange?.({
              ready: true,
              modelPath: '/workspace/model/round1/pretrained_model',
              serviceType: 'lerobot',
              policyType: 'act',
              rlEpoch: 1,
              lineageMode: 'advance',
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
              rlEpoch: currentPolicyEpoch + 1,
              lineageMode: 'advance',
            })}
          >
            Mark MultiTaskDiT training complete
          </button>
          <button
            type="button"
            onClick={() => onDeploymentStateChange?.({
              ready: true,
              modelPath: '/workspace/model/imitation/act/pretrained_model',
              serviceType: 'lerobot',
              policyType: 'act',
              rlEpoch: 0,
              lineageMode: 'new',
            })}
          >
            Mark IL training complete
          </button>
          <button
            type="button"
            onClick={() => onDeploymentStateChange?.({
              ready: true,
              artifactKind: 'rlt_bundle',
              modelPath: '/workspace/model/groot/showroom',
              rltBundlePath: '/workspace/checkpoint/rlt/stage2/round_0001',
              serviceType: 'groot',
              policyType: 'n17',
              rlEpoch: currentPolicyEpoch + 1,
              lineageMode: 'advance',
            })}
          >
            Mark RLT training complete
          </button>
        </div>
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

const openTrainingDrawer = () => {
  const trigger = screen.getByRole('button', { name: 'Training' });
  if (trigger.getAttribute('aria-expanded') !== 'true') {
    fireEvent.click(trigger);
  }
  return {
    trigger,
    drawer: screen.getByTestId('offline-rl-training-drawer'),
  };
};

beforeEach(() => {
  window.sessionStorage.clear();
});

test('renders the TD3 and ACT workflow in pipeline order', () => {
  renderPage();
  const frameworkNavigation = screen.getByRole('navigation', {
    name: 'Playground sections',
  });
  const workflowSteps = screen.getByTestId('offline-rl-workflow-steps');

  expect(screen.queryByRole('heading', {
    name: 'Offline Reinforcement Learning',
  })).not.toBeInTheDocument();
  expect(screen.getByRole('heading', {
    name: 'Inference Workspace',
  })).toBeInTheDocument();
  expect(within(frameworkNavigation).getByRole('button', {
    name: 'Environment',
  })).toHaveAttribute('aria-current', 'page');
  expect(workflowSteps).toHaveAttribute('data-panel-state', 'environment');
  openTrainingDrawer();
  expect(within(workflowSteps).getByText('Replay Buffer')).toBeInTheDocument();
  expect(within(workflowSteps).getByText('Data Collection')).toBeInTheDocument();
  expect(within(workflowSteps).getByText('Dataset Conversion')).toBeInTheDocument();
  expect(within(workflowSteps).getByText('LeRobot Dataset')).toBeInTheDocument();
  expect(within(screen.getByTestId('offline-rl-training-content')).getByRole(
    'heading', { name: 'Training' }
  )).toBeInTheDocument();
  expect(screen.getByLabelText('Training policy RL Epoch 0 to 1'))
    .toHaveTextContent('RL Epoch E0000 → E0001');
  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-policy-epoch', '0');
  expect(screen.getByTestId('replay-buffer-stack')).toBeInTheDocument();
  expect(screen.getByTestId('dataset-conversion')).toBeInTheDocument();
  expect(screen.getByTestId('lerobot-dataset-stack')).toBeInTheDocument();
  expect(screen.getByTestId('workflow-training-controller')).toBeInTheDocument();
  expect(screen.getByTestId('offline-rl-deployment')).toBeInTheDocument();

  const content = workflowSteps.textContent;
  expect(content.indexOf('Data Collection')).toBeLessThan(
    content.indexOf('Dataset Conversion')
  );
  expect(content.indexOf('Dataset Conversion')).toBeLessThan(
    content.indexOf('LeRobot Dataset')
  );
  expect(content.indexOf('LeRobot Dataset')).toBeLessThan(
    content.indexOf('Training')
  );
  expect(screen.getByTestId('offline-rl-workflow-grid'))
    .toHaveAttribute('data-layout', 'environment-canvas');
  expect(screen.getByTestId('offline-rl-main')).toHaveClass('overflow-hidden');
  expect(screen.getByTestId('offline-rl-workflow-grid'))
    .toHaveClass('h-full', 'min-h-0');
  expect(screen.getByTestId('offline-rl-environment-canvas'))
    .toHaveClass('h-full', 'w-full', 'overflow-y-auto');
  expect(screen.getByTestId('offline-rl-workflow-steps'))
    .toHaveAttribute('data-panel-state', 'training');
  expect(screen.getByTestId('offline-rl-dataset-pipeline')).toHaveClass(
    'flex-col', 'justify-between', 'gap-3', 'overflow-y-auto'
  );
  const dataCollectionStep = screen.getByTestId('offline-rl-pipeline-step-01');
  const datasetConversionStep = screen.getByTestId('offline-rl-pipeline-step-02');
  const lerobotDatasetStep = screen.getByTestId('offline-rl-pipeline-step-03');
  [dataCollectionStep, datasetConversionStep, lerobotDatasetStep].forEach((step) => {
    expect(step).toHaveClass('w-full', 'shrink-0', 'min-h-0');
    expect(step).not.toHaveClass('min-h-[240px]');
  });
  expect(screen.getByTestId('offline-rl-dataset-pipeline').lastElementChild)
    .toBe(lerobotDatasetStep);
});

test('shows the Playground rail and collapses it to icon-only navigation', () => {
  renderPage();

  const rail = screen.getByRole('complementary', {
    name: 'Playground navigation',
  });
  const workspace = screen.getByTestId('offline-rl-workspace');

  expect(rail).toHaveAttribute('data-collapsed', 'false');
  expect(screen.getByRole('button', { name: 'Environment' }))
    .toHaveAttribute('aria-current', 'page');
  expect(workspace).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', {
    name: 'Collapse Playground menu',
  }));

  expect(rail).toHaveAttribute('data-collapsed', 'true');
  expect(screen.getByRole('button', {
    name: 'Expand Playground menu',
  })).toBeInTheDocument();
  expect(workspace).toBeInTheDocument();
});

test('slides the Replay Buffer data workflow over the mounted environment', () => {
  renderPage();

  const workspace = screen.getByTestId('offline-rl-workspace');
  const replayStack = screen.getByTestId('replay-buffer-stack');
  const conversion = screen.getByTestId('dataset-conversion');
  const lerobotDataset = screen.getByTestId('lerobot-dataset-stack');
  const drawer = screen.getByTestId('offline-rl-replay-drawer');
  const replayButton = screen.getByRole('button', { name: 'Replay Buffer' });

  expect(drawer).toHaveAttribute('data-panel-state', 'closed');
  expect(drawer).toHaveAttribute('aria-hidden', 'true');
  expect(drawer).toHaveAttribute('inert');
  expect(replayButton).toHaveAttribute(
    'aria-controls', 'offline-rl-replay-drawer'
  );
  expect(replayButton).toHaveAttribute('aria-expanded', 'false');
  expect(replayStack).toHaveAttribute('data-active', 'true');
  expect(conversion).toHaveAttribute('data-active', 'true');
  expect(lerobotDataset).toHaveAttribute('data-active', 'true');
  expect(drawer).toHaveClass(
    'absolute', 'w-[calc(100%_-_2rem)]', 'lg:w-1/2'
  );
  expect(screen.getByTestId('offline-rl-workflow-steps'))
    .not.toHaveClass('xl:contents');
  expect(screen.getByTestId('offline-rl-environment-canvas'))
    .toHaveClass('w-full', 'flex-1');

  fireEvent.click(replayButton);

  expect(drawer).toHaveAttribute('data-panel-state', 'open');
  expect(drawer).toHaveAttribute('aria-hidden', 'false');
  expect(drawer).not.toHaveAttribute('inert');
  expect(drawer).toHaveClass('absolute', 'lg:w-1/2');
  expect(drawer).not.toHaveClass('xl:static', 'xl:mr-4');
  expect(screen.getByTestId('offline-rl-dataset-pipeline'))
    .toHaveClass('min-h-0', 'w-full', 'flex-1', 'overflow-y-auto');
  expect(screen.getByTestId('offline-rl-workflow-steps'))
    .toHaveAttribute('data-panel-state', 'replay');
  expect(replayButton).toHaveAttribute('aria-current', 'page');
  expect(replayButton).toHaveAttribute('aria-expanded', 'true');
  expect(screen.getByRole('button', {
    name: 'Close Replay Buffer panel',
  })).toHaveFocus();
  const replayCloseButton = screen.getByRole('button', {
    name: 'Close Replay Buffer panel',
  });
  expect(screen.getByTestId('offline-rl-replay-drawer-header').firstElementChild)
    .toBe(replayCloseButton);
  expect(screen.getByTestId('replay-drawer-toggle-glyph'))
    .toHaveClass('bg-[#f3f0e8]');
  expect(screen.getByTestId('replay-drawer-toggle-accent'))
    .toHaveClass('left-0', 'w-[20%]', 'bg-[#627d68]');
  expect(replayStack).toHaveAttribute('data-active', 'true');
  expect(conversion).toHaveAttribute('data-active', 'true');
  expect(lerobotDataset).toHaveAttribute('data-active', 'true');
  expect(workspace).toBeInTheDocument();
  expect(screen.getByTestId('offline-rl-environment-canvas'))
    .toHaveClass('w-full', 'flex-1');

  fireEvent.click(replayButton);
  expect(drawer).toHaveAttribute('data-panel-state', 'closed');
  expect(replayButton).toHaveAttribute('aria-expanded', 'false');
  expect(screen.getByRole('button', { name: 'Environment' }))
    .toHaveAttribute('aria-current', 'page');

  fireEvent.click(replayButton);
  fireEvent.click(screen.getByRole('button', {
    name: 'Close Replay Buffer panel',
  }));

  expect(drawer).toHaveAttribute('data-panel-state', 'closed');
  expect(replayButton).toHaveAttribute('aria-expanded', 'false');
  expect(replayButton).toHaveFocus();
  expect(screen.getByRole('button', { name: 'Environment' }))
    .toHaveAttribute('aria-current', 'page');
  expect(screen.getByTestId('replay-buffer-stack')).toBe(replayStack);
  expect(screen.getByTestId('dataset-conversion')).toBe(conversion);
  expect(screen.getByTestId('lerobot-dataset-stack')).toBe(lerobotDataset);
  expect(replayStack).toHaveAttribute('data-active', 'true');
  expect(workspace).toBeInTheDocument();

  fireEvent.click(replayButton);
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(drawer).toHaveAttribute('data-panel-state', 'closed');
  expect(replayButton).toHaveFocus();
});

test('keeps Replay Buffer and Training open together and closes them independently', () => {
  renderPage();

  const workspace = screen.getByTestId('offline-rl-workspace');
  const replayDrawer = screen.getByTestId('offline-rl-replay-drawer');
  const trainingDrawer = screen.getByTestId('offline-rl-training-drawer');
  const replayButton = screen.getByRole('button', { name: 'Replay Buffer' });
  const trainingButton = screen.getByRole('button', { name: 'Training' });
  const trainingController = screen.getByTestId('workflow-training-controller');
  const deployment = screen.getByTestId('offline-rl-deployment');

  expect(trainingDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(trainingDrawer).toHaveAttribute('aria-hidden', 'true');
  expect(trainingDrawer).toHaveAttribute('inert');
  expect(trainingButton).toHaveAttribute(
    'aria-controls', 'offline-rl-training-drawer'
  );
  expect(trainingButton).toHaveAttribute('aria-expanded', 'false');
  expect(trainingController).toHaveAttribute('data-active', 'true');
  expect(trainingDrawer).toHaveClass(
    'absolute', 'min-h-0', 'w-[calc(100%_-_2rem)]', 'lg:w-1/2'
  );
  expect(trainingDrawer.style.width).toBe('');

  fireEvent.click(replayButton);
  expect(replayDrawer).toHaveAttribute('data-panel-state', 'open');

  fireEvent.click(trainingButton);

  expect(replayDrawer).toHaveAttribute('data-panel-state', 'open');
  expect(replayDrawer).toHaveAttribute('aria-hidden', 'false');
  expect(replayDrawer).not.toHaveAttribute('inert');
  expect(replayButton).toHaveAttribute('aria-expanded', 'true');
  expect(replayButton).not.toHaveAttribute('aria-current');
  expect(trainingDrawer).toHaveAttribute('data-panel-state', 'open');
  expect(trainingDrawer).toHaveAttribute('aria-hidden', 'false');
  expect(trainingDrawer).not.toHaveAttribute('inert');
  expect(replayDrawer).toHaveClass('lg:w-[calc(50%_-_1.5rem)]');
  expect(trainingDrawer).toHaveClass(
    'absolute', 'lg:w-[calc(50%_-_1.5rem)]'
  );
  expect(trainingDrawer).not.toHaveClass('xl:static', 'xl:ml-4');
  expect(within(trainingDrawer).getByRole('heading', { name: 'Training Pipeline' }))
    .toBeInTheDocument();
  expect(trainingButton).toHaveAttribute('aria-current', 'page');
  expect(trainingButton).toHaveAttribute('aria-expanded', 'true');
  expect(screen.getByRole('button', {
    name: 'Close Training panel',
  })).toHaveFocus();
  const trainingCloseButton = screen.getByRole('button', {
    name: 'Close Training panel',
  });
  expect(screen.getByTestId('offline-rl-training-drawer-header').firstElementChild)
    .toBe(trainingCloseButton);
  expect(screen.getByTestId('training-drawer-toggle-glyph'))
    .toHaveClass('bg-[#f3f0e8]');
  expect(screen.getByTestId('training-drawer-toggle-accent'))
    .toHaveClass('right-0', 'w-[20%]', 'bg-[#627d68]');
  expect(screen.getByTestId('training-drawer-toggle-accent'))
    .not.toHaveClass('left-0');
  expect(screen.getByTestId('offline-rl-training-content'))
    .toHaveClass('min-h-0', 'w-full', 'flex-1', 'overflow-y-auto');
  expect(screen.getByTestId('offline-rl-training-stage'))
    .toHaveClass('h-full', 'min-h-[640px]', 'w-full', 'shrink-0');
  expect(deployment).toHaveClass('w-full', 'shrink-0');
  expect(workspace).toBeInTheDocument();
  expect(screen.getByTestId('offline-rl-environment-canvas'))
    .toHaveClass('w-full', 'flex-1');
  expect(screen.getByTestId('offline-rl-workflow-steps'))
    .toHaveAttribute('data-panel-state', 'both');

  fireEvent.click(screen.getByRole('button', {
    name: 'Close Replay Buffer panel',
  }));

  expect(replayDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(replayDrawer).toHaveAttribute('inert');
  expect(replayButton).toHaveAttribute('aria-expanded', 'false');
  expect(replayButton).toHaveFocus();
  expect(trainingDrawer).toHaveAttribute('data-panel-state', 'open');
  expect(trainingButton).toHaveAttribute('aria-expanded', 'true');
  expect(trainingDrawer).toHaveClass('lg:w-1/2');
  expect(screen.getByTestId('offline-rl-workflow-steps'))
    .toHaveAttribute('data-panel-state', 'training');

  fireEvent.click(trainingButton);
  expect(trainingDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(trainingButton).toHaveAttribute('aria-expanded', 'false');
  expect(trainingButton).toHaveFocus();
  expect(screen.getByRole('button', { name: 'Environment' }))
    .toHaveAttribute('aria-current', 'page');
  expect(screen.getByTestId('workflow-training-controller')).toBe(trainingController);
  expect(screen.getByTestId('offline-rl-deployment')).toBe(deployment);
  expect(trainingController).toHaveAttribute('data-active', 'true');

  fireEvent.click(replayButton);
  fireEvent.click(trainingButton);
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(trainingDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(replayDrawer).toHaveAttribute('data-panel-state', 'open');
  expect(replayButton).toHaveAttribute('aria-expanded', 'true');
  expect(trainingButton).toHaveFocus();

  fireEvent.keyDown(document, { key: 'Escape' });
  expect(replayDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(replayButton).toHaveFocus();

  fireEvent.click(replayButton);
  fireEvent.click(trainingButton);
  fireEvent.click(screen.getByRole('button', { name: 'Environment' }));
  expect(replayDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(trainingDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(screen.getByRole('button', { name: 'Environment' }))
    .toHaveAttribute('aria-current', 'page');
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
  openTrainingDrawer();

  expect(screen.getAllByText('TD3').length).toBeGreaterThan(0);
  expect(screen.getAllByText('ACT').length).toBeGreaterThan(0);
  expect(screen.getByRole('button', { name: 'GR00T' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Pi0.5' })).toBeDisabled();
  expect(screen.queryByText('RLT')).not.toBeInTheDocument();

  const trainingSection = within(
    screen.getByTestId('offline-rl-training-content')
  ).getByRole('heading', { name: 'Training' }).closest('section');
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
  openTrainingDrawer();
  const architecture = screen.getByTestId('offline-rl-training-architecture');
  const footer = screen.getByTestId('offline-rl-training-footer');

  expect(footer).toHaveClass('mt-auto', 'shrink-0');
  expect(container.textContent.indexOf(architecture.textContent))
    .toBeLessThan(container.textContent.indexOf('Training progress'));
  expect(within(footer).getByText('Training progress')).toBeInTheDocument();
  expect(within(footer).getByText('Training action')).toBeInTheDocument();
});

test('pulls Training progress and Policy Deploy upward for compact workflows', () => {
  renderPage();
  openTrainingDrawer();
  const stage = screen.getByTestId('offline-rl-training-stage');
  const deployment = screen.getByTestId('offline-rl-deployment');

  expect(stage).toHaveAttribute('data-compact-layout', 'false');
  expect(stage).toHaveClass('h-full', 'min-h-[640px]');

  fireEvent.click(screen.getByRole('button', { name: 'Compact training layout' }));

  expect(stage).toHaveAttribute('data-compact-layout', 'true');
  expect(stage).toHaveClass('min-h-0');
  expect(stage).not.toHaveClass('h-full', 'min-h-[640px]');
  expect(stage.compareDocumentPosition(deployment) & Node.DOCUMENT_POSITION_FOLLOWING)
    .toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: 'Restore training layout' }));
  expect(stage).toHaveAttribute('data-compact-layout', 'false');
});

test('shows three camera slots without enabling unfinished actions', () => {
  const { container } = renderPage();
  openTrainingDrawer();

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
  expect(statusButton.querySelector('[data-robot-lab-icon="true"]'))
    .toBeInTheDocument();

  fireEvent.click(statusButton);
  expect(screen.getByRole('dialog', {
    name: 'Inference Workspace Status',
  })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Back' }));
  expect(screen.queryByRole('dialog', {
    name: 'Inference Workspace Status',
  })).not.toBeInTheDocument();
});

test('opens Training Guide from the Step 04 icon without closing Training', () => {
  renderPage();
  const { drawer } = openTrainingDrawer();
  const guideButton = screen.getByRole('button', {
    name: 'Open Training Guide',
  });

  guideButton.focus();
  fireEvent.click(guideButton);

  expect(screen.getByRole('dialog', { name: 'Training Guide' }))
    .toBeInTheDocument();
  expect(screen.getByTestId('offline-rl-training-guide-backdrop').parentElement)
    .toBe(document.body);

  fireEvent.keyDown(document, { key: 'Escape' });

  expect(screen.queryByRole('dialog', { name: 'Training Guide' }))
    .not.toBeInTheDocument();
  expect(drawer).toHaveAttribute('data-panel-state', 'open');
  expect(guideButton).toHaveFocus();
});

test('opens Data Conversion Guide from the Step 01 icon without closing Replay Buffer', () => {
  renderPage();
  const replayButton = screen.getByRole('button', { name: 'Replay Buffer' });
  fireEvent.click(replayButton);
  const replayDrawer = screen.getByTestId('offline-rl-replay-drawer');
  const guideButton = screen.getByRole('button', {
    name: 'Open Data Conversion Guide',
  });

  guideButton.focus();
  fireEvent.click(guideButton);

  expect(screen.getByRole('dialog', { name: 'Data Conversion Guide' }))
    .toBeInTheDocument();
  expect(screen.getByTestId('offline-rl-data-conversion-guide-backdrop').parentElement)
    .toBe(document.body);

  fireEvent.keyDown(window, { key: 'Escape' });

  expect(screen.queryByRole('dialog', { name: 'Data Conversion Guide' }))
    .not.toBeInTheDocument();
  expect(replayDrawer).toHaveAttribute('data-panel-state', 'open');
  expect(guideButton).toHaveFocus();
});

test('deploys a completed model into the shared inference Model path', () => {
  const { testStore } = renderPage();
  openTrainingDrawer();

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

test('deploys an RLT bundle without replacing the GR00T policy or TensorRT settings', () => {
  const { testStore } = renderPage();
  openTrainingDrawer();

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/groot/showroom',
      serviceType: 'groot',
      policyType: 'n17',
      accelerationMode: 'tensorrt_dit',
      accelerationEnginePath: '/workspace/model/groot/showroom/engines/dit.plan',
      rltEnabled: true,
      rltBundlePath: '/workspace/checkpoint/rlt/stage2/round_0000',
      rltRobotOverride: true,
      actionPolicyMode: 'rlt',
    }));
  });

  fireEvent.click(screen.getByRole('button', { name: 'Mark RLT training complete' }));
  const deployButton = screen.getByRole('button', { name: 'Deploy RLT Bundle' });
  expect(deployButton).not.toBeDisabled();
  fireEvent.click(deployButton);

  expect(testStore.getState().tasks.inferenceTaskInfo).toMatchObject({
    policyPath: '/workspace/model/groot/showroom',
    serviceType: 'groot',
    policyType: 'n17',
    accelerationMode: 'tensorrt_dit',
    accelerationEnginePath: '/workspace/model/groot/showroom/engines/dit.plan',
    rltEnabled: true,
    rltBundlePath: '/workspace/checkpoint/rlt/stage2/round_0001',
    rltRobotOverride: false,
    actionPolicyMode: 'base',
  });
  expect(screen.getByRole('button', { name: 'Deploy RLT Bundle' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Discard RLT Bundle' })).not.toBeDisabled();
});

test('discards a deployed RLT bundle and restores the previous RLT runtime state', () => {
  const { testStore } = renderPage();
  openTrainingDrawer();

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/groot/showroom',
      serviceType: 'groot',
      policyType: 'n17',
      accelerationMode: 'tensorrt_dit',
      accelerationEnginePath: '/workspace/model/groot/showroom/engines/dit.plan',
      rltEnabled: true,
      rltBundlePath: '/workspace/checkpoint/rlt/stage2/round_0000',
      rltRobotOverride: true,
      actionPolicyMode: 'rlt',
    }));
  });

  fireEvent.click(screen.getByRole('button', { name: 'Mark RLT training complete' }));
  fireEvent.click(screen.getByRole('button', { name: 'Deploy RLT Bundle' }));
  fireEvent.click(screen.getByRole('button', { name: 'Discard RLT Bundle' }));

  expect(testStore.getState().tasks.inferenceTaskInfo).toMatchObject({
    policyPath: '/workspace/model/groot/showroom',
    serviceType: 'groot',
    policyType: 'n17',
    accelerationMode: 'tensorrt_dit',
    accelerationEnginePath: '/workspace/model/groot/showroom/engines/dit.plan',
    rltEnabled: true,
    rltBundlePath: '/workspace/checkpoint/rlt/stage2/round_0000',
    rltRobotOverride: true,
    actionPolicyMode: 'rlt',
  });
  expect(screen.getByRole('button', { name: 'Discard Policy' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Deploy RLT Bundle' })).not.toBeDisabled();
});

test('does not discard an RLT bundle after the user selects another bundle manually', () => {
  const { testStore } = renderPage();
  openTrainingDrawer();

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/groot/showroom',
      serviceType: 'groot',
      policyType: 'n17',
      rltEnabled: true,
      rltBundlePath: '/workspace/checkpoint/rlt/stage2/round_0000',
    }));
  });
  fireEvent.click(screen.getByRole('button', { name: 'Mark RLT training complete' }));
  fireEvent.click(screen.getByRole('button', { name: 'Deploy RLT Bundle' }));

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      rltBundlePath: '/workspace/checkpoint/rlt/stage2/manual_selection',
    }));
  });

  expect(screen.getByRole('button', { name: 'Discard Policy' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Discard Policy' }));
  expect(testStore.getState().tasks.inferenceTaskInfo.rltBundlePath)
    .toBe('/workspace/checkpoint/rlt/stage2/manual_selection');
});

test('deploys imitation learning as a new Epoch 0 policy lineage', () => {
  window.sessionStorage.setItem(OFFLINE_RL_LINEAGE_STORAGE_KEY, JSON.stringify({
    policyEpoch: 3,
    policyPath: '/workspace/model/rl_epoch_0003/pretrained_model',
    forceFresh: false,
    lineageId: 'existing-lineage',
  }));
  const { testStore } = renderPage();
  openTrainingDrawer();

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/rl_epoch_0003/pretrained_model',
      serviceType: 'lerobot',
      policyType: 'act',
    }));
  });
  expect(screen.getByLabelText('Training policy RL Epoch 3 to 4'))
    .toHaveTextContent('RL Epoch E0003 → E0004');

  fireEvent.click(screen.getByRole('button', { name: 'Select IL' }));
  expect(screen.getByLabelText('Imitation Learning base policy RL Epoch 0'))
    .toHaveTextContent('Base Policy E0000');
  expect(screen.queryByLabelText('Training policy RL Epoch 3 to 4'))
    .not.toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Mark IL training complete' }));
  fireEvent.click(screen.getByRole('button', { name: 'Deploy Policy' }));

  expect(testStore.getState().tasks.inferenceTaskInfo.policyPath)
    .toBe('/workspace/model/imitation/act/pretrained_model');
  expect(screen.getByTestId('offline-rl-workspace'))
    .toHaveAttribute('data-policy-epoch', '0');
  const newLineage = JSON.parse(
    window.sessionStorage.getItem(OFFLINE_RL_LINEAGE_STORAGE_KEY)
  );
  expect(newLineage).toMatchObject({
    policyEpoch: 0,
    policyPath: '/workspace/model/imitation/act/pretrained_model',
    forceFresh: true,
  });
  expect(newLineage.lineageId).not.toBe('existing-lineage');
});

test('keeps the current RL Epoch unchanged during independent critic warm-up', () => {
  window.sessionStorage.setItem(OFFLINE_RL_LINEAGE_STORAGE_KEY, JSON.stringify({
    policyEpoch: 3,
    policyPath: '/workspace/model/rl_epoch_0003/pretrained_model',
    forceFresh: false,
    lineageId: 'existing-lineage',
  }));
  renderPage();
  openTrainingDrawer();

  expect(screen.getByLabelText('Training policy RL Epoch 3 to 4'))
    .toHaveTextContent('RL Epoch E0003 → E0004');
  fireEvent.click(screen.getByRole('button', { name: 'Select Critic' }));

  expect(screen.getByLabelText('Critic Warm-up policy RL Epoch 3 unchanged'))
    .toHaveTextContent('Critic · E0003');
  expect(screen.queryByLabelText('Training policy RL Epoch 3 to 4'))
    .not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Deploy Policy' })).toBeDisabled();
  expect(JSON.parse(
    window.sessionStorage.getItem(OFFLINE_RL_LINEAGE_STORAGE_KEY)
  )).toMatchObject({
    policyEpoch: 3,
    policyPath: '/workspace/model/rl_epoch_0003/pretrained_model',
    forceFresh: false,
    lineageId: 'existing-lineage',
  });
});

test('starts a new RL lineage at Epoch 0 without deleting or clearing selected paths', () => {
  const { testStore } = renderPage();
  openTrainingDrawer();
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
  openTrainingDrawer();

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
  openTrainingDrawer();

  act(() => {
    testStore.dispatch(setInferenceTaskInfo({
      policyPath: '/workspace/model/groot/original',
      serviceType: 'groot',
      policyType: 'n17',
      accelerationMode: 'tensorrt_dit',
      accelerationEnginePath: '/workspace/model/groot/original/engines/dit.plan',
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
    accelerationMode: 'pytorch',
    accelerationEnginePath: '',
  });
  expect(screen.getByRole('button', { name: 'Deploy Policy' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Discard Policy' })).not.toBeDisabled();

  fireEvent.click(screen.getByRole('button', { name: 'Discard Policy' }));

  expect(testStore.getState().tasks.inferenceTaskInfo).toMatchObject({
    policyPath: '/workspace/model/groot/original',
    serviceType: 'groot',
    policyType: 'n17',
    accelerationMode: 'tensorrt_dit',
    accelerationEnginePath: '/workspace/model/groot/original/engines/dit.plan',
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
  openTrainingDrawer();

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
  openTrainingDrawer();

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

test('keeps closed Replay and Training controllers mounted outside the environment canvas', () => {
  renderPage();

  const workflowGrid = screen.getByTestId('offline-rl-workflow-grid');
  const workflowSteps = screen.getByTestId('offline-rl-workflow-steps');
  const replayDrawer = screen.getByTestId('offline-rl-replay-drawer');
  const trainingDrawer = screen.getByTestId('offline-rl-training-drawer');
  expect(workflowGrid).toHaveAttribute('data-layout', 'environment-canvas');
  expect(screen.getByTestId('offline-rl-environment-canvas'))
    .toHaveClass('w-full');
  expect(workflowSteps).toHaveClass(
    'absolute', 'inset-0', 'overflow-hidden'
  );
  expect(workflowSteps).not.toHaveClass('xl:contents');
  expect(workflowSteps).toHaveAttribute('data-panel-state', 'environment');
  expect(replayDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(replayDrawer).toHaveAttribute('inert');
  expect(trainingDrawer).toHaveAttribute('data-panel-state', 'closed');
  expect(trainingDrawer).toHaveAttribute('aria-hidden', 'true');
  expect(trainingDrawer).toHaveAttribute('inert');
  expect(screen.getByTestId('replay-buffer-stack')).toBeInTheDocument();
  expect(screen.getByTestId('dataset-conversion')).toBeInTheDocument();
  expect(screen.getByTestId('lerobot-dataset-stack')).toBeInTheDocument();
  expect(screen.getByTestId('workflow-training-controller')).toBeInTheDocument();
  expect(screen.getByTestId('workflow-training-controller'))
    .toHaveAttribute('data-active', 'true');
  expect(screen.getByTestId('offline-rl-deployment')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Hide workflow panel' }))
    .not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Show workflow panel' }))
    .not.toBeInTheDocument();
});

test('uses white bordered workflow surfaces without shrinking the environment', () => {
  renderPage();

  const environment = screen.getByTestId('offline-rl-environment-stage');
  const environmentCanvas = screen.getByTestId('offline-rl-environment-canvas');
  const replayButton = screen.getByRole('button', { name: 'Replay Buffer' });

  expect(environment).toHaveClass('border', 'bg-[#fbfaf6]');
  expect(environmentCanvas).toHaveClass('w-full', 'flex-1');

  fireEvent.click(replayButton);

  const collectionCard = screen.getByRole('heading', {
    name: 'Data Collection',
  }).closest('section');
  expect(collectionCard).toHaveClass('border', 'bg-[#fbfaf6]');
  expect(environmentCanvas).toHaveClass('w-full', 'flex-1');

  fireEvent.click(screen.getByRole('button', { name: 'Training' }));
  expect(screen.getByTestId('offline-rl-training-stage'))
    .toHaveClass('border', 'bg-[#fbfaf6]');
  expect(screen.getByTestId('offline-rl-deployment'))
    .toHaveClass('border', 'bg-[#fbfaf6]');
  expect(environmentCanvas).toHaveClass('w-full', 'flex-1');
});
