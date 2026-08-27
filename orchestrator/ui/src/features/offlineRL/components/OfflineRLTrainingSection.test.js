import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Provider } from 'react-redux';
import { InferencePhase } from '../../../constants/taskPhases';
import taskReducer, {
  selectRobotType,
  setInferenceTaskInfo,
} from '../../tasks/taskSlice';
import editDatasetReducer, {
  setConversionStatus,
} from '../../editDataset/editDatasetSlice';
import offlineRLReducer, {
  setOfflineRLCheckpointPath,
  setOfflineRLDatasetPath,
  setOfflineRLDatasetSelection,
  setOfflineRLDatasetSelections,
} from '../offlineRLSlice';
import OfflineRLTrainingSection, {
  resolveTrainingPolicyModel,
} from './OfflineRLTrainingSection';
import {
  getACTTD3CriticWarmupStatus,
  getFlowSDEPPOValueWarmupStatus,
  getImitationLearningStatus,
  getOfflineRLDatasetInfo,
  getOfflineRLStatus,
  startACTTD3CriticWarmup,
  startFlowSDEPPOValueWarmup,
  startImitationLearningTraining,
  startOfflineRLTraining,
  stopACTTD3CriticWarmup,
  stopFlowSDEPPOValueWarmup,
  stopImitationLearningTraining,
  stopOfflineRLTraining,
} from '../../../utils/offlineRlApi';

jest.mock('../../../utils/offlineRlApi', () => ({
  getACTTD3CriticWarmupStatus: jest.fn(),
  getFlowSDEPPOValueWarmupStatus: jest.fn(),
  getImitationLearningStatus: jest.fn(),
  getOfflineRLDatasetInfo: jest.fn(),
  getOfflineRLStatus: jest.fn(),
  startACTTD3CriticWarmup: jest.fn(),
  startFlowSDEPPOValueWarmup: jest.fn(),
  startImitationLearningTraining: jest.fn(),
  startOfflineRLTraining: jest.fn(),
  stopACTTD3CriticWarmup: jest.fn(),
  stopFlowSDEPPOValueWarmup: jest.fn(),
  stopImitationLearningTraining: jest.fn(),
  stopOfflineRLTraining: jest.fn(),
}));

const deferred = () => {
  let resolve;
  const promise = new Promise((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
};

const renderSection = (props = {}) => {
  const testStore = configureStore({
    reducer: {
      tasks: taskReducer,
      editDataset: editDatasetReducer,
      offlineRL: offlineRLReducer,
    },
  });
  testStore.dispatch(selectRobotType('ffw_sg2_rev1'));
  const view = render(
    <Provider store={testStore}>
      <OfflineRLTrainingSection {...props} />
    </Provider>
  );
  return { ...view, testStore };
};

describe('OfflineRLTrainingSection', () => {
  test.each([
    ['lerobot', 'act', 'act'],
    ['lerobot', 'multi_task_dit', 'multi_task_dit'],
    ['lerobot', 'pi05', 'pi05'],
    ['groot', 'n17', 'groot'],
    ['lerobot', 'diffusion', null],
  ])('maps Inference %s:%s to the supported Training model %s', (
    serviceType,
    policyType,
    expected
  ) => {
    expect(resolveTrainingPolicyModel({ serviceType, policyType })).toBe(expected);
  });

  beforeEach(() => {
    getACTTD3CriticWarmupStatus.mockResolvedValue({
      status: 'idle',
      percentage: 0,
      total_critic_updates: 5000,
    });
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    getImitationLearningStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    getOfflineRLDatasetInfo.mockImplementation(async (path) => ({
      dataset_path: path,
      name: String(path).split('/').filter(Boolean).pop() || 'dataset',
      version: 'v3.0',
      total_episodes: 0,
      success_count: 0,
      failure_count: 0,
      unlabeled_count: 0,
      episodes: [],
    }));
    getOfflineRLStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    startImitationLearningTraining.mockResolvedValue({ status: 'starting', percentage: 0 });
    startOfflineRLTraining.mockResolvedValue({ status: 'starting', percentage: 0 });
    startACTTD3CriticWarmup.mockResolvedValue({
      status: 'running',
      percentage: 0,
      job_id: 'act-critic-warmup-job-1',
      total_critic_updates: 5000,
    });
    startFlowSDEPPOValueWarmup.mockResolvedValue({
      status: 'running',
      percentage: 0,
      job_id: 'warmup-job-1',
    });
    stopImitationLearningTraining.mockResolvedValue({ status: 'running', percentage: 10 });
    stopOfflineRLTraining.mockResolvedValue({ status: 'running', percentage: 10 });
    stopACTTD3CriticWarmup.mockResolvedValue({
      status: 'running',
      percentage: 10,
      job_id: 'act-critic-warmup-job-1',
    });
    stopFlowSDEPPOValueWarmup.mockResolvedValue({
      status: 'running',
      percentage: 10,
      job_id: 'warmup-job-1',
    });
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  test('shows the bounded round and editable valid TD3 schedule', async () => {
    renderSection();

    expect(screen.getByText('200 episodes')).toBeInTheDocument();
    expect(screen.getByText('Auto · 1–50')).toBeInTheDocument();
    expect(screen.getByLabelText('Critic epochs')).toHaveValue(10);
    expect(screen.getByLabelText('Actor equivalent epochs')).toHaveValue(5);
    expect(screen.getByLabelText('Batch size')).toHaveValue(4);
    expect(screen.getByRole('option', { name: /SAC.*Coming soon/i })).toBeDisabled();
    expect(screen.getByRole('option', { name: /RLT.*Coming soon/i })).toBeDisabled();
    await screen.findByRole('button', { name: 'Start Training' });
  });

  test('keeps backend critic-source diagnostics out of the compact TD3 card', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'running',
      percentage: 10,
      job_id: 'td3-policy-warmup',
      critic_source: 'policy_warmup',
      critic_checkpoint: '/workspace/model/act/critic/latest.pt',
    });

    renderSection({ variant: 'workflow' });

    await screen.findByText('Training…');
    expect(screen.queryByLabelText('TD3 critic initialization')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('TD3 critic source')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('TD3 critic checkpoint')).not.toBeInTheDocument();
  });

  test('keeps inputs and Start locked until the first successful status response', async () => {
    const initialStatus = deferred();
    getOfflineRLStatus.mockReturnValueOnce(initialStatus.promise);

    renderSection();

    expect(screen.getByLabelText('LeRobot v3 Dataset Path')).toBeDisabled();
    expect(screen.getByLabelText('Original ACT Checkpoint')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Checking status…' })).toBeDisabled();

    await act(async () => {
      initialStatus.resolve({ status: 'idle', percentage: 0 });
      await initialStatus.promise;
    });

    expect(screen.getByLabelText('LeRobot v3 Dataset Path')).not.toBeDisabled();
    expect(screen.getByLabelText('Original ACT Checkpoint')).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();
  });

  test('starts TD3 with the dataset, original ACT, optional parent, and robot type', async () => {
    renderSection();
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.change(screen.getByLabelText('LeRobot v3 Dataset Path'), {
      target: { value: '/workspace/lerobot/task_lerobot_v30' },
    });
    fireEvent.change(screen.getByLabelText('Original ACT Checkpoint'), {
      target: { value: '/workspace/model/lerobot/base/pretrained_model' },
    });
    fireEvent.change(screen.getByRole('textbox', { name: /Previous Round Checkpoint/ }), {
      target: { value: '/workspace/model/lerobot/round1/training_state/act_td3.pt' },
    });
    fireEvent.change(screen.getByLabelText('Critic epochs'), {
      target: { value: '6' },
    });
    fireEvent.change(screen.getByLabelText('Actor equivalent epochs'), {
      target: { value: '3' },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith({
      dataset_path: '/workspace/lerobot/task_lerobot_v30',
      dataset_paths: ['/workspace/lerobot/task_lerobot_v30'],
      act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
      parent_checkpoint: '/workspace/model/lerobot/round1/training_state/act_td3.pt',
      algorithm: 'td3',
      actor_objective: 'td3_bc',
      robot_type: 'ffw_sg2_rev1',
      critic_epochs: 6,
      actor_equivalent_epochs: 3,
      batch_size: 4,
      actor_trainable_groups: [
        'visual_backbone',
        'cvae_encoder',
        'transformer_encoder',
        'action_decoder',
      ],
    }));
  });

  test('switches to pure TD3 and freezes the CVAE encoder in the submitted contract', async () => {
    renderSection();
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.change(screen.getByLabelText('LeRobot v3 Dataset Path'), {
      target: { value: '/workspace/lerobot/task_lerobot_v30' },
    });
    fireEvent.change(screen.getByLabelText('Original ACT Checkpoint'), {
      target: { value: '/workspace/model/lerobot/base/pretrained_model' },
    });
    fireEvent.change(screen.getByLabelText('Loss option'), {
      target: { value: 'td3' },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        algorithm: 'td3',
        actor_objective: 'td3',
        actor_trainable_groups: [
          'visual_backbone',
          'transformer_encoder',
          'action_decoder',
        ],
      })
    ));
  });

  test('hydrates the TD3 loss option from an existing pure-TD3 job', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'running',
      percentage: 12,
      job_id: 'pure-td3-running',
      algorithm: 'td3',
      actor_objective: 'td3',
      actor_trainable_groups: [
        'visual_backbone',
        'transformer_encoder',
        'action_decoder',
      ],
    });

    renderSection({ variant: 'workflow' });

    const pureTD3Option = await screen.findByRole('button', { name: 'TD3 loss' });
    await waitFor(() => expect(pureTD3Option).toHaveAttribute('aria-pressed', 'true'));
    expect(screen.getByRole('button', { name: 'TD3-BC loss' }))
      .toHaveAttribute('aria-pressed', 'false');
    const cvae = screen.getByRole('button', { name: /CVAE encoder: Frozen/i });
    expect(cvae).toHaveAttribute('aria-pressed', 'false');
    expect(cvae).toBeDisabled();
    expect(cvae).toHaveTextContent('Frozen · TD3');
  });

  test('does not let status polling overwrite an explicit TD3 loss selection', async () => {
    jest.useFakeTimers();
    getOfflineRLStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'previous-td3-bc',
      algorithm: 'td3',
      actor_objective: 'td3_bc',
    });

    renderSection({ variant: 'workflow' });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const pureTD3Option = screen.getByRole('button', { name: 'TD3 loss' });
    expect(screen.getByRole('button', { name: 'TD3-BC loss' }))
      .toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(pureTD3Option);
    expect(pureTD3Option).toHaveAttribute('aria-pressed', 'true');

    await act(async () => {
      jest.advanceTimersByTime(2000);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(pureTD3Option).toHaveAttribute('aria-pressed', 'true');
  });

  test('submits chronologically ordered immutable Data Epoch roots', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelections([
        {
          path: '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 1,
        },
        {
          path: '/workspace/lerobot/data_epoch_0002/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 2,
        },
      ]));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/base/pretrained_model',
      }));
    });

    expect(screen.getByText(/2 Data Epochs/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        dataset_path: '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
        dataset_paths: [
          '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
          '/workspace/lerobot/data_epoch_0002/task_lerobot_v30',
        ],
      })
    ));
  });

  test('renders the compact workflow controls and keeps progress/action in one footer', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    const policyGroup = screen.getByRole('group', { name: 'Policy model' });
    expect(policyGroup).toHaveTextContent('ACT');
    expect(screen.getByRole('button', { name: 'GR00T' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Pi0.5' })).not.toBeDisabled();
    const lossGroup = screen.getByRole('group', { name: 'Loss option' });
    const trainingLoop = screen.getByTestId('act-td3-training-loop');
    expect(trainingLoop).toContainElement(lossGroup);
    expect(screen.getByTestId('act-td3-algorithm-card')).toContainElement(lossGroup);
    expect(screen.getByRole('button', { name: 'TD3-BC loss' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'TD3' })).not.toBeDisabled();
    expect(within(screen.getByRole('group', { name: 'RL algorithm' }))
      .queryByRole('button', { name: 'TD3+BC' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i }))
      .toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i }))
      .toBeDisabled();
    expect(screen.getByRole('button', { name: 'SAC' })).toBeDisabled();
    const methodGroup = screen.getByRole('group', { name: 'Training method' });
    expect(within(methodGroup).getByRole('button', { name: 'Reinforcement Learning' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(within(methodGroup).getByRole('button', { name: 'Reinforcement Learning' }))
      .toHaveTextContent(/^RL$/);
    expect(within(methodGroup).getByRole('button', { name: 'Imitation Learning' }))
      .toHaveAttribute('aria-pressed', 'false');
    expect(within(methodGroup).getByRole('button', { name: 'Imitation Learning' }))
      .toHaveTextContent(/^IL$/);
    expect(within(methodGroup).getByRole('button', { name: 'Critic Warm-up' }))
      .toHaveAttribute('aria-pressed', 'false');
    expect(within(methodGroup).getByRole('button', { name: 'Critic Warm-up' }))
      .toHaveTextContent(/^Critic$/);
    expect(within(methodGroup).getAllByRole('button').map((button) => button.textContent))
      .toEqual(['IL', 'Critic', 'RL']);
    expect(screen.getByTestId('act-td3-training-loop')).toBeInTheDocument();
    expect(screen.getByTestId('training-replay-buffer-card')).toBeInTheDocument();
    expect(screen.getByTestId('act-td3-algorithm-card')).toHaveTextContent('Critic Network');
    expect(screen.getByLabelText('Critic epochs')).toHaveValue(10);
    expect(screen.getByLabelText('Actor epochs')).toHaveValue(5);
    expect(screen.getByLabelText('Batch size')).toHaveValue(4);

    const workflow = screen.getByTestId('offline-rl-workflow-training');
    expect(workflow).toHaveClass(
      'grid',
      'flex-none',
      'grid-rows-[auto_auto_auto]',
      'overflow-hidden'
    );

    const architecture = screen.getByTestId('offline-rl-training-architecture');
    expect(architecture).toHaveClass(
      'flex-1',
      'items-stretch',
      'min-h-0',
      'overflow-y-auto',
      'overscroll-contain'
    );
    expect(architecture).toContainElement(screen.getByTestId('act-td3-training-loop'));

    const footer = screen.getByTestId('offline-rl-training-footer');
    expect(footer).toHaveClass(
      'mt-3',
      'shrink-0',
      'items-stretch',
      'xl:grid-cols-[minmax(0,1fr)_220px]'
    );
    expect(footer).not.toHaveClass('mt-auto');
    expect(workflow.lastElementChild).toBe(footer);
    expect(footer).toHaveTextContent('Training progress');
    expect(footer).toHaveTextContent('Training loss');
    expect(footer).toHaveTextContent('Training action');
    expect(footer).toHaveTextContent('ETA');
    const progressCard = screen.getByTestId('offline-rl-training-progress-card');
    expect(progressCard).toHaveClass('bg-[#f8f5ef]');
    expect(within(progressCard).getByTestId('training-loss-chart')).toHaveClass('bg-white');
  });

  test('keeps ACT Policy and Replay Buffer fixed while IL, Critic, and RL replace only the training card', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    const policyStage = screen.getByTestId('act-td3-policy-stage');
    const replayStage = screen.getByTestId('training-replay-buffer-card');
    expect(screen.getByTestId('act-td3-algorithm-card')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));
    await screen.findByTestId('act-imitation-algorithm-card');
    expect(screen.getByTestId('act-td3-policy-stage')).toBe(policyStage);
    expect(screen.getByTestId('training-replay-buffer-card')).toBe(replayStage);
    expect(screen.queryByTestId('act-td3-algorithm-card')).not.toBeInTheDocument();

    const criticMethod = screen.getByRole('button', { name: 'Critic Warm-up' });
    await waitFor(() => expect(criticMethod).not.toBeDisabled());
    fireEvent.click(criticMethod);
    await screen.findByTestId('act-critic-warmup-card');
    expect(screen.getByTestId('act-td3-policy-stage')).toBe(policyStage);
    expect(screen.getByTestId('training-replay-buffer-card')).toBe(replayStage);
    expect(screen.queryByTestId('act-imitation-algorithm-card')).not.toBeInTheDocument();

    const reinforcementMethod = screen.getByRole('button', { name: 'Reinforcement Learning' });
    await waitFor(() => expect(reinforcementMethod).not.toBeDisabled());
    fireEvent.click(reinforcementMethod);
    await screen.findByTestId('act-td3-algorithm-card');
    expect(screen.getByTestId('act-td3-policy-stage')).toBe(policyStage);
    expect(screen.getByTestId('training-replay-buffer-card')).toBe(replayStage);
    expect(screen.queryByTestId('act-critic-warmup-card')).not.toBeInTheDocument();
  });

  test('hydrates selected dataset outcome counts in the ACT training replay buffer', async () => {
    getOfflineRLDatasetInfo.mockResolvedValue({
      dataset_path: '/workspace/lerobot/data_epoch_0003/showroom_v30',
      name: 'showroom_v30',
      version: 'v3.0',
      total_episodes: 42,
      success_count: 31,
      failure_count: 9,
      unlabeled_count: 2,
      episodes: [],
    });

    const { testStore } = renderSection({ variant: 'workflow' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelections([{
        path: '/workspace/lerobot/data_epoch_0003/showroom_v30',
        version: 'v3.0',
        dataEpoch: 3,
      }]));
    });

    await waitFor(() => expect(getOfflineRLDatasetInfo).toHaveBeenCalledWith(
      '/workspace/lerobot/data_epoch_0003/showroom_v30'
    ));
    const replayCard = screen.getByTestId('training-replay-buffer-card');
    await waitFor(() => expect(within(replayCard).getByText('42 / 200 episodes'))
      .toBeInTheDocument());
    expect(within(replayCard).getByText('21%')).toBeInTheDocument();
    expect(within(replayCard).getByText('Success 31')).toBeInTheDocument();
    expect(within(replayCard).getByText('Failure 9')).toBeInTheDocument();
    expect(within(replayCard).getByText('Unlabeled 2')).toBeInTheDocument();
  });

  test('renders backend ACT-TD3 latest losses and exposes RL epoch metrics', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'running',
      job_id: 'act-td3-loss-history-job',
      algorithm: 'td3',
      actor_objective: 'td3_bc',
      percentage: 37.5,
      completed_critic_updates: 20,
      actor_loss: -0.45,
      critic_loss: 0.8,
      loss_history: [
        { step: 5, actor_loss: null, critic_loss: 2.0 },
        { step: 10, actor_loss: -0.2, critic_loss: 1.4 },
        { step: 20, actor_loss: -0.45, critic_loss: 0.8 },
      ],
      rl_metric_history: [{
        rl_epoch: 1,
        actor_loss_mean: -0.45,
        critic_loss_mean: 0.8,
        replay_average_reward: 0.75,
      }],
    });

    renderSection({ variant: 'workflow' });

    const chart = await screen.findByTestId('training-loss-chart');
    await waitFor(() => {
      expect(within(chart).getByLabelText('Latest actor loss'))
        .toHaveTextContent('-0.45');
      expect(within(chart).getByLabelText('Latest critic loss'))
        .toHaveTextContent('0.80');
    });
    expect(within(chart).getByRole('progressbar', { name: 'Training loss progress' }))
      .toHaveAttribute('aria-valuenow', '37.5');
    expect(within(chart).getByLabelText('Training percentage')).toHaveTextContent('37.5%');
    expect(within(chart).queryByRole('img')).not.toBeInTheDocument();
    expect(within(chart).getByRole('button', { name: 'Expand training metrics' }))
      .toBeInTheDocument();
  });

  test('reports the selected IL, RL, or Critic method to the workflow layout', async () => {
    const methodListener = jest.fn();
    renderSection({
      variant: 'workflow',
      onTrainingMethodStateChange: methodListener,
    });
    await screen.findByRole('button', { name: 'Start Training' });

    await waitFor(() => expect(methodListener).toHaveBeenLastCalledWith('reinforcement'));
    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));
    await waitFor(() => expect(methodListener).toHaveBeenLastCalledWith('imitation'));
    expect(screen.queryByRole('button', { name: 'Expand training metrics' }))
      .not.toBeInTheDocument();
    const criticMethod = screen.getByRole('button', { name: 'Critic Warm-up' });
    await waitFor(() => expect(criticMethod).not.toBeDisabled());
    fireEvent.click(criticMethod);
    await waitFor(() => expect(methodListener).toHaveBeenLastCalledWith('critic'));
    await screen.findByRole('button', { name: 'Start Critic Warm-up' });
    expect(screen.queryByRole('button', { name: 'Expand training metrics' }))
      .not.toBeInTheDocument();
  });

  test('can leave Critic when its status channel is unavailable', async () => {
    getACTTD3CriticWarmupStatus.mockRejectedValue(
      new Error('critic status unavailable')
    );
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));
    await waitFor(() => expect(getACTTD3CriticWarmupStatus).toHaveBeenCalled());

    const imitationMethod = screen.getByRole('button', { name: 'Imitation Learning' });
    const reinforcementMethod = screen.getByRole('button', { name: 'Reinforcement Learning' });
    expect(imitationMethod).not.toBeDisabled();
    expect(reinforcementMethod).not.toBeDisabled();

    fireEvent.click(imitationMethod);
    await waitFor(() => {
      expect(imitationMethod).toHaveAttribute('aria-pressed', 'true');
      expect(getImitationLearningStatus).toHaveBeenCalled();
    });
  });

  test('keeps every model selectable while Critic remains unavailable for preview-only models', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));
    await screen.findByRole('button', { name: 'Start Critic Warm-up' });

    const policyGroup = screen.getByRole('group', { name: 'Policy model' });
    expect(within(policyGroup).getByRole('button', { name: 'ACT' }))
      .toHaveAttribute('aria-pressed', 'true');
    for (const modelName of ['Diffusion Transformer', 'GR00T', 'Pi0.5']) {
      expect(within(policyGroup).getByRole('button', { name: modelName }))
        .not.toBeDisabled();
    }

    fireEvent.click(within(policyGroup).getByRole('button', { name: 'Pi0.5' }));
    await waitFor(() => {
      expect(within(policyGroup).getByRole('button', { name: 'Pi0.5' }))
        .toHaveAttribute('aria-pressed', 'true');
      expect(screen.getByRole('button', { name: 'Critic Warm-up' }))
        .toHaveAttribute('aria-pressed', 'true');
    });
    expect(screen.getByText(/Pi0.5 critic warm-up is not connected/))
      .toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start Critic Warm-up' }))
      .toBeDisabled();
  });

  test('keeps Pi0.5 selectable in IL as an explicit preview-only model', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));
    const policyGroup = screen.getByRole('group', { name: 'Policy model' });
    const piButton = within(policyGroup).getByRole('button', { name: 'Pi0.5' });
    await waitFor(() => expect(piButton).not.toBeDisabled());
    fireEvent.click(piButton);

    expect(piButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Imitation Learning' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('pi05-architecture-diagram')).toBeInTheDocument();
    const loop = screen.getByTestId('policy-training-loop');
    expect(loop).toHaveAttribute('data-policy-model', 'pi05');
    expect(loop).toHaveAttribute('data-fit-content', 'true');
    expect(within(loop).getByTestId('training-replay-buffer-card')).toBeInTheDocument();
    expect(within(loop).getByTestId('pi05-imitation-algorithm-card')).toBeInTheDocument();
    expect(within(loop).getByTestId('policy-training-loop-connectors')).toBeInTheDocument();
    expect(within(loop).getByText('Flow-Matching Action Reconstruction')).toBeInTheDocument();
    expect(screen.getByText(/Pi0.5 imitation-learning preview is available/))
      .toBeInTheDocument();
    const startTraining = screen.getByRole('button', { name: 'Start Training' });
    expect(startTraining).toBeDisabled();
    fireEvent.click(startTraining);
    expect(startImitationLearningTraining).not.toHaveBeenCalled();
  });

  test('keeps GR00T selectable in IL as an explicit preview-only model', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));
    const policyGroup = screen.getByRole('group', { name: 'Policy model' });
    const grootButton = within(policyGroup).getByRole('button', { name: 'GR00T' });
    await waitFor(() => expect(grootButton).not.toBeDisabled());
    fireEvent.click(grootButton);

    expect(grootButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Imitation Learning' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('groot-architecture-diagram')).toBeInTheDocument();
    const loop = screen.getByTestId('policy-training-loop');
    expect(loop).toHaveAttribute('data-policy-model', 'groot');
    expect(loop).toHaveAttribute('data-fit-content', 'true');
    expect(within(loop).getByTestId('training-replay-buffer-card')).toBeInTheDocument();
    expect(within(loop).getByTestId('groot-imitation-algorithm-card')).toBeInTheDocument();
    expect(within(loop).getByTestId('policy-training-loop-connectors')).toBeInTheDocument();
    expect(within(loop).getByText('Flow-Matching Action Reconstruction')).toBeInTheDocument();
    expect(screen.getByText(/GR00T imitation-learning preview is available/))
      .toBeInTheDocument();
    const startTraining = screen.getByRole('button', { name: 'Start Training' });
    expect(startTraining).toBeDisabled();
    fireEvent.click(startTraining);
    expect(startImitationLearningTraining).not.toHaveBeenCalled();
  });

  test('starts independent ACT critic warm-up with the actor frozen and ordered replay', async () => {
    const onFreshLineageConsumed = jest.fn();
    const { testStore } = renderSection({
      variant: 'workflow',
      forceFreshLineage: true,
      onFreshLineageConsumed,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelections([
        {
          path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 0,
        },
        {
          path: '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 1,
        },
      ]));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/imitation/act/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));
    await waitFor(() => expect(getACTTD3CriticWarmupStatus).toHaveBeenCalled());

    expect(screen.getByRole('button', { name: 'Critic Warm-up' }))
      .toHaveTextContent(/^Critic$/);
    expect(screen.getByRole('button', { name: 'Critic Warm-up' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'TD3' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'TD3' }))
      .toHaveAttribute('aria-pressed', 'false');
    const criticPolicyGroup = screen.getByRole('group', { name: 'Policy model' });
    expect(within(criticPolicyGroup).getByRole('button', {
      name: 'Diffusion Transformer',
    })).not.toBeDisabled();
    expect(within(criticPolicyGroup).getByRole('button', { name: 'GR00T' }))
      .not.toBeDisabled();
    expect(within(criticPolicyGroup).getByRole('button', { name: 'Pi0.5' }))
      .not.toBeDisabled();
    for (const label of [
      'Visual backbone',
      'CVAE encoder',
      'Action Module',
    ]) {
      const actorBlock = screen.getByRole('button', {
        name: new RegExp(`${label}: Frozen`, 'i'),
      });
      expect(actorBlock).toBeDisabled();
      expect(actorBlock).toHaveAttribute('aria-pressed', 'false');
    }
    expect(screen.getByLabelText('ACT actor: Frozen; no gradients')).toBeInTheDocument();
    expect(screen.queryByText('ACT ← maximize Q1')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Critic warm-up batch size')).toHaveValue(4);
    expect(screen.getByLabelText('Critic warm-up total updates')).toHaveValue(5000);
    expect(screen.getByLabelText('Critic warm-up ACT actor mode')).toHaveTextContent('Frozen');
    expect(screen.getByLabelText('Critic checkpoint path')).toHaveTextContent(
      'Resolved by backend under selected ACT policy/critic/latest.pt'
    );
    expect(screen.queryByLabelText(/ACT-TD3 policy RL Epoch/)).not.toBeInTheDocument();

    const criticUpdates = screen.getByLabelText('Critic warm-up total updates');
    await waitFor(() => expect(criticUpdates).not.toBeDisabled());
    fireEvent.change(criticUpdates, {
      target: { value: '1200' },
    });
    const startCritic = await screen.findByRole('button', { name: 'Start Critic Warm-up' });
    await waitFor(() => expect(startCritic).not.toBeDisabled());
    fireEvent.click(startCritic);

    await waitFor(() => expect(startACTTD3CriticWarmup).toHaveBeenCalledWith({
      dataset_path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
      dataset_paths: [
        '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
        '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
      ],
      act_checkpoint: '/workspace/model/lerobot/imitation/act/pretrained_model',
      robot_type: 'ffw_sg2_rev1',
      batch_size: 4,
      critic_updates: 1200,
    }));
    await screen.findByRole('button', { name: 'Warming Critic…' });
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
    expect(startImitationLearningTraining).not.toHaveBeenCalled();
    expect(onFreshLineageConsumed).not.toHaveBeenCalled();
  });

  test('enables the independent Critic method for Diffusion Transformer', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));

    const criticMethod = screen.getByRole('button', { name: 'Critic Warm-up' });
    expect(criticMethod).not.toBeDisabled();
    fireEvent.click(criticMethod);

    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());
    expect(criticMethod).toHaveAttribute('aria-pressed', 'true');
    expect(await screen.findByRole('button', { name: 'Start Critic Warm-up' }))
      .toBeInTheDocument();
  });

  test('shows ACT critic progress and stops only the exact warm-up job', async () => {
    getACTTD3CriticWarmupStatus.mockResolvedValue({
      status: 'running',
      percentage: 40,
      job_id: 'act-critic-job-exact',
      completed_critic_updates: 2000,
      total_critic_updates: 5000,
      critic_loss: 0.012345,
      target_mean: 0.75,
      eta_seconds: 125,
      actor_exactly_unchanged: true,
      checkpoint_path: '/workspace/model/lerobot/imitation/act/pretrained_model/critic/latest.pt',
      act_checkpoint: '/workspace/model/lerobot/imitation/act/pretrained_model',
      batch_size: 4,
    });
    const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelection({
        path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
        version: 'v3.0',
        dataEpoch: 0,
      }));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/imitation/act',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));

    const criticProgress = await screen.findByTestId('training-loss-chart');
    await waitFor(() => {
      expect(within(criticProgress).getByLabelText('Training percentage'))
        .toHaveTextContent('40.0%');
    });
    expect(within(criticProgress).getByText('Training critic')).toBeInTheDocument();
    expect(within(criticProgress).getByLabelText('Training update detail'))
      .toHaveTextContent('Update 2,000/5,000');
    expect(within(criticProgress).getByLabelText('Training ETA'))
      .toHaveTextContent('ETA 2m 05s');
    expect(screen.getByRole('progressbar', { name: 'ACT critic warm-up progress' }))
      .toHaveAttribute('aria-valuenow', '40');
    expect(screen.getByText('Critic loss').parentElement).toHaveTextContent('0.012345');
    expect(screen.getByText('Target mean').parentElement).toHaveTextContent('0.75000');
    expect(within(screen.getByTestId('offline-rl-training-footer')).getByText('Actor').parentElement)
      .toHaveTextContent('Unchanged');
    expect(screen.getByLabelText('Critic checkpoint path')).toHaveTextContent(
      '/workspace/model/lerobot/imitation/act/pretrained_model/critic/latest.pt'
    );
    expect(screen.getByRole('button', { name: 'Reinforcement Learning' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Imitation Learning' })).toBeDisabled();
    expect(within(screen.getByRole('group', { name: 'Policy model' })).getByRole(
      'button', { name: 'Diffusion Transformer' }
    )).toBeDisabled();

    const stopCritic = screen.getByRole('button', { name: 'Stop Critic Warm-up' });
    await waitFor(() => expect(stopCritic).not.toBeDisabled());
    fireEvent.click(stopCritic);
    await waitFor(() => {
      expect(stopACTTD3CriticWarmup).toHaveBeenCalledWith('act-critic-job-exact');
    });
    expect(stopOfflineRLTraining).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  test('hides a critic checkpoint reported for a different ACT policy', async () => {
    getACTTD3CriticWarmupStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'stale-act-critic-job',
      checkpoint_path: '/workspace/model/lerobot/other/pretrained_model/critic/latest.pt',
      act_checkpoint: '/workspace/model/lerobot/other/pretrained_model',
    });
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/current/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));

    const criticCheckpoint = await screen.findByLabelText('Critic checkpoint path');
    await waitFor(() => {
      expect(criticCheckpoint).toHaveTextContent(
        'Saved critic belongs to a different or unverified ACT policy'
      );
    });
    expect(criticCheckpoint).not.toHaveTextContent(
      '/workspace/model/lerobot/other/pretrained_model/critic/latest.pt'
    );
  });

  test('never publishes a completed critic warm-up as a deployable policy', async () => {
    getACTTD3CriticWarmupStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'act-critic-complete',
      completed_critic_updates: 5000,
      total_critic_updates: 5000,
      actor_exactly_unchanged: true,
      checkpoint_path: '/workspace/model/lerobot/imitation/act/pretrained_model/critic/latest.pt',
      act_checkpoint: '/workspace/model/lerobot/imitation/act/pretrained_model',
    });
    const deploymentListener = jest.fn();
    const { testStore } = renderSection({
      variant: 'workflow',
      currentPolicyEpoch: 3,
      onDeploymentStateChange: deploymentListener,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/imitation/act/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));

    await waitFor(() => expect(deploymentListener).toHaveBeenLastCalledWith({
      ready: false,
      modelPath: '',
      serviceType: 'lerobot',
      policyType: 'act',
      rlEpoch: 3,
      lineageMode: 'unchanged',
    }));
    expect(screen.queryByLabelText('ACT-TD3 policy RL Epoch 3 to 4'))
      .not.toBeInTheDocument();
  });

  test('maps Diffusion Transformer exclusively to Flow-SDE PPO and gates Start by readiness', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelection({
        path: '/workspace/lerobot/showroom_lerobot_v30',
        version: 'v3.0',
        dataEpoch: 0,
      }));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));

    expect(await screen.findByTestId('multi-task-dit-architecture-diagram'))
      .toBeInTheDocument();
    expect(screen.getByTestId('flow-sde-ppo-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Diffusion Transformer' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'TD3' }))
      .toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', { name: 'TD3' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent(/backend is not ready/i);
    const gatedStart = await screen.findByRole('button', { name: 'Start Training' });
    expect(gatedStart).toBeDisabled();
    fireEvent.click(gatedStart);
    expect(startOfflineRLTraining).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'TD3' }));
    expect(screen.getByTestId('multi-task-dit-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Diffusion Transformer' }))
      .toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(screen.getByRole('button', { name: 'ACT' }));
    expect(await screen.findByTestId('act-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'ACT' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'TD3-BC loss' }))
      .toHaveAttribute('aria-pressed', 'true');
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'TD3' })).not.toBeDisabled();
    });
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i })).toBeDisabled();
  });

  test('follows supported Inference model changes without aliasing plain Diffusion', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        serviceType: 'lerobot',
        policyType: 'multi_task_dit',
        policyPath: '/workspace/model/multi_task_dit/showroom/pretrained_model',
      }));
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Diffusion Transformer' }))
        .toHaveAttribute('aria-pressed', 'true');
    });
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'TD3' })).toBeDisabled();

    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        serviceType: 'lerobot',
        policyType: 'diffusion',
      }));
    });
    expect(screen.getByRole('button', { name: 'Diffusion Transformer' }))
      .toHaveAttribute('aria-pressed', 'true');

    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        serviceType: 'groot',
        policyType: 'n17',
        policyPath: '/workspace/model/groot/showroom',
      }));
    });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'GR00T' }))
        .toHaveAttribute('aria-pressed', 'true');
    });
    expect(screen.getByRole('button', { name: 'TD3' })).toBeDisabled();
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i })).toBeDisabled();

    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        serviceType: 'lerobot',
        policyType: 'act',
        policyPath: '/workspace/model/act/showroom/pretrained_model',
      }));
    });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'ACT' }))
        .toHaveAttribute('aria-pressed', 'true');
    });
    expect(screen.getByRole('button', { name: 'TD3-BC loss' }))
      .toHaveAttribute('aria-pressed', 'true');
  });

  test.each([
    [InferencePhase.LOADING, 'LOADING'],
    [InferencePhase.INFERENCING, 'INFERENCING'],
    [InferencePhase.PAUSED, 'PAUSED'],
  ])('blocks Flow-SDE PPO while inference phase is %s', async (inferencePhase, phaseName) => {
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({
      status: 'idle',
      percentage: 0,
    });
    const { testStore } = renderSection({
      variant: 'workflow',
      inferencePhase,
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: jest.fn(),
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    const blockedStart = await screen.findByRole('button', { name: 'Start Training' });

    expect(screen.getByRole('alert')).toHaveTextContent(
      `Flow-SDE PPO requires Inference READY (current: ${phaseName}).`
    );
    expect(blockedStart).toBeDisabled();
  });

  test('does not apply the inference-phase guard to ACT-TD3', async () => {
    const { testStore } = renderSection({
      variant: 'workflow',
      inferencePhase: InferencePhase.INFERENCING,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelection({
        path: '/workspace/lerobot/showroom_lerobot_v30',
        version: 'v3.0',
        dataEpoch: 0,
      }));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/base/pretrained_model',
      }));
    });

    expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();
    expect(screen.queryByText(/requires Inference READY/)).not.toBeInTheDocument();
  });

  test('starts dedicated on-policy Flow-SDE PPO without requiring a LeRobot dataset', async () => {
    const startFlowSDEPPO = jest.fn().mockResolvedValue({
      status: 'starting',
      percentage: 0,
      job_id: 'flow-sde-job-1',
    });
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({ status: 'idle', percentage: 0 });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: startFlowSDEPPO,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      // Flow-SDE PPO is live on-policy: even an unrelated v2.1 selection and
      // an active converter must not become training inputs or block Start.
      testStore.dispatch(setOfflineRLDatasetSelection({
        path: '/workspace/lerobot/ignored_lerobot_v21',
        version: 'v2.1',
        dataEpoch: 0,
      }));
      testStore.dispatch(setConversionStatus({ status: 'running' }));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });

    const ppoButton = screen.getByRole('button', { name: /PPO.*Flow-SDE/i });
    expect(ppoButton).toBeDisabled();
    fireEvent.click(ppoButton);
    expect(screen.getByRole('button', { name: 'ACT' }))
      .toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(ppoButton).not.toBeDisabled());
    expect(ppoButton).toHaveAttribute('aria-pressed', 'true');
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();
    });
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startFlowSDEPPO).toHaveBeenCalledWith({
      policy_type: 'multi_task_dit',
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
    }));
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('starts independent Diffusion critic warm-up with checked replay roots', async () => {
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({ status: 'idle', percentage: 0 });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: jest.fn(),
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelections([
        {
          path: '/workspace/lerobot/showroom_test_3_success_v30',
          version: 'v3.0',
          dataEpoch: 1,
        },
        {
          path: '/workspace/lerobot/data_epoch_0002/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 2,
        },
      ]));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());
    const criticPanel = screen.getByRole('region', {
      name: 'Diffusion Policy critic warm-up',
    });
    expect(criticPanel).toBeInTheDocument();
    expect(within(criticPanel).getByText('Value initialization')).toBeInTheDocument();
    expect(within(criticPanel).getByText('Critic Warm-up')).toBeInTheDocument();
    expect(within(criticPanel).getByText('Diffusion Policy')).toBeInTheDocument();
    expect(within(criticPanel).getByText('Value Critic Network')).toBeInTheDocument();
    expect(within(criticPanel).getByText(/State value V\(s\)/)).toBeInTheDocument();
    expect(within(criticPanel).getByLabelText('Critic warm-up Diffusion policy mode'))
      .toHaveTextContent('Frozen');
    expect(within(criticPanel).getByText('Trainable')).toBeInTheDocument();
    expect(getACTTD3CriticWarmupStatus).not.toHaveBeenCalled();

    expect(screen.getByLabelText('Critic warm-up steps')).toHaveValue(2000);
    expect(screen.getByLabelText('Critic warm-up batch size')).toHaveValue(8);
    expect(screen.getByLabelText('Critic warm-up value learning rate')).toHaveValue(0.0001);
    expect(screen.getByLabelText('Critic warm-up discount')).toHaveValue(0.99);

    const trainCritic = await screen.findByRole('button', { name: 'Start Critic Warm-up' });
    await waitFor(() => expect(trainCritic).not.toBeDisabled());
    fireEvent.click(trainCritic);

    await waitFor(() => expect(startFlowSDEPPOValueWarmup).toHaveBeenCalledWith({
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      dataset_paths: [
        '/workspace/lerobot/showroom_test_3_success_v30',
        '/workspace/lerobot/data_epoch_0002/task_lerobot_v30',
      ],
      policy_type: 'multi_task_dit',
      task_instruction: 'Pick up the jelly bag',
      steps: 2000,
      batch_size: 8,
      value_learning_rate: 0.0001,
      discount: 0.99,
    }));
    expect(startACTTD3CriticWarmup).not.toHaveBeenCalled();
  });

  test('shows completed critic warm-up progress, critic loss, ETA, and bundle path', async () => {
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      step: 2000,
      total_steps: 2000,
      value_loss: 0.012345,
      eta_seconds: 0,
      job_id: 'warmup-job-1',
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      task_instruction: 'Pick up the jelly bag',
      bundle_path: '/workspace/checkpoint/multi_task_dit/value_warmup/warmup-job-1',
    });
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({ status: 'idle', percentage: 0 });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: jest.fn(),
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());

    expect(await screen.findByText(/Complete · 100% · ETA 0s/)).toBeInTheDocument();
    expect(screen.getByText(/Step 2,000\/2,000 · Critic loss 0.012345/)).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Diffusion critic warm-up progress' }))
      .toHaveAttribute('aria-valuenow', '100');
    expect(screen.getByLabelText('Critic warm-up bundle path')).toHaveTextContent(
      '/workspace/checkpoint/multi_task_dit/value_warmup/warmup-job-1'
    );
  });

  test('keeps the PPO screen free of a nested critic warm-up toggle', async () => {
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({ status: 'idle', percentage: 0 });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: jest.fn(),
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());

    expect(screen.queryByRole('group', { name: 'Critic warm-up' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Train Critic' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();
  });

  test('automatically submits the compatible completed warm-up bundle to online PPO', async () => {
    const bundlePath = '/workspace/checkpoint/multi_task_dit/value_warmup/warmup-job-1';
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'aa1bbac0c479494d8cac9fdcdb1bc683',
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      task_instruction: 'Pick up the jelly bag',
      bundle_path: bundlePath,
      checkpoint_path: `${bundlePath}/training_state/value_warmup.pt`,
      model_path: `${bundlePath}/pretrained_model`,
    });
    const startFlowSDEPPO = jest.fn().mockResolvedValue({
      status: 'starting',
      percentage: 0,
      job_id: 'flow-sde-job-1',
    });
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({ status: 'idle', percentage: 0 });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: startFlowSDEPPO,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();
    });
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startFlowSDEPPO).toHaveBeenCalledWith({
      policy_type: 'multi_task_dit',
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      value_warmup_bundle: bundlePath,
      task_instruction: 'Pick up the jelly bag',
    }));
  });

  test('prefers the latest compatible PPO trainer state over the offline critic warm-up', async () => {
    const basePolicy = '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model';
    const priorModel = '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/ppo-job-2/pretrained_model';
    const priorTrainer = '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/ppo-job-2/training_state/trainer_state.pt';
    const warmupBundle = '/workspace/checkpoint/multi_task_dit/value_warmup/warmup-job-1';
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'warmup-job-1',
      policy_checkpoint: basePolicy,
      task_instruction: 'Pick up the jelly bag',
      bundle_path: warmupBundle,
      checkpoint_path: `${warmupBundle}/training_state/value_warmup.pt`,
      model_path: `${warmupBundle}/pretrained_model`,
    });
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'ppo-job-2',
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/ppo-job-1/pretrained_model',
      lineage_policy_checkpoint: basePolicy,
      task_instruction: 'Pick up the jelly bag',
      checkpoint_path: priorTrainer,
      model_path: priorModel,
    });
    const startFlowSDEPPO = jest.fn().mockResolvedValue({
      status: 'running',
      percentage: 0,
      job_id: 'ppo-job-3',
    });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: startFlowSDEPPO,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: basePolicy,
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startFlowSDEPPO).toHaveBeenCalledWith({
      policy_type: 'multi_task_dit',
      policy_checkpoint: priorModel,
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      resume_checkpoint: priorTrainer,
      task_instruction: 'Pick up the jelly bag',
    }));
    expect(startFlowSDEPPO.mock.calls[0][0]).not.toHaveProperty('value_warmup_bundle');
  });

  test('starts PPO with a fresh critic when the recovered PPO has no trainer artifacts', async () => {
    const basePolicy = '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model';
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'ppo-job-2',
      policy_checkpoint: basePolicy,
      lineage_policy_checkpoint: basePolicy,
      task_instruction: 'Pick up the jelly bag',
      checkpoint_path: '',
      model_path: '',
    });
    const startFlowSDEPPO = jest.fn().mockResolvedValue({
      status: 'running',
      percentage: 0,
      job_id: 'ppo-job-fresh',
    });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: startFlowSDEPPO,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: basePolicy,
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());
    const start = await screen.findByRole('button', { name: 'Start Training' });
    await waitFor(() => expect(start).not.toBeDisabled());
    fireEvent.click(start);

    await waitFor(() => expect(startFlowSDEPPO).toHaveBeenCalledWith({
      policy_type: 'multi_task_dit',
      policy_checkpoint: basePolicy,
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
    }));
    expect(startFlowSDEPPO.mock.calls[0][0]).not.toHaveProperty('resume_checkpoint');
    expect(startFlowSDEPPO.mock.calls[0][0]).not.toHaveProperty('value_warmup_bundle');
  });

  test.each([
    [
      'policy lineage',
      {
        lineage_policy_checkpoint: '/workspace/checkpoint/multi_task_dit/another/pretrained_model',
        policy_checkpoint: '/workspace/checkpoint/multi_task_dit/another/pretrained_model',
        task_instruction: 'Pick up the jelly bag',
      },
      /different policy lineage/i,
    ],
    [
      'task instruction',
      {
        lineage_policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        task_instruction: 'Move the jelly bag',
      },
      /different task instruction/i,
    ],
  ])('ignores a stale completed PPO %s and starts with a fresh critic', async (
    _label,
    lineage,
    _expectedMessage
  ) => {
    const basePolicy = '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model';
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'stale-ppo-job',
      checkpoint_path: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/stale/training_state/trainer_state.pt',
      model_path: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/stale/pretrained_model',
      ...lineage,
    });
    const startFlowSDEPPO = jest.fn();
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: startFlowSDEPPO,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: basePolicy,
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());
    const start = screen.getByRole('button', { name: 'Start Training' });
    await waitFor(() => expect(start).not.toBeDisabled());
    fireEvent.click(start);

    await waitFor(() => expect(startFlowSDEPPO).toHaveBeenCalledWith({
      policy_type: 'multi_task_dit',
      policy_checkpoint: basePolicy,
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
    }));
    expect(startFlowSDEPPO.mock.calls[0][0]).not.toHaveProperty('resume_checkpoint');
    expect(startFlowSDEPPO.mock.calls[0][0]).not.toHaveProperty('value_warmup_bundle');
  });

  test('falls back to a compatible warm-up when the recovered PPO belongs to another lineage', async () => {
    const basePolicy = '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model';
    const warmupBundle = '/workspace/checkpoint/multi_task_dit/value_warmup/compatible-warmup';
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'compatible-warmup',
      policy_checkpoint: basePolicy,
      task_instruction: 'Pick up the jelly bag',
      bundle_path: warmupBundle,
      checkpoint_path: `${warmupBundle}/training_state/value_warmup.pt`,
      model_path: `${warmupBundle}/pretrained_model`,
    });
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'unrelated-ppo-job',
      lineage_policy_checkpoint: '/workspace/checkpoint/multi_task_dit/old/pretrained_model',
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/old/pretrained_model',
      task_instruction: 'Pick up the jelly bag',
      checkpoint_path: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/old/training_state/trainer_state.pt',
      model_path: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/old/pretrained_model',
    });
    const startFlowSDEPPO = jest.fn().mockResolvedValue({
      status: 'running',
      percentage: 0,
      job_id: 'new-ppo-job',
    });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: startFlowSDEPPO,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: basePolicy,
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    const start = screen.getByRole('button', { name: 'Start Training' });
    expect(start).not.toBeDisabled();
    fireEvent.click(start);

    await waitFor(() => expect(startFlowSDEPPO).toHaveBeenCalledWith({
      policy_type: 'multi_task_dit',
      policy_checkpoint: basePolicy,
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      value_warmup_bundle: warmupBundle,
      task_instruction: 'Pick up the jelly bag',
    }));
    expect(startFlowSDEPPO.mock.calls[0][0]).not.toHaveProperty('resume_checkpoint');
  });

  test.each([
    [
      'policy checkpoint',
      {
        policy_checkpoint: '/workspace/checkpoint/multi_task_dit/another/pretrained_model',
        task_instruction: 'Pick up the jelly bag',
      },
      /different policy checkpoint/i,
    ],
    [
      'task instruction',
      {
        policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        task_instruction: 'Move the jelly bag',
      },
      /different task instruction/i,
    ],
  ])('ignores a stale warm-up with a different %s and starts with a fresh critic', async (
    _label,
    lineage,
    _message
  ) => {
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'stale-warmup-job',
      bundle_path: '/workspace/checkpoint/multi_task_dit/value_warmup/stale-warmup-job',
      ...lineage,
    });
    const startFlowSDEPPO = jest.fn();
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({ status: 'idle', percentage: 0 });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: startFlowSDEPPO,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    const start = screen.getByRole('button', { name: 'Start Training' });
    await waitFor(() => expect(start).not.toBeDisabled());
    fireEvent.click(start);

    await waitFor(() => expect(startFlowSDEPPO).toHaveBeenCalledWith({
      policy_type: 'multi_task_dit',
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
    }));
    expect(startFlowSDEPPO.mock.calls[0][0]).not.toHaveProperty('resume_checkpoint');
    expect(startFlowSDEPPO.mock.calls[0][0]).not.toHaveProperty('value_warmup_bundle');
  });

  test('stops only the running critic warm-up job', async () => {
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({
      status: 'running',
      percentage: 40,
      job_id: 'warmup-job-exact',
      step: 800,
      total_steps: 2000,
    });
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({ status: 'idle', percentage: 0 });
    const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: jest.fn(),
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    fireEvent.click(screen.getByRole('button', { name: 'Critic Warm-up' }));
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());

    const stopButton = await screen.findByRole('button', { name: 'Stop Critic Warm-up' });
    await waitFor(() => expect(stopButton).not.toBeDisabled());
    fireEvent.click(stopButton);

    await waitFor(() => {
      expect(stopFlowSDEPPOValueWarmup).toHaveBeenCalledWith('warmup-job-exact');
    });
    confirmSpy.mockRestore();
  });

  test.each([
    ['Success', 'success'],
    ['Fail', 'fail'],
    ['Cancel', 'cancel'],
  ])('submits %s to the exact running Flow-SDE PPO job', async (label, outcome) => {
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({
      status: 'running',
      percentage: 25,
      job_id: 'flow-sde-job-live-1',
      awaiting_outcome: true,
    });
    const submitFlowSDEPPOOutcome = jest.fn().mockImplementation((jobId) => (
      Promise.resolve({
        status: 'running',
        percentage: 25,
        job_id: jobId,
        awaiting_outcome: false,
      })
    ));
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: jest.fn(),
      onStopFlowSDEPPO: jest.fn(),
      onSubmitFlowSDEPPOOutcome: submitFlowSDEPPOOutcome,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    expect(await screen.findByRole('group', { name: 'Flow-SDE episode outcome' }))
      .toBeInTheDocument();

    const outcomeButton = screen.getByRole('button', { name: label });
    expect(outcomeButton).not.toBeDisabled();
    fireEvent.click(outcomeButton);
    await waitFor(() => expect(submitFlowSDEPPOOutcome).toHaveBeenCalledWith(
      'flow-sde-job-live-1',
      outcome
    ));
  });

  test('unlocks labeling only when a cancelled episode reaches the next retry', async () => {
    jest.useFakeTimers();
    const getFlowSDEPPOStatus = jest.fn()
      .mockResolvedValueOnce({
        status: 'running',
        percentage: 20,
        job_id: 'flow-sde-retry-job',
        episode: 1,
        awaiting_outcome: true,
      })
      .mockResolvedValue({
        status: 'running',
        percentage: 20,
        job_id: 'flow-sde-retry-job',
        episode: 2,
        awaiting_outcome: true,
      });
    const submitFlowSDEPPOOutcome = jest.fn().mockResolvedValue({
      status: 'running',
      percentage: 20,
      job_id: 'flow-sde-retry-job',
      episode: 1,
      phase: 'retrying',
      awaiting_outcome: false,
    });
    const { testStore } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: jest.fn(),
      onStopFlowSDEPPO: jest.fn(),
      onSubmitFlowSDEPPOOutcome: submitFlowSDEPPOOutcome,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      }));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));

    const cancelButton = await screen.findByRole('button', { name: 'Cancel' });
    expect(cancelButton).not.toBeDisabled();
    await act(async () => {
      fireEvent.click(cancelButton);
      await Promise.resolve();
    });
    expect(submitFlowSDEPPOOutcome).toHaveBeenCalledWith(
      'flow-sde-retry-job',
      'cancel'
    );
    expect(screen.getByRole('button', { name: 'Success' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fail' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();

    await act(async () => {
      jest.advanceTimersByTime(2000);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(getFlowSDEPPOStatus).toHaveBeenCalledTimes(2);
    expect(screen.getByRole('button', { name: 'Success' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Fail' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).not.toBeDisabled();
  });

  test('publishes a completed MultiTaskDiT policy as lerobot:multi_task_dit', async () => {
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({
      status: 'completed',
      percentage: 100,
      model_path: '/workspace/model/multi_task_dit/ppo/pretrained_model',
    });
    const deploymentListener = jest.fn();
    renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus,
      onStartFlowSDEPPO: jest.fn(),
      onStopFlowSDEPPO: jest.fn(),
      onSubmitFlowSDEPPOOutcome: jest.fn(),
      onDeploymentStateChange: deploymentListener,
    });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));

    await waitFor(() => expect(deploymentListener).toHaveBeenLastCalledWith({
      ready: true,
      modelPath: '/workspace/model/multi_task_dit/ppo/pretrained_model',
      serviceType: 'lerobot',
      policyType: 'multi_task_dit',
      rlEpoch: 1,
      lineageMode: 'advance',
    }));
  });

  test('uses ACT imitation learning with editable full-ACT settings by default', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    expect(await screen.findByTestId('act-imitation-learning-diagram')).toBeInTheDocument();
    expect(screen.getByLabelText('Imitation steps')).toHaveValue(80000);
    expect(screen.getByLabelText('Imitation batch size')).toHaveValue(8);
    expect(screen.getByLabelText('Imitation save frequency')).toHaveValue(10000);
    expect(screen.getByLabelText('Imitation action chunk')).toHaveValue(30);
    expect(screen.getByRole('button', { name: 'TD3' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'SAC' })).toBeDisabled();

    await waitFor(() => expect(screen.getByRole('button', {
      name: /Visual backbone: Trainable/i,
    })).not.toBeDisabled());

    [
      /Visual backbone: Trainable/i,
      /CVAE encoder: Trainable/i,
      /Action Module: Trainable/i,
    ].forEach((name) => {
      const block = screen.getByRole('button', { name });
      expect(block).not.toBeDisabled();
      expect(block).toHaveAttribute('aria-pressed', 'true');
    });

    fireEvent.click(screen.getByRole('button', { name: /CVAE encoder: Trainable/i }));
    expect(screen.getByRole('button', {
      name: /CVAE encoder: Frozen/i,
    })).toHaveAttribute('aria-pressed', 'false');
  });

  test('starts ACT imitation learning on every selected Data Epoch without a base checkpoint', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelections([
        {
          path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 0,
        },
        {
          path: '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 1,
        },
      ]));
      testStore.dispatch(setInferenceTaskInfo({ policyPath: '' }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));
    const cvaeBlock = screen.getByRole('button', { name: /CVAE encoder: Trainable/i });
    await waitFor(() => expect(cvaeBlock).not.toBeDisabled());
    fireEvent.click(cvaeBlock);
    fireEvent.change(screen.getByLabelText('Imitation action chunk'), {
      target: { value: '24' },
    });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startImitationLearningTraining).toHaveBeenCalledWith({
      dataset_path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
      dataset_paths: [
        '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
        '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
      ],
      policy_type: 'act',
      steps: 80000,
      batch_size: 8,
      save_freq: 10000,
      chunk_size: 24,
      trainable_groups: [
        'visual_backbone',
        'transformer_encoder',
        'action_decoder',
      ],
    }));
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('blocks all-frozen and CVAE-only ACT imitation-learning configurations', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));
    const visualBlock = screen.getByRole('button', { name: /Visual backbone/i });
    await waitFor(() => expect(visualBlock).not.toBeDisabled());
    fireEvent.click(visualBlock);
    fireEvent.click(screen.getByRole('button', { name: /Action Module/i }));

    expect(screen.getByRole('alert')).toHaveTextContent(/CVAE-only is not supported/i);
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: /CVAE encoder/i }));
    expect(screen.getByRole('alert')).toHaveTextContent(/At least one ACT network block/i);
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(startImitationLearningTraining).not.toHaveBeenCalled();
  });

  test('keeps Diffusion Transformer selected for flow-matching imitation learning', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelection({
        path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
        version: 'v3.0',
        dataEpoch: 0,
      }));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '',
        taskInstruction: ['Pick up the jelly bag'],
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Imitation Learning' }))
      .not.toBeDisabled());
    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    expect(await screen.findByText('Diffusion Transformer imitation learning'))
      .toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Diffusion Transformer' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('multi-task-dit-architecture-diagram')).toBeInTheDocument();
    const loop = screen.getByTestId('policy-training-loop');
    expect(loop).toHaveAttribute('data-policy-model', 'multi_task_dit');
    expect(loop).toHaveAttribute('data-fit-content', 'true');
    expect(within(loop).getByTestId('training-replay-buffer-card')).toBeInTheDocument();
    expect(within(loop).getByTestId('multi_task_dit-imitation-algorithm-card'))
      .toBeInTheDocument();
    expect(within(loop).getByTestId('policy-training-loop-connectors')).toBeInTheDocument();
    expect(within(loop).getByText('Flow-Matching Reconstruction')).toBeInTheDocument();
    expect(screen.getByLabelText('Imitation action chunk')).toHaveValue(16);
    expect(screen.getByLabelText('Imitation action chunk')).toBeDisabled();
    expect(screen.getByText(/Supervised flow-matching/i)).toBeInTheDocument();
    expect(screen.getByText(/no reward or outcome labels required/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'GR00T' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Pi0.5' })).not.toBeDisabled();
    expect(await screen.findByRole('button', { name: 'Start Training' }))
      .not.toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    await waitFor(() => expect(startImitationLearningTraining).toHaveBeenCalledWith({
      dataset_path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
      dataset_paths: ['/workspace/lerobot/data_epoch_0000/task_lerobot_v30'],
      policy_type: 'multi_task_dit',
      steps: 80000,
      batch_size: 8,
      save_freq: 10000,
      chunk_size: 16,
      task_instruction: 'Pick up the jelly bag',
    }));
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('does not silently switch an unsupported policy to ACT when imitation learning is selected', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'GR00T' }));
    expect(screen.getByRole('button', { name: 'GR00T' }))
      .toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    expect(screen.getByRole('button', { name: 'GR00T' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'ACT' }))
      .toHaveAttribute('aria-pressed', 'false');
    expect(await screen.findByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(screen.getByRole('alert')).toHaveTextContent(
      /GR00T imitation-learning preview is available.*backend is not connected/i
    );
  });

  test('validates imitation learning settings independently from TD3', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelection({
        path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
        version: 'v3.0',
        dataEpoch: 0,
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));
    const saveFrequency = await screen.findByLabelText('Imitation save frequency');
    await waitFor(() => expect(saveFrequency).not.toBeDisabled());
    fireEvent.change(saveFrequency, {
      target: { value: '90000' },
    });
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start Training' }))
      .toBeDisabled());
    expect(screen.getByRole('button', { name: 'TD3' })).toBeDisabled();
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i })).toBeDisabled();

    fireEvent.change(saveFrequency, {
      target: { value: '10000' },
    });
    await waitFor(() => expect(screen.getByRole('button', { name: 'Start Training' }))
      .not.toBeDisabled());
  });

  test('shows ACT imitation losses and locks the method while stopping the exact job', async () => {
    getImitationLearningStatus.mockResolvedValue({
      status: 'running',
      percentage: 25,
      step: 20000,
      total_steps: 80000,
      loss: 0.12,
      l1_loss: 0.08,
      kld_loss: 0.004,
      eta_seconds: 125,
      batch_size: 8,
      save_freq: 10000,
      job_id: 'il-job-visible-123',
    });
    stopImitationLearningTraining.mockResolvedValue({
      status: 'running',
      percentage: 25,
      job_id: 'il-job-visible-123',
      message: 'Stopping ACT imitation learning',
    });
    const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    const imitationProgress = await screen.findByTestId('training-loss-chart');
    await waitFor(() => {
      expect(within(imitationProgress).getByText('Training')).toBeInTheDocument();
      expect(within(imitationProgress).getByLabelText('Training percentage'))
        .toHaveTextContent('25.0%');
      expect(within(imitationProgress).getByLabelText('Training update detail'))
        .toHaveTextContent('Step 20,000/80,000');
      expect(within(imitationProgress).getByLabelText('Training ETA'))
        .toHaveTextContent('ETA 2m 05s');
      expect(within(imitationProgress).getByText('Total loss').parentElement)
        .toHaveTextContent('0.12000');
      expect(within(imitationProgress).getByText('L1 loss').parentElement)
        .toHaveTextContent('0.080000');
      expect(within(imitationProgress).getByText('KLD loss').parentElement)
        .toHaveTextContent('0.0040000');
    });
    expect(screen.getByRole('button', { name: 'Imitation Learning' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Reinforcement Learning' })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Stop Training' }));
    await waitFor(() => {
      expect(stopImitationLearningTraining).toHaveBeenCalledWith('il-job-visible-123');
    });
    expect(stopOfflineRLTraining).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  test('publishes a completed imitation policy through the existing deployment callback', async () => {
    getImitationLearningStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      step: 80000,
      total_steps: 80000,
      model_path: '/workspace/model/lerobot/imitation/act/checkpoints/080000/pretrained_model',
    });
    const deploymentListener = jest.fn();

    renderSection({
      variant: 'workflow',
      onDeploymentStateChange: deploymentListener,
      currentPolicyEpoch: 3,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    await waitFor(() => expect(deploymentListener).toHaveBeenLastCalledWith({
      ready: true,
      modelPath: '/workspace/model/lerobot/imitation/act/checkpoints/080000/pretrained_model',
      serviceType: 'lerobot',
      policyType: 'act',
      rlEpoch: 0,
      lineageMode: 'new',
    }));
  });

  test('publishes completed Diffusion Transformer imitation as multi_task_dit', async () => {
    getImitationLearningStatus.mockResolvedValue({
      status: 'completed',
      policy_type: 'multi_task_dit',
      percentage: 100,
      step: 80000,
      total_steps: 80000,
      loss: 0.012,
      model_path: '/workspace/model/lerobot/imitation/multi_task_dit/checkpoints/080000/pretrained_model',
    });
    const deploymentListener = jest.fn();

    renderSection({
      variant: 'workflow',
      onDeploymentStateChange: deploymentListener,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    await waitFor(() => expect(deploymentListener).toHaveBeenLastCalledWith({
      ready: true,
      modelPath: '/workspace/model/lerobot/imitation/multi_task_dit/checkpoints/080000/pretrained_model',
      serviceType: 'lerobot',
      policyType: 'multi_task_dit',
      rlEpoch: 0,
      lineageMode: 'new',
    }));
  });

  test('shows Diffusion Transformer imitation flow loss without ACT-only losses', async () => {
    getImitationLearningStatus.mockResolvedValue({
      status: 'running',
      policy_type: 'multi_task_dit',
      percentage: 25,
      step: 20000,
      total_steps: 80000,
      loss: 0.012,
      eta_seconds: 125,
      batch_size: 1,
      save_freq: 10000,
      job_id: 'dit-il-job-visible-123',
    });

    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    await screen.findByTestId('multi-task-dit-architecture-diagram');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Imitation Learning' }))
      .not.toBeDisabled());
    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    const imitationProgress = await screen.findByTestId('training-loss-chart');
    expect(within(imitationProgress).getByText('Training')).toBeInTheDocument();
    expect(within(imitationProgress).getByLabelText('Training percentage'))
      .toHaveTextContent('25.0%');
    expect(within(imitationProgress).getByLabelText('Training update detail'))
      .toHaveTextContent('Step 20,000/80,000');
    expect(within(imitationProgress).getByLabelText('Training ETA'))
      .toHaveTextContent('ETA 2m 05s');
    expect(screen.getByText('Flow loss').parentElement).toHaveTextContent('0.012000');
    expect(screen.queryByText('L1 loss')).not.toBeInTheDocument();
    expect(screen.queryByText('KLD loss')).not.toBeInTheDocument();
  });

  test('uses the ACT-style Policy, Replay Buffer, and Algorithm loop for every other policy', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    const cases = [
      {
        button: 'Diffusion Transformer',
        model: 'multi_task_dit',
        policyDiagram: 'multi-task-dit-architecture-diagram',
        algorithmDiagram: 'flow-sde-ppo-architecture-diagram',
      },
      {
        button: 'GR00T',
        model: 'groot',
        policyDiagram: 'groot-architecture-diagram',
        algorithmDiagram: 'rlt-architecture-diagram',
      },
      {
        button: 'Pi0.5',
        model: 'pi05',
        policyDiagram: 'pi05-architecture-diagram',
        algorithmDiagram: 'rlt-architecture-diagram',
      },
    ];

    for (const item of cases) {
      fireEvent.click(screen.getByRole('button', { name: item.button }));
      const loop = await screen.findByTestId('policy-training-loop');
      await waitFor(() => expect(loop).toHaveAttribute('data-policy-model', item.model));

      const policyStage = within(loop).getByTestId('training-policy-stage');
      const replayStage = within(loop).getByTestId('training-replay-buffer-card');
      const algorithmStage = within(loop).getByTestId('training-algorithm-stage');
      expect(policyStage).toContainElement(screen.getByTestId(item.policyDiagram));
      expect(algorithmStage).toContainElement(screen.getByTestId(item.algorithmDiagram));
      expect(loop).toContainElement(replayStage);
      expect(within(loop).getAllByTestId('training-replay-buffer-card')).toHaveLength(1);

      const progressMetrics = within(screen.getByTestId('offline-rl-training-footer'))
        .getByTestId('training-progress-metrics');
      if (item.model === 'multi_task_dit') {
        expect(within(progressMetrics).getByText('Critic loss')).toBeInTheDocument();
        expect(within(progressMetrics).queryByText('Value loss')).not.toBeInTheDocument();
      } else {
        expect(progressMetrics).toHaveClass('grid-cols-2');
        expect(within(progressMetrics).getByText('Critic loss')).toBeInTheDocument();
        expect(within(progressMetrics).getByText('Action MLP loss')).toBeInTheDocument();
        expect(within(progressMetrics).queryByText('Action chunk')).not.toBeInTheDocument();
      }
    }
  });

  test('previews GR00T and Pi0.5 diagrams without calling the ACT-TD3 backend', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetPath('/workspace/lerobot/task_lerobot_v30'));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/base/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'GR00T' }));
    expect(screen.getByTestId('groot-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'GR00T' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent(
      /GR00T \+ RLT architecture.*backend is not connected/i
    );
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'TD3' })).toBeDisabled();
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    expect(startOfflineRLTraining).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Pi0.5' }));
    expect(screen.getByTestId('pi05-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Pi0.5' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent(
      /Pi0.5 \+ RLT architecture.*backend is not connected/i
    );
    expect(screen.getByRole('alert')).toHaveTextContent(
      /Pi0.5-compatible RLT checkpoint is required.*GR00T-only/i
    );
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    expect(startOfflineRLTraining).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'ACT' }));
    expect(screen.getByTestId('act-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'ACT' }))
      .toHaveAttribute('aria-pressed', 'true');
  });

  test('enables RLT only for GR00T and Pi0.5 and exposes both trainable RLT blocks', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    const rltButton = screen.getByRole('button', { name: 'RLT' });
    expect(rltButton).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'GR00T' }));
    await waitFor(() => expect(rltButton).not.toBeDisabled());
    expect(rltButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('groot-architecture-diagram'))
      .toHaveAttribute('data-architecture-mode', 'all-frozen');
    const rltLoop = screen.getByTestId('policy-training-loop');
    const workflow = screen.getByTestId('offline-rl-workflow-training');
    const trainingFooter = screen.getByTestId('offline-rl-training-footer');
    expect(screen.getByTestId('rlt-architecture-diagram')).toBeInTheDocument();
    expect(rltLoop).toHaveAttribute('data-fit-content', 'true');
    expect(rltLoop).toHaveClass('self-start');
    expect(workflow).toHaveClass('flex-none');
    expect(workflow).toHaveClass('grid-rows-[auto_auto_auto]');
    expect(trainingFooter).toHaveClass('mt-3');
    expect(trainingFooter).not.toHaveClass('mt-auto');
    expect(screen.getByTestId('training-algorithm-card')).toHaveClass('h-fit');
    expect(screen.getByTestId('training-algorithm-card')).not.toHaveClass('h-full');
    expect(screen.getByText('Action MLP loss')).toBeInTheDocument();
    const grootProgressMetrics = within(trainingFooter)
      .getByTestId('training-progress-metrics');
    expect(grootProgressMetrics).toHaveClass('grid-cols-2');
    expect(within(grootProgressMetrics).queryByText('Action chunk')).not.toBeInTheDocument();
    expect(screen.getByRole('button', {
      name: 'RL Token Encoder: Frozen; make trainable',
    })).toBeInTheDocument();
    expect(screen.getByRole('button', {
      name: 'Action MLP: Trainable; freeze',
    })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', {
      name: 'RL Token Encoder: Frozen; make trainable',
    }));
    expect(screen.getByRole('button', {
      name: 'RL Token Encoder: Trainable; freeze',
    })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', {
      name: 'Action MLP: Trainable; freeze',
    })).toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(screen.getByRole('button', { name: 'Pi0.5' }));
    await waitFor(() => expect(rltButton).not.toBeDisabled());
    expect(rltButton).toHaveAttribute('aria-pressed', 'true');
    const piProgressMetrics = within(trainingFooter)
      .getByTestId('training-progress-metrics');
    expect(piProgressMetrics).toHaveClass('grid-cols-2');
    expect(within(piProgressMetrics).queryByText('Action chunk')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'ACT' }));
    await waitFor(() => expect(rltButton).toBeDisabled());
    expect(screen.queryByTestId('rlt-architecture-diagram')).not.toBeInTheDocument();
  });

  test('submits the selected ACT trainable groups from the workflow graph', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetPath('/workspace/lerobot/task_lerobot_v30'));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/base/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: /CVAE encoder/i }));
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        actor_trainable_groups: [
          'visual_backbone',
          'transformer_encoder',
          'action_decoder',
        ],
      })
    ));
  });

  test('never submits a hidden Redux checkpoint from the compact workflow', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetPath('/workspace/lerobot/task_lerobot_v30'));
      testStore.dispatch(setOfflineRLCheckpointPath('/workspace/model/stale/act_td3.pt'));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/base/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
        parent_checkpoint: '',
      })
    ));
  });

  test('auto-resumes a completed workflow round with its server-reported lineage', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'completed',
      algorithm: 'td3',
      actor_objective: 'td3_bc',
      percentage: 100,
      dataset_path: '/workspace/lerobot/task_lerobot_v30',
      dataset_paths: ['/workspace/lerobot/task_lerobot_v30'],
      act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
      checkpoint_path: '/workspace/model/round1/training_state/act_td3.pt',
      model_path: '/workspace/model/round1/pretrained_model',
      round_index: 1,
    });
    const deploymentListener = jest.fn();
    const { testStore } = renderSection({
      variant: 'workflow',
      onDeploymentStateChange: deploymentListener,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelections([
        {
          path: '/workspace/lerobot/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 0,
        },
        {
          path: '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 1,
        },
      ]));
      // Deploying the result changes this inference path, but must not change
      // the immutable base ACT checkpoint used for cumulative training.
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/round1/pretrained_model',
      }));
    });

    await waitFor(() => expect(deploymentListener).toHaveBeenLastCalledWith({
      ready: true,
      modelPath: '/workspace/model/round1/pretrained_model',
      serviceType: 'lerobot',
      policyType: 'act',
      rlEpoch: 1,
      lineageMode: 'advance',
    }));
    expect(screen.getAllByText('Policy').length).toBeGreaterThan(0);
    expect(screen.queryByText('Checkpoint')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        dataset_paths: [
          '/workspace/lerobot/task_lerobot_v30',
          '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
        ],
        act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
        parent_checkpoint: '/workspace/model/round1/training_state/act_td3.pt',
      })
    ));
  });

  test('starts a fresh lineage when the selected TD3 loss differs from the completed round', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'completed',
      algorithm: 'td3',
      actor_objective: 'td3_bc',
      percentage: 100,
      dataset_path: '/workspace/lerobot/task_lerobot_v30',
      dataset_paths: ['/workspace/lerobot/task_lerobot_v30'],
      act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
      checkpoint_path: '/workspace/model/round1/training_state/act_td3.pt',
      model_path: '/workspace/model/round1/pretrained_model',
      round_index: 1,
    });
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelections([
        {
          path: '/workspace/lerobot/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 0,
        },
        {
          path: '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 1,
        },
      ]));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/round1/pretrained_model',
      }));
    });

    const pureTD3Option = screen.getByRole('button', { name: 'TD3 loss' });
    fireEvent.click(pureTD3Option);
    expect(pureTD3Option).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        algorithm: 'td3',
        actor_objective: 'td3',
        act_checkpoint: '/workspace/model/round1/pretrained_model',
        parent_checkpoint: '',
      })
    ));
  });

  test('shows policy RL Epoch separately from critic replay progress', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'running',
      algorithm: 'td3',
      actor_objective: 'td3_bc',
      percentage: 40,
      round_index: 2,
      completed_epochs: 4,
      total_epochs: 10,
      job_id: 'td3-round-2',
    });

    renderSection({
      variant: 'workflow',
      currentPolicyEpoch: 1,
    });

    expect(await screen.findByLabelText('ACT-TD3 policy RL Epoch 1 to 2'))
      .toHaveTextContent('RL Epoch E0001 → E0002');
    expect(screen.getByText(/Critic replay 4\/10/)).toBeInTheDocument();
  });

  test('starts a new lineage without submitting the compatible previous round', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      dataset_path: '/workspace/lerobot/task_lerobot_v30',
      dataset_paths: ['/workspace/lerobot/task_lerobot_v30'],
      act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
      checkpoint_path: '/workspace/model/round1/training_state/act_td3.pt',
      model_path: '/workspace/model/round1/pretrained_model',
      round_index: 1,
    });
    const onFreshLineageConsumed = jest.fn();
    const { testStore } = renderSection({
      variant: 'workflow',
      currentPolicyEpoch: 0,
      forceFreshLineage: true,
      onFreshLineageConsumed,
    });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelections([
        {
          path: '/workspace/lerobot/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 0,
        },
        {
          path: '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
          version: 'v3.0',
          dataEpoch: 1,
        },
      ]));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/round1/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        act_checkpoint: '/workspace/model/round1/pretrained_model',
        parent_checkpoint: '',
      })
    ));
    expect(onFreshLineageConsumed).toHaveBeenCalledTimes(1);
  });

  test('starts a fresh lineage when the selected ACT model does not match the completed job', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'completed',
      percentage: 100,
      dataset_path: '/workspace/lerobot/task_lerobot_v30',
      act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
      checkpoint_path: '/workspace/model/round1/training_state/act_td3.pt',
      model_path: '/workspace/model/round1/pretrained_model',
    });
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    act(() => {
      testStore.dispatch(setOfflineRLDatasetPath('/workspace/lerobot/task_lerobot_v30'));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/new_experiment/pretrained_model',
      }));
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({
        act_checkpoint: '/workspace/model/new_experiment/pretrained_model',
        parent_checkpoint: '',
      })
    ));
  });

  test('restores a running job trainability contract after a page reload', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'running',
      percentage: 12,
      actor_trainable_groups: ['transformer_encoder', 'action_decoder'],
    });

    renderSection({ variant: 'workflow' });

    expect(await screen.findByRole('button', {
      name: /Visual backbone: Frozen/i,
    })).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', {
      name: /CVAE encoder: Frozen/i,
    })).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', {
      name: /Action Module: Trainable/i,
    })).toHaveAttribute('aria-pressed', 'true');
  });

  test('restores a running ACT imitation-learning trainability contract', async () => {
    getImitationLearningStatus.mockResolvedValue({
      status: 'running',
      percentage: 12,
      policy_type: 'act',
      trainable_groups: ['transformer_encoder', 'action_decoder'],
    });

    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    expect(await screen.findByRole('button', {
      name: /Visual backbone: Frozen/i,
    })).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', {
      name: /CVAE encoder: Frozen/i,
    })).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', {
      name: /Action Module: Trainable/i,
    })).toHaveAttribute('aria-pressed', 'true');
  });

  test('blocks all-frozen and CVAE-only TD3 actor configurations', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: /Visual backbone/i }));
    fireEvent.click(screen.getByRole('button', { name: /Action Module/i }));

    expect(screen.getByRole('alert')).toHaveTextContent(/CVAE-only is not supported/i);
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: /CVAE encoder/i }));
    expect(screen.getByRole('alert')).toHaveTextContent(/At least one ACT network block/i);
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('rejects an actor schedule that violates TD3 policy delay 2', async () => {
    renderSection();
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.change(screen.getByLabelText('Actor equivalent epochs'), {
      target: { value: '4' },
    });

    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('validates and submits the selected TD3 batch size', async () => {
    renderSection();
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.change(screen.getByLabelText('LeRobot v3 Dataset Path'), {
      target: { value: '/workspace/lerobot/task_lerobot_v30' },
    });
    fireEvent.change(screen.getByLabelText('Original ACT Checkpoint'), {
      target: { value: '/workspace/model/lerobot/base/pretrained_model' },
    });
    fireEvent.change(screen.getByLabelText('Batch size'), {
      target: { value: '0' },
    });
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();

    fireEvent.change(screen.getByLabelText('Batch size'), {
      target: { value: '8' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith(
      expect.objectContaining({ batch_size: 8 })
    ));
  });

  test('shows ETA beside progress and stops only the displayed running job', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'running',
      percentage: 25,
      eta_seconds: 125,
      batch_size: 8,
      job_id: 'job-visible-123',
      actor_trainable_groups: [
        'visual_backbone',
        'cvae_encoder',
        'transformer_encoder',
        'action_decoder',
      ],
    });
    stopOfflineRLTraining.mockResolvedValue({
      status: 'running',
      percentage: 25,
      eta_seconds: 125,
      batch_size: 8,
      job_id: 'job-visible-123',
      message: 'Stopping ACT-TD3 training',
    });
    const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

    renderSection({ variant: 'workflow' });

    const lossChart = await screen.findByTestId('training-loss-chart');
    expect(within(lossChart).getByText('Training')).toBeInTheDocument();
    expect(within(lossChart).getByLabelText('Training percentage')).toHaveTextContent('25.0%');
    expect(within(lossChart).getByLabelText('Training ETA')).toHaveTextContent('ETA 2m 05s');
    await waitFor(() => expect(screen.getByLabelText('Batch size')).toHaveValue(8));
    const stopButton = screen.getByRole('button', { name: 'Stop Training' });
    expect(stopButton).not.toBeDisabled();
    fireEvent.click(stopButton);

    await waitFor(() => {
      expect(stopOfflineRLTraining).toHaveBeenCalledWith('job-visible-123');
    });
    expect(screen.getByRole('button', { name: 'Stopping…' })).toBeDisabled();
    confirmSpy.mockRestore();
  });

  test('blocks training while dataset conversion is running', async () => {
    const { testStore } = renderSection();
    await screen.findByRole('button', { name: 'Start Training' });

    act(() => {
      testStore.dispatch(setConversionStatus({ status: 'running' }));
    });

    expect(await screen.findByText(/Dataset conversion is running/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('shows v2.1 in Step 3 but blocks it from TD3 training', async () => {
    const { testStore } = renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    act(() => {
      testStore.dispatch(setOfflineRLDatasetSelection({
        path: '/workspace/lerobot/task_lerobot_v21',
        version: 'v2.1',
      }));
      testStore.dispatch(setInferenceTaskInfo({
        policyPath: '/workspace/model/lerobot/base/pretrained_model',
      }));
    });

    expect(screen.getByRole('alert')).toHaveTextContent(
      /TD3 requires LeRobot v3\.0.*v2\.1 dataset is view only/i
    );
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('renders backend progress, metrics, and generated paths', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'complete',
      percentage: 100,
      episode_count: 50,
      round_index: 1,
      round_episode_count: 30,
      success_count: 42,
      failure_count: 8,
      completed_epochs: 10,
      total_epochs: 10,
      completed_critic_updates: 320,
      total_critic_updates: 320,
      completed_actor_updates: 160,
      total_actor_updates: 160,
      critic_loss: 0.001,
      actor_loss: 0.02,
      eta_seconds: 0,
      model_path: '/workspace/model/lerobot/result/pretrained_model',
      checkpoint_path: '/workspace/model/lerobot/result/training_state/act_td3.pt',
    });

    renderSection();

    expect(await screen.findByText('Complete')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
    expect(screen.getByText('50 / 200')).toBeInTheDocument();
    expect(screen.getByText('1 / 30')).toBeInTheDocument();
    expect(screen.getByText('42 / 8')).toBeInTheDocument();
    expect(screen.getByText('/workspace/model/lerobot/result/pretrained_model')).toBeInTheDocument();
    expect(screen.getByText('/workspace/model/lerobot/result/training_state/act_td3.pt')).toBeInTheDocument();
  });

  test('ignores an older GET that resolves after a newer POST result', async () => {
    jest.useFakeTimers();
    const staleStatus = deferred();
    getOfflineRLStatus
      .mockResolvedValueOnce({ status: 'idle', percentage: 0 })
      .mockReturnValueOnce(staleStatus.promise);
    startOfflineRLTraining.mockResolvedValue({
      status: 'running',
      percentage: 12,
      message: 'New training job',
    });

    const view = renderSection();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(getOfflineRLStatus).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();

    act(() => {
      jest.advanceTimersByTime(2000);
    });
    expect(getOfflineRLStatus).toHaveBeenCalledTimes(2);
    act(() => {
      jest.advanceTimersByTime(6000);
    });
    expect(getOfflineRLStatus).toHaveBeenCalledTimes(2);

    fireEvent.change(screen.getByLabelText('LeRobot v3 Dataset Path'), {
      target: { value: '/workspace/lerobot/task_lerobot_v30' },
    });
    fireEvent.change(screen.getByLabelText('Original ACT Checkpoint'), {
      target: { value: '/workspace/model/lerobot/base/pretrained_model' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByText('12%')).toBeInTheDocument();

    await act(async () => {
      staleStatus.resolve({ status: 'idle', percentage: 0 });
      await staleStatus.promise;
    });
    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByText('12%')).toBeInTheDocument();

    view.unmount();
  });

  test('keeps controls locked after an ambiguous POST failure until status reconciliation reports running', async () => {
    const reconciliationStatus = deferred();
    getOfflineRLStatus
      .mockResolvedValueOnce({ status: 'idle', percentage: 0 })
      .mockReturnValueOnce(reconciliationStatus.promise);
    startOfflineRLTraining.mockRejectedValueOnce(new Error('Connection lost'));

    renderSection();
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.change(screen.getByLabelText('LeRobot v3 Dataset Path'), {
      target: { value: '/workspace/lerobot/task_lerobot_v30' },
    });
    fireEvent.change(screen.getByLabelText('Original ACT Checkpoint'), {
      target: { value: '/workspace/model/lerobot/base/pretrained_model' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));

    await waitFor(() => expect(getOfflineRLStatus).toHaveBeenCalledTimes(2));
    expect(screen.getByLabelText('LeRobot v3 Dataset Path')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Checking status…' })).toBeDisabled();

    await act(async () => {
      reconciliationStatus.resolve({ status: 'running', percentage: 7 });
      await reconciliationStatus.promise;
    });

    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByText('7%')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Training…' })).toBeDisabled();
  });

  test('locks controls again when a later current status request fails', async () => {
    jest.useFakeTimers();
    getOfflineRLStatus
      .mockResolvedValueOnce({ status: 'idle', percentage: 0 })
      .mockRejectedValueOnce(new Error('Status unavailable'));

    const view = renderSection();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();

    await act(async () => {
      jest.advanceTimersByTime(2000);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(getOfflineRLStatus).toHaveBeenCalledTimes(2);
    expect(screen.getByLabelText('LeRobot v3 Dataset Path')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Checking status…' })).toBeDisabled();

    view.unmount();
  });

  test('uses the compact outer loop for ACT and Diffusion Transformer workflows', async () => {
    const onCompactLayoutChange = jest.fn();
    const { unmount } = renderSection({
      variant: 'workflow',
      flowSdePpoReady: true,
      getFlowSDEPPOStatus: jest.fn().mockResolvedValue({ status: 'idle', percentage: 0 }),
      onStartFlowSDEPPO: jest.fn(),
      onCompactLayoutChange,
    });
    await screen.findByRole('button', { name: 'Start Training' });

    const workflow = screen.getByTestId('offline-rl-workflow-training');
    const footer = screen.getByTestId('offline-rl-training-footer');
    const actLoop = screen.getByTestId('act-td3-training-loop');
    expect(actLoop).toHaveAttribute('data-fit-content', 'true');
    expect(actLoop).toHaveClass('self-start');
    expect(workflow).toHaveClass('flex-none', 'grid-rows-[auto_auto_auto]');
    expect(footer).toHaveClass('mt-3');
    expect(footer).not.toHaveClass('mt-auto');
    await waitFor(() => expect(onCompactLayoutChange).toHaveBeenLastCalledWith(true));

    fireEvent.click(screen.getByRole('button', { name: 'Diffusion Transformer' }));
    const diffusionLoop = await screen.findByTestId('policy-training-loop');
    expect(diffusionLoop).toHaveAttribute('data-fit-content', 'true');
    expect(diffusionLoop).toHaveClass('self-start');
    expect(screen.getByTestId('training-algorithm-card')).toHaveClass('h-fit');
    expect(workflow).toHaveClass('flex-none', 'grid-rows-[auto_auto_auto]');
    expect(footer).toHaveClass('mt-3');
    await waitFor(() => expect(onCompactLayoutChange).toHaveBeenLastCalledWith(true));

    unmount();
    expect(onCompactLayoutChange).toHaveBeenLastCalledWith(false);
  });
});
