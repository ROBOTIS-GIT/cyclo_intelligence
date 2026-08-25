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
import OfflineRLTrainingSection from './OfflineRLTrainingSection';
import {
  getFlowSDEPPOValueWarmupStatus,
  getImitationLearningStatus,
  getOfflineRLStatus,
  startFlowSDEPPOValueWarmup,
  startImitationLearningTraining,
  startOfflineRLTraining,
  stopFlowSDEPPOValueWarmup,
  stopImitationLearningTraining,
  stopOfflineRLTraining,
} from '../../../utils/offlineRlApi';

jest.mock('../../../utils/offlineRlApi', () => ({
  getFlowSDEPPOValueWarmupStatus: jest.fn(),
  getImitationLearningStatus: jest.fn(),
  getOfflineRLStatus: jest.fn(),
  startFlowSDEPPOValueWarmup: jest.fn(),
  startImitationLearningTraining: jest.fn(),
  startOfflineRLTraining: jest.fn(),
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
  beforeEach(() => {
    getFlowSDEPPOValueWarmupStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    getImitationLearningStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    getOfflineRLStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    startImitationLearningTraining.mockResolvedValue({ status: 'starting', percentage: 0 });
    startOfflineRLTraining.mockResolvedValue({ status: 'starting', percentage: 0 });
    startFlowSDEPPOValueWarmup.mockResolvedValue({
      status: 'running',
      percentage: 0,
      job_id: 'warmup-job-1',
    });
    stopImitationLearningTraining.mockResolvedValue({ status: 'running', percentage: 10 });
    stopOfflineRLTraining.mockResolvedValue({ status: 'running', percentage: 10 });
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
    expect(screen.getByRole('button', { name: 'TD3' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: /PPO.*Flow-SDE/i }))
      .toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByRole('button', { name: 'SAC' })).toBeDisabled();
    const methodGroup = screen.getByRole('group', { name: 'Training method' });
    expect(within(methodGroup).getByRole('button', { name: 'Reinforcement Learning' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(within(methodGroup).getByRole('button', { name: 'Imitation Learning' }))
      .toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByTestId('td3-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByLabelText('Q1 critic: Fire; Trainable; fixed')).toBeInTheDocument();
    expect(screen.getByLabelText('Q2 critic: Fire; Trainable; fixed')).toBeInTheDocument();
    expect(screen.getByText('ACT ← maximize Q1')).toBeInTheDocument();
    expect(screen.getByLabelText('Critic epochs')).toHaveValue(10);
    expect(screen.getByLabelText('Actor equivalent epochs')).toHaveValue(5);
    expect(screen.getByLabelText('Batch size')).toHaveValue(4);

    const architecture = screen.getByTestId('offline-rl-training-architecture');
    expect(architecture).toHaveClass('flex-1', 'items-stretch');
    expect(screen.getByTestId('td3-architecture-diagram').parentElement)
      .toHaveClass('h-full', 'flex-col');

    const footer = screen.getByTestId('offline-rl-training-footer');
    expect(footer).toHaveClass('mt-auto', 'shrink-0');
    expect(footer).toHaveTextContent('Training progress');
    expect(footer).toHaveTextContent('Training action');
    expect(footer).toHaveTextContent('ETA');
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
    expect(screen.getByRole('alert')).toHaveTextContent(/backend is not ready/i);
    const gatedStart = await screen.findByRole('button', { name: 'Start Training' });
    expect(gatedStart).toBeDisabled();
    fireEvent.click(gatedStart);
    expect(startOfflineRLTraining).not.toHaveBeenCalled();

    await waitFor(() => expect(screen.getByRole('button', { name: 'TD3' })).not.toBeDisabled());
    fireEvent.click(screen.getByRole('button', { name: 'TD3' }));
    expect(await screen.findByTestId('act-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'ACT' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'TD3' }))
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

    fireEvent.click(screen.getByRole('button', { name: /PPO.*Flow-SDE/i }));
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

  test('keeps critic warm-up off by default and submits checked replay roots when enabled', async () => {
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
    await waitFor(() => expect(getFlowSDEPPOStatus).toHaveBeenCalled());
    await screen.findByRole('button', { name: 'Start Training' });
    const warmupGroup = screen.getByRole('group', { name: 'Critic warm-up' });
    expect(within(warmupGroup).getByRole('button', { name: 'No' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();

    fireEvent.click(within(warmupGroup).getByRole('button', { name: 'Yes' }));
    await waitFor(() => expect(getFlowSDEPPOValueWarmupStatus).toHaveBeenCalled());

    expect(screen.getByLabelText('Critic warm-up steps')).toHaveValue(2000);
    expect(screen.getByLabelText('Critic warm-up batch size')).toHaveValue(8);
    expect(screen.getByLabelText('Critic warm-up value learning rate')).toHaveValue(0.0001);
    expect(screen.getByLabelText('Critic warm-up discount')).toHaveValue(0.99);
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(/Train the critic to completion/i);
    });
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();

    const trainCritic = screen.getByRole('button', { name: 'Train Critic' });
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
  });

  test('shows completed critic warm-up progress, value loss, ETA, and bundle path', async () => {
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
    await screen.findByRole('button', { name: 'Start Training' });
    const warmupGroup = await screen.findByRole('group', { name: 'Critic warm-up' });
    await waitFor(() => {
      expect(within(warmupGroup).getByRole('button', { name: 'Yes' })).not.toBeDisabled();
    });
    fireEvent.click(within(warmupGroup).getByRole('button', { name: 'Yes' }));

    expect(await screen.findByText(/Complete · 100% · ETA 0s/)).toBeInTheDocument();
    expect(screen.getByText(/Step 2,000\/2,000 · Value loss 0.012345/)).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Critic warm-up progress' }))
      .toHaveAttribute('aria-valuenow', '100');
    expect(screen.getByLabelText('Critic warm-up bundle path')).toHaveTextContent(
      '/workspace/checkpoint/multi_task_dit/value_warmup/warmup-job-1'
    );
    expect(screen.getByTestId('critic-warmup-source')).toHaveTextContent(
      'Critic source: Warm-up · warmup-j · Ready for online PPO'
    );
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
    const warmupGroup = await screen.findByRole('group', { name: 'Critic warm-up' });
    const warmupYes = within(warmupGroup).getByRole('button', { name: 'Yes' });
    await waitFor(() => expect(warmupYes).not.toBeDisabled());
    fireEvent.click(warmupYes);

    await screen.findByTestId('critic-warmup-source');
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
    const warmupGroup = await screen.findByRole('group', { name: 'Critic warm-up' });
    const warmupYes = within(warmupGroup).getByRole('button', { name: 'Yes' });
    await waitFor(() => expect(warmupYes).not.toBeDisabled());
    fireEvent.click(warmupYes);

    expect(await screen.findByTestId('critic-warmup-source')).toHaveTextContent(
      'Critic source: PPO · ppo-job- · Ready to continue'
    );
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

  test('keeps a fresh critic when warm-up is No even if a completed PPO state exists', async () => {
    const basePolicy = '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model';
    const getFlowSDEPPOStatus = jest.fn().mockResolvedValue({
      status: 'completed',
      percentage: 100,
      job_id: 'ppo-job-2',
      policy_checkpoint: basePolicy,
      lineage_policy_checkpoint: basePolicy,
      task_instruction: 'Pick up the jelly bag',
      checkpoint_path: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/ppo-job-2/training_state/trainer_state.pt',
      model_path: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/ppo-job-2/pretrained_model',
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
  ])('blocks a stale completed PPO %s when no compatible warm-up exists', async (
    _label,
    lineage,
    expectedMessage
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
    const warmupGroup = await screen.findByRole('group', { name: 'Critic warm-up' });
    const warmupYes = within(warmupGroup).getByRole('button', { name: 'Yes' });
    await waitFor(() => expect(warmupYes).not.toBeDisabled());
    fireEvent.click(warmupYes);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(expectedMessage);
    });
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    expect(startFlowSDEPPO).not.toHaveBeenCalled();
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
    const warmupGroup = await screen.findByRole('group', { name: 'Critic warm-up' });
    const warmupYes = within(warmupGroup).getByRole('button', { name: 'Yes' });
    await waitFor(() => expect(warmupYes).not.toBeDisabled());
    fireEvent.click(warmupYes);

    expect(await screen.findByTestId('critic-warmup-source')).toHaveTextContent(
      'Critic source: Warm-up · compatib · Ready for online PPO'
    );
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
  ])('blocks a stale warm-up with a different %s', async (_label, lineage, message) => {
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
    const warmupGroup = await screen.findByRole('group', { name: 'Critic warm-up' });
    const warmupYes = within(warmupGroup).getByRole('button', { name: 'Yes' });
    await waitFor(() => expect(warmupYes).not.toBeDisabled());
    fireEvent.click(warmupYes);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent(message);
    });
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    expect(startFlowSDEPPO).not.toHaveBeenCalled();
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
    await screen.findByRole('button', { name: 'Start Training' });
    const warmupGroup = await screen.findByRole('group', { name: 'Critic warm-up' });
    await waitFor(() => {
      expect(within(warmupGroup).getByRole('button', { name: 'Yes' })).not.toBeDisabled();
    });
    fireEvent.click(within(warmupGroup).getByRole('button', { name: 'Yes' }));

    const stopButton = await screen.findByRole('button', { name: 'Stop' });
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
      rlEpoch: 0,
    }));
  });

  test('uses ACT imitation learning with fixed full-ACT settings by default', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    expect(await screen.findByTestId('act-imitation-learning-diagram')).toBeInTheDocument();
    expect(screen.getByLabelText('Imitation steps')).toHaveValue(80000);
    expect(screen.getByLabelText('Imitation batch size')).toHaveValue(8);
    expect(screen.getByLabelText('Imitation save frequency')).toHaveValue(10000);
    expect(screen.getByLabelText('Imitation action chunk')).toHaveTextContent('30');
    expect(screen.getByRole('button', { name: 'TD3' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'SAC' })).toBeDisabled();

    [
      /Visual backbone: Trainable/i,
      /CVAE encoder: Trainable/i,
      /Transformer encoder: Trainable/i,
      /Action decoder: Trainable/i,
    ].forEach((name) => {
      const block = screen.getByRole('button', { name });
      expect(block).toBeDisabled();
      expect(block).toHaveAttribute('aria-pressed', 'true');
    });
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
      chunk_size: 30,
    }));
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
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
    expect(screen.getByLabelText('Imitation action chunk')).toHaveTextContent('16');
    expect(screen.getByText(/Supervised flow-matching/i)).toBeInTheDocument();
    expect(screen.getByText(/no reward or outcome labels required/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'GR00T' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Pi0.5' })).toBeDisabled();
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
      /Training is available for ACT and Diffusion Transformer/i
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
    await screen.findByLabelText('Imitation steps');
    fireEvent.change(screen.getByLabelText('Imitation save frequency'), {
      target: { value: '90000' },
    });
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();

    fireEvent.change(screen.getByLabelText('Imitation save frequency'), {
      target: { value: '10000' },
    });
    expect(screen.getByRole('button', { name: 'Start Training' })).not.toBeDisabled();
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

    expect(await screen.findByText(/Training · 25%.*Step 20,000.*80,000.*ETA 2m 5s/))
      .toBeInTheDocument();
    expect(screen.getByText('Total loss').parentElement).toHaveTextContent('0.12000');
    expect(screen.getByText('L1 loss').parentElement).toHaveTextContent('0.080000');
    expect(screen.getByText('KLD loss').parentElement).toHaveTextContent('0.0040000');
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
    });
    await screen.findByRole('button', { name: 'Start Training' });
    fireEvent.click(screen.getByRole('button', { name: 'Imitation Learning' }));

    await waitFor(() => expect(deploymentListener).toHaveBeenLastCalledWith({
      ready: true,
      modelPath: '/workspace/model/lerobot/imitation/act/checkpoints/080000/pretrained_model',
      serviceType: 'lerobot',
      policyType: 'act',
      rlEpoch: 0,
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

    expect(await screen.findByText(/Training · 25%.*Step 20,000.*80,000.*ETA 2m 5s/))
      .toBeInTheDocument();
    expect(screen.getByText('Flow loss').parentElement).toHaveTextContent('0.012000');
    expect(screen.queryByText('L1 loss')).not.toBeInTheDocument();
    expect(screen.queryByText('KLD loss')).not.toBeInTheDocument();
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
      /GR00T diagram preview only.*backend is not connected/i
    );
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    expect(startOfflineRLTraining).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Pi0.5' }));
    expect(screen.getByTestId('pi05-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Pi0.5' }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent(
      /Pi0.5 diagram preview only.*backend is not connected/i
    );
    expect(screen.getByRole('button', { name: 'Start Training' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Start Training' }));
    expect(startOfflineRLTraining).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'ACT' }));
    expect(screen.getByTestId('act-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'ACT' }))
      .toHaveAttribute('aria-pressed', 'true');
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

  test('shows policy RL Epoch separately from critic replay progress', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'running',
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
      name: /Transformer encoder: Trainable/i,
    })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', {
      name: /Action decoder: Trainable/i,
    })).toHaveAttribute('aria-pressed', 'true');
  });

  test('blocks all-frozen and CVAE-only TD3 actor configurations', async () => {
    renderSection({ variant: 'workflow' });
    await screen.findByRole('button', { name: 'Start Training' });

    fireEvent.click(screen.getByRole('button', { name: /Visual backbone/i }));
    fireEvent.click(screen.getByRole('button', { name: /Transformer encoder/i }));
    fireEvent.click(screen.getByRole('button', { name: /Action decoder/i }));

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

    expect(await screen.findByText(/Training · 25% · ETA 2m 5s/)).toBeInTheDocument();
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
});
