import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import { InferencePhase } from '../../../constants/taskPhases';
import taskReducer, {
  selectRobotType,
  setInferenceTaskInfo,
} from '../../tasks/taskSlice';
import FlowSDEPPOInferenceControls from './FlowSDEPPOInferenceControls';

const POLICY = '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model';
const BUNDLE = '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/abc/rollouts/bundle';

const renderControls = (props = {}, taskOverrides = {}) => {
  const store = configureStore({ reducer: { tasks: taskReducer } });
  store.dispatch(selectRobotType('ffw_sg2_rev1'));
  store.dispatch(setInferenceTaskInfo({
    serviceType: 'lerobot',
    policyType: 'multi_task_dit',
    policyPath: POLICY,
    taskInstruction: ['Pick up the jelly bag'],
    ...taskOverrides,
  }));
  const defaults = {
    getRolloutStatus: jest.fn().mockResolvedValue({
      status: 'idle',
      operation: 'combined',
    }),
    onStartRollout: jest.fn().mockResolvedValue({
      status: 'running',
      operation: 'collect',
      job_id: 'rollout-job',
    }),
    onStopRollout: jest.fn(),
    onSubmitOutcome: jest.fn(),
    getValueWarmupStatus: jest.fn().mockResolvedValue({ status: 'idle' }),
  };
  const merged = { ...defaults, ...props };
  return {
    ...render(
      <Provider store={store}>
        <FlowSDEPPOInferenceControls {...merged} />
      </Provider>
    ),
    store,
    props: merged,
  };
};

describe('FlowSDEPPOInferenceControls', () => {
  test('starts one rollout with a compatible critic warm-up source', async () => {
    const warmupBundle = '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/value_warmup/a';
    const view = renderControls({
      getValueWarmupStatus: jest.fn().mockResolvedValue({
        status: 'completed',
        policy_checkpoint: POLICY,
        task_instruction: 'Pick up the jelly bag',
        bundle_path: warmupBundle,
      }),
    });

    const rollout = await screen.findByRole('button', { name: 'PPO Rollout' });
    await waitFor(() => expect(rollout).not.toBeDisabled());
    fireEvent.click(rollout);

    await waitFor(() => expect(view.props.onStartRollout).toHaveBeenCalledWith({
      policy_checkpoint: POLICY,
      policy_type: 'multi_task_dit',
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
      episodes: 1,
      value_warmup_bundle: warmupBundle,
    }));
  });

  test('continues actor, critic, and optimizer state from the latest PPO update', async () => {
    const trainer = '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/a/training_state/trainer_state.pt';
    const view = renderControls({
      getRolloutStatus: jest.fn().mockResolvedValue({
        status: 'completed',
        operation: 'update',
        policy_checkpoint: POLICY,
        lineage_policy_checkpoint: POLICY,
        task_instruction: 'Pick up the jelly bag',
        checkpoint_path: trainer,
        model_path: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/a/pretrained_model',
      }),
    });

    const rollout = await screen.findByRole('button', { name: 'PPO Rollout' });
    await waitFor(() => expect(rollout).not.toBeDisabled());
    fireEvent.click(rollout);

    await waitFor(() => expect(view.props.onStartRollout).toHaveBeenCalledWith(
      expect.objectContaining({ resume_checkpoint: trainer, episodes: 1 })
    ));
    expect(view.props.onStartRollout.mock.calls[0][0])
      .not.toHaveProperty('value_warmup_bundle');
  });

  test.each([
    ['Success', 'success'],
    ['Fail', 'fail'],
    ['Cancel', 'cancel'],
  ])('submits %s to the exact collector job', async (label, outcome) => {
    const submit = jest.fn().mockResolvedValue({
      status: 'running',
      operation: 'collect',
      job_id: 'collector-exact',
      awaiting_outcome: false,
    });
    renderControls({
      getRolloutStatus: jest.fn().mockResolvedValue({
        status: 'running',
        operation: 'collect',
        job_id: 'collector-exact',
        awaiting_outcome: true,
      }),
      onSubmitOutcome: submit,
    });

    const button = await screen.findByRole('button', { name: label });
    expect(button).not.toBeDisabled();
    fireEvent.click(button);
    await waitFor(() => expect(submit).toHaveBeenCalledWith('collector-exact', outcome));
  });

  test('hands a sealed rollout to Training without reporting it as a trained policy', async () => {
    const onBundle = jest.fn();
    renderControls({
      getRolloutStatus: jest.fn().mockResolvedValue({
        status: 'completed',
        operation: 'collect',
        job_id: 'collector-complete',
        rollout_bundles: [BUNDLE],
      }),
      onRolloutBundleChange: onBundle,
    });

    await screen.findByText(/Rollout sealed/);
    await waitFor(() => expect(onBundle).toHaveBeenLastCalledWith(BUNDLE));
    expect(screen.getByRole('button', { name: 'PPO Rollout' })).toBeDisabled();
  });

  test('requires standard VLA inference to stop before PPO rollout', async () => {
    renderControls({ inferencePhase: InferencePhase.INFERENCING });
    const rollout = await screen.findByRole('button', { name: 'PPO Rollout' });
    await waitFor(() => {
      expect(rollout).toBeDisabled();
      expect(rollout).toHaveAttribute(
        'title',
        'Clear standard VLA inference before PPO rollout'
      );
    });
    expect(screen.getByRole('button', { name: 'VLA Action' }))
      .toHaveAttribute('aria-pressed', 'true');
  });

  test('switching back to VLA stops only the active rollout job', async () => {
    const stop = jest.fn().mockResolvedValue({
      status: 'stopped',
      operation: 'collect',
      job_id: 'collector-stop',
    });
    renderControls({
      getRolloutStatus: jest.fn().mockResolvedValue({
        status: 'running',
        operation: 'collect',
        job_id: 'collector-stop',
        awaiting_outcome: false,
      }),
      onStopRollout: stop,
    });

    const vla = await screen.findByRole('button', { name: 'VLA Action' });
    await waitFor(() => expect(vla).not.toBeDisabled());
    await act(async () => fireEvent.click(vla));
    await waitFor(() => expect(stop).toHaveBeenCalledWith('collector-stop'));
  });

  test('is hidden for policies that do not use Flow-SDE PPO', () => {
    renderControls({}, { policyType: 'act' });
    expect(screen.queryByTestId('flow-sde-ppo-inference-controls'))
      .not.toBeInTheDocument();
  });
});
