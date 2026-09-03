import {
  cancelOfflineRLTraining,
  deleteOfflineRLDatasetEpisodes,
  getACTTD3CriticWarmupStatus,
  getFlowSDEPPOValueWarmupStatus,
  getFlowSDEPPOPolicyRolloutStatus,
  getFlowSDEPPOUpdateStatus,
  getFlowSDEPPOStatus,
  getImitationLearningStatus,
  getOfflineRLDatasetInfo,
  getOfflineRLDatasetEpisodeData,
  getOfflineRLDatasets,
  getOfflineRLStatus,
  getRLTStage1Status,
  getRLTStage2Status,
  reserveOfflineRLDataEpoch,
  startACTTD3CriticWarmup,
  startFlowSDEPPOTraining,
  startFlowSDEPPOValueWarmup,
  startFlowSDEPPOPolicyRollout,
  startFlowSDEPPOUpdate,
  startImitationLearningTraining,
  startOfflineRLTraining,
  startRLTStage1Training,
  startRLTStage2Training,
  stopImitationLearningTraining,
  stopACTTD3CriticWarmup,
  stopFlowSDEPPOTraining,
  stopFlowSDEPPOValueWarmup,
  stopFlowSDEPPOPolicyRollout,
  stopFlowSDEPPOUpdate,
  stopOfflineRLTraining,
  stopRLTStage1Training,
  stopRLTStage2Training,
  submitFlowSDEPPOOutcome,
  submitFlowSDEPPOPolicyRolloutOutcome,
} from './offlineRlApi';

const jsonResponse = (value, { ok = true, status = 200 } = {}) => ({
  ok,
  status,
  text: jest.fn().mockResolvedValue(JSON.stringify(value)),
});

describe('offline RL API', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    delete global.fetch;
  });

  test('starts a job with the supplied request', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ status: 'starting' }));
    const request = {
      dataset_path: '/workspace/lerobot/task_lerobot_v30',
      act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
      parent_checkpoint: '',
      algorithm: 'td3',
      robot_type: 'ffw_sg2_rev1',
      actor_trainable_groups: [
        'visual_backbone',
        'cvae_encoder',
        'transformer_encoder',
        'action_decoder',
      ],
    };

    await expect(startOfflineRLTraining(request)).resolves.toEqual({
      status: 'starting',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/offline-rl/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  });

  test('reads status without browser caching', async () => {
    global.fetch.mockResolvedValue(
      jsonResponse({ status: 'running', percentage: 25 })
    );

    await expect(getOfflineRLStatus()).resolves.toEqual({
      status: 'running',
      percentage: 25,
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/offline-rl/status', {
      cache: 'no-store',
    });
  });

  test('stops only the observed offline RL job', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      job_id: 'job-123',
      message: 'Stopping ACT-TD3 training',
    }));

    await expect(stopOfflineRLTraining('job-123')).resolves.toEqual({
      status: 'running',
      job_id: 'job-123',
      message: 'Stopping ACT-TD3 training',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/offline-rl/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: 'job-123' }),
    });
  });

  test('cancels only the observed stopped offline RL job', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'cancelled',
      job_id: 'job-123',
      message: 'Cancelled ACT-TD3 training artifacts',
    }));

    await expect(cancelOfflineRLTraining('job-123')).resolves.toEqual({
      status: 'cancelled',
      job_id: 'job-123',
      message: 'Cancelled ACT-TD3 training artifacts',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/offline-rl/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: 'job-123' }),
    });
  });

  test('starts ACT-TD3 critic warm-up with the supplied request', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      job_id: 'critic-job-123',
    }));
    const request = {
      dataset_path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
      dataset_paths: [
        '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
        '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
      ],
      act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
      robot_type: 'ffw_sg2_rev1',
      batch_size: 4,
    };

    await expect(startACTTD3CriticWarmup(request)).resolves.toEqual({
      status: 'running',
      job_id: 'critic-job-123',
    });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/offline-rl/critic-warmup/start',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      }
    );
  });

  test('reads ACT-TD3 critic warm-up status without browser caching', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      percentage: 25,
      completed_critic_updates: 1250,
    }));

    await expect(getACTTD3CriticWarmupStatus()).resolves.toEqual({
      status: 'running',
      percentage: 25,
      completed_critic_updates: 1250,
    });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/offline-rl/critic-warmup/status',
      { cache: 'no-store' }
    );
  });

  test('stops only the observed ACT-TD3 critic warm-up job', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      job_id: 'critic-job-123',
      message: 'Stopping ACT-TD3 critic warm-up',
    }));

    await expect(stopACTTD3CriticWarmup('critic-job-123')).resolves.toEqual({
      status: 'running',
      job_id: 'critic-job-123',
      message: 'Stopping ACT-TD3 critic warm-up',
    });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/offline-rl/critic-warmup/stop',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: 'critic-job-123' }),
      }
    );
  });

  test('starts ACT imitation learning with the supplied request', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ status: 'starting' }));
    const request = {
      dataset_paths: [
        '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
        '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
      ],
      dataset_path: '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
      policy_type: 'act',
      steps: 80000,
      batch_size: 8,
      save_freq: 10000,
      chunk_size: 30,
      trainable_groups: [
        'visual_backbone',
        'transformer_encoder',
        'action_decoder',
      ],
    };

    await expect(startImitationLearningTraining(request)).resolves.toEqual({
      status: 'starting',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/imitation-learning/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  });

  test('reads imitation learning status without browser caching', async () => {
    global.fetch.mockResolvedValue(
      jsonResponse({ status: 'running', percentage: 25 })
    );

    await expect(getImitationLearningStatus()).resolves.toEqual({
      status: 'running',
      percentage: 25,
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/imitation-learning/status', {
      cache: 'no-store',
    });
  });

  test('stops only the observed imitation learning job', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      job_id: 'il-job-123',
      message: 'Stopping ACT imitation learning',
    }));

    await expect(stopImitationLearningTraining('il-job-123')).resolves.toEqual({
      status: 'running',
      job_id: 'il-job-123',
      message: 'Stopping ACT imitation learning',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/imitation-learning/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: 'il-job-123' }),
    });
  });

  test('starts RLT Stage 1 with frozen-GR00T feature training inputs', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'starting',
      job_id: 'rlt-stage1-job-123',
    }));
    const request = {
      dataset_paths: [
        '/workspace/lerobot/data_epoch_0000/task_lerobot_v30',
        '/workspace/lerobot/data_epoch_0001/task_lerobot_v30',
      ],
      groot_checkpoint: '/workspace/model/groot/showroom',
      steps: 10000,
      batch_size: 1,
      save_freq: 1000,
    };

    await expect(startRLTStage1Training(request)).resolves.toEqual({
      status: 'starting',
      job_id: 'rlt-stage1-job-123',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/rlt-stage1/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  });

  test('reads RLT Stage 1 status without browser caching', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      percentage: 25,
      reconstruction_loss: 0.012,
    }));

    await expect(getRLTStage1Status()).resolves.toEqual({
      status: 'running',
      percentage: 25,
      reconstruction_loss: 0.012,
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/rlt-stage1/status', {
      cache: 'no-store',
    });
  });

  test('stops only the observed RLT Stage 1 job', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      job_id: 'rlt-stage1-job-123',
      message: 'Stopping RLT Stage 1 training',
    }));

    await expect(stopRLTStage1Training('rlt-stage1-job-123')).resolves.toEqual({
      status: 'running',
      job_id: 'rlt-stage1-job-123',
      message: 'Stopping RLT Stage 1 training',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/rlt-stage1/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: 'rlt-stage1-job-123' }),
    });
  });

  test('starts RLT Stage 2 with an explicit initialization contract', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'starting',
      job_id: 'rlt-stage2-job-123',
    }));
    const request = {
      initialization_mode: 'new',
      dataset_paths: ['/workspace/lerobot/data_epoch_0000/task_lerobot_v30'],
      groot_checkpoint: '/workspace/model/groot/showroom',
      rl_token_encoder_path: '/workspace/checkpoint/rlt/stage1/run/artifacts/rl_token_encoder.pt',
      rlt_bundle_path: '',
      steps: 10000,
      batch_size: 1,
      save_freq: 1000,
    };

    await expect(startRLTStage2Training(request)).resolves.toEqual({
      status: 'starting',
      job_id: 'rlt-stage2-job-123',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/rlt-stage2/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  });

  test('starts RLT Stage 2 resume without inventing New-lineage sources', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'starting',
      job_id: 'rlt-stage2-resume-123',
    }));
    const request = {
      initialization_mode: 'resume',
      dataset_paths: ['/workspace/lerobot/data_epoch_0001/task_lerobot_v30'],
      groot_checkpoint: '',
      rl_token_encoder_path: '',
      rlt_bundle_path: '/workspace/checkpoint/rlt/stage2/round_0002',
      steps: 10000,
      batch_size: 1,
      save_freq: 1000,
    };

    await startRLTStage2Training(request);
    expect(global.fetch).toHaveBeenCalledWith('/api/rlt-stage2/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  });

  test('reads RLT Stage 2 status without browser caching', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      percentage: 25,
      actor_loss: -0.18,
    }));

    await expect(getRLTStage2Status()).resolves.toEqual({
      status: 'running',
      percentage: 25,
      actor_loss: -0.18,
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/rlt-stage2/status', {
      cache: 'no-store',
    });
  });

  test('stops only the observed RLT Stage 2 job', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      job_id: 'rlt-stage2-job-123',
    }));

    await expect(stopRLTStage2Training('rlt-stage2-job-123')).resolves.toEqual({
      status: 'running',
      job_id: 'rlt-stage2-job-123',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/rlt-stage2/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: 'rlt-stage2-job-123' }),
    });
  });

  test('starts live Flow-SDE PPO without an offline dataset payload', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ status: 'running', job_id: 'flow-job-1' }));
    const request = {
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      policy_type: 'multi_task_dit',
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
    };

    await expect(startFlowSDEPPOTraining(request)).resolves.toEqual({
      status: 'running',
      job_id: 'flow-job-1',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  });

  test('forwards an explicitly selected value warm-up bundle to live Flow-SDE PPO', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ status: 'running', job_id: 'flow-job-2' }));
    const request = {
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      policy_type: 'multi_task_dit',
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
      value_warmup_bundle: '/workspace/checkpoint/multi_task_dit/value_warmup/warmup-job-1',
    };

    await expect(startFlowSDEPPOTraining(request)).resolves.toEqual({
      status: 'running',
      job_id: 'flow-job-2',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  });

  test('forwards an explicitly selected PPO trainer state for continuation', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ status: 'running', job_id: 'flow-job-3' }));
    const request = {
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/flow-job-2/pretrained_model',
      policy_type: 'multi_task_dit',
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
      resume_checkpoint: '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/flow-job-2/training_state/trainer_state.pt',
    };

    await expect(startFlowSDEPPOTraining(request)).resolves.toEqual({
      status: 'running',
      job_id: 'flow-job-3',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
  });

  test('reads Flow-SDE PPO status without browser caching', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ ready: true, status: 'idle' }));

    await expect(getFlowSDEPPOStatus()).resolves.toEqual({
      ready: true,
      status: 'idle',
    });
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/status', {
      cache: 'no-store',
    });
  });

  test('stops the exact Flow-SDE PPO job', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ status: 'running', job_id: 'flow-job-1' }));

    await stopFlowSDEPPOTraining('flow-job-1');

    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: 'flow-job-1' }),
    });
  });

  test.each(['success', 'fail', 'cancel'])(
    'submits the %s outcome to the exact Flow-SDE PPO job',
    async (outcome) => {
      global.fetch.mockResolvedValue(jsonResponse({
        status: 'running',
        job_id: 'flow-job-1',
      }));

      await submitFlowSDEPPOOutcome('flow-job-1', outcome);

      expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/outcome', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: 'flow-job-1', outcome }),
      });
    }
  );

  test('uses the dedicated rollout API for on-policy collection', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      operation: 'collect',
      job_id: 'rollout-job-1',
    }));
    const request = {
      policy_checkpoint: '/workspace/checkpoint/multi_task_dit/showroom/pretrained_model',
      policy_type: 'multi_task_dit',
      algorithm: 'flow_sde_ppo',
      robot_type: 'ffw_sg2_rev1',
      task_instruction: 'Pick up the jelly bag',
      episodes: 1,
    };

    await startFlowSDEPPOPolicyRollout(request);
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/rollout/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    global.fetch.mockClear();
    global.fetch.mockResolvedValue(jsonResponse({ status: 'running' }));
    await getFlowSDEPPOPolicyRolloutStatus();
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/rollout/status', {
      cache: 'no-store',
    });
    global.fetch.mockClear();
    await stopFlowSDEPPOPolicyRollout('rollout-job-1');
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/rollout/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: 'rollout-job-1' }),
    });
  });

  test.each(['success', 'fail', 'cancel'])(
    'submits %s only through the rollout outcome API',
    async (outcome) => {
      global.fetch.mockResolvedValue(jsonResponse({ status: 'running' }));
      await submitFlowSDEPPOPolicyRolloutOutcome('rollout-job-1', outcome);
      expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/rollout/outcome', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: 'rollout-job-1', outcome }),
      });
    }
  );

  test('updates from only the selected sealed rollout bundle', async () => {
    const bundle = '/workspace/checkpoint/multi_task_dit/flow_sde_ppo/a/rollouts/b';
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'running',
      operation: 'update',
      job_id: 'update-job-1',
    }));

    await startFlowSDEPPOUpdate(bundle);
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/update/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rollout_bundle: bundle }),
    });
    global.fetch.mockClear();
    global.fetch.mockResolvedValue(jsonResponse({ status: 'running' }));
    await getFlowSDEPPOUpdateStatus();
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/update/status', {
      cache: 'no-store',
    });
    global.fetch.mockClear();
    await stopFlowSDEPPOUpdate('update-job-1');
    expect(global.fetch).toHaveBeenCalledWith('/api/flow-sde-ppo/update/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: 'update-job-1' }),
    });
  });

  test('starts offline Flow-SDE PPO value warm-up with the selected replay roots', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ status: 'running', job_id: 'warmup-job-1' }));
    const request = {
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
    };

    await expect(startFlowSDEPPOValueWarmup(request)).resolves.toEqual({
      status: 'running',
      job_id: 'warmup-job-1',
    });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/flow-sde-ppo/value-warmup/start',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      }
    );
  });

  test('reads Flow-SDE PPO value warm-up status without browser caching', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      status: 'completed',
      percentage: 100,
      bundle_path: '/workspace/checkpoint/multi_task_dit/value_warmup/job-1',
    }));

    await expect(getFlowSDEPPOValueWarmupStatus()).resolves.toEqual({
      status: 'completed',
      percentage: 100,
      bundle_path: '/workspace/checkpoint/multi_task_dit/value_warmup/job-1',
    });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/flow-sde-ppo/value-warmup/status',
      { cache: 'no-store' }
    );
  });

  test('stops the exact Flow-SDE PPO value warm-up job', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ status: 'running', job_id: 'warmup-job-1' }));

    await stopFlowSDEPPOValueWarmup('warmup-job-1');

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/flow-sde-ppo/value-warmup/stop',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: 'warmup-job-1' }),
      }
    );
  });

  test('reads one LeRobot dataset summary without caching', async () => {
    global.fetch.mockResolvedValue(
      jsonResponse({ codebase_version: 'v3.0', total_episodes: 2 })
    );

    await getOfflineRLDatasetInfo('/workspace/lerobot/task v30');

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/offline-rl/dataset?dataset_path=%2Fworkspace%2Flerobot%2Ftask+v30',
      { cache: 'no-store' }
    );
  });

  test('reads one LeRobot episode trajectory without caching', async () => {
    global.fetch.mockResolvedValue(
      jsonResponse({ joint_names: ['arm_l_joint1'], joint_timestamps: [0] })
    );

    await getOfflineRLDatasetEpisodeData(
      '/workspace/lerobot/task v30',
      7
    );

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/offline-rl/dataset/episode-data?dataset_path=%2Fworkspace%2Flerobot%2Ftask+v30&episode_index=7',
      { cache: 'no-store' }
    );
  });

  test('discovers LeRobot datasets recursively below the selected destination', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ datasets: [] }));

    await getOfflineRLDatasets('/workspace/lerobot/round 1');

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/offline-rl/datasets?root_path=%2Fworkspace%2Flerobot%2Fround+1',
      { cache: 'no-store' }
    );
  });

  test('reserves an immutable Data Epoch before Offline RL conversion', async () => {
    global.fetch.mockResolvedValue(jsonResponse({
      data_epoch: 2,
      epoch_name: 'data_epoch_0002',
    }));
    const request = {
      destination_root: '/workspace/lerobot/RLTEST',
      source_mcap: '/workspace/rosbag2/Task_01',
      behavior_policy_path: '/workspace/model/lerobot/act',
      boundary_reason: 'manual_conversion',
      fps: 15,
      formats: ['v3.0'],
    };

    await expect(reserveOfflineRLDataEpoch(request)).resolves.toEqual({
      data_epoch: 2,
      epoch_name: 'data_epoch_0002',
    });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/offline-rl/data-epochs/reserve',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      }
    );
  });

  test('requests transactional LeRobot episode deletion', async () => {
    global.fetch.mockResolvedValue(jsonResponse({ total_episodes: 1 }));

    await deleteOfflineRLDatasetEpisodes('/workspace/lerobot/task', [1]);

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/offline-rl/dataset/delete-episodes',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_path: '/workspace/lerobot/task',
          episode_indices: [1],
        }),
      }
    );
  });

  test('surfaces backend error details', async () => {
    global.fetch.mockResolvedValue(
      jsonResponse({ detail: 'Episode count exceeds 200' }, { ok: false, status: 400 })
    );

    await expect(getOfflineRLStatus()).rejects.toThrow('Episode count exceeds 200');
  });
});
