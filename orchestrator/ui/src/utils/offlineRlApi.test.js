import {
  deleteOfflineRLDatasetEpisodes,
  getFlowSDEPPOValueWarmupStatus,
  getFlowSDEPPOStatus,
  getImitationLearningStatus,
  getOfflineRLDatasetInfo,
  getOfflineRLDatasets,
  getOfflineRLStatus,
  reserveOfflineRLDataEpoch,
  startFlowSDEPPOTraining,
  startFlowSDEPPOValueWarmup,
  startImitationLearningTraining,
  startOfflineRLTraining,
  stopImitationLearningTraining,
  stopFlowSDEPPOTraining,
  stopFlowSDEPPOValueWarmup,
  stopOfflineRLTraining,
  submitFlowSDEPPOOutcome,
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
