import { getOfflineRLStatus, startOfflineRLTraining } from './offlineRlApi';

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

  test('surfaces backend error details', async () => {
    global.fetch.mockResolvedValue(
      jsonResponse({ detail: 'Episode count exceeds 200' }, { ok: false, status: 400 })
    );

    await expect(getOfflineRLStatus()).rejects.toThrow('Episode count exceeds 200');
  });
});
