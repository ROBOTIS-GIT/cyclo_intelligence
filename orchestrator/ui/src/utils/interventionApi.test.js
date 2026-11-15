import {
  getInterventionState,
  sendInterventionAction,
  sendOperatorPose,
} from './interventionApi';

const response = (payload, ok = true) => ({
  ok,
  text: jest.fn().mockResolvedValue(JSON.stringify(payload)),
});

describe('interventionApi', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    delete global.fetch;
  });

  test('reads shared intervention state without browser caching', async () => {
    global.fetch.mockResolvedValue(response({ active: false }));

    await expect(getInterventionState()).resolves.toEqual({ active: false });
    expect(global.fetch).toHaveBeenCalledWith('/api/intervention/state', {
      cache: 'no-store',
    });
  });

  test('forwards the selected Mini leader', async () => {
    global.fetch.mockResolvedValue(response({ ok: true }));

    await sendInterventionAction('start', { model: 'mini' });

    expect(global.fetch).toHaveBeenCalledWith('/api/intervention/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'mini' }),
    });
  });

  test('surfaces the controller safety reason', async () => {
    global.fetch.mockResolvedValue(response({
      detail: { message: 'Intervention safety gate failed' },
    }, false));

    await expect(sendInterventionAction('start', { model: 'a3' }))
      .rejects.toThrow('Intervention safety gate failed');
  });

  test('forwards operator-entered whole-body pose lists', async () => {
    global.fetch.mockResolvedValue(response({ ok: true }));
    const pose = {
      model: 'f2',
      left_arm: [0, 0, 0, 0, 0, 0, 0, 0],
      right_arm: [0, 0, 0, 0, 0, 0, 0, 0],
      head: [0, 0],
      lift: [0],
      duration: 8,
    };

    await sendOperatorPose(pose);

    expect(global.fetch).toHaveBeenCalledWith('/api/intervention/pose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pose),
    });
  });
});
