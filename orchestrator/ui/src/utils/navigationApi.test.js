import {
  cancelNavigateToPoseGoal,
  configureDesignLocalizationAmcl,
  getMapAnnotations,
  getPgmImage,
  getServiceStatus,
  requestGlobalLocalization,
  saveMapAnnotations,
  saveNavigationMap,
  sendInitialPoseEstimate,
  requestNoMotionUpdate,
  sendNavigateToPoseGoalAndWait,
  sendNavigateThroughPosesGoalsAndWait,
  startNavigation,
  stopNavigation,
} from './navigationApi';

beforeEach(() => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ ok: true }),
  });
});

afterEach(() => {
  jest.restoreAllMocks();
});

test('uses the cyclo_intelligence same-origin navigation API', async () => {
  await getServiceStatus();

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/status',
    expect.any(Object)
  );
});

test('maps a mapping restart to the self-hosted start endpoint', async () => {
  await startNavigation('map', 'factory');

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/start',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ mode: 'map', map_name: 'factory' }),
    })
  );
});

test('maps localization starts to the supervisor localize mode', async () => {
  await startNavigation('localize', 'factory');

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/start',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ mode: 'localize', map_name: 'factory' }),
    })
  );
});

test('keeps the navigation stop request alive while the page exits', async () => {
  await stopNavigation({ keepalive: true });

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/stop',
    expect.objectContaining({
      method: 'POST',
      keepalive: true,
    })
  );
});

test('saves maps with the requested basename', async () => {
  await saveNavigationMap('factory');

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/save-map',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ map_name: 'factory' }),
    })
  );
});

test('uses self-hosted endpoints for map files and action cancellation', async () => {
  await getPgmImage('warehouse/map.pgm');
  await cancelNavigateToPoseGoal();

  expect(global.fetch.mock.calls[0][0]).toBe(
    '/api/navigation/maps/pgm?path=warehouse%2Fmap.pgm'
  );
  expect(global.fetch.mock.calls[1][0]).toBe('/api/navigation/cancel');
});

test('uses sidecar endpoints for map annotations', async () => {
  const annotations = [{
    id: 'mark_1',
    label: 'Dock',
    color: '#C96442',
    pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
  }];

  await getMapAnnotations('warehouse/map.pgm');
  await saveMapAnnotations('warehouse/map.pgm', annotations);

  expect(global.fetch.mock.calls[0][0]).toBe(
    '/api/navigation/maps/annotations?path=warehouse%2Fmap.pgm'
  );
  expect(global.fetch.mock.calls[1]).toEqual([
    '/api/navigation/maps/annotations/save',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ path: 'warehouse/map.pgm', annotations }),
    }),
  ]);
});

test('sends initial pose estimates through the supervisor API', async () => {
  await sendInitialPoseEstimate({
    x: 1.2,
    y: -0.4,
    yaw: 0.7,
    mapName: 'factory',
  });

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/initial-pose',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        x: 1.2,
        y: -0.4,
        yaw: 0.7,
        frame_id: 'map',
        map_name: 'factory',
      }),
    })
  );
});

test('requests a no-motion AMCL update through the supervisor API', async () => {
  await requestNoMotionUpdate();

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/nomotion-update',
    expect.objectContaining({ method: 'POST' })
  );
});

test('requests AMCL global localization through the supervisor API', async () => {
  await requestGlobalLocalization();

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/global-localization',
    expect.objectContaining({ method: 'POST' })
  );
});

test('configures design localization AMCL parameters through the supervisor API', async () => {
  await configureDesignLocalizationAmcl();

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/amcl/design-localization-params',
    expect.objectContaining({ method: 'POST' })
  );
});

test('waits for the tracked NavigateToPose result with cancellation support', async () => {
  const controller = new AbortController();
  const goal = { pose: { header: { frame_id: 'map' }, pose: {} } };

  await sendNavigateToPoseGoalAndWait(goal, controller.signal);

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/goal/wait',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify(goal),
      signal: controller.signal,
    })
  );
});

test('waits for a NavigateThroughPoses result with cancellation support', async () => {
  const controller = new AbortController();
  const goals = {
    poses: [
      { header: { frame_id: 'map' }, pose: {} },
      { header: { frame_id: 'map' }, pose: {} },
    ],
  };

  await sendNavigateThroughPosesGoalsAndWait(goals, controller.signal);

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/goals/wait',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify(goals),
      signal: controller.signal,
    })
  );
});
