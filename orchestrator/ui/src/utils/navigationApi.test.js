import {
  cancelNavigateToPoseGoal,
  getPgmImage,
  getServiceStatus,
  startNavigation,
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

test('uses self-hosted endpoints for map files and action cancellation', async () => {
  await getPgmImage('warehouse/map.pgm');
  await cancelNavigateToPoseGoal();

  expect(global.fetch.mock.calls[0][0]).toBe(
    '/api/navigation/maps/pgm?path=warehouse%2Fmap.pgm'
  );
  expect(global.fetch.mock.calls[1][0]).toBe('/api/navigation/cancel');
});
