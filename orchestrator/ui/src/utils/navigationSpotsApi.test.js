import {
  createNavigationSpot,
  deleteNavigationSpot,
  getNavigationSpots,
  updateNavigationSpot,
} from './navigationSpotsApi';

beforeEach(() => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ spots: [] }),
  });
});

afterEach(() => {
  jest.restoreAllMocks();
});

test('uses same-origin Mission Canvas spot endpoints', async () => {
  await getNavigationSpots('factory');
  await createNavigationSpot({
    map_name: 'factory',
    label: 'Table A',
    pose: { frame_id: 'map', x: 1, y: 2, yaw: 0 },
  });
  await updateNavigationSpot('table_a', {
    map_name: 'factory',
    label: 'Prep Table',
  });
  await deleteNavigationSpot('table_a', 'factory');

  expect(global.fetch.mock.calls[0][0]).toBe(
    '/api/navigation/spots?map_name=factory'
  );
  expect(global.fetch.mock.calls[1][0]).toBe('/api/navigation/spots');
  expect(global.fetch.mock.calls[1][1]).toEqual(expect.objectContaining({
    method: 'POST',
  }));
  expect(global.fetch.mock.calls[2][0]).toBe('/api/navigation/spots/table_a');
  expect(global.fetch.mock.calls[2][1]).toEqual(expect.objectContaining({
    method: 'PATCH',
  }));
  expect(global.fetch.mock.calls[3][0]).toBe(
    '/api/navigation/spots/table_a?map_name=factory'
  );
  expect(global.fetch.mock.calls[3][1]).toEqual(expect.objectContaining({
    method: 'DELETE',
  }));
});
