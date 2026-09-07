import {
  deleteNavigationMission,
  deleteNavigationMissionBtFile,
  duplicateNavigationMission,
  renameNavigationMission,
  saveNavigationMissionBtFile,
  setNavigationMissionDefaultBtFile,
} from './navigationMissionsApi';

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

test('keeps the existing BT file save request unchanged without waypoint ownership', async () => {
  await saveNavigationMissionBtFile(
    'warehouse map',
    'locals/start.xml',
    '<root/>',
    'night shift',
  );

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/missions/warehouse%20map/bt?mission_name=night%20shift',
    expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({
        path: 'locals/start.xml',
        content: '<root/>',
      }),
    }),
  );
});

test('registers a saved BT file with its owning waypoint when requested', async () => {
  await saveNavigationMissionBtFile(
    'warehouse',
    'locals/start/alternate.xml',
    '<root/>',
    'inspection',
    { waypointId: 'start', expectedRevision: 7 },
  );

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/missions/warehouse/bt?mission_name=inspection',
    expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({
        path: 'locals/start/alternate.xml',
        content: '<root/>',
        waypoint_id: 'start',
        expected_revision: 7,
      }),
    }),
  );
});

test('persists the selected default BT file for a waypoint', async () => {
  await setNavigationMissionDefaultBtFile(
    'warehouse map',
    'start point',
    'locals/start point/default.xml',
    'night shift',
    { expectedRevision: 8 },
  );

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/missions/warehouse%20map/bt/default?mission_name=night%20shift',
    expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({
        waypoint_id: 'start point',
        path: 'locals/start point/default.xml',
        expected_revision: 8,
      }),
    }),
  );
});

test('deletes only the mission revision the user confirmed', async () => {
  await deleteNavigationMission(
    'warehouse map',
    'night shift',
    { expectedRevision: 9 },
  );

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/missions/warehouse%20map?mission_name=night%20shift&expected_revision=9',
    expect.objectContaining({ method: 'DELETE' }),
  );
});

test('deletes an unowned BT file at the expected mission revision', async () => {
  await deleteNavigationMissionBtFile(
    'warehouse',
    'legacy.xml',
    'night-shift',
    { expectedRevision: 12 },
  );

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/missions/warehouse/bt?path=legacy.xml&mission_name=night-shift&expected_revision=12',
    expect.objectContaining({ method: 'DELETE' }),
  );
});

test('renames only the mission revision the user loaded', async () => {
  await renameNavigationMission(
    'warehouse',
    'morning',
    'evening',
    { expectedRevision: 10 },
  );

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/missions/warehouse/rename',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        mission_name: 'morning',
        new_name: 'evening',
        expected_revision: 10,
      }),
    }),
  );
});

test('duplicates a stable source mission revision', async () => {
  await duplicateNavigationMission(
    'warehouse',
    'morning',
    'morning-copy',
    { expectedRevision: 11 },
  );

  expect(global.fetch).toHaveBeenCalledWith(
    '/api/navigation/missions/warehouse/duplicate',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        mission_name: 'morning',
        new_name: 'morning-copy',
        expected_revision: 11,
      }),
    }),
  );
});
