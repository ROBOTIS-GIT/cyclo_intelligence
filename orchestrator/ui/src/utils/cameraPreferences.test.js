import { readCameraPreferences, writeCameraPreferences } from './cameraPreferences';

beforeEach(() => localStorage.clear());
afterEach(() => jest.restoreAllMocks());

test('keeps robot preferences separate, including all cameras hidden', () => {
  writeCameraPreferences('sg2', [], { '/cam': 270 });
  writeCameraPreferences('bg2', ['/cam'], { '/cam': 90 });
  expect(readCameraPreferences('sg2')).toEqual({ topics: [], rotations: { '/cam': 270 } });
  expect(readCameraPreferences('bg2')).toEqual({ topics: ['/cam'], rotations: { '/cam': 90 } });
  expect(readCameraPreferences('new')).toBeNull();
});

test('does not create preferences until camera selection is known', () => {
  writeCameraPreferences('sg2', null, {});
  writeCameraPreferences('', [], {});
  expect(localStorage.length).toBe(0);
});

test.each(['{bad', '{}', '{"topics":[null]}', '{"topics":[""]}'])(
  'ignores corrupt preferences: %s', (value) => {
    writeCameraPreferences('sg2', [], {});
    localStorage.setItem(localStorage.key(0), value);
    expect(readCameraPreferences('sg2')).toBeNull();
  }
);

test('deduplicates topics and rejects malformed angles', () => {
  writeCameraPreferences('sg2', ['/a', '/a'], { '/a': 90, '/b': '180', '/c': -90, '/d': null });
  expect(readCameraPreferences('sg2')).toEqual({ topics: ['/a'], rotations: { '/a': 90 } });
});

test('storage security/quota errors do not break controls', () => {
  jest.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
  jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('quota'); });
  expect(readCameraPreferences('sg2')).toBeNull();
  expect(() => writeCameraPreferences('sg2', ['/a'], {})).not.toThrow();
});
