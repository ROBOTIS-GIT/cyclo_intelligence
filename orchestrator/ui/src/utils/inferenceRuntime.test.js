import {
  formatZmqEndpoint,
  getBackendArch,
  isArmBackend,
  parseZmqEndpoint,
} from './inferenceRuntime';

describe('inference runtime backend arch helpers', () => {
  it('uses explicit backend arch when provided', () => {
    expect(getBackendArch({
      arch: 'arm64',
      image: 'robotis/rldx-zenoh:0.1.1-amd64',
    })).toBe('arm64');
  });

  it('falls back to parsing the image tag', () => {
    expect(getBackendArch({
      image: 'robotis/rldx-zenoh:0.1.1-arm64',
    })).toBe('arm64');
    expect(getBackendArch({
      image: 'robotis/rldx-zenoh:0.1.1-amd64',
    })).toBe('amd64');
  });

  it('detects arm backends', () => {
    expect(isArmBackend({ arch: 'arm64' })).toBe(true);
    expect(isArmBackend({ image: 'robotis/rldx-zenoh:0.1.1-amd64' })).toBe(false);
  });

  it('treats missing backend status as unknown arch', () => {
    expect(getBackendArch(null)).toBe('');
    expect(isArmBackend(null)).toBe(false);
  });
});

describe('ZMQ endpoint helpers', () => {
  it('formats host and port as a copyable endpoint', () => {
    expect(formatZmqEndpoint('192.168.60.150', 5555)).toBe('192.168.60.150:5555');
    expect(formatZmqEndpoint('fe80::1', 5555)).toBe('[fe80::1]:5555');
  });

  it('parses endpoints pasted from the server panel', () => {
    expect(parseZmqEndpoint('192.168.60.150:5555')).toEqual({
      host: '192.168.60.150',
      port: 5555,
      hasPort: true,
      isValidPort: true,
    });
    expect(parseZmqEndpoint('tcp://10.0.0.2:6000')).toEqual({
      host: '10.0.0.2',
      port: 6000,
      hasPort: true,
      isValidPort: true,
    });
    expect(parseZmqEndpoint('[fe80::1]:5555')).toEqual({
      host: 'fe80::1',
      port: 5555,
      hasPort: true,
      isValidPort: true,
    });
  });

  it('keeps host-only input editable', () => {
    expect(parseZmqEndpoint('192.168.60.150')).toEqual({
      host: '192.168.60.150',
      port: '',
      hasPort: false,
      isValidPort: true,
    });
  });
});
