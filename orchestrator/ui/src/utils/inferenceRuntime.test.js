import {
  buildRuntimeRequestFields,
  formatZmqEndpoint,
  getBackendArch,
  getRuntimeValidationErrors,
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

describe('RLDX runtime request fields', () => {
  it('uses remote defaults for client mode', () => {
    expect(buildRuntimeRequestFields({
      serviceType: 'rldx',
      rldxRuntimeMode: 'client',
    })).toEqual({
      remote_host: '127.0.0.1',
      remote_port: 5555,
      remote_timeout_ms: 300000,
    });
  });

  it('clears remote fields for server-only mode', () => {
    expect(buildRuntimeRequestFields({
      serviceType: 'rldx',
      rldxRuntimeMode: 'server',
      remoteHost: '10.0.0.2',
      remotePort: 6000,
    })).toEqual({
      remote_host: '',
      remote_port: 0,
      remote_timeout_ms: 0,
    });
  });

  it('rejects invalid explicit endpoint values', () => {
    expect(getRuntimeValidationErrors({
      serviceType: 'rldx',
      rldxRuntimeMode: 'client',
      remoteHost: '10.0.0.2',
      remotePort: 70000,
      remoteTimeoutMs: -1,
    })).toEqual(['ZMQ Endpoint port', 'Timeout ms']);
  });
});
