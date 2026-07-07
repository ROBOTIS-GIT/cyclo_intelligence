export const REMOTE_ZMQ_DEFAULTS = {
  remoteHost: '127.0.0.1',
  remotePort: 5555,
  remoteTimeoutMs: 300000,
};

export function getBackendArch(status = {}) {
  const safeStatus = status || {};
  const explicit = String(safeStatus.arch || '').trim().toLowerCase();
  if (explicit) return explicit;

  const image = String(safeStatus.image || '').trim().toLowerCase();
  if (/(^|[-_:])arm64($|[-_:])/.test(image) || /aarch64/.test(image)) {
    return 'arm64';
  }
  if (/(^|[-_:])amd64($|[-_:])/.test(image) || /x86_64/.test(image)) {
    return 'amd64';
  }
  return '';
}

export function isArmBackend(status = {}) {
  const arch = getBackendArch(status);
  return arch === 'arm64' || arch === 'aarch64';
}

export function formatZmqEndpoint(host, port) {
  const cleanHost = String(host || '').trim();
  const cleanPort = Number(port || 0);
  if (!cleanHost && !cleanPort) return '';
  if (!cleanPort) return cleanHost;
  const displayHost = cleanHost.includes(':') && !cleanHost.startsWith('[')
    ? `[${cleanHost}]`
    : cleanHost;
  return `${displayHost}:${cleanPort}`;
}

export function parseZmqEndpoint(value) {
  const raw = String(value || '').trim();
  if (!raw) {
    return {
      host: '',
      port: '',
      hasPort: false,
      isValidPort: true,
    };
  }

  const withoutScheme = raw.replace(/^[a-z][a-z0-9+.-]*:\/\//i, '');
  const slashIndex = withoutScheme.indexOf('/');
  const endpoint = slashIndex >= 0
    ? withoutScheme.slice(0, slashIndex)
    : withoutScheme;

  let host = endpoint;
  let port = '';
  let hasPort = false;
  const bracketMatch = endpoint.match(/^\[([^\]]+)\](?::([^:]+))?$/);
  if (bracketMatch) {
    host = bracketMatch[1];
    port = bracketMatch[2] || '';
    hasPort = bracketMatch[2] !== undefined;
  } else {
    const colonIndex = endpoint.lastIndexOf(':');
    const hasSingleColon = colonIndex > -1 && endpoint.indexOf(':') === colonIndex;
    if (hasSingleColon) {
      host = endpoint.slice(0, colonIndex);
      port = endpoint.slice(colonIndex + 1);
      hasPort = true;
    }
  }

  const portNumber = Number(port);
  const isValidPort = !hasPort ||
    (Number.isInteger(portNumber) && portNumber >= 1 && portNumber <= 65535);
  return {
    host: host.trim(),
    port: hasPort && isValidPort ? portNumber : port,
    hasPort,
    isValidPort,
  };
}

export function shouldUseRemoteRuntime(taskInfo = {}) {
  return String(taskInfo.serviceType || '').trim().toLowerCase() === 'rldx' &&
    String(taskInfo.rldxRuntimeMode || 'client').trim().toLowerCase() !== 'server';
}

export function withRuntimeDefaults(taskInfo = {}) {
  if (!shouldUseRemoteRuntime(taskInfo)) {
    return {
      ...taskInfo,
      remoteHost: '',
      remotePort: 0,
      remoteTimeoutMs: 0,
    };
  }

  return {
    ...taskInfo,
    remoteHost: taskInfo.remoteHost || REMOTE_ZMQ_DEFAULTS.remoteHost,
    remotePort: taskInfo.remotePort || REMOTE_ZMQ_DEFAULTS.remotePort,
    remoteTimeoutMs: taskInfo.remoteTimeoutMs || REMOTE_ZMQ_DEFAULTS.remoteTimeoutMs,
  };
}

export function getRuntimeValidationErrors(taskInfo = {}) {
  if (!shouldUseRemoteRuntime(taskInfo)) return [];

  const runtime = withRuntimeDefaults(taskInfo);
  const missingFields = [];
  const host = String(runtime.remoteHost || '').trim();
  const port = Number(runtime.remotePort || 0);
  const timeoutMs = Number(runtime.remoteTimeoutMs || 0);

  if (!host) missingFields.push('ZMQ Endpoint');
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    missingFields.push('ZMQ Endpoint port');
  }
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1) {
    missingFields.push('Timeout ms');
  }
  return missingFields;
}

export function buildRuntimeRequestFields(taskInfo = {}) {
  if (!shouldUseRemoteRuntime(taskInfo)) {
    return {
      remote_host: '',
      remote_port: 0,
      remote_timeout_ms: 0,
    };
  }

  const runtime = withRuntimeDefaults(taskInfo);
  return {
    remote_host: String(runtime.remoteHost || '').trim(),
    remote_port: Number(runtime.remotePort || 0),
    remote_timeout_ms: Number(runtime.remoteTimeoutMs || 0),
  };
}
