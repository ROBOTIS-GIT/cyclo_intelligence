import { useCallback, useEffect, useMemo, useState } from 'react';

const API_BASE = '/api';
const DEFAULT_POLL_MS = 2000;

export const POLICY_BACKEND_SERVICE_LABELS = {
  'main-runtime': 'Main',
  'engine-process': 'Engine',
  'inference-server': 'Inference',
  'control-publisher': 'Control',
};

const POLICY_BACKEND_SERVICE_GROUPS = [
  ['main-runtime', 'engine-process'],
  ['inference-server', 'control-publisher'],
];

export const getPolicyBackendName = (serviceType) => (
  serviceType === 'groot' ? 'groot' : 'lerobot'
);

export function getPolicyBackendServiceLabel(name) {
  return POLICY_BACKEND_SERVICE_LABELS[name] || name;
}

export function getPolicyBackendServices(status) {
  const services = status?.services || [];
  if (services.length === 0) return [];

  for (const group of POLICY_BACKEND_SERVICE_GROUPS) {
    if (group.some((name) => services.some((service) => service.name === name))) {
      return group.map((name) => (
        services.find((service) => service.name === name) || {
          name,
          state: 'unknown',
          raw: 'not reported',
        }
      ));
    }
  }

  return services;
}

export function getPolicyBackendStaleReason(status) {
  const rawState = status?.raw_state || '';
  if (
    rawState === 'stale_image' ||
    rawState === 'image_mismatch'
  ) {
    return 'stale_image';
  }
  if (
    rawState === 'workspace_mount_mismatch' ||
    rawState.startsWith('missing_required_mounts=')
  ) {
    return rawState;
  }
  if (status?.image_status === 'stale') {
    return 'stale_image';
  }
  return null;
}

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export function getPolicyBackendReadiness(status) {
  if (!status) {
    return {
      ready: false,
      state: 'checking',
      message: 'Checking backend...',
    };
  }
  const staleReason = getPolicyBackendStaleReason(status);
  const imageStatus = status.image_status ||
    (staleReason ? 'stale' : (
      status.image_pulled ? 'current' : 'missing'
    ));
  if (imageStatus === 'stale' || staleReason) {
    return {
      ready: false,
      state: 'update_required',
      message: staleReason === 'stale_image'
        ? 'Policy Docker image changed. Update container before starting.'
        : 'Policy Docker container changed. Update container before starting.',
    };
  }
  if (!status.image_pulled) {
    return {
      ready: false,
      state: 'missing_image',
      message: 'Policy image is not available',
    };
  }
  if (status.container_state !== 'running') {
    return {
      ready: false,
      state: 'stopped',
      message: 'Policy Docker is off',
    };
  }

  const services = getPolicyBackendServices(status);
  if (services.length === 0 || services.some((service) => service.state !== 'up')) {
    return {
      ready: false,
      state: 'warming',
      message: 'Backend processes are starting...',
    };
  }

  return {
    ready: true,
    state: 'ready',
    message: 'Backend ready',
  };
}

export default function usePolicyBackendStatus(
  serviceType,
  { enabled = true, intervalMs = DEFAULT_POLL_MS } = {}
) {
  const backend = useMemo(() => getPolicyBackendName(serviceType), [serviceType]);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const refreshStatus = useCallback(async ({ quiet = true } = {}) => {
    if (!enabled) return null;
    if (!quiet) setIsRefreshing(true);
    try {
      const response = await fetch(`${API_BASE}/backends/${backend}/status`);
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || `status failed (${response.status})`);
      }
      setStatus(data);
      setError('');
      return data;
    } catch (err) {
      const message = err?.message || 'status failed';
      setError(message);
      setStatus({
        container_state: 'unknown',
        image_pulled: false,
        raw_state: message,
      });
      return null;
    } finally {
      if (!quiet) setIsRefreshing(false);
    }
  }, [backend, enabled]);

  useEffect(() => {
    if (!enabled) return undefined;
    refreshStatus({ quiet: true });
    const id = setInterval(() => refreshStatus({ quiet: true }), intervalMs);
    return () => clearInterval(id);
  }, [enabled, intervalMs, refreshStatus]);

  return {
    backend,
    status,
    error,
    isRefreshing,
    refreshStatus,
    readiness: getPolicyBackendReadiness(status),
  };
}
