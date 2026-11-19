import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { readBackendStatus, subscribeBackendStatus } from '../utils/backendStatusPolling';
import useDocumentVisible from './useDocumentVisible';

const DEFAULT_POLL_MS = 2000;

export const POLICY_BACKEND_SERVICE_LABELS = {
  'main-runtime': 'Main',
  'engine-process': 'Engine',
  'inference-server': 'Inference',
  'control-publisher': 'Control',
};

const POLICY_BACKEND_SERVICE_GROUPS = [
  ['engine-process'],
  ['main-runtime', 'engine-process'],
  ['inference-server', 'control-publisher'],
];

export const getPolicyBackendName = (serviceType) => (
  String(serviceType || '').trim()
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
  if (status.worker_compatible === false) {
    return {
      ready: false,
      state: status.worker_readiness === 'error' ? 'error' : 'warming',
      message: status.worker_message || 'Waiting for a compatible model worker...',
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
  const visible = useDocumentVisible();
  const generation = useRef(0);

  const accept = useCallback((data, err) => {
    setError(err?.message || '');
    setStatus(err ? {
      container_state: 'unknown', image_pulled: false, raw_state: err.message,
    } : data);
  }, []);

  const refreshStatus = useCallback(async ({ quiet = true } = {}) => {
    if (!enabled || !backend) return null;
    const current = generation.current;
    if (!quiet) setIsRefreshing(true);
    try {
      const data = await readBackendStatus(backend);
      if (current === generation.current) accept(data);
      return data;
    } catch (err) {
      if (current === generation.current) accept(null, err);
      return null;
    } finally {
      if (!quiet && current === generation.current) setIsRefreshing(false);
    }
  }, [backend, enabled, accept]);

  useEffect(() => {
    generation.current += 1;
    setStatus(null);
    setError('');
    setIsRefreshing(false);
    const unsubscribe = enabled && visible && backend
      ? subscribeBackendStatus(backend, accept, intervalMs) : () => {};
    return () => { generation.current += 1; unsubscribe(); };
  }, [backend, enabled, visible, intervalMs, accept]);

  return {
    backend,
    status,
    error,
    isRefreshing,
    refreshStatus,
    readiness: getPolicyBackendReadiness(status),
  };
}
