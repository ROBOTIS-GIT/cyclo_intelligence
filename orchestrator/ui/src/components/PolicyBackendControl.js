// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdBuild,
  MdKey,
  MdPowerSettingsNew,
  MdRefresh,
  MdStop,
} from 'react-icons/md';
import Tooltip from './Tooltip';
import TokenInputPopup from './TokenInputPopup';
import {
  getPolicyBackendReadiness,
  getPolicyBackendServiceLabel,
  getPolicyBackendServices,
  getPolicyBackendStaleReason,
} from '../hooks/usePolicyBackendStatus';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

const API_BASE = '/api';
const HUGGINGFACE_ENDPOINT = 'https://huggingface.co';

const stateLabels = {
  running: 'Running',
  exited: 'Stopped',
  not_created: 'Not created',
  unknown: 'Unknown',
};

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export default function PolicyBackendControl({ serviceType, runtime = null }) {
  return <WorkerControl key={runtime?.id || serviceType} serviceType={serviceType} runtime={runtime} />;
}

function WorkerControl({ serviceType, runtime }) {
  const backend = runtime?.id || String(serviceType || '').trim();
  const label = useMemo(
    () => `${runtime?.label || 'Policy'} Docker`,
    [runtime]
  );

  const [status, setStatus] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);
  const [statusError, setStatusError] = useState('');
  const alive = useRef(true);
  const statusSequence = useRef(0);
  const actionPending = useRef(false);
  const [showTokenPopup, setShowTokenPopup] = useState(false);
  const [isRegisteringToken, setIsRegisteringToken] = useState(false);
  const [tokenRegistered, setTokenRegistered] = useState(false);
  const { registerHFUser, listHFEndpoints } = useRosServiceCaller();

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  const refreshStatus = useCallback(async ({ quiet = false } = {}) => {
    if (!backend) return;
    const sequence = ++statusSequence.current;
    if (!quiet) setIsRefreshing(true);
    try {
      const response = await fetch(`${API_BASE}/backends/${backend}/status`);
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || `status failed (${response.status})`);
      }
      if (!alive.current || sequence !== statusSequence.current) return;
      setStatus(data);
      setStatusError('');
    } catch (error) {
      if (!alive.current || sequence !== statusSequence.current) return;
      setStatusError(error.message);
      if (!quiet) toast.error(`${label} status failed: ${error.message}`);
    } finally {
      if (alive.current && !quiet) setIsRefreshing(false);
    }
  }, [backend, label]);

  useEffect(() => {
    let cancelled = false;
    let timer;
    const poll = async () => {
      await refreshStatus({ quiet: true });
      if (!cancelled) timer = setTimeout(poll, 5000);
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [refreshStatus]);

  useEffect(() => {
    let cancelled = false;
    if (!runtime?.capabilities?.requires_hf_token) return undefined;
    listHFEndpoints()
      .then((result) => {
        if (cancelled || !result?.success) return;
        const endpoints = result.endpoints || [];
        setTokenRegistered(
          endpoints.some((entry) => entry.endpoint === HUGGINGFACE_ENDPOINT)
        );
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [listHFEndpoints, runtime]);

  const callBackend = useCallback(async (action, successLabel) => {
    if (actionPending.current) return;
    actionPending.current = true;
    setPendingAction(action);
    try {
      const response = await fetch(`${API_BASE}/backends/${backend}/${action}`, {
        method: 'POST',
      });
      const data = await readJsonResponse(response);
      if (!response.ok || data.ok === false) {
        throw new Error(data.detail || data.message || `${action} failed`);
      }
      if (!alive.current) return;
      toast.success(`${label} ${successLabel}`);
      await refreshStatus({ quiet: true });
    } catch (error) {
      if (!alive.current) return;
      toast.error(`${label} ${action} failed: ${error.message}`);
      await refreshStatus({ quiet: true });
    } finally {
      actionPending.current = false;
      if (alive.current) setPendingAction(null);
    }
  }, [backend, label, refreshStatus]);

  const buildBackendImage = useCallback(async () => {
    if (actionPending.current) return;
    actionPending.current = true;
    ++statusSequence.current;
    setPendingAction('build');
    try {
      const response = await fetch(`${API_BASE}/backends/${backend}/build`, {
        method: 'POST',
      });
      const data = await readJsonResponse(response);
      if (!alive.current) return;
      if (!response.ok) {
        throw new Error(data.detail || data.message || `build failed (${response.status})`);
      }
      ++statusSequence.current;
      setStatus(previous => ({ ...previous, build: data }));
      await refreshStatus({ quiet: true });
    } catch (error) {
      if (!alive.current) return;
      toast.error(`${label} build failed: ${error.message}`);
      await refreshStatus({ quiet: true });
    } finally {
      actionPending.current = false;
      if (alive.current) setPendingAction(null);
    }
  }, [backend, label, refreshStatus]);

  const handleTokenSubmit = useCallback(async ({ token, label: tokenLabel = '' }) => {
    if (!token || !token.trim()) {
      toast.error('Please enter a Hugging Face token');
      return;
    }
    setIsRegisteringToken(true);
    try {
      const result = await registerHFUser({
        endpoint: HUGGINGFACE_ENDPOINT,
        label: tokenLabel || 'Hugging Face',
        token,
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Token registration failed');
      }
      setTokenRegistered(true);
      setShowTokenPopup(false);
      toast.success('Hugging Face token registered');
    } catch (error) {
      toast.error(`HF token registration failed: ${error.message}`);
    } finally {
      setIsRegisteringToken(false);
    }
  }, [registerHFUser]);

  const state = status?.container_state || 'unknown';
  const hasStatus = Boolean(status);
  const build = status?.build;
  const isBuilding = build?.state === 'building' || pendingAction === 'build';
  const isBusy = Boolean(pendingAction) || isRefreshing || isBuilding || Boolean(statusError);
  const isRunning = state === 'running';
  const imagePulled = Boolean(status?.image_pulled);
  const staleReason = getPolicyBackendStaleReason(status);
  const isStaleContainer = Boolean(staleReason);
  const showBuildButton = hasStatus && !imagePulled;
  const showUpdateButton = hasStatus && imagePulled && isStaleContainer;
  const showRuntimeControls = !isStaleContainer &&
    (imagePulled || (hasStatus && state !== 'not_created'));
  const showTokenControl = Boolean(runtime?.capabilities?.requires_hf_token);
  const readiness = useMemo(() => getPolicyBackendReadiness(status), [status]);
  const isWarming = isRunning && !readiness.ready &&
    (readiness.state === 'checking' || readiness.state === 'warming');
  const statusLabel = isStaleContainer
    ? 'Update required'
    : (
      isWarming
        ? 'Warming up'
        : stateLabels[state] || stateLabels.unknown
    );
  const serviceRows = useMemo(
    () => getPolicyBackendServices(status),
    [status]
  );

  const statusClass = clsx(
    'text-xs',
    'font-semibold',
    'px-2',
    'py-0.5',
    'rounded-full',
    {
      'bg-green-100 text-green-700': isRunning && readiness.ready,
      'bg-red-100 text-red-700': isStaleContainer,
      'bg-gray-100 text-gray-600': !isStaleContainer &&
        (state === 'exited' || state === 'not_created'),
      'bg-yellow-100 text-yellow-700': !isStaleContainer &&
        (state === 'unknown' || isWarming),
    }
  );

  const serviceStatusClass = (serviceState) => clsx(
    'text-xs',
    'font-semibold',
    'px-2',
    'py-0.5',
    'rounded-full',
    {
      'bg-green-100 text-green-700': serviceState === 'up',
      'bg-red-100 text-red-700': serviceState === 'down',
      'bg-yellow-100 text-yellow-700': serviceState !== 'up' && serviceState !== 'down',
    }
  );

  const buttonClass = (variant) => clsx(
    'h-8',
    'px-2.5',
    'rounded-md',
    'text-sm',
    'font-semibold',
    'flex',
    'items-center',
    'justify-center',
    'gap-1',
    'transition-colors',
    'disabled:opacity-40',
    'disabled:cursor-not-allowed',
    {
      'bg-blue-500 text-white hover:bg-blue-600': variant === 'on',
      'bg-gray-500 text-white hover:bg-gray-600': variant === 'restart',
      'bg-red-500 text-white hover:bg-red-600': variant === 'off',
      'bg-emerald-500 text-white hover:bg-emerald-600': variant === 'build',
      'bg-amber-500 text-white hover:bg-amber-600': variant === 'update',
      'bg-violet-500 text-white hover:bg-violet-600': variant === 'token',
    }
  );

  return (
    <div className="mb-3">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="text-sm font-semibold text-gray-700">{label}</div>
        <div className="flex flex-wrap items-center gap-1.5">
          {imagePulled && (
            <Tooltip position="bottom" content="Rebuild Worker image from local source">
              <button type="button" className="flex h-8 w-8 shrink-0 items-center justify-center rounded text-gray-600 hover:bg-gray-100 disabled:opacity-40"
                disabled={isBusy} onClick={buildBackendImage} aria-label={`${label} rebuild image`}>
                <MdBuild size={16} />
              </button>
            </Tooltip>
          )}
          <span className={statusClass}>
            {statusLabel}
          </span>
          {isStaleContainer && (
            <Tooltip
              position="bottom"
              content={staleReason === 'stale_image'
                ? `Container was created from an older image. Expected ${status?.image || 'the current backend image'}.`
                : 'Container workspace or required mounts are out of date. Update it before starting.'}
            >
              <span className="text-xs font-semibold text-red-700 bg-red-100 px-2 py-0.5 rounded-full">
                Container outdated
              </span>
            </Tooltip>
          )}
          {!imagePulled && (
            <Tooltip
              position="bottom"
              content="Build the Worker image from the local Cyclo checkout before first start."
            >
              <span className="text-xs font-semibold text-yellow-700 bg-yellow-100 px-2 py-0.5 rounded-full">
                Image missing
              </span>
            </Tooltip>
          )}
        </div>
      </div>
      <div className={clsx(
        'grid',
        'gap-2',
        showUpdateButton || (showBuildButton && !showRuntimeControls) ? 'grid-cols-1' : (
          showBuildButton ? 'grid-cols-2' : (
            showTokenControl ? 'grid-cols-2' : 'grid-cols-3'
          )
        )
      )}
      >
        {showBuildButton && (
          <button
            type="button"
            className={buttonClass('build')}
            disabled={isBusy}
            onClick={buildBackendImage}
            aria-label={`${label} build image`}
          >
            <MdBuild size={16} />
            {isBuilding ? 'Building...' : 'Build'}
          </button>
        )}
        {showUpdateButton && (
          <button
            type="button"
            className={buttonClass('update')}
            disabled={isBusy}
            onClick={() => callBackend('recreate', 'container updated')}
            aria-label={`${label} update container`}
          >
            <MdRefresh size={16} />
            Update Container
          </button>
        )}
        {showRuntimeControls && (
          <>
            {imagePulled && (
              <>
                <button
                  type="button"
                  className={buttonClass('on')}
                  disabled={isBusy}
                  onClick={() => callBackend('start', 'started')}
                  aria-label={`${label} on`}
                >
                  <MdPowerSettingsNew size={16} />
                  ON
                </button>
                {showTokenControl && (
                  <button
                    type="button"
                    className={buttonClass('token')}
                    disabled={isBusy || isRegisteringToken}
                    onClick={() => setShowTokenPopup(true)}
                    aria-label={`${label} Hugging Face token`}
                  >
                    <MdKey size={16} />
                    HF Token
                  </button>
                )}
                <button
                  type="button"
                  className={buttonClass('restart')}
                  disabled={isBusy}
                  onClick={() => callBackend('restart', 'restarted')}
                  aria-label={`${label} restart`}
                >
                  <MdRefresh size={16} />
                  Restart
                </button>
              </>
            )}
            <button
              type="button"
              className={buttonClass('off')}
              disabled={isBusy}
              onClick={() => callBackend('stop', 'stopped')}
              aria-label={`${label} off`}
            >
              <MdStop size={16} />
              OFF
            </button>
          </>
        )}
      </div>
      {statusError && <div role="alert" className="mt-2 break-words text-xs text-red-700">{statusError}</div>}
      {build && build.state !== 'idle' && (
        <div className="mt-2 min-w-0 text-xs">
          <div role={build.state === 'failed' ? 'alert' : 'status'}
            className={clsx('break-words', build.state === 'failed' ? 'text-red-700' : 'text-gray-600')}>
            {build.message}
          </div>
          {build.log_tail?.length > 0 && (
            <details className="mt-1">
              <summary className="cursor-pointer text-gray-600">Build log</summary>
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-all bg-gray-50 p-2 text-[11px]">{build.log_tail.join('\n')}</pre>
            </details>
          )}
        </div>
      )}
      {isRunning && (
        <>
          <div className="mt-2 grid grid-cols-2 gap-2">
            {serviceRows.map((service) => {
              const displayState = service.state === 'up'
                ? 'Up'
                : service.state === 'down'
                  ? 'Down'
                  : 'Unknown';
              return (
                <Tooltip
                  key={service.name}
                  position="bottom"
                  content={service.raw || displayState}
                >
                  <div className="flex items-center justify-between gap-2 rounded-md bg-gray-50 px-2 py-1">
                    <span className="text-xs font-medium text-gray-600">
                      {getPolicyBackendServiceLabel(service.name)}
                    </span>
                    <span className={serviceStatusClass(service.state)}>
                      {displayState}
                    </span>
                  </div>
                </Tooltip>
              );
            })}
          </div>
          {!readiness.ready && (
            <div className={`mt-1 text-xs font-medium ${readiness.state === 'error' ? 'text-red-600' : 'text-yellow-700'}`}>
              {readiness.message}
            </div>
          )}
        </>
      )}
      {showTokenControl && tokenRegistered && (
        <div className="mt-2 text-xs text-green-700">
          Hugging Face token registered for GR00T gated backbone downloads.
        </div>
      )}
      <TokenInputPopup
        isOpen={showTokenPopup}
        onClose={() => setShowTokenPopup(false)}
        onSubmit={handleTokenSubmit}
        isLoading={isRegisteringToken}
        title="Register Hugging Face Token"
        endpoint={HUGGINGFACE_ENDPOINT}
        defaultLabel="Hugging Face"
      />
    </div>
  );
}
