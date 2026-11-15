// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdCloudDownload,
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
import {
  createDockerPullProgressTracker,
  parseDockerPullSseBlock,
} from '../utils/dockerPullProgress';

const API_BASE = '/api';
const HUGGINGFACE_ENDPOINT = 'https://huggingface.co';

const stateLabels = {
  running: 'Running',
  exited: 'Stopped',
  not_created: 'Not created',
  unknown: 'Unknown',
};

const getBackendLabel = (serviceType) => {
  if (serviceType === 'groot') return 'GR00T Docker';
  if (serviceType === 'lerobot') return 'LeRobot Docker';
  return 'Policy Docker';
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

function getPullErrorMessage(data) {
  return data?.message ||
    data?.error ||
    data?.errorDetail?.message ||
    'Image pull failed';
}

async function readPullStream(response, onProgress) {
  if (!response.body) {
    onProgress({ percent: null, message: 'Image pull started...', layers: 0 });
    return;
  }

  const tracker = createDockerPullProgressTracker();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = false;

  const handleBlock = (block) => {
    const item = parseDockerPullSseBlock(block);
    if (!item) return;
    if (item.event === 'error' || item.data?.error) {
      throw new Error(getPullErrorMessage(item.data));
    }
    if (item.event === 'done') {
      completed = true;
      onProgress(tracker.complete(item.data?.message || 'Image pull complete'));
      return;
    }
    onProgress(tracker.update(item.data));
  };

  while (true) {
    const { done, value } = await reader.read();
    if (value) {
      buffer += decoder.decode(value, { stream: !done });
    }
    if (done) {
      buffer += decoder.decode();
    }

    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || '';
    for (const block of blocks) {
      handleBlock(block);
    }

    if (done) break;
  }

  if (buffer.trim()) {
    handleBlock(buffer);
  }
  if (!completed) {
    onProgress(tracker.complete());
  }
}

export default function PolicyBackendControl({
  serviceType,
  variant = 'default',
  children = null,
}) {
  const backend = serviceType === 'groot' ? 'groot' : 'lerobot';
  const isOfflineRL = variant === 'offlineRL';
  const label = useMemo(
    () => getBackendLabel(serviceType),
    [serviceType]
  );

  const [status, setStatus] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);
  const [pullProgress, setPullProgress] = useState(null);
  const [showTokenPopup, setShowTokenPopup] = useState(false);
  const [isRegisteringToken, setIsRegisteringToken] = useState(false);
  const [tokenRegistered, setTokenRegistered] = useState(false);
  const { registerHFUser, listHFEndpoints } = useRosServiceCaller();

  useEffect(() => {
    if (!pullProgress || pendingAction === 'pull') return undefined;
    if (pullProgress.percent !== 100 || pullProgress.error) return undefined;
    const id = setTimeout(() => setPullProgress(null), 3000);
    return () => clearTimeout(id);
  }, [pendingAction, pullProgress]);

  const refreshStatus = useCallback(async ({ quiet = false } = {}) => {
    if (!quiet) setIsRefreshing(true);
    try {
      const response = await fetch(`${API_BASE}/backends/${backend}/status`);
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || `status failed (${response.status})`);
      }
      setStatus(data);
    } catch (error) {
      setStatus({
        container_state: 'unknown',
        image_pulled: false,
        raw_state: error.message,
      });
      if (!quiet) toast.error(`${label} status failed: ${error.message}`);
    } finally {
      if (!quiet) setIsRefreshing(false);
    }
  }, [backend, label]);

  useEffect(() => {
    refreshStatus({ quiet: true });
    const id = setInterval(() => refreshStatus({ quiet: true }), 5000);
    return () => clearInterval(id);
  }, [refreshStatus]);

  useEffect(() => {
    let cancelled = false;
    if (backend !== 'groot') return undefined;
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
  }, [backend, listHFEndpoints]);

  const callBackend = useCallback(async (action, successLabel) => {
    setPendingAction(action);
    try {
      const response = await fetch(`${API_BASE}/backends/${backend}/${action}`, {
        method: 'POST',
      });
      const data = await readJsonResponse(response);
      if (!response.ok || data.ok === false) {
        throw new Error(data.detail || data.message || `${action} failed`);
      }
      toast.success(`${label} ${successLabel}`);
      await refreshStatus({ quiet: true });
    } catch (error) {
      toast.error(`${label} ${action} failed: ${error.message}`);
      await refreshStatus({ quiet: true });
    } finally {
      setPendingAction(null);
    }
  }, [backend, label, refreshStatus]);

  const pullBackendImage = useCallback(async () => {
    setPendingAction('pull');
    setPullProgress({
      percent: null,
      message: 'Preparing image pull...',
      layers: 0,
    });
    try {
      const response = await fetch(`${API_BASE}/backends/${backend}/pull`, {
        method: 'POST',
      });
      if (!response.ok) {
        const data = await readJsonResponse(response);
        throw new Error(data.detail || data.message || `pull failed (${response.status})`);
      }
      await readPullStream(response, setPullProgress);
      await refreshStatus({ quiet: true });
      setPullProgress((previous) => ({
        ...(previous || {}),
        percent: 100,
        layerPercent: null,
        message: 'Image ready. Press ON.',
        detail: '',
      }));
      toast.success(`${label} image ready`);
    } catch (error) {
      setPullProgress((previous) => ({
        ...(previous || {}),
        percent: previous?.percent ?? null,
        message: error.message || 'Image pull failed',
        error: true,
      }));
      toast.error(`${label} pull failed: ${error.message}`);
      await refreshStatus({ quiet: true });
    } finally {
      setPendingAction(null);
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
  const isBusy = Boolean(pendingAction) || isRefreshing;
  const isPulling = pendingAction === 'pull';
  const isRunning = state === 'running';
  const imagePulled = Boolean(status?.image_pulled);
  const staleReason = getPolicyBackendStaleReason(status);
  const isStaleContainer = Boolean(staleReason);
  const showPullButton = isPulling || (hasStatus && !imagePulled);
  const showUpdateButton = hasStatus && imagePulled && isStaleContainer;
  const showRuntimeControls = !isStaleContainer &&
    (imagePulled || (hasStatus && state !== 'not_created'));
  const showTokenControl = backend === 'groot';
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
    isOfflineRL ? 'text-[9px]' : 'text-xs',
    'font-semibold',
    'px-2',
    'py-0.5',
    'rounded-full',
    {
      [isOfflineRL
        ? 'bg-[#e5ece5] text-[#607563]'
        : 'bg-green-100 text-green-700']: isRunning && readiness.ready,
      [isOfflineRL
        ? 'bg-[#f4e5e1] text-[#995e54]'
        : 'bg-red-100 text-red-700']: isStaleContainer,
      [isOfflineRL
        ? 'bg-[#ece8df] text-[#756e63]'
        : 'bg-gray-100 text-gray-600']: !isStaleContainer &&
        (state === 'exited' || state === 'not_created'),
      [isOfflineRL
        ? 'bg-[#f1ead9] text-[#8b724d]'
        : 'bg-yellow-100 text-yellow-700']: !isStaleContainer &&
        (state === 'unknown' || isWarming),
    }
  );

  const serviceStatusClass = (serviceState) => clsx(
    isOfflineRL ? 'text-[9px]' : 'text-xs',
    'font-semibold',
    'px-2',
    'py-0.5',
    'rounded-full',
    {
      [isOfflineRL
        ? 'bg-[#e5ece5] text-[#607563]'
        : 'bg-green-100 text-green-700']: serviceState === 'up',
      [isOfflineRL
        ? 'bg-[#f4e5e1] text-[#995e54]'
        : 'bg-red-100 text-red-700']: serviceState === 'down',
      [isOfflineRL
        ? 'bg-[#f1ead9] text-[#8b724d]'
        : 'bg-yellow-100 text-yellow-700']:
        serviceState !== 'up' && serviceState !== 'down',
    }
  );

  const buttonClass = (variant) => clsx(
    'h-8',
    'px-2.5',
    isOfflineRL ? 'rounded-lg' : 'rounded-md',
    isOfflineRL ? 'text-[10px]' : 'text-sm',
    'font-semibold',
    'flex',
    'items-center',
    'justify-center',
    'gap-1',
    'transition-colors',
    'disabled:opacity-40',
    'disabled:cursor-not-allowed',
    {
      [isOfflineRL
        ? 'bg-[#69866f] text-white hover:bg-[#5c7663]'
        : 'bg-blue-500 text-white hover:bg-blue-600']: variant === 'on',
      [isOfflineRL
        ? 'bg-[#81796d] text-white hover:bg-[#70695f]'
        : 'bg-gray-500 text-white hover:bg-gray-600']: variant === 'restart',
      [isOfflineRL
        ? 'bg-[#a86b68] text-white hover:bg-[#965d5a]'
        : 'bg-red-500 text-white hover:bg-red-600']: variant === 'off',
      [isOfflineRL
        ? 'bg-[#71806b] text-white hover:bg-[#64725f]'
        : 'bg-emerald-500 text-white hover:bg-emerald-600']: variant === 'pull',
      [isOfflineRL
        ? 'bg-[#a8795b] text-white hover:bg-[#966a4f]'
        : 'bg-amber-500 text-white hover:bg-amber-600']: variant === 'update',
      [isOfflineRL
        ? 'bg-[#7b7180] text-white hover:bg-[#6c6371]'
        : 'bg-violet-500 text-white hover:bg-violet-600']: variant === 'token',
    }
  );

  const pullPercent = Number.isFinite(pullProgress?.percent)
    ? pullProgress.percent
    : null;
  const layerPercent = Number.isFinite(pullProgress?.layerPercent)
    ? pullProgress.layerPercent
    : null;
  const pullProgressLabel = pullPercent === null
    ? (layerPercent === null ? '...' : `Layer ${layerPercent}%`)
    : `${pullPercent}%`;
  const pullBarWidth = `${pullPercent ?? layerPercent ?? (isPulling ? 8 : 0)}%`;

  return (
    <div
      className={clsx(
        'mb-3',
        isOfflineRL
          ? 'rounded-lg border border-[#ded8cc] bg-[#fbfaf6] px-2.5 py-2.5'
          : 'border-t border-b border-gray-200 py-2'
      )}
      data-appearance={isOfflineRL ? 'offline-rl' : 'default'}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className={clsx(
          'font-semibold',
          isOfflineRL ? 'text-[10px] text-[#5f584e]' : 'text-sm text-gray-700'
        )}
        >
          {label}
        </div>
        <div className="flex items-center gap-1.5">
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
              content="Image is not available locally. Connect internet and pull/install before first start."
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
        showUpdateButton || (showPullButton && !showRuntimeControls) ? 'grid-cols-1' : (
          showPullButton ? 'grid-cols-2' : (
            showTokenControl ? 'grid-cols-2' : 'grid-cols-3'
          )
        )
      )}
      >
        {showPullButton && (
          <button
            type="button"
            className={buttonClass('pull')}
            disabled={isBusy}
            onClick={pullBackendImage}
            aria-label={`${label} pull image`}
          >
            <MdCloudDownload size={16} />
            Pull
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
      {pullProgress && (
        <Tooltip
          position="bottom"
          content={pullProgress.message}
          disabled={!pullProgress.message}
          className="w-full"
        >
          <div className={clsx(
            'mt-2',
            'w-full',
            'rounded-md',
            isOfflineRL ? 'bg-[#f3efe7]' : 'bg-gray-50',
            'px-2',
            'py-2',
            pullProgress.error && 'bg-red-50'
          )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className={clsx(
                'min-w-0',
                'truncate',
                'text-xs',
                'font-medium',
                pullProgress.error ? 'text-red-700' : 'text-gray-600'
              )}
              >
                {pullProgress.message || 'Pulling image...'}
              </span>
              <span className={clsx(
                'shrink-0',
                'text-xs',
                'font-semibold',
                pullProgress.error ? 'text-red-700' : 'text-gray-700'
              )}
              >
                {pullProgressLabel}
              </span>
            </div>
            {pullProgress.detail && (
              <div className={clsx(
                'mt-0.5',
                'truncate',
                'text-[11px]',
                'font-medium',
                pullProgress.error ? 'text-red-600' : 'text-gray-500'
              )}
              >
                {pullProgress.detail}
              </div>
            )}
            <div className={clsx(
              'mt-1 h-1.5 overflow-hidden rounded-full',
              isOfflineRL ? 'bg-[#e2dcd1]' : 'bg-gray-200'
            )}
            >
              <div
                className={clsx(
                  'h-full',
                  'transition-all',
                  'duration-300',
                  pullProgress.error ? 'bg-red-500' : 'bg-emerald-500'
                )}
                style={{ width: pullBarWidth }}
              />
            </div>
          </div>
        </Tooltip>
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
                  <div className={clsx(
                    'flex items-center justify-between gap-2 rounded-md px-2 py-1',
                    isOfflineRL ? 'bg-[#f3efe7]' : 'bg-gray-50'
                  )}
                  >
                    <span className={clsx(
                      'font-medium',
                      isOfflineRL ? 'text-[9px] text-[#746d62]' : 'text-xs text-gray-600'
                    )}
                    >
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
            <div className="mt-1 text-xs font-medium text-yellow-700">
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
      {children}
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
