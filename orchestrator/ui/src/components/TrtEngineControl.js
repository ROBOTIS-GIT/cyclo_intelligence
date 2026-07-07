// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { MdBuild, MdRefresh } from 'react-icons/md';
import Tooltip from './Tooltip';

const API_BASE = '/api';

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function buildStatusUrl(modelPath, enginePath) {
  const params = new URLSearchParams();
  params.set('model_path', modelPath);
  if (enginePath) params.set('engine_path', enginePath);
  return `${API_BASE}/backends/groot/trt/status?${params.toString()}`;
}

function statusLabel(status) {
  if (status === 'ready') return 'Ready';
  if (status === 'building') return 'Building';
  if (status === 'failed') return 'Failed';
  if (status === 'unknown') return 'Unknown';
  return 'Missing';
}

function statusClass(status) {
  return clsx(
    'inline-flex',
    'items-center',
    'justify-center',
    'h-7',
    'min-w-20',
    'px-2',
    'rounded-md',
    'text-xs',
    'font-semibold',
    {
      'bg-emerald-100 text-emerald-700': status === 'ready',
      'bg-blue-100 text-blue-700': status === 'building',
      'bg-red-100 text-red-700': status === 'failed',
      'bg-gray-100 text-gray-600': !status || status === 'missing' || status === 'unknown',
    }
  );
}

export default function TrtEngineControl({
  modelPath,
  enginePath = '',
  robotType = '',
  taskInstruction = '',
  disabled = false,
  labelClassName = '',
}) {
  const [status, setStatus] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isBuilding, setIsBuilding] = useState(false);

  const trimmedModelPath = useMemo(() => String(modelPath || '').trim(), [modelPath]);
  const trimmedEnginePath = useMemo(() => String(enginePath || '').trim(), [enginePath]);
  const canQuery = trimmedModelPath.length > 0;
  const isStatusBuilding = status?.status === 'building';

  const refreshStatus = useCallback(async ({ quiet = false } = {}) => {
    if (!canQuery) {
      setStatus(null);
      return null;
    }
    if (!quiet) setIsRefreshing(true);
    try {
      const response = await fetch(buildStatusUrl(trimmedModelPath, trimmedEnginePath));
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || data.message || `status failed (${response.status})`);
      }
      setStatus(data);
      return data;
    } catch (error) {
      const failed = {
        status: 'unknown',
        message: error.message,
      };
      setStatus(failed);
      if (!quiet) toast.error(`TRT status failed: ${error.message}`);
      return failed;
    } finally {
      if (!quiet) setIsRefreshing(false);
    }
  }, [canQuery, trimmedModelPath, trimmedEnginePath]);

  useEffect(() => {
    refreshStatus({ quiet: true });
  }, [refreshStatus]);

  useEffect(() => {
    if (!canQuery) return undefined;
    const intervalMs = isStatusBuilding ? 2000 : 5000;
    const id = setInterval(() => refreshStatus({ quiet: true }), intervalMs);
    return () => clearInterval(id);
  }, [canQuery, isStatusBuilding, refreshStatus]);

  const handleBuild = useCallback(async () => {
    if (!trimmedModelPath) {
      toast.error('Select a policy path first');
      return;
    }
    if (!robotType) {
      toast.error('Select a robot type first');
      return;
    }
    setIsBuilding(true);
    try {
      const response = await fetch(`${API_BASE}/backends/groot/trt/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: trimmedModelPath,
          engine_path: trimmedEnginePath,
          robot_type: robotType,
          task_instruction: taskInstruction || 'dummy task',
          force: status?.status === 'ready',
        }),
      });
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || data.message || `build failed (${response.status})`);
      }
      setStatus(data);
      toast.success(data.status === 'ready' ? 'TRT engine ready' : 'TRT build started');
    } catch (error) {
      toast.error(`TRT build failed: ${error.message}`);
      await refreshStatus({ quiet: true });
    } finally {
      setIsBuilding(false);
    }
  }, [
    trimmedModelPath,
    trimmedEnginePath,
    robotType,
    taskInstruction,
    status?.status,
    refreshStatus,
  ]);

  const currentStatus = status?.status || 'missing';
  const message = status?.message || '';
  const buildDisabled = disabled || isBuilding || isStatusBuilding || !trimmedModelPath || !robotType;
  const refreshDisabled = disabled || isRefreshing || !trimmedModelPath;
  const buildText = currentStatus === 'ready' ? 'Rebuild' : 'Build';

  return (
    <div className={clsx('flex', 'items-center', 'mb-2.5')}>
      <span className={labelClassName}>TRT Engine</span>
      <div className="flex flex-1 min-w-0 items-center gap-2">
        <Tooltip
          content={message || status?.engine_path || ''}
          position="bottom"
          disabled={!message && !status?.engine_path}
        >
          <span className={statusClass(currentStatus)}>
            {statusLabel(currentStatus)}
          </span>
        </Tooltip>
        <button
          type="button"
          onClick={handleBuild}
          disabled={buildDisabled}
          className="h-8 px-2 rounded-md bg-blue-500 text-white text-sm font-semibold hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-1"
        >
          <MdBuild size={16} />
          {isBuilding ? 'Starting' : buildText}
        </button>
        <button
          type="button"
          onClick={() => refreshStatus()}
          disabled={refreshDisabled}
          className="h-8 w-8 rounded-md bg-gray-200 text-gray-600 hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          aria-label="Refresh TRT engine status"
        >
          <MdRefresh className={isRefreshing ? 'animate-spin' : ''} size={18} />
        </button>
      </div>
    </div>
  );
}
