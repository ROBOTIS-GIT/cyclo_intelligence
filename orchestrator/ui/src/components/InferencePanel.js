// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Kiwoong Park

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { shallowEqual, useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdContentCopy,
  MdFolderOpen,
  MdHourglassEmpty,
  MdInfoOutline,
  MdPrecisionManufacturing,
  MdSync,
  MdViewInAr,
  MdWarningAmber,
} from 'react-icons/md';
import FileBrowserModal from './FileBrowserModal';
import InferenceModelSelector from './InferenceModelSelector';
import PolicyBackendControl from './PolicyBackendControl';
import TrtEngineControl from './TrtEngineControl';
import Tooltip from './Tooltip';
import usePolicyBackendStatus from '../hooks/usePolicyBackendStatus';
import { InferencePhase } from '../constants/taskPhases';
import { DEFAULT_PATHS } from '../constants/paths';
import {
  markLocalTaskInfoEdited,
  markInferenceTaskInfoSyncFailed,
  markInferenceTaskInfoSyncing,
  markInferenceTaskInfoSyncPending,
  markInferenceTaskInfoSyncSuccess,
  selectInferenceTaskInfo,
  setInferenceMode,
  setInferenceTaskInfo,
} from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { requiresInstruction } from '../constants/policyCapabilities';
import {
  formatZmqEndpoint,
  isArmBackend,
  parseZmqEndpoint,
  REMOTE_ZMQ_DEFAULTS,
  shouldUseRemoteRuntime,
} from '../utils/inferenceRuntime';
import { getInferenceTaskInfoKey } from '../utils/taskInfoSync';

const AUTO_SYNC_DELAY_MS = 700;

async function copyTextToClipboard(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand('copy');
  } finally {
    document.body.removeChild(textarea);
  }
}

const InferencePanel = () => {
  const dispatch = useDispatch();

  const info = useSelector(selectInferenceTaskInfo, shallowEqual);
  const taskInfoSync = useSelector((state) => state.tasks.inferenceTaskInfoSync);
  const robotType = useSelector((state) => state.tasks.robotType);
  const inferenceStatus = useSelector((state) => state.tasks.inferenceStatus);
  const showInstruction = requiresInstruction(info.serviceType, info.policyType);

  const [isTaskStatusPaused, setIsTaskStatusPaused] = useState(false);
  const [lastTaskStatusUpdate, setLastTaskStatusUpdate] = useState(Date.now());
  const [showPolicyBrowser, setShowPolicyBrowser] = useState(false);

  // InferencePage's lock — only the inference-side phase matters here.
  // Record phase is the InfoPanel's concern (D18, plan record-zippy-sunrise).
  const isTaskRunning = inferenceStatus.inferencePhase !== InferencePhase.READY;
  const isInferencing =
    inferenceStatus.inferencePhase === InferencePhase.INFERENCING;
  const inferenceMode = info.inferenceMode || 'simulation';
  const isRobotMode = inferenceMode === 'robot';
  const actionRequestMode =
    String(info.actionRequestMode || '').trim().toLowerCase() === 'sync'
      ? 'sync'
      : 'async';
  const isGrootModel = info.serviceType === 'groot';
  const isTensorRtEnabled = info.accelerationMode === 'tensorrt_dit';
  const trtTaskInstruction = (info.taskInstruction?.[0] || '').trim();
  const isModeSwitchLocked =
    inferenceStatus.inferencePhase === InferencePhase.LOADING;
  const isModelActive = [
    InferencePhase.INFERENCING,
    InferencePhase.PAUSED,
  ].includes(inferenceStatus.inferencePhase);
  const isRldx = info.serviceType === 'rldx';
  const rldxRuntimeMode = String(info.rldxRuntimeMode || 'client').toLowerCase();
  const rldxServerHost = (
    typeof window !== 'undefined' &&
    window.location.hostname
  )
    ? window.location.hostname
    : '127.0.0.1';
  const rldxServerPort = rldxRuntimeMode === 'server'
    ? REMOTE_ZMQ_DEFAULTS.remotePort
    : (info.remotePort || REMOTE_ZMQ_DEFAULTS.remotePort);
  const rldxServerEndpoint = formatZmqEndpoint(rldxServerHost, rldxServerPort);
  const remoteEndpointValue = formatZmqEndpoint(
    info.remoteHost || REMOTE_ZMQ_DEFAULTS.remoteHost,
    info.remotePort || REMOTE_ZMQ_DEFAULTS.remotePort
  );
  const { status: rldxBackendStatus } = usePolicyBackendStatus('rldx', {
    enabled: isRldx,
    intervalMs: 5000,
  });
  const isArmRldxRuntime = isRldx && isArmBackend(rldxBackendStatus);
  const isRldxClientMode = isRldx && rldxRuntimeMode === 'client';
  const isRldxServerStackMode = isRldx &&
    (rldxRuntimeMode === 'server' || rldxRuntimeMode === 'local');
  const disabled = isTaskRunning;
  const [isEditable, setIsEditable] = useState(!disabled);
  const [isUpdatingInstruction, setIsUpdatingInstruction] = useState(false);
  const [remoteEndpointDraft, setRemoteEndpointDraft] = useState('');
  const [isRemoteEndpointFocused, setIsRemoteEndpointFocused] = useState(false);
  const syncGenerationRef = useRef(0);
  const syncTimerRef = useRef(null);

  const { sendRecordCommand } = useRosServiceCaller();

  const handleChange = useCallback(
    (field, value) => {
      // taskInstruction stays editable while inference is running so a
      // multi-task language-conditioned policy can be re-conditioned via
      // the "Update Task Instruction" button below.
      if (field !== 'taskInstruction' && !isEditable) return;
      dispatch(setInferenceTaskInfo({ [field]: value }));
      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    },
    [isEditable, dispatch]
  );

  useEffect(() => {
    if (!isRemoteEndpointFocused) {
      setRemoteEndpointDraft(remoteEndpointValue);
    }
  }, [isRemoteEndpointFocused, remoteEndpointValue]);

  const applyRemoteEndpoint = useCallback(
    (value, { notify = false } = {}) => {
      if (!isEditable) return false;
      const parsed = parseZmqEndpoint(value);
      if (!parsed.host) {
        toast.error('ZMQ endpoint requires a host, e.g. 192.168.60.150:5555');
        return false;
      }
      if (!parsed.isValidPort) {
        toast.error('ZMQ endpoint port must be between 1 and 65535');
        return false;
      }

      const nextInfo = {
        remoteHost: parsed.host,
      };
      if (parsed.hasPort) {
        nextInfo.remotePort = parsed.port;
      }
      dispatch(setInferenceTaskInfo(nextInfo));
      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
      setRemoteEndpointDraft(formatZmqEndpoint(
        parsed.host,
        parsed.hasPort ? parsed.port : (info.remotePort || REMOTE_ZMQ_DEFAULTS.remotePort)
      ));
      if (notify) {
        toast.success('ZMQ endpoint applied');
      }
      return true;
    },
    [dispatch, info.remotePort, isEditable]
  );

  const handleRemoteEndpointChange = useCallback(
    (value) => {
      setRemoteEndpointDraft(value);
      const parsed = parseZmqEndpoint(value);
      if (parsed.host && parsed.hasPort && parsed.isValidPort) {
        dispatch(setInferenceTaskInfo({
          remoteHost: parsed.host,
          remotePort: parsed.port,
        }));
        dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
      }
    },
    [dispatch]
  );

  const handleRemoteEndpointBlur = useCallback(() => {
    setIsRemoteEndpointFocused(false);
    if (!remoteEndpointDraft.trim()) {
      setRemoteEndpointDraft(remoteEndpointValue);
      return;
    }
    applyRemoteEndpoint(remoteEndpointDraft);
  }, [applyRemoteEndpoint, remoteEndpointDraft, remoteEndpointValue]);

  const handleCopyServerEndpoint = useCallback(async () => {
    try {
      await copyTextToClipboard(rldxServerEndpoint);
      toast.success(`Copied ${rldxServerEndpoint}`);
    } catch (error) {
      toast.error(`Failed to copy endpoint: ${error.message}`);
    }
  }, [rldxServerEndpoint]);

  const taskSyncKey = useMemo(
    () => getInferenceTaskInfoKey(info),
    [info]
  );

  useEffect(
    () => () => {
      if (syncTimerRef.current) {
        clearTimeout(syncTimerRef.current);
      }
    },
    []
  );

  useEffect(() => {
    if (syncTimerRef.current) {
      clearTimeout(syncTimerRef.current);
      syncTimerRef.current = null;
    }

    if (taskInfoSync.conflict) {
      return;
    }

    if (!taskInfoSync.dirty) {
      return;
    }

    const generation = syncGenerationRef.current + 1;
    syncGenerationRef.current = generation;
    const submittedTaskKey = taskSyncKey;
    dispatch(markInferenceTaskInfoSyncPending());

    syncTimerRef.current = setTimeout(async () => {
      dispatch(markInferenceTaskInfoSyncing());
      try {
        const result = await sendRecordCommand('set_task_info', {
          autofillEmptyTaskFields: false,
        });
        if (syncGenerationRef.current !== generation) return;
        if (result && result.success) {
          dispatch(markInferenceTaskInfoSyncSuccess({ taskKey: submittedTaskKey }));
        } else {
          dispatch(markInferenceTaskInfoSyncFailed(
            (result && result.message) || 'Inference task info not synced.'
          ));
        }
      } catch (error) {
        if (syncGenerationRef.current !== generation) return;
        dispatch(markInferenceTaskInfoSyncFailed(
          `Inference task info not synced. ${error.message || error}`
        ));
      }
    }, AUTO_SYNC_DELAY_MS);

    return () => {
      if (syncTimerRef.current) {
        clearTimeout(syncTimerRef.current);
        syncTimerRef.current = null;
      }
    };
  }, [
    dispatch,
    sendRecordCommand,
    taskInfoSync.conflict,
    taskInfoSync.dirty,
    taskSyncKey,
  ]);

  const handleDeployModeChange = useCallback(
    async (mode) => {
      if (mode === inferenceMode || isModeSwitchLocked) return;

      if (isModelActive) {
        const result = await sendRecordCommand('finish').catch((error) => {
          toast.error(`Deploy target reset failed: ${error.message || error}`);
          return null;
        });
        if (!result || result.success === false) {
          toast.error(result?.message || 'Inference reset failed');
          return;
        }
        toast('Inference reset before deploy target switch');
      }

      dispatch(setInferenceMode(mode));
      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    },
    [
      dispatch,
      inferenceMode,
      isModeSwitchLocked,
      isModelActive,
      sendRecordCommand,
    ]
  );

  const handleRldxRuntimeModeChange = useCallback(
    (mode) => {
      if (!isEditable || rldxRuntimeMode === mode) return;
      if (isArmRldxRuntime && (mode === 'server' || mode === 'local')) {
        toast.error(
          'RLDX Server/Local mode is not supported on ARM. Run the server on an x86 GPU PC and use Client mode.'
        );
        return;
      }
      const nextInfo = {
        ...info,
        rldxRuntimeMode: mode,
      };
      if (mode === 'local') {
        nextInfo.remoteHost = REMOTE_ZMQ_DEFAULTS.remoteHost;
        nextInfo.remotePort = REMOTE_ZMQ_DEFAULTS.remotePort;
        nextInfo.remoteTimeoutMs =
          info.remoteTimeoutMs || REMOTE_ZMQ_DEFAULTS.remoteTimeoutMs;
      }
      dispatch(setInferenceTaskInfo(nextInfo));
      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    },
    [dispatch, info, isArmRldxRuntime, isEditable, rldxRuntimeMode]
  );

  useEffect(() => {
    if (
      !isRldx ||
      !isArmRldxRuntime ||
      (rldxRuntimeMode !== 'server' && rldxRuntimeMode !== 'local')
    ) {
      return;
    }
    dispatch(setInferenceTaskInfo({ rldxRuntimeMode: 'client' }));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    toast.error(
      'RLDX Server/Local mode is not supported on ARM. Switched to Client mode.'
    );
  }, [dispatch, isArmRldxRuntime, isRldx, rldxRuntimeMode]);

  const currentInstruction = (info.taskInstruction?.[0] || '').trim();
  const canUpdateInstruction =
    isInferencing && currentInstruction !== '' && !isUpdatingInstruction;
  const showRemoteRuntimeFields =
    shouldUseRemoteRuntime(info) && isRldxClientMode;
  const showPolicyPath =
    !isRldx || isRldxServerStackMode;
  const showRldxClientControl = !isRldx || rldxRuntimeMode !== 'server';
  const showRldxServerControl = isRldx &&
    (rldxRuntimeMode === 'server' || rldxRuntimeMode === 'local');
  const rldxServerStartPayload = useMemo(() => {
    if (!isRldxServerStackMode) return null;
    const modelPath = String(info.policyPath || '').trim();
    return modelPath ? { model_path: modelPath } : null;
  }, [info.policyPath, isRldxServerStackMode]);

  const handleUpdateInstruction = useCallback(async () => {
    if (!canUpdateInstruction) return;
    setIsUpdatingInstruction(true);
    try {
      const result = await sendRecordCommand('update_instruction');
      if (result?.success) {
        toast.success(
          `Instruction updated: "${currentInstruction.slice(0, 60)}${
            currentInstruction.length > 60 ? '…' : ''
          }"`
        );
      } else {
        toast.error(result?.message || 'Failed to update instruction');
      }
    } catch (err) {
      toast.error(err?.message || 'Failed to update instruction');
    } finally {
      setIsUpdatingInstruction(false);
    }
  }, [canUpdateInstruction, sendRecordCommand, currentInstruction]);

  const handlePolicyFolderSelect = useCallback((item) => {
    if (!isEditable) return;
    const fullPath = item?.full_path || '';
    if (fullPath) {
      dispatch(setInferenceTaskInfo({ policyPath: fullPath }));
      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    }
    setShowPolicyBrowser(false);
  }, [isEditable, dispatch]);

  const policyBrowserPath =
    info.serviceType === 'groot'
      ? DEFAULT_PATHS.GROOT_CHECKPOINTS_PATH
      : info.serviceType === 'rldx'
        ? DEFAULT_PATHS.RLDX_CHECKPOINTS_PATH
      : DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH;

  // Update isEditable state when the disabled prop changes
  useEffect(() => {
    setIsEditable(!disabled);
  }, [disabled]);

  // track task status update
  useEffect(() => {
    if (inferenceStatus) {
      setLastTaskStatusUpdate(Date.now());
      setIsTaskStatusPaused(false);
    }
  }, [inferenceStatus]);

  // Check if task status updates are paused (considered paused if no updates for 1 second)
  useEffect(() => {
    const UPDATE_PAUSE_THRESHOLD = 1000;
    const timer = setInterval(() => {
      const timeSinceLastUpdate = Date.now() - lastTaskStatusUpdate;
      const isPaused = timeSinceLastUpdate >= UPDATE_PAUSE_THRESHOLD;
      if (isPaused !== isTaskStatusPaused) {
        setIsTaskStatusPaused(isPaused);
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [lastTaskStatusUpdate, isTaskStatusPaused]);

  const classLabel = clsx('text-sm', 'text-gray-600', 'w-28', 'flex-shrink-0', 'font-medium');

  const classInfoPanel = clsx(
    'bg-white',
    'border',
    'border-gray-200',
    'rounded-2xl',
    'shadow-md',
    'p-4',
    'w-full',
    'max-w-[350px]',
    'relative',
    'overflow-y-auto',
    'scrollbar-thin'
  );

  // taskInstruction stays always-editable so multi-task LLM-conditioned
  // policies can be re-conditioned mid-run. No isEditable gate here.
  const classTaskInstructionTextarea = clsx(
    'text-sm',
    'resize-y',
    'min-h-10',
    'max-h-20',
    'w-full',
    'p-2',
    'border',
    'border-gray-300',
    'rounded-md',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-blue-500',
    'focus:border-transparent',
    'bg-white',
  );

  const classPolicyPathTextarea = clsx(
    'text-sm',
    'resize-y',
    'min-h-16',
    'max-h-24',
    'w-full',
    'p-2',
    'border',
    'border-gray-300',
    'rounded-md',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-blue-500',
    'focus:border-transparent',
    {
      'bg-gray-100 cursor-not-allowed': !isEditable,
      'bg-white': isEditable,
    }
  );

  const classTextInput = clsx(
    'text-sm',
    'w-full',
    'h-8',
    'p-2',
    'border',
    'border-gray-300',
    'rounded-md',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-blue-500',
    'focus:border-transparent',
    {
      'bg-gray-100 cursor-not-allowed': !isEditable,
      'bg-white': isEditable,
    }
  );

  const deployButtonClass = (active, danger = false, unavailable = false) => clsx(
    'h-9',
    'min-w-0',
    'px-2',
    'rounded-md',
    'flex',
    'items-center',
    'justify-center',
    'gap-1.5',
    'text-xs',
    'font-semibold',
    'whitespace-nowrap',
    'transition-colors',
    'focus:outline-none',
    'focus:ring-2',
    active
      ? danger
        ? 'bg-orange-500 text-white focus:ring-orange-300'
        : 'bg-emerald-500 text-white focus:ring-emerald-300'
      : 'bg-white text-gray-600 hover:bg-gray-50 focus:ring-gray-300 border border-gray-200',
    {
      'opacity-50 cursor-not-allowed': isModeSwitchLocked || unavailable,
      'cursor-pointer': !isModeSwitchLocked && !unavailable,
    }
  );

  const actionModeButtonClass = (active) => clsx(
    'h-8',
    'min-w-0',
    'px-2',
    'rounded-md',
    'flex',
    'items-center',
    'justify-center',
    'gap-1.5',
    'text-xs',
    'font-semibold',
    'whitespace-nowrap',
    'transition-colors',
    'focus:outline-none',
    'focus:ring-2',
    active
      ? 'bg-blue-500 text-white focus:ring-blue-300'
      : 'bg-white text-gray-600 hover:bg-gray-50 focus:ring-gray-300 border border-gray-200',
    {
      'opacity-50 cursor-not-allowed': !isEditable,
      'cursor-pointer': isEditable,
    }
  );

  return (
    <div className={classInfoPanel}>
      <div className={clsx('text-lg', 'font-semibold', 'mb-3', 'text-gray-800')}>
        Task Information
      </div>

      <InferenceModelSelector readonly={!isEditable} />

      {isRldx && (
        <div className="mb-3 rounded-lg border border-gray-200 bg-gray-50 p-2">
          <div className="flex items-center justify-between gap-2 mb-1.5">
            <span className="text-sm font-medium text-gray-600">Runtime Role</span>
            <span className="inline-flex rounded-md bg-white px-2 py-0.5 text-xs font-semibold text-gray-600 border border-gray-200">
              {rldxRuntimeMode === 'server'
                ? 'Server'
                : rldxRuntimeMode === 'local'
                  ? 'Local Stack'
                  : 'Client'}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-1">
            {[
              ['local', 'Local'],
              ['client', 'Client'],
              ['server', 'Server'],
            ].map(([mode, label]) => {
              const unavailable = isArmRldxRuntime &&
                (mode === 'server' || mode === 'local');
              const disabledReason = unavailable
                ? 'RLDX Server/Local mode is not supported on ARM. Run the server on an x86 GPU PC and use Client mode.'
                : !isEditable
                  ? 'Runtime Role cannot be changed while inference is loading or running. Stop or Clear first.'
                  : '';
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => handleRldxRuntimeModeChange(mode)}
                  disabled={!isEditable}
                  aria-disabled={unavailable ? 'true' : undefined}
                  className={deployButtonClass(rldxRuntimeMode === mode, false, unavailable)}
                  aria-label={disabledReason
                    ? `Use RLDX ${label} Mode disabled: ${disabledReason}`
                    : `Use RLDX ${label} Mode`}
                  title={disabledReason || `RLDX ${label} Mode`}
                >
                  <span className="truncate">{label}</span>
                </button>
              );
            })}
          </div>
          {isArmRldxRuntime && (
            <div className="mt-2 rounded-md bg-amber-50 px-2 py-1 text-xs font-medium text-amber-700 border border-amber-200">
              ARM devices support RLDX Client mode only.
            </div>
          )}
          {(rldxRuntimeMode === 'server' || rldxRuntimeMode === 'local') && (
            <div className="mt-2 flex items-center justify-between gap-2 rounded-md bg-white px-2 py-1 text-xs text-gray-600 border border-gray-200">
              <span className="font-medium">Server endpoint</span>
              <div className="flex min-w-0 items-center gap-1.5">
                <span className="font-mono truncate">
                  {rldxServerEndpoint}
                </span>
                <button
                  type="button"
                  onClick={handleCopyServerEndpoint}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-gray-500 hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
                  aria-label={`Copy RLDX server endpoint ${rldxServerEndpoint}`}
                  title="Copy endpoint"
                >
                  <MdContentCopy size={14} />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {showRldxClientControl && (
        <PolicyBackendControl
          serviceType={info.serviceType}
          label={isRldx ? 'RLDX Client Docker' : undefined}
        />
      )}

      {showRldxServerControl && (
        <PolicyBackendControl
          serviceType="rldx-server"
          backendName="rldx-server"
          label="RLDX Server Docker"
          actionPayload={rldxServerStartPayload}
        />
      )}

      <div className="mb-3 rounded-lg border border-gray-200 bg-gray-50 p-2">
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <span className="text-sm font-medium text-gray-600">Deploy Target</span>
          <span className={clsx(
            'inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold whitespace-nowrap',
            isRobotMode
              ? 'bg-orange-100 text-orange-700'
              : 'bg-emerald-100 text-emerald-700'
          )}>
            {isRobotMode ? <MdWarningAmber size={14} /> : <MdViewInAr size={14} />}
            {isRobotMode ? 'Commands Enabled' : 'Commands Blocked'}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-1">
          <button
            type="button"
            onClick={() => handleDeployModeChange('simulation')}
            disabled={isModeSwitchLocked}
            className={deployButtonClass(!isRobotMode)}
            aria-label="Use 3D Sim Deploy"
            title="3D Sim Deploy"
          >
            <MdViewInAr size={17} className="shrink-0" />
            <span className="truncate">3D Sim Deploy</span>
          </button>
          <button
            type="button"
            onClick={() => handleDeployModeChange('robot')}
            disabled={isModeSwitchLocked}
            className={deployButtonClass(isRobotMode, true)}
            aria-label="Use Real Robot Deploy"
            title="Real Robot Deploy"
          >
            <MdPrecisionManufacturing size={17} className="shrink-0" />
            <span className="truncate">Real Robot Deploy</span>
          </button>
        </div>
      </div>

      {showRemoteRuntimeFields && (
        <>
          <div className={clsx('flex', 'items-center', 'mb-2.5')}>
            <span className={classLabel}>ZMQ Endpoint</span>
            <input
              className={classTextInput}
              type="text"
              value={remoteEndpointDraft}
              onFocus={() => setIsRemoteEndpointFocused(true)}
              onChange={(e) => handleRemoteEndpointChange(e.target.value)}
              onBlur={handleRemoteEndpointBlur}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.currentTarget.blur();
                }
              }}
              onPaste={(e) => {
                const pasted = e.clipboardData.getData('text');
                const parsed = parseZmqEndpoint(pasted);
                if (parsed.host && parsed.hasPort && parsed.isValidPort) {
                  e.preventDefault();
                  applyRemoteEndpoint(pasted, { notify: true });
                }
              }}
              disabled={!isEditable}
              placeholder={`${REMOTE_ZMQ_DEFAULTS.remoteHost}:${REMOTE_ZMQ_DEFAULTS.remotePort}`}
              aria-label="ZMQ Endpoint"
              title="Paste the server endpoint, e.g. 192.168.60.150:5555"
            />
          </div>
          <div className={clsx('flex', 'items-center', 'mb-2.5')}>
            <span className={classLabel}>Timeout ms</span>
            <input
              className={classTextInput}
              type="number"
              min="1"
              step="1000"
              value={info.remoteTimeoutMs || ''}
              onChange={(e) => {
                const value = e.target.value;
                handleChange(
                  'remoteTimeoutMs',
                  value === '' ? '' : Number(value)
                );
              }}
              disabled={!isEditable}
              placeholder={String(REMOTE_ZMQ_DEFAULTS.remoteTimeoutMs)}
            />
          </div>
        </>
      )}

      {/* Edit mode indicator */}
      <div
        className={clsx('mb-3', 'p-2', 'rounded-md', 'text-sm', 'font-medium', {
          'bg-green-100 text-green-800': isEditable,
          'bg-gray-100 text-gray-600': !isEditable,
        })}
      >
        {isEditable ? (
          '✏️ Edit mode'
        ) : (
          <div className="leading-tight">
            <div>🔒 Read only</div>
            <div className="text-xs mt-1 opacity-80">task is running or robot is not connected</div>
          </div>
        )}
      </div>

      {/* Task Instruction — only shown for language-conditioned policies.
          Whitelist lives in constants/policyCapabilities.js. */}
      {showInstruction && (
        <>
          <div className={clsx('flex', 'items-start', 'mb-1')}>
            <span className={clsx(classLabel, 'pt-2')}>Task Instruction</span>
            <textarea
              className={classTaskInstructionTextarea}
              value={(info.taskInstruction && info.taskInstruction[0]) || ''}
              onChange={(e) => handleChange('taskInstruction', [e.target.value])}
              placeholder="Enter Task Instruction"
            />
          </div>
          <div className={clsx('flex', 'items-center', 'mb-2.5')}>
            <span className={classLabel} />
            <button
              type="button"
              onClick={handleUpdateInstruction}
              disabled={!canUpdateInstruction}
              className={clsx(
                'px-3 py-1.5 text-sm rounded-md',
                'bg-blue-500 text-white hover:bg-blue-600',
                'disabled:bg-gray-300 disabled:text-gray-500 disabled:cursor-not-allowed',
                'transition-colors'
              )}
              title={
                isInferencing
                  ? 'Send the current Task Instruction to the running policy'
                  : 'Inference must be running to update the instruction'
              }
            >
              {isUpdatingInstruction ? 'Updating…' : 'Update Task Instruction'}
            </button>
          </div>
        </>
      )}

      {showPolicyPath && (
        <>
          {/* Policy Path */}
          <div className={clsx('flex', 'items-start', 'mb-2.5')}>
            <span className={clsx(classLabel, 'pt-2')}>
              {isRldxServerStackMode ? 'Server Policy' : 'Policy Path'}
            </span>
            <div className="flex flex-row items-start gap-2 flex-1 min-w-0">
              <textarea
                className={classPolicyPathTextarea}
                value={info.policyPath || ''}
                onChange={(e) => handleChange('policyPath', e.target.value)}
                disabled={!isEditable}
                placeholder={
                  isRldxServerStackMode
                    ? 'Enter server model path'
                    : 'Enter Policy Path or Repo ID'
                }
              />
              <button
                type="button"
                onClick={() => setShowPolicyBrowser(true)}
                disabled={!isEditable}
                className="flex items-center justify-center w-9 h-9 text-blue-500 bg-gray-200 rounded-md hover:text-blue-700 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                aria-label="Browse for policy model folder"
                title="Browse for policy model folder"
              >
                <MdFolderOpen className="w-5 h-5" />
              </button>
            </div>
          </div>
        </>
      )}

      {isGrootModel && (
        <>
          <div className={clsx('flex', 'items-center', 'mb-2.5')}>
            <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
              <Tooltip content="Run GR00T with DiT TensorRT acceleration." position="bottom">
                <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
              </Tooltip>
              <span>TensorRT</span>
            </div>
            <label className={clsx('flex', 'items-center', 'gap-2', 'text-sm')}>
              <input
                type="checkbox"
                className={clsx('w-4 h-4', {
                  'cursor-not-allowed opacity-50': !isEditable,
                  'cursor-pointer': isEditable,
                })}
                checked={isTensorRtEnabled}
                onChange={(e) => handleChange(
                  'accelerationMode',
                  e.target.checked ? 'tensorrt_dit' : 'pytorch'
                )}
                disabled={!isEditable}
              />
              <span className="text-gray-500">Enable</span>
            </label>
          </div>
          {isTensorRtEnabled && (
            <TrtEngineControl
              modelPath={info.policyPath}
              enginePath={info.accelerationEnginePath}
              robotType={robotType}
              taskInstruction={trtTaskInstruction}
              disabled={!isEditable}
              labelClassName={classLabel}
            />
          )}
        </>
      )}

      <div className="w-full h-1 my-2 border-t border-gray-300"></div>

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip content="Choose when the next action chunk is requested." position="bottom">
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Action Request</span>
        </div>
        <div className="grid grid-cols-2 gap-1 flex-1 min-w-0">
          <button
            type="button"
            onClick={() => handleChange('actionRequestMode', 'async')}
            disabled={!isEditable}
            className={actionModeButtonClass(actionRequestMode !== 'sync')}
            aria-label="Use async action requests"
            title="Async"
          >
            <MdSync size={16} className="shrink-0" />
            <span className="truncate">Async</span>
          </button>
          <button
            type="button"
            onClick={() => handleChange('actionRequestMode', 'sync')}
            disabled={!isEditable}
            className={actionModeButtonClass(actionRequestMode === 'sync')}
            aria-label="Use sync action requests"
            title="Sync"
          >
            <MdHourglassEmpty size={16} className="shrink-0" />
            <span className="truncate">Sync</span>
          </button>
        </div>
      </div>

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip content="Model output rate. Match training data rate." position="bottom">
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Inference Hz</span>
        </div>
        <input
          className={classTextInput}
          type="number"
          step="1"
          min="1"
          value={info.inferenceHz || ''}
          onChange={(e) => {
            const v = e.target.value;
            handleChange('inferenceHz', v === '' ? '' : Number(v));
          }}
          disabled={!isEditable}
        />
      </div>

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip content="Rate of commands sent to the robot." position="bottom">
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Control Hz</span>
        </div>
        <input
          className={classTextInput}
          type="number"
          step="5"
          min="1"
          value={info.controlHz || ''}
          onChange={(e) => {
            const v = e.target.value;
            handleChange('controlHz', v === '' ? '' : Number(v));
          }}
          disabled={!isEditable}
        />
      </div>

      <FileBrowserModal
        isOpen={showPolicyBrowser}
        onClose={() => setShowPolicyBrowser(false)}
        onFileSelect={handlePolicyFolderSelect}
        title="Select policy model folder"
        selectButtonText="Select"
        allowDirectorySelect={true}
        allowFileSelect={false}
        initialPath={policyBrowserPath}
        defaultPath={policyBrowserPath}
        homePath={DEFAULT_PATHS.POLICY_CHECKPOINTS_PATH}
      />
    </div>
  );
};

export default InferencePanel;
