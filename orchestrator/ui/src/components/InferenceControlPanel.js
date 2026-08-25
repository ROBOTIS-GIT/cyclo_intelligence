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
// Author: Dongyun Kim

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { shallowEqual, useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import toast, { useToasterStore } from 'react-hot-toast';
import {
  MdPlayArrow,
  MdStop,
  MdDeleteOutline,
  MdClose,
  MdPrecisionManufacturing,
  MdViewInAr,
  MdWarningAmber,
} from 'react-icons/md';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import Tooltip from './Tooltip';
import InferenceRecordingControls from './InferenceRecordingControls';
import { InferencePhase } from '../constants/taskPhases';
import {
  markLocalTaskInfoEdited,
  selectInferenceRecordingControl,
  selectInferenceTaskInfo,
  setInferenceMode,
  setRecordInferenceMode,
  setInferenceStatus,
} from '../features/tasks/taskSlice';
import { requiresInstruction } from '../constants/policyCapabilities';
import usePolicyBackendStatus, {
  getPolicyBackendReadiness,
} from '../hooks/usePolicyBackendStatus';

const phaseGuideMessages = {
  [InferencePhase.READY]: 'Ready to start',
  [InferencePhase.LOADING]: 'Loading model / downloading assets...',
  [InferencePhase.INFERENCING]: 'Inferencing',
  [InferencePhase.PAUSED]: 'Paused',
};

const buildRequiredFields = (serviceType, policyType) => {
  const fields = [{ key: 'policyPath', label: 'Policy Path' }];
  if (requiresInstruction(serviceType, policyType)) {
    fields.unshift({ key: 'taskInstruction', label: 'Task Instruction' });
  }
  return fields;
};

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
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

function buildTrtStatusUrl(modelPath, enginePath) {
  const params = new URLSearchParams();
  params.set('model_path', modelPath);
  if (enginePath) params.set('engine_path', enginePath);
  return `${API_BASE}/backends/groot/trt/status?${params.toString()}`;
}

export default function InferenceControlPanel({
  showRecordingControls = true,
  variant = 'default',
  policyEpoch = null,
}) {
  const dispatch = useDispatch();
  const taskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const inferenceStatus = useSelector((state) => state.tasks.inferenceStatus);
  const recordingControl = useSelector(selectInferenceRecordingControl);
  const rosHost = useSelector((state) => state.ros.rosHost);

  const [hovered, setHovered] = useState(null);
  const [pressed, setPressed] = useState(null);
  const [lastPolicyPath, setLastPolicyPath] = useState('');
  const [spinnerIndex, setSpinnerIndex] = useState(0);
  const [pendingRobotDeployIntent, setPendingRobotDeployIntent] = useState(null);

  const { sendRecordCommand } = useRosServiceCaller();

  const { toasts } = useToasterStore();
  const TOAST_LIMIT = 3;

  const phase = inferenceStatus.inferencePhase;
  const isIdle = phase === InferencePhase.READY;
  const isLoading = phase === InferencePhase.LOADING;
  const isInferencing = phase === InferencePhase.INFERENCING;
  const isPaused = phase === InferencePhase.PAUSED;
  const inferencePhaseRef = useRef(phase);
  const isModelLoaded = isInferencing || isPaused;
  const shouldCheckBackend = isIdle || isPaused;

  const {
    readiness: backendReadiness,
    refreshStatus: refreshBackendStatus,
  } = usePolicyBackendStatus(taskInfo.serviceType, {
    enabled: shouldCheckBackend,
    intervalMs: 2000,
  });

  const isBackendStartBlocked = shouldCheckBackend && !backendReadiness.ready;
  const isBackendWarming = isBackendStartBlocked &&
    (backendReadiness.state === 'checking' || backendReadiness.state === 'warming');

  useEffect(() => {
    inferencePhaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    toasts
      .filter((t) => t.visible)
      .filter((_, i) => i >= TOAST_LIMIT)
      .forEach((t) => toast.dismiss(t.id));
  }, [toasts]);

  // Spin while loading / inferencing — independent of taskStatus update
  // cadence. The old (`}, [taskStatus]`) pattern only ticked when a new
  // TaskStatus message arrived, which happens once per phase transition
  // in cyclo_intelligence (LOADING → INFERENCING) — so the spinner
  // sat frozen between transitions. setInterval gives a steady visual
  // beat while either banner is on screen.
  useEffect(() => {
    if (!isLoading && !isInferencing && !isBackendWarming) return undefined;
    const id = setInterval(() => {
      setSpinnerIndex((prev) => (prev + 1) % spinnerFrames.length);
    }, 100);
    return () => clearInterval(id);
  }, [isLoading, isInferencing, isBackendWarming]);

  const validateTaskInfo = useCallback(() => {
    const missingFields = [];
    const fields = buildRequiredFields(taskInfo.serviceType, taskInfo.policyType);
    for (const field of fields) {
      const value = taskInfo[field.key];
      if (
        value === null ||
        value === undefined ||
        value === '' ||
        (typeof value === 'string' && value.trim() === '') ||
        (Array.isArray(value) && value.length === 0) ||
        (Array.isArray(value) && value.every((item) => item.trim() === ''))
      ) {
        missingFields.push(field.label);
      }
    }
    return { isValid: missingFields.length === 0, missingFields };
  }, [taskInfo]);

  const ensureTensorRtReady = useCallback(async () => {
    if (
      taskInfo.serviceType !== 'groot' ||
      taskInfo.accelerationMode !== 'tensorrt_dit'
    ) {
      return true;
    }
    const policyPath = String(taskInfo.policyPath || '').trim();
    if (!policyPath) return true;
    try {
      const response = await fetch(buildTrtStatusUrl(
        policyPath,
        String(taskInfo.accelerationEnginePath || '').trim()
      ));
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || data.message || `status failed (${response.status})`);
      }
      if (data.status === 'ready') return true;
      toast.error(
        data.status === 'building'
          ? 'TRT engine is still building'
          : 'Build TRT engine before starting inference'
      );
      return false;
    } catch (error) {
      toast.error(`TRT engine status failed: ${error.message}`);
      return false;
    }
  }, [
    taskInfo.serviceType,
    taskInfo.accelerationMode,
    taskInfo.policyPath,
    taskInfo.accelerationEnginePath,
  ]);

  const executeCommand = useCallback(
    async (commandName, commandString, options = {}) => {
      const isStartTimeoutDuringLoading = (message = '') => (
        commandString === 'start_inference' &&
        String(message).toLowerCase().includes('timeout') &&
        inferencePhaseRef.current === InferencePhase.LOADING
      );

      try {
        const result = await sendRecordCommand(commandString, {
          ...options,
          taskSource: 'inference',
        });
        if (result && result.success === false) {
          if (isStartTimeoutDuringLoading(result.message)) {
            toast('Model loading is still running. Large downloads can take several minutes.');
            return result;
          }
          toast.error(`Command failed: ${result.message || 'Unknown error'}`);
          // Backend may have left phase in LOADING/INFERENCING after a failed
          // setup; force the local phase back to READY so the panel becomes
          // editable and the user can retry.
          if (commandString === 'start_inference') {
            dispatch(setInferenceStatus({ inferencePhase: InferencePhase.READY }));
          }
        } else if (result && result.success === true) {
          toast.success(`${commandName} executed successfully`);
        } else {
          toast.error(`${commandName} completed with uncertain status`);
          if (commandString === 'start_inference') {
            dispatch(setInferenceStatus({ inferencePhase: InferencePhase.READY }));
          }
        }
        return result;
      } catch (error) {
        let errorMessage = error.message || error.toString();
        if (
          errorMessage.includes('ROS connection failed') ||
          errorMessage.includes('WebSocket')
        ) {
          toast.error(`ROS connection failed: rosbridge server is not running (${rosHost})`);
        } else if (isStartTimeoutDuringLoading(errorMessage)) {
          toast('Model loading is still running. Large downloads can take several minutes.');
          return {
            success: true,
            message: 'Model loading is still running',
          };
        } else if (errorMessage.includes('timeout')) {
          toast.error(`Command timeout [${commandName}]: Server did not respond`);
        } else {
          toast.error(`Command failed [${commandName}]: ${errorMessage}`);
        }
        // Same reasoning as the success===false branch above.
        if (commandString === 'start_inference') {
          dispatch(setInferenceStatus({ inferencePhase: InferencePhase.READY }));
        }
        return null;
      }
    },
    [sendRecordCommand, rosHost, dispatch]
  );

  const executeStartIntent = useCallback(async (intent, inferenceMode) => {
    if (!intent) return;
    if (intent.policyPath) {
      setLastPolicyPath(intent.policyPath);
    }
    await executeCommand(intent.commandName, intent.commandString, {
      inferenceMode,
    });
  }, [executeCommand]);

  const handleStart = useCallback(async () => {
    let readiness = backendReadiness;
    if (!readiness.ready) {
      const refreshedStatus = await refreshBackendStatus({ quiet: true });
      readiness = getPolicyBackendReadiness(refreshedStatus);
    }
    if (!readiness.ready) {
      const message = readiness.message || 'Policy backend is not ready yet';
      if (readiness.state === 'warming' || readiness.state === 'checking') {
        toast(message);
      } else {
        toast.error(message);
      }
      return;
    }

    let startIntent;
    if (isPaused && taskInfo.policyPath === lastPolicyPath) {
      startIntent = {
        commandName: 'Resume',
        commandString: 'resume_inference',
        policyPath: '',
      };
    } else {
      const validation = validateTaskInfo();
      if (!validation.isValid) {
        toast.error(`Missing required fields: ${validation.missingFields.join(', ')}`);
        return;
      }
      startIntent = {
        commandName: 'Start Inference',
        commandString: 'start_inference',
        policyPath: taskInfo.policyPath,
      };
    }

    if (
      startIntent.commandString === 'start_inference' &&
      !(await ensureTensorRtReady())
    ) {
      return;
    }

    const inferenceMode = taskInfo.inferenceMode || 'simulation';
    if (inferenceMode === 'robot') {
      setPendingRobotDeployIntent(startIntent);
      return;
    }

    await executeStartIntent(startIntent, 'simulation');
  }, [
    backendReadiness,
    refreshBackendStatus,
    isPaused,
    taskInfo.policyPath,
    taskInfo.inferenceMode,
    lastPolicyPath,
    executeStartIntent,
    ensureTensorRtReady,
    validateTaskInfo,
  ]);

  const handleConfirmRobotDeploy = useCallback(async () => {
    const intent = pendingRobotDeployIntent;
    setPendingRobotDeployIntent(null);
    await executeStartIntent(intent, 'robot');
  }, [executeStartIntent, pendingRobotDeployIntent]);

  const handleUseSimDeploy = useCallback(async () => {
    const intent = pendingRobotDeployIntent;
    setPendingRobotDeployIntent(null);
    dispatch(setRecordInferenceMode(false));
    dispatch(setInferenceMode('simulation'));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    await executeStartIntent(intent, 'simulation');
  }, [dispatch, executeStartIntent, pendingRobotDeployIntent]);

  const handleCloseRobotDeployWarning = useCallback(() => {
    setPendingRobotDeployIntent(null);
  }, []);

  const handleStop = useCallback(async () => {
    await executeCommand('Stop', 'stop_inference');
  }, [executeCommand]);

  const handleClear = useCallback(async () => {
    const result = await executeCommand('Clear', 'finish');
    if (result?.success) {
      setLastPolicyPath('');
    }
  }, [executeCommand]);

  const startEnabled = shouldCheckBackend && backendReadiness.ready;
  const stopEnabled = isInferencing;
  const clearEnabled = isModelLoaded && !recordingControl.lifecycleLocked;
  const startDescription = isBackendStartBlocked
    ? backendReadiness.message
    : isPaused
      ? 'Resume inference'
      : 'Start inference';
  const guideMessage = isBackendStartBlocked
    ? backendReadiness.message
    : phaseGuideMessages[phase] || '';
  const showGuideSpinner = isInferencing || isLoading || isBackendWarming;

  const handleKeyAction = useCallback(
    (e) => {
      if (e.key === ' ' || e.key === 'Spacebar' || e.code === 'Space') {
        if (startEnabled) return 'Start';
      }
      if (
        (e.ctrlKey || e.metaKey) &&
        e.shiftKey &&
        (e.key === 's' || e.key === 'S')
      ) {
        if (stopEnabled) return 'Stop';
      }
      if (e.key === 'Escape') {
        if (clearEnabled) return 'Clear';
      }
      return null;
    },
    [startEnabled, stopEnabled, clearEnabled]
  );

  useEffect(() => {
    const isInputFocused = () => {
      const el = document.activeElement;
      if (!el) return false;
      const tag = el.tagName.toLowerCase();
      return tag === 'input' || tag === 'textarea' || tag === 'select' ||
        el.contentEditable === 'true';
    };

    const handleKeyDown = (e) => {
      if (e.repeat || isInputFocused()) return;
      const action = handleKeyAction(e);
      if (action) {
        e.preventDefault();
        setPressed(action);
      }
    };

    const handleKeyUp = (e) => {
      setPressed(null);
      if (isInputFocused()) return;
      const action = handleKeyAction(e);
      if (action === 'Start') handleStart();
      else if (action === 'Stop') handleStop();
      else if (action === 'Clear') handleClear();
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleKeyAction, handleStart, handleStop, handleClear]);

  const isOfflineRL = variant === 'offlineRL';
  const isOfflineRLCombined = isOfflineRL && showRecordingControls;

  const classBody = clsx(
    'flex flex-row items-center border',
    isOfflineRL
      ? 'gap-1.5 rounded-xl border-[#d9d2c5] bg-[#f6f3ec] px-2 py-1.5 shadow-[0_2px_8px_rgba(69,61,47,0.045)]'
      : 'gap-1.5 rounded-full border-gray-100 bg-white/90 px-3 py-1 shadow-md backdrop-blur-sm'
  );

  const classBtn = (label, isDisabled) =>
    clsx(
      isOfflineRLCombined ? 'h-9 w-full' : (isOfflineRL ? 'h-8' : 'h-full'),
      'rounded-lg',
      'border-none',
      'cursor-pointer',
      'px-2.5',
      'flex',
      'items-center',
      'justify-center',
      'gap-1',
      isOfflineRL ? 'bg-[#ebe6dc] text-[#49443c]' : 'bg-gray-100',
      'transition-all',
      'duration-150',
      'font-semibold',
      isOfflineRL ? 'text-[11px]' : 'text-lg',
      isOfflineRLCombined ? 'min-w-0' : 'shrink-0',
      {
        [isOfflineRL ? 'bg-[#d5cdc0]' : 'bg-gray-400']:
          pressed === label && !isDisabled,
        [isOfflineRL ? 'bg-[#e0d9cd]' : 'bg-gray-200']:
          hovered === label && pressed !== label && !isDisabled,
        [isOfflineRL
          ? 'cursor-not-allowed bg-[#f0ece4] text-[#aaa295] opacity-55'
          : 'opacity-30 cursor-not-allowed bg-gray-50']:
          isDisabled,
      }
    );

  const controlButtons = [
    {
      label: 'Start',
      icon: MdPlayArrow,
      color: isOfflineRL ? '#667b69' : '#1976d2',
      enabled: startEnabled,
      handler: handleStart,
      description: startDescription,
      shortcut: 'Space',
    },
    {
      label: 'Stop',
      icon: MdStop,
      color: isOfflineRL ? '#a8795b' : '#f57c00',
      enabled: stopEnabled,
      handler: handleStop,
      description: 'Pause inference (model stays loaded)',
      shortcut: 'Ctrl+Shift+S',
    },
    {
      label: 'Clear',
      icon: MdDeleteOutline,
      color: isOfflineRL ? '#a86b68' : '#d32f2f',
      enabled: clearEnabled,
      handler: handleClear,
      description: 'Stop inference and unload model',
      shortcut: 'Escape',
    },
  ];

  const renderControlButton = ({
    label,
    icon: Icon,
    color,
    enabled,
    handler,
    description,
    shortcut,
  }) => {
    const isDisabled = !enabled;
    return (
      <Tooltip
        key={label}
        position="bottom"
        content={
          <div className="text-center">
            <div className="font-semibold">{description}</div>
            {!isDisabled && (
              <div className="text-sm mt-1 text-gray-300">
                <span className="font-mono bg-gray-700 px-1 rounded">{shortcut}</span>
              </div>
            )}
          </div>
        }
        disabled={false}
        className={clsx(
          'relative h-full',
          isOfflineRLCombined ? 'min-w-0' : 'shrink-0'
        )}
      >
        <button
          className={classBtn(label, isDisabled)}
          onClick={() => !isDisabled && handler()}
          onMouseEnter={() => !isDisabled && setHovered(label)}
          onMouseLeave={() => { setHovered(null); setPressed(null); }}
          onMouseDown={() => !isDisabled && setPressed(label)}
          onMouseUp={() => setPressed(null)}
          disabled={isDisabled}
          aria-label={description}
        >
          <Icon
            style={{ fontSize: isOfflineRL ? '0.95rem' : '1.1rem' }}
            color={isDisabled ? '#9ca3af' : color}
          />
          {label}
        </button>
      </Tooltip>
    );
  };

  return (
    <>
      {isOfflineRLCombined ? (
        <InferenceRecordingControls
          variant="workspace"
          inferenceActions={controlButtons.map(renderControlButton)}
          guideMessage={guideMessage}
          guideSpinner={showGuideSpinner ? spinnerFrames[spinnerIndex] : ''}
          policyEpoch={policyEpoch}
        />
      ) : (
        <div
          className="flex shrink-0 flex-col items-end gap-2"
          data-appearance={isOfflineRL ? 'offline-rl' : 'default'}
        >
          <div
            className={classBody}
            role="group"
            aria-label="Inference controls"
          >
            <span className={clsx(
              'shrink-0 whitespace-nowrap px-1 font-semibold',
              isOfflineRL
                ? 'text-[10px] uppercase tracking-[0.12em] text-[#81796d]'
                : 'text-lg text-gray-500'
            )}
            >
              Inference
            </span>
            <div className={clsx(
              'h-5 w-px shrink-0',
              isOfflineRL ? 'bg-[#d7d0c4]' : 'bg-gray-300'
            )}
            />
            {controlButtons.map(renderControlButton)}

            {(guideMessage || showGuideSpinner) && (
              <>
                <div className={clsx(
                  'h-5 w-px shrink-0',
                  isOfflineRL ? 'bg-[#cbc3b6]' : 'bg-gray-400'
                )}
                />
                <div className="flex items-center gap-1 shrink-0">
                  <span className={clsx(
                    'whitespace-nowrap font-semibold',
                    isOfflineRL ? 'text-[10px] text-[#665f54]' : 'text-lg text-gray-600'
                  )}
                  >
                    {guideMessage}
                  </span>
                  {showGuideSpinner && (
                    <span className={clsx(
                      'font-mono text-sm',
                      isOfflineRL ? 'text-[#69866f]' : 'text-blue-500'
                    )}
                    >
                      {spinnerFrames[spinnerIndex]}
                    </span>
                  )}
                </div>
              </>
            )}
          </div>

          {showRecordingControls && <InferenceRecordingControls />}
        </div>
      )}

      {pendingRobotDeployIntent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 px-4">
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="real-robot-deploy-title"
            className="w-full max-w-md rounded-lg bg-white shadow-2xl border border-orange-200 overflow-hidden"
          >
            <div className="flex items-center justify-between gap-3 px-4 py-3 bg-orange-50 border-b border-orange-200">
              <div className="flex items-center gap-2 min-w-0">
                <MdWarningAmber className="text-orange-600 shrink-0" size={22} />
                <h2 id="real-robot-deploy-title" className="text-base font-bold text-orange-900 truncate">
                  Real Robot Deploy
                </h2>
              </div>
              <button
                type="button"
                onClick={handleCloseRobotDeployWarning}
                className="w-8 h-8 rounded-md flex items-center justify-center text-orange-700 hover:bg-orange-100 focus:outline-none focus:ring-2 focus:ring-orange-300"
                aria-label="Close deploy warning"
              >
                <MdClose size={20} />
              </button>
            </div>
            <div className="px-4 py-4 text-sm text-gray-700 space-y-3">
              <p className="font-semibold text-gray-900">
                Real Robot Deploy sends policy actions to the physical robot.
              </p>
              <p>
                Keep people clear of the robot workspace before continuing.
                For first-time inference, test with 3D Sim Deploy before switching to Real Robot Deploy.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-2 px-4 py-3 bg-gray-50 border-t border-gray-200">
              <button
                type="button"
                onClick={handleConfirmRobotDeploy}
                className="flex-1 h-10 rounded-md bg-orange-600 text-white font-semibold hover:bg-orange-700 focus:outline-none focus:ring-2 focus:ring-orange-300 flex items-center justify-center gap-1.5"
              >
                <MdPrecisionManufacturing size={18} />
                Real Robot Deploy
              </button>
              <button
                type="button"
                onClick={handleUseSimDeploy}
                className="flex-1 h-10 rounded-md bg-emerald-600 text-white font-semibold hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-300 flex items-center justify-center gap-1.5"
              >
                <MdViewInAr size={18} />
                3D Sim Deploy
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
