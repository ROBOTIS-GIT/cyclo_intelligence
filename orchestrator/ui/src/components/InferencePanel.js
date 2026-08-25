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
  MdClose,
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
import { InferencePhase } from '../constants/taskPhases';
import { DEFAULT_PATHS } from '../constants/paths';
import {
  markLocalTaskInfoEdited,
  markInferenceTaskInfoSyncFailed,
  markInferenceTaskInfoSyncing,
  markInferenceTaskInfoSyncPending,
  markInferenceTaskInfoSyncSuccess,
  selectInferenceRecordingControl,
  selectInferenceTaskInfo,
  setInferenceMode,
  setRecordInferenceMode,
  setInferenceTaskInfo,
} from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { requiresInstruction } from '../constants/policyCapabilities';
import { getInferenceTaskInfoKey } from '../utils/taskInfoSync';
import {
  getInferenceRecordingFolderName,
  getInferenceRecordingSessionId,
} from '../utils/inferenceRecordingFolder';

const AUTO_SYNC_DELAY_MS = 700;

const InferencePanel = ({
  title = 'Task Information',
  embedded = false,
  showPolicyPath = true,
  showRecordingSettings = true,
  modelLabel = 'Model',
  variant = 'default',
  settingsAside = null,
}) => {
  const dispatch = useDispatch();

  const info = useSelector(selectInferenceTaskInfo, shallowEqual);
  const taskInfoSync = useSelector((state) => state.tasks.inferenceTaskInfoSync);
  const robotType = useSelector((state) => state.tasks.robotType);
  const inferenceStatus = useSelector((state) => state.tasks.inferenceStatus);
  const recordingControl = useSelector(selectInferenceRecordingControl);
  const showInstruction = requiresInstruction(info.serviceType, info.policyType);
  const isOfflineRL = variant === 'offlineRL';

  const [isTaskStatusPaused, setIsTaskStatusPaused] = useState(false);
  const [lastTaskStatusUpdate, setLastTaskStatusUpdate] = useState(Date.now());
  const [showPolicyBrowser, setShowPolicyBrowser] = useState(false);
  const [showRecordingFolderBrowser, setShowRecordingFolderBrowser] = useState(false);

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
    inferenceStatus.inferencePhase === InferencePhase.LOADING ||
    recordingControl.lifecycleLocked;
  const isModelActive = [
    InferencePhase.INFERENCING,
    InferencePhase.PAUSED,
  ].includes(inferenceStatus.inferencePhase);
  const disabled = isTaskRunning;
  const [isEditable, setIsEditable] = useState(!disabled);
  const [isUpdatingInstruction, setIsUpdatingInstruction] = useState(false);
  const syncGenerationRef = useRef(0);
  const syncTimerRef = useRef(null);

  const { sendRecordCommand } = useRosServiceCaller();

  useEffect(() => {
    if (!isRobotMode && info.recordInferenceMode) {
      dispatch(setRecordInferenceMode(false));
      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    }
  }, [dispatch, info.recordInferenceMode, isRobotMode]);

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
          taskSource: 'inference',
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
        const result = await sendRecordCommand('finish', {
          taskSource: 'inference',
        }).catch((error) => {
          toast.error(`Deploy target reset failed: ${error.message || error}`);
          return null;
        });
        if (!result || result.success === false) {
          toast.error(result?.message || 'Inference reset failed');
          return;
        }
        toast('Inference reset before deploy target switch');
      }

      if (mode === 'simulation') {
        dispatch(setRecordInferenceMode(false));
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

  const currentInstruction = (info.taskInstruction?.[0] || '').trim();
  const canUpdateInstruction =
    isInferencing && currentInstruction !== '' && !isUpdatingInstruction;

  const handleUpdateInstruction = useCallback(async () => {
    if (!canUpdateInstruction) return;
    setIsUpdatingInstruction(true);
    try {
      const result = await sendRecordCommand('update_instruction', {
        taskSource: 'inference',
      });
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

  const handleRecordingFolderSelect = useCallback((item) => {
    const fullPath = String(item?.full_path || '').trim();
    if (!getInferenceRecordingSessionId(fullPath)) {
      toast.error(
        'Select a Task_*_inference_MCAP folder directly under /workspace/rosbag2'
      );
      return;
    }
    dispatch(setInferenceTaskInfo({ recordingFolder: fullPath }));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    setShowRecordingFolderBrowser(false);
  }, [dispatch]);

  const clearRecordingFolder = useCallback(() => {
    dispatch(setInferenceTaskInfo({ recordingFolder: '' }));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
  }, [dispatch]);

  const policyBrowserPath =
    info.serviceType === 'groot'
      ? DEFAULT_PATHS.GROOT_CHECKPOINTS_PATH
      : DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH;
  const recordingFolderName = getInferenceRecordingFolderName(
    info.recordingFolder
  );
  const recordingFolderLocked = !isEditable || recordingControl.lifecycleLocked;

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

  const classLabel = clsx(
    'w-28 flex-shrink-0 font-medium',
    isOfflineRL ? 'text-[10px] text-[#6e675c]' : 'text-sm text-gray-600'
  );

  const classInfoPanel = clsx(
    isOfflineRL ? 'bg-transparent' : 'rounded-2xl bg-white',
    'w-full',
    'relative',
    isOfflineRL ? 'overflow-visible' : 'overflow-y-auto scrollbar-thin',
    !embedded && [
      'border',
      'border-gray-200',
      'shadow-md',
      'p-4',
      'max-w-[350px]',
    ]
  );

  // taskInstruction stays always-editable so multi-task LLM-conditioned
  // policies can be re-conditioned mid-run. No isEditable gate here.
  const classTaskInstructionTextarea = clsx(
    isOfflineRL ? 'text-[11px] text-[#403b34]' : 'text-sm',
    'resize-y',
    'min-h-10',
    'max-h-20',
    'w-full',
    'p-2',
    'border',
    isOfflineRL ? 'border-[#d9d2c5]' : 'border-gray-300',
    isOfflineRL ? 'rounded-lg' : 'rounded-md',
    'focus:outline-none',
    'focus:ring-2',
    isOfflineRL ? 'focus:ring-[#c7beb0]' : 'focus:ring-blue-500',
    'focus:border-transparent',
    isOfflineRL ? 'bg-[#fffefa]' : 'bg-white',
  );

  const classPolicyPathTextarea = clsx(
    isOfflineRL ? 'text-[11px] text-[#403b34]' : 'text-sm',
    'resize-y',
    'min-h-16',
    'max-h-24',
    'w-full',
    'p-2',
    'border',
    isOfflineRL ? 'border-[#d9d2c5]' : 'border-gray-300',
    isOfflineRL ? 'rounded-lg' : 'rounded-md',
    'focus:outline-none',
    'focus:ring-2',
    isOfflineRL ? 'focus:ring-[#c7beb0]' : 'focus:ring-blue-500',
    'focus:border-transparent',
    {
      [isOfflineRL
        ? 'cursor-not-allowed bg-[#ece8df] text-[#8f877b]'
        : 'bg-gray-100 cursor-not-allowed']: !isEditable,
      [isOfflineRL ? 'bg-[#fffefa]' : 'bg-white']: isEditable,
    }
  );

  const classTextInput = clsx(
    isOfflineRL ? 'text-[11px] text-[#403b34]' : 'text-sm',
    'w-full',
    'h-8',
    'p-2',
    'border',
    isOfflineRL ? 'border-[#d9d2c5]' : 'border-gray-300',
    isOfflineRL ? 'rounded-lg' : 'rounded-md',
    'focus:outline-none',
    'focus:ring-2',
    isOfflineRL ? 'focus:ring-[#c7beb0]' : 'focus:ring-blue-500',
    'focus:border-transparent',
    {
      [isOfflineRL
        ? 'cursor-not-allowed bg-[#ece8df] text-[#8f877b]'
        : 'bg-gray-100 cursor-not-allowed']: !isEditable,
      [isOfflineRL ? 'bg-[#fffefa]' : 'bg-white']: isEditable,
    }
  );

  const deployButtonClass = (active, danger = false) => clsx(
    'h-9',
    'min-w-0',
    'px-2',
    isOfflineRL ? 'rounded-lg' : 'rounded-md',
    'flex',
    'items-center',
    'justify-center',
    'gap-1.5',
    isOfflineRL ? 'text-[10px]' : 'text-xs',
    'font-semibold',
    'whitespace-nowrap',
    'transition-colors',
    'focus:outline-none',
    'focus:ring-2',
    active
      ? danger
        ? (isOfflineRL
          ? 'bg-[#a8795b] text-white focus:ring-[#d9bba8]'
          : 'bg-orange-500 text-white focus:ring-orange-300')
        : (isOfflineRL
          ? 'bg-[#69866f] text-white focus:ring-[#b9c5b6]'
          : 'bg-emerald-500 text-white focus:ring-emerald-300')
      : (isOfflineRL
        ? 'border border-[#d9d2c5] bg-[#fffefa] text-[#6e675c] hover:bg-[#f0ece4] focus:ring-[#c7beb0]'
        : 'bg-white text-gray-600 hover:bg-gray-50 focus:ring-gray-300 border border-gray-200'),
    {
      'opacity-50 cursor-not-allowed': isModeSwitchLocked,
      'cursor-pointer': !isModeSwitchLocked,
    }
  );

  const actionModeButtonClass = (active) => clsx(
    'h-8',
    'min-w-0',
    'px-2',
    isOfflineRL ? 'rounded-lg' : 'rounded-md',
    'flex',
    'items-center',
    'justify-center',
    'gap-1.5',
    isOfflineRL ? 'text-[10px]' : 'text-xs',
    'font-semibold',
    'whitespace-nowrap',
    'transition-colors',
    'focus:outline-none',
    'focus:ring-2',
    active
      ? (isOfflineRL
        ? 'bg-[#71806b] text-white focus:ring-[#b9c5b6]'
        : 'bg-blue-500 text-white focus:ring-blue-300')
      : (isOfflineRL
        ? 'border border-[#d9d2c5] bg-[#fffefa] text-[#6e675c] hover:bg-[#f0ece4] focus:ring-[#c7beb0]'
        : 'bg-white text-gray-600 hover:bg-gray-50 focus:ring-gray-300 border border-gray-200'),
    {
      'opacity-50 cursor-not-allowed': !isEditable,
      'cursor-pointer': isEditable,
    }
  );

  const taskInstructionControls = showInstruction ? (
    <div className={isOfflineRL ? 'mt-2 border-t border-[#e2dcd1] pt-2' : ''}>
      <div className={clsx(
        isOfflineRL ? 'mb-1.5' : 'mb-1 flex items-start'
      )}>
        <span className={clsx(
          classLabel,
          isOfflineRL ? 'mb-1 block w-auto' : 'pt-2'
        )}>
          Task Instruction
        </span>
        <textarea
          className={classTaskInstructionTextarea}
          value={(info.taskInstruction && info.taskInstruction[0]) || ''}
          onChange={(e) => handleChange('taskInstruction', [e.target.value])}
          placeholder="Enter Task Instruction"
        />
      </div>
      <div className={clsx(
        isOfflineRL ? 'flex justify-end' : 'mb-2.5 flex items-center'
      )}>
        {!isOfflineRL && <span className={classLabel} />}
        <button
          type="button"
          onClick={handleUpdateInstruction}
          disabled={!canUpdateInstruction}
          className={clsx(
            'px-3 py-1.5 font-semibold transition-colors disabled:cursor-not-allowed',
            isOfflineRL
              ? 'rounded-lg bg-[#71806b] text-[10px] text-white hover:bg-[#64725f] disabled:bg-[#d7d0c4] disabled:text-[#8f877b]'
              : 'rounded-md bg-blue-500 text-sm text-white hover:bg-blue-600 disabled:bg-gray-300 disabled:text-gray-500'
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
    </div>
  ) : null;

  const tensorRtControls = isGrootModel ? (
    <div className={isOfflineRL ? 'mt-1 border-t border-[#e2dcd1] pt-2' : ''}>
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
          <span className={isOfflineRL ? 'text-[10px] text-[#756e63]' : 'text-gray-500'}>
            Enable
          </span>
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
          variant={variant}
        />
      )}
    </div>
  ) : null;

  return (
    <div
      className={classInfoPanel}
      data-appearance={isOfflineRL ? 'offline-rl' : 'default'}
    >
      <div className={clsx(
        'mb-3 font-semibold',
        isOfflineRL
          ? 'text-[10px] uppercase tracking-[0.12em] text-[#81796d]'
          : 'text-lg text-gray-800'
      )}
      >
        {title}
      </div>

      <InferenceModelSelector
        readonly={!isEditable}
        label={modelLabel}
        variant={variant}
      />

      <div
        className={isOfflineRL
          ? 'grid min-w-0 items-stretch gap-2 md:grid-cols-2'
          : 'contents'}
        data-testid={isOfflineRL ? 'offline-rl-runtime-row' : undefined}
      >
        <PolicyBackendControl
          serviceType={info.serviceType}
          variant={variant}
        >
          {isOfflineRL && tensorRtControls}
        </PolicyBackendControl>

        <div className={clsx(
          'mb-3 rounded-lg border p-2',
          isOfflineRL
            ? 'border-[#ded8cc] bg-[#fbfaf6]'
            : 'border-gray-200 bg-gray-50'
        )}
        >
          <div className="flex items-center justify-between gap-2 mb-1.5">
            <span className={clsx(
              'font-medium',
              isOfflineRL ? 'text-[10px] text-[#6e675c]' : 'text-sm text-gray-600'
            )}
            >
              Deploy Target
            </span>
            <span className={clsx(
              'inline-flex items-center gap-1 rounded-md px-2 py-0.5 font-semibold whitespace-nowrap',
              isOfflineRL ? 'text-[9px]' : 'text-xs',
              isRobotMode
                ? (isOfflineRL
                  ? 'bg-[#f1e5dc] text-[#93654e]'
                  : 'bg-orange-100 text-orange-700')
                : (isOfflineRL
                  ? 'bg-[#e5ece5] text-[#607563]'
                  : 'bg-emerald-100 text-emerald-700')
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
          {isOfflineRL && taskInstructionControls}
        </div>
      </div>

      {/* The compact Offline RL workspace communicates its lock state through
          the controls themselves; keep the legacy status banner only on the
          standalone Inference page. */}
      {!isOfflineRL && (
        <div
          className={clsx(
            'mb-3 rounded-lg p-2 text-sm font-medium',
            {
              'bg-green-100 text-green-800': isEditable,
              'bg-gray-100 text-gray-600': !isEditable,
            }
          )}
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
      )}

      {/* Task Instruction — only shown for language-conditioned policies.
          Whitelist lives in constants/policyCapabilities.js. */}
      {!isOfflineRL && taskInstructionControls}

      {/* Policy Path */}
      {showPolicyPath && (
        <div className={clsx('flex', 'items-start', 'mb-2.5')}>
          <span className={clsx(classLabel, 'pt-2')}>Policy Path</span>
          <div className="flex flex-row items-start gap-2 flex-1 min-w-0">
            <textarea
              className={classPolicyPathTextarea}
              value={info.policyPath || ''}
              onChange={(e) => handleChange('policyPath', e.target.value)}
              disabled={!isEditable}
              placeholder="Enter Policy Path or Repo ID"
            />
            <button
              type="button"
              onClick={() => setShowPolicyBrowser(true)}
              disabled={!isEditable}
              className="flex items-center justify-center w-9 h-9 text-blue-500 bg-gray-200 rounded-md hover:text-blue-700 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
              aria-label="Browse for policy model folder"
            >
              <MdFolderOpen className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}

      {!isOfflineRL && tensorRtControls}

      <div
        className={clsx(
          isOfflineRL && settingsAside &&
            'grid min-w-0 items-stretch gap-2 md:grid-cols-2'
        )}
        data-testid={isOfflineRL && settingsAside
          ? 'offline-rl-action-path-row'
          : undefined}
      >
        <div className={clsx(
          'min-w-0',
          isOfflineRL && settingsAside &&
            'rounded-xl border border-[#e2dcd1] bg-[#fbfaf6] p-3'
        )}>
          {isOfflineRL && settingsAside ? (
            <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#8f877b]">
              Action &amp; timing
            </div>
          ) : (
            <div className={clsx(
              'my-2 h-1 w-full border-t',
              isOfflineRL ? 'border-[#ded8cc]' : 'border-gray-300'
            )}
            />
          )}

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
        </div>

        {isOfflineRL && settingsAside}
      </div>

      {showRecordingSettings && (
        <>
          <div
            role="separator"
            aria-label="RL Recording settings"
            className="w-full h-1 my-2 border-t border-gray-300"
          />

          <div className="rounded-md border border-gray-200">
            <div className="flex h-9 items-center justify-between gap-3 px-2.5">
              <span className="text-sm font-medium text-gray-600">RL Recording</span>
              <label className="relative inline-flex cursor-pointer items-center">
                <input
                  type="checkbox"
                  className="peer sr-only"
                  checked={isRobotMode && Boolean(info.recordInferenceMode)}
                  onChange={(event) => {
                    if (!isRobotMode || !isEditable || recordingControl.lifecycleLocked) {
                      return;
                    }
                    dispatch(setRecordInferenceMode(event.target.checked));
                    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
                  }}
                  disabled={
                    !isRobotMode || !isEditable || recordingControl.lifecycleLocked
                  }
                  aria-label="Enable RL Recording"
                />
                <span className="h-5 w-9 rounded-full bg-gray-300 transition-colors peer-checked:bg-red-500 peer-disabled:cursor-not-allowed peer-disabled:opacity-50 after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full after:bg-white after:transition-transform peer-checked:after:translate-x-4" />
              </label>
            </div>

            {isRobotMode && info.recordInferenceMode && (
              <div className="border-t border-gray-200 px-2.5 py-2">
                <div className="mb-1 text-xs font-medium text-gray-500">Save Folder</div>
                <div className="flex min-w-0 items-center gap-1.5">
                  <span
                    className="min-w-0 flex-1 truncate text-xs text-gray-700"
                    title={info.recordingFolder || 'Automatic new folder'}
                  >
                    {recordingFolderName || 'Automatic new folder'}
                  </span>
                  {recordingFolderName && (
                    <button
                      type="button"
                      onClick={clearRecordingFolder}
                      disabled={recordingFolderLocked}
                      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-gray-500 hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
                      aria-label="Clear RL Recording folder"
                      title="Use automatic new folder"
                    >
                      <MdClose size={17} />
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setShowRecordingFolderBrowser(true)}
                    disabled={recordingFolderLocked}
                    className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-gray-100 text-blue-600 hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-50"
                    aria-label="Select RL Recording folder"
                    title="Select existing RL Recording folder"
                  >
                    <MdFolderOpen size={18} />
                  </button>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {showPolicyPath && (
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
      )}
      {showRecordingSettings && (
        <FileBrowserModal
          isOpen={showRecordingFolderBrowser}
          onClose={() => setShowRecordingFolderBrowser(false)}
          onFileSelect={handleRecordingFolderSelect}
          title="Select RL Recording folder"
          selectButtonText="Use Folder"
          allowDirectorySelect={true}
          allowFileSelect={false}
          initialPath={DEFAULT_PATHS.ROSBAG2_PATH}
          defaultPath={DEFAULT_PATHS.ROSBAG2_PATH}
          homePath={DEFAULT_PATHS.ROSBAG2_PATH}
        />
      )}
    </div>
  );
};

export default InferencePanel;
