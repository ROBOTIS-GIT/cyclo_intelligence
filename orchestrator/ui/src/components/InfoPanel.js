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
import { MdInfoOutline } from 'react-icons/md';
import { RecordPhase } from '../constants/taskPhases';
import {
  applyServerTaskInfo,
  markLocalTaskInfoEdited,
  markTaskInfoSyncFailed,
  markTaskInfoSyncing,
  markTaskInfoSyncMissing,
  markTaskInfoSyncPending,
  markTaskInfoSyncSuccess,
  selectRecordTaskInfo,
  setRecordTaskInfo,
} from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { getRecordTaskInfoKey } from '../utils/taskInfoSync';
import Tooltip from './Tooltip';

const AUTO_SYNC_DELAY_MS = 700;

const InfoPanel = ({ variant = 'card' }) => {
  const dispatch = useDispatch();
  const embedded = variant === 'embedded';

  const info = useSelector(selectRecordTaskInfo, shallowEqual);
  const recordStatus = useSelector((state) => state.tasks.recordStatus);
  const taskInfoSync = useSelector((state) => state.tasks.taskInfoSync);

  const [isTaskStatusPaused, setIsTaskStatusPaused] = useState(false);
  const [lastTaskStatusUpdate, setLastTaskStatusUpdate] = useState(Date.now());

  // RecordPage's lock — only the record-side phase matters here. Inference
  // phase is the InferencePanel's concern (D18, plan record-zippy-sunrise).
  const isTaskRunning = recordStatus.recordPhase !== RecordPhase.READY;
  const disabled = isTaskRunning;
  const [isEditable, setIsEditable] = useState(!disabled);

  const [isPreparing, setIsPreparing] = useState(false);
  const syncGenerationRef = useRef(0);
  const syncTimerRef = useRef(null);

  const { sendRecordCommand } = useRosServiceCaller();

  const handleChange = useCallback(
    (field, value) => {
      if (!isEditable) return; // Block changes when not editable
      dispatch(setRecordTaskInfo({ [field]: value }));
      dispatch(markLocalTaskInfoEdited());
    },
    [isEditable, dispatch]
  );

  // Update isEditable state when the disabled prop changes
  useEffect(() => {
    setIsEditable(!disabled);
  }, [disabled]);

  const hasTaskIdentity =
    Boolean(String(info.taskNum || '').trim()) &&
    Boolean(String(info.taskName || '').trim());

  const taskSyncKey = useMemo(
    () => getRecordTaskInfoKey(info),
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

    if (disabled) {
      return;
    }

    if (!hasTaskIdentity) {
      syncGenerationRef.current += 1;
      dispatch(markTaskInfoSyncMissing());
      return;
    }

    if (taskInfoSync.conflict) {
      return;
    }

    if (!taskInfoSync.dirty) {
      return;
    }

    const generation = syncGenerationRef.current + 1;
    syncGenerationRef.current = generation;
    dispatch(markTaskInfoSyncPending());

    syncTimerRef.current = setTimeout(async () => {
      dispatch(markTaskInfoSyncing());
      try {
        const result = await sendRecordCommand('set_task_info');
        if (syncGenerationRef.current !== generation) return;
        if (result && result.success) {
          dispatch(markTaskInfoSyncSuccess());
        } else {
          dispatch(markTaskInfoSyncFailed(
            (result && result.message) || 'Task info not synced; robot button may use old task.'
          ));
        }
      } catch (error) {
        if (syncGenerationRef.current !== generation) return;
        dispatch(markTaskInfoSyncFailed(
          `Task info not synced; robot button may use old task. ${error.message || error}`
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
    disabled,
    dispatch,
    hasTaskIdentity,
    sendRecordCommand,
    taskInfoSync.conflict,
    taskInfoSync.dirty,
    taskSyncKey,
  ]);

  const taskSyncStatus = taskInfoSync.syncStatus;
  const taskSyncMessage = taskInfoSync.syncMessage;
  const isPrepared =
    hasTaskIdentity &&
    !taskInfoSync.dirty &&
    !taskInfoSync.conflict &&
    taskInfoSync.serverTaskKey === taskSyncKey &&
    taskSyncStatus === 'synced';
  const isAutoSyncing = taskSyncStatus === 'pending' || taskSyncStatus === 'syncing';
  const canShowSyncAction = !disabled && hasTaskIdentity;

  const canPrepare =
    canShowSyncAction &&
    !isPreparing &&
    !isAutoSyncing;

  const handlePrepareSession = useCallback(async () => {
    if (!canPrepare) {
      if (!hasTaskIdentity) {
        toast.error('Fill in Task Num and Task Name first.');
      }
      return;
    }
    if (syncTimerRef.current) {
      clearTimeout(syncTimerRef.current);
      syncTimerRef.current = null;
    }
    const generation = syncGenerationRef.current + 1;
    syncGenerationRef.current = generation;
    setIsPreparing(true);
    dispatch(markTaskInfoSyncing());
    try {
      const syncResult = await sendRecordCommand('set_task_info');
      if (syncGenerationRef.current !== generation) return;
      if (!syncResult || !syncResult.success) {
        dispatch(markTaskInfoSyncFailed(
          (syncResult && syncResult.message) || 'Task info not synced; robot button may use old task.'
        ));
        toast.error(
          `Sync failed: ${(syncResult && syncResult.message) || 'Unknown error'}`
        );
        return;
      }

      const result = await sendRecordCommand('prepare_session');
      if (syncGenerationRef.current !== generation) return;
      if (result && result.success) {
        dispatch(markTaskInfoSyncSuccess());
        toast.success(result.message || 'Session prepared — use the leader to start.');
      } else {
        dispatch(markTaskInfoSyncFailed(
          (result && result.message) || 'Task info not synced; robot button may use old task.'
        ));
        toast.error(
          `Prepare failed: ${(result && result.message) || 'Unknown error'}`
        );
      }
    } catch (error) {
      if (syncGenerationRef.current !== generation) return;
      dispatch(markTaskInfoSyncFailed(
        `Task info not synced; robot button may use old task. ${error.message || error}`
      ));
      toast.error(`Prepare failed: ${error.message || error}`);
    } finally {
      if (syncGenerationRef.current === generation) {
        setIsPreparing(false);
      }
    }
  }, [canPrepare, dispatch, hasTaskIdentity, sendRecordCommand]);

  const handleUseServerInfo = useCallback(() => {
    if (syncTimerRef.current) {
      clearTimeout(syncTimerRef.current);
      syncTimerRef.current = null;
    }
    syncGenerationRef.current += 1;
    dispatch(applyServerTaskInfo());
  }, [dispatch]);

  // track task status update
  useEffect(() => {
    if (recordStatus) {
      setLastTaskStatusUpdate(Date.now());
      setIsTaskStatusPaused(false);
    }
  }, [recordStatus]);

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

  const classInfoPanel = embedded
    ? clsx('rounded-lg', 'p-2', 'border', 'border-gray-200')
    : clsx(
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

  const classTaskNameTextarea = clsx(
    'text-sm',
    'resize-y',
    'min-h-8',
    'max-h-20',
    'h-10',
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

  const classTaskInstructionTextarea = clsx(
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

  const classSyncStatus = clsx(
    'mb-2',
    'p-2',
    'rounded-md',
    'border',
    'text-xs',
    'font-medium',
    {
      'bg-green-50 border-green-200 text-green-700': taskSyncStatus === 'synced',
      'bg-amber-50 border-amber-200 text-amber-700':
        taskSyncStatus === 'pending' || taskSyncStatus === 'syncing',
      'bg-red-50 border-red-200 text-red-700':
        taskSyncStatus === 'failed' || taskSyncStatus === 'conflict',
      'bg-gray-50 border-gray-200 text-gray-500':
        taskSyncStatus === 'missing' || taskSyncStatus === 'idle',
    }
  );

  return (
    <div className={classInfoPanel}>
      {!embedded && (
        <div className={clsx('text-lg', 'font-semibold', 'mb-3', 'text-gray-800')}>
          Task Information
        </div>
      )}

      {/* Edit mode indicator */}
      {!embedded && (
        <div
          className={clsx('mb-3', 'p-2', 'rounded-md', 'text-sm', 'font-medium', {
            'bg-green-100 text-green-800': isEditable,
            'bg-gray-100 text-gray-600': !isEditable,
          })}
        >
          {isEditable ? (
            'Edit mode'
          ) : (
            <div className="leading-tight">
              <div>Read only</div>
              <div className="text-xs mt-1 opacity-80">task is running or robot is not connected</div>
            </div>
          )}
        </div>
      )}

      {/* Task Num */}
      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <span className={classLabel}>Task Num</span>
        <textarea
          className={classTaskNameTextarea}
          value={info.taskNum || ''}
          onChange={(e) => handleChange('taskNum', e.target.value)}
          disabled={!isEditable}
          placeholder="Enter Task Num"
        />
      </div>

      {/* Task Name */}
      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <span className={classLabel}>Task Name</span>
        <textarea
          className={classTaskNameTextarea}
          value={info.taskName || ''}
          onChange={(e) => handleChange('taskName', e.target.value)}
          disabled={!isEditable}
          placeholder="Enter Task Name"
        />
      </div>

      {/* Task Instruction */}
      <div className={clsx('flex', 'items-start', 'mb-2.5')}>
        <span className={clsx(classLabel, 'pt-2')}>Task Instruction</span>
        <textarea
          className={classTaskInstructionTextarea}
          value={(info.taskInstruction && info.taskInstruction[0]) || ''}
          onChange={(e) => handleChange('taskInstruction', [e.target.value])}
          disabled={!isEditable}
          placeholder="Enter Task Instruction"
        />
      </div>

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        {/* ROBOTIS license stamp — opt-in. Default off because recording
            outputs are the user's intellectual property, not ROBOTIS'. */}
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip
            content={
              'Off by default — recording outputs are the user’s, not ROBOTIS’. ' +
              'Tick on for ROBOTIS-internal captures: README gets the Apache 2.0 license ' +
              'header (Copyright ROBOTIS CO., LTD.) baked in and rides through conversion ' +
              'and HF Hub upload.'
            }
            position="bottom"
          >
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Add License</span>
        </div>
        <input
          type="checkbox"
          checked={Boolean(info.includeRobotisLicense)}
          onChange={(e) => handleChange('includeRobotisLicense', e.target.checked)}
          disabled={!isEditable}
          className="rounded"
        />
      </div>

      <div className={classSyncStatus}>
        {taskSyncMessage || 'Task info will sync automatically.'}
      </div>

      {taskInfoSync.conflict && taskInfoSync.serverTaskInfo && (
        <button
          type="button"
          onClick={handleUseServerInfo}
          disabled={disabled}
          className={clsx(
            'w-full',
            'mb-2',
            'px-2',
            'py-1.5',
            'rounded-md',
            'border',
            'text-xs',
            'font-semibold',
            'transition-colors',
            disabled
              ? 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed'
              : 'bg-white border-red-300 text-red-700 hover:bg-red-50'
          )}
        >
          Use server info: Task_{taskInfoSync.serverTaskInfo.taskNum}_{taskInfoSync.serverTaskInfo.taskName}_MCAP
        </button>
      )}

      {/* Prepare-session button. Doubles as the "saved as" indicator:
          clicking arms the orchestrator with this task_info so the
          leader joystick can drive episode 0 without anyone touching
          the UI's RECORD button. Auto-sync normally keeps this current;
          clicking forces an immediate prepare/sync. */}
      <button
        type="button"
        onClick={handlePrepareSession}
        disabled={!canPrepare}
        title={
          !hasTaskIdentity
            ? 'Fill in Task Num and Task Name to enable.'
            : disabled
              ? 'Task is running; task info cannot be changed right now.'
              : isAutoSyncing || isPreparing
                ? 'Task info is syncing.'
                : isPrepared
                  ? 'Task info is synced — use the leader joystick to start recording.'
                  : 'Sync task info now so the leader joystick uses the latest values.'
        }
        className={clsx(
          'flex',
          'flex-col',
          'items-center',
          'w-full',
          'text-xs',
          'mt-3',
          'leading-relaxed',
          'p-2',
          'rounded-md',
          'border',
          'transition-colors',
          'focus:outline-none',
          'focus:ring-2',
          'focus:ring-blue-400',
          {
            'bg-amber-50 border-amber-300 text-amber-700 hover:bg-amber-100 cursor-pointer':
              canShowSyncAction &&
              !isPrepared &&
              taskSyncStatus !== 'failed' &&
              taskSyncStatus !== 'conflict',
            'bg-red-50 border-red-300 text-red-700 hover:bg-red-100 cursor-pointer':
              canShowSyncAction &&
              (taskSyncStatus === 'failed' || taskSyncStatus === 'conflict'),
            'bg-green-50 border-green-300 text-green-700 hover:bg-green-100 cursor-pointer':
              canShowSyncAction && isPrepared,
            'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed': !canShowSyncAction,
            'cursor-not-allowed': canShowSyncAction && !canPrepare,
          }
        )}
      >
        <div>
          {isPreparing
            ? 'Preparing…'
            : isAutoSyncing
              ? 'Syncing task info…'
              : isPrepared
                ? 'Task info synced — leader ready'
                : taskSyncStatus === 'failed'
                  ? 'Task info not synced — click to retry:'
                  : taskSyncStatus === 'conflict'
                    ? 'Local draft differs — click to overwrite server as:'
                  : 'Sync now as:'}
        </div>
        <div
          className={clsx('font-bold', 'break-all', {
            'text-amber-700':
              !isPrepared && taskSyncStatus !== 'failed' && taskSyncStatus !== 'conflict',
            'text-red-700': taskSyncStatus === 'failed' || taskSyncStatus === 'conflict',
            'text-green-700': isPrepared,
          })}
        >
          Task_{info.taskNum}_{info.taskName}_MCAP
        </div>
      </button>

    </div>
  );
};

export default InfoPanel;
