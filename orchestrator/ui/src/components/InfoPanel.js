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

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { MdInfoOutline } from 'react-icons/md';
import { RecordPhase } from '../constants/taskPhases';
import { setTaskInfo } from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import Tooltip from './Tooltip';

const InfoPanel = () => {
  const dispatch = useDispatch();

  const info = useSelector((state) => state.tasks.taskInfo);
  const recordStatus = useSelector((state) => state.tasks.recordStatus);

  const [isTaskStatusPaused, setIsTaskStatusPaused] = useState(false);
  const [lastTaskStatusUpdate, setLastTaskStatusUpdate] = useState(Date.now());

  // RecordPage's lock — only the record-side phase matters here. Inference
  // phase is the InferencePanel's concern (D18, plan record-zippy-sunrise).
  const isTaskRunning = recordStatus.recordPhase !== RecordPhase.READY;
  const disabled = isTaskRunning;
  const [isEditable, setIsEditable] = useState(!disabled);

  // Tracks whether prepare_session has succeeded for the current task
  // identity. Reset whenever taskNum / taskName change so the button
  // re-arms — user needs to confirm again after editing.
  const [isPrepared, setIsPrepared] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);

  const { sendRecordCommand } = useRosServiceCaller();

  const handleChange = useCallback(
    (field, value) => {
      if (!isEditable) return; // Block changes when not editable
      dispatch(setTaskInfo({ ...info, [field]: value }));
    },
    [isEditable, info, dispatch]
  );

  // Update isEditable state when the disabled prop changes
  useEffect(() => {
    setIsEditable(!disabled);
  }, [disabled]);

  // Re-arm the prepare button whenever the task identity changes so the
  // user has to re-confirm. Solo-recording flow relies on this — editing
  // task_num/name mid-session should not silently keep an old prep.
  useEffect(() => {
    setIsPrepared(false);
  }, [info.taskNum, info.taskName]);

  const canPrepare =
    !disabled &&
    !isPreparing &&
    Boolean(String(info.taskNum || '').trim()) &&
    Boolean(String(info.taskName || '').trim());

  const handlePrepareSession = useCallback(async () => {
    if (!canPrepare) {
      if (!String(info.taskNum || '').trim() || !String(info.taskName || '').trim()) {
        toast.error('Fill in Task Num and Task Name first.');
      }
      return;
    }
    setIsPreparing(true);
    try {
      const result = await sendRecordCommand('prepare_session');
      if (result && result.success) {
        setIsPrepared(true);
        toast.success(result.message || 'Session prepared — use the leader to start.');
      } else {
        setIsPrepared(false);
        toast.error(
          `Prepare failed: ${(result && result.message) || 'Unknown error'}`
        );
      }
    } catch (error) {
      setIsPrepared(false);
      toast.error(`Prepare failed: ${error.message || error}`);
    } finally {
      setIsPreparing(false);
    }
  }, [canPrepare, info.taskNum, info.taskName, sendRecordCommand]);

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

  return (
    <div className={classInfoPanel}>
      <div className={clsx('text-lg', 'font-semibold', 'mb-3', 'text-gray-800')}>
        Task Information
      </div>

      {/* Edit mode indicator */}
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

      {/* ROBOTIS license stamp — opt-in. Default off because recording
          outputs are the user's intellectual property, not ROBOTIS'.
          Tick on for ROBOTIS-internal captures so the Apache 2.0
          header rides through to HF Hub. */}
      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
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

      {/* Prepare-session button. Doubles as the "saved as" indicator:
          clicking arms the orchestrator with this task_info so the
          leader joystick can drive episode 0 without anyone touching
          the UI's RECORD button. Re-arms on task_num/name edit. */}
      <button
        type="button"
        onClick={handlePrepareSession}
        disabled={!canPrepare}
        title={
          !canPrepare && !isPreparing
            ? 'Fill in Task Num and Task Name to enable.'
            : isPrepared
              ? 'Session prepared — use the leader joystick to start recording.'
              : 'Click to arm this task on the orchestrator so the leader joystick can start episode 0.'
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
            'bg-gray-100 border-gray-200 text-gray-500 hover:bg-blue-50 hover:border-blue-300 cursor-pointer':
              canPrepare && !isPrepared,
            'bg-green-50 border-green-300 text-green-700 hover:bg-green-100 cursor-pointer':
              canPrepare && isPrepared,
            'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed': !canPrepare,
          }
        )}
      >
        <div>
          {isPreparing
            ? 'Preparing…'
            : isPrepared
              ? 'Session ready — use leader to record'
              : 'Click to prepare session as:'}
        </div>
        <div
          className={clsx('font-bold', 'break-all', {
            'text-blue-500': !isPrepared,
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
