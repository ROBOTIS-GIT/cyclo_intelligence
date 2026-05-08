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
import { MdInfoOutline } from 'react-icons/md';
import { RecordPhase } from '../constants/taskPhases';
import { setTaskInfo } from '../features/tasks/taskSlice';
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

      {/* Dataset save path indicator */}
      <div className="flex flex-col items-center text-xs text-gray-500 mt-3 leading-relaxed bg-gray-100 p-2 rounded-md">
        <div>Dataset will be saved as:</div>
        <div className="text-blue-500 font-bold break-all">
          Task_{info.taskNum}_{info.taskName}_MCAP
        </div>
      </div>

    </div>
  );
};

export default InfoPanel;
