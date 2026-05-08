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
import { MdFolderOpen, MdInfoOutline } from 'react-icons/md';
import FileBrowserModal from './FileBrowserModal';
import InferenceModelSelector from './InferenceModelSelector';
import Tooltip from './Tooltip';
import { InferencePhase } from '../constants/taskPhases';
import { DEFAULT_PATHS } from '../constants/paths';
import { setTaskInfo } from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { requiresInstruction } from '../constants/policyCapabilities';

const InferencePanel = () => {
  const dispatch = useDispatch();

  const info = useSelector((state) => state.tasks.taskInfo);
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
  const disabled = isTaskRunning;
  const [isEditable, setIsEditable] = useState(!disabled);
  const [isUpdatingInstruction, setIsUpdatingInstruction] = useState(false);

  const { sendRecordCommand } = useRosServiceCaller();

  const handleChange = useCallback(
    (field, value) => {
      // taskInstruction stays editable while inference is running so a
      // multi-task language-conditioned policy can be re-conditioned via
      // the "Update Task Instruction" button below.
      if (field !== 'taskInstruction' && !isEditable) return;
      dispatch(setTaskInfo({ ...info, [field]: value }));
    },
    [isEditable, info, dispatch]
  );

  const currentInstruction = (info.taskInstruction?.[0] || '').trim();
  const canUpdateInstruction =
    isInferencing && currentInstruction !== '' && !isUpdatingInstruction;

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
      dispatch(setTaskInfo({ ...info, policyPath: fullPath }));
    }
    setShowPolicyBrowser(false);
  }, [isEditable, info, dispatch]);

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

  return (
    <div className={classInfoPanel}>
      <div className={clsx('text-lg', 'font-semibold', 'mb-3', 'text-gray-800')}>
        Task Information
      </div>

      <InferenceModelSelector readonly={!isEditable} />

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
              value={info.taskInstruction || ''}
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

      {/* Policy Path */}
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

      <div className="w-full h-1 my-2 border-t border-gray-300"></div>

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

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip content="How far ahead inference can jump when joining chunks. Keep small (~0.3s) for loop trajectories." position="bottom">
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Max Skip Ahead (s)</span>
        </div>
        <input
          className={classTextInput}
          type="number"
          step="0.05"
          min="0"
          value={info.chunkAlignWindowS ?? ''}
          onChange={(e) => {
            const v = e.target.value;
            handleChange('chunkAlignWindowS', v === '' ? '' : Number(v));
          }}
          disabled={!isEditable}
        />
      </div>

      {/* Record during inference toggle */}
      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <span className={classLabel}>Record</span>
        <label className={clsx('flex', 'items-center', 'gap-2', 'text-sm')}>
          <input
            type="checkbox"
            className={clsx('w-4 h-4', {
              'cursor-not-allowed opacity-50': !isEditable,
              'cursor-pointer': isEditable,
            })}
            checked={!!info.recordInferenceMode}
            onChange={(e) => handleChange('recordInferenceMode', e.target.checked)}
            disabled={!isEditable}
          />
          <span className="text-gray-500">
            {info.recordInferenceMode ? 'Enabled' : 'Disabled'}
          </span>
        </label>
      </div>

      {/* Recording-only fields */}
      {info.recordInferenceMode && (
        <>
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

          {/* Dataset save path indicator */}
          <div className="flex flex-col items-center text-xs text-gray-500 mt-3 leading-relaxed bg-gray-100 p-2 rounded-md">
            <div>Dataset will be saved as:</div>
            <div className="text-blue-500 font-bold break-all">
              Task_{info.taskNum}_{info.taskName}_Inference_MCAP
            </div>
          </div>
        </>
      )}

      <FileBrowserModal
        isOpen={showPolicyBrowser}
        onClose={() => setShowPolicyBrowser(false)}
        onFileSelect={handlePolicyFolderSelect}
        title="Select policy model folder"
        selectButtonText="Select"
        allowDirectorySelect={true}
        allowFileSelect={false}
        initialPath={DEFAULT_PATHS.HF_MODEL_DOWNLOAD_PATH}
        defaultPath={DEFAULT_PATHS.HF_MODEL_DOWNLOAD_PATH}
        homePath=""
      />
    </div>
  );
};

export default InferencePanel;
