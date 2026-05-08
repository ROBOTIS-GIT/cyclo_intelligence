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
import { useSelector } from 'react-redux';
import clsx from 'clsx';
import toast, { useToasterStore } from 'react-hot-toast';
import { MdFiberManualRecord, MdSave, MdClose } from 'react-icons/md';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import Tooltip from './Tooltip';
import { RecordPhase } from '../constants/taskPhases';

const phaseGuideMessages = {
  [RecordPhase.READY]: 'Ready to start',
  [RecordPhase.RECORDING]: 'Recording',
  [RecordPhase.SAVING]: 'Saving...',
};

const requiredFieldsForRecord = [];

const requiredFieldsForRecordInferenceMode = [
  { key: 'policyPath', label: 'Policy Path' },
];

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];

export default function RecordControlPanel() {
  const taskInfo = useSelector((state) => state.tasks.taskInfo);
  const recordStatus = useSelector((state) => state.tasks.recordStatus);
  const rosHost = useSelector((state) => state.ros.rosHost);

  const [hovered, setHovered] = useState(null);
  const [pressed, setPressed] = useState(null);
  const [spinnerIndex, setSpinnerIndex] = useState(0);

  const { sendRecordCommand } = useRosServiceCaller();

  const { toasts } = useToasterStore();
  const TOAST_LIMIT = 3;

  const phase = recordStatus.recordPhase;
  const isRecording = phase === RecordPhase.RECORDING;
  const isSaving = phase === RecordPhase.SAVING;
  const isBusy = isRecording || isSaving;

  useEffect(() => {
    toasts
      .filter((t) => t.visible)
      .filter((_, i) => i >= TOAST_LIMIT)
      .forEach((t) => toast.dismiss(t.id));
  }, [toasts]);

  useEffect(() => {
    setSpinnerIndex((prev) => (prev + 1) % spinnerFrames.length);
  }, [recordStatus]);

  const validateTaskInfo = useCallback(() => {
    const requiredFields = taskInfo.recordInferenceMode
      ? requiredFieldsForRecordInferenceMode
      : requiredFieldsForRecord;

    const missingFields = [];
    for (const field of requiredFields) {
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

  const showStartWarningToast = useCallback((message) => {
    toast.error(message, {
      id: 'start-warning-toast',
      duration: 6000,
    });
  }, []);

  const executeCommand = useCallback(
    async (commandName, commandString) => {
      try {
        const result = await sendRecordCommand(commandString);
        if (result && result.success === false) {
          const message = `Command failed: ${result.message || 'Unknown error'}`;
          if (commandString === 'start_record') {
            showStartWarningToast(message);
          } else {
            toast.error(message);
          }
        } else if (result && result.success === true) {
          toast.success(`${commandName} executed successfully`);
        } else {
          toast.error(`${commandName} completed with uncertain status`);
        }
        return result;
      } catch (error) {
        let errorMessage = error.message || error.toString();
        if (
          errorMessage.includes('ROS connection failed') ||
          errorMessage.includes('WebSocket')
        ) {
          const message = `ROS connection failed: rosbridge server is not running (${rosHost})`;
          if (commandString === 'start_record') {
            showStartWarningToast(message);
          } else {
            toast.error(message);
          }
        } else if (errorMessage.includes('timeout')) {
          const message = `Command timeout [${commandName}]: Server did not respond`;
          if (commandString === 'start_record') {
            showStartWarningToast(message);
          } else {
            toast.error(message);
          }
        } else {
          const message = `Command failed [${commandName}]: ${errorMessage}`;
          if (commandString === 'start_record') {
            showStartWarningToast(message);
          } else {
            toast.error(message);
          }
        }
        return null;
      }
    },
    [sendRecordCommand, rosHost, showStartWarningToast]
  );

  const handleStart = useCallback(async () => {
    const validation = validateTaskInfo();
    if (!validation.isValid) {
      toast.error(`Missing required fields: ${validation.missingFields.join(', ')}`);
      return;
    }
    await executeCommand('Start', 'start_record');
  }, [executeCommand, validateTaskInfo]);

  const handleFinish = useCallback(async () => {
    await executeCommand('Finish', 'finish');
  }, [executeCommand]);

  const handleCancel = useCallback(async () => {
    await executeCommand('Cancel', 'cancel');
  }, [executeCommand]);

  const startEnabled = !isBusy;
  const finishEnabled = isRecording;
  const cancelEnabled = isRecording;

  const handleKeyAction = useCallback(
    (e) => {
      if (e.key === ' ' || e.key === 'Spacebar' || e.code === 'Space') {
        if (startEnabled) return 'Record';
      }
      if (
        (e.ctrlKey || e.metaKey) &&
        e.shiftKey &&
        (e.key === 'x' || e.key === 'X')
      ) {
        if (finishEnabled) return 'Save';
      }
      if (e.key === 'Escape') {
        if (cancelEnabled) return 'Discard';
      }
      return null;
    },
    [startEnabled, finishEnabled, cancelEnabled]
  );

  useEffect(() => {
    const isInputFocused = () => {
      const el = document.activeElement;
      if (!el) return false;
      const tag = el.tagName.toLowerCase();
      return tag === 'input' || tag === 'textarea' || tag === 'select' || el.contentEditable === 'true';
    };

    const handleKeyDown = (e) => {
      if (e.repeat || isInputFocused()) return;
      const action = handleKeyAction(e);
      if (action) setPressed(action);
    };

    const handleKeyUp = (e) => {
      setPressed(null);
      if (isInputFocused()) return;
      const action = handleKeyAction(e);
      if (action === 'Record') handleStart();
      else if (action === 'Save') handleFinish();
      else if (action === 'Discard') handleCancel();
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [handleKeyAction, handleStart, handleFinish, handleCancel]);

  const classBody = clsx(
    'bg-white/90',
    'backdrop-blur-sm',
    'rounded-full',
    'px-3',
    'py-1',
    'flex',
    'flex-row',
    'items-center',
    'gap-1.5',
    'shadow-md',
    'border',
    'border-gray-100'
  );

  const classBtn = (label, isDisabled) =>
    clsx(
      'h-full',
      'rounded-lg',
      'border-none',
      'cursor-pointer',
      'px-2.5',
      'flex',
      'items-center',
      'justify-center',
      'gap-1',
      'bg-gray-100',
      'transition-all',
      'duration-150',
      'font-semibold',
      'text-lg',
      'shrink-0',
      {
        'bg-gray-400': pressed === label && !isDisabled,
        'bg-gray-200': hovered === label && pressed !== label && !isDisabled,
        'opacity-30 cursor-not-allowed bg-gray-50': isDisabled,
      }
    );

  const controlButtons = [
    {
      label: 'Record',
      icon: MdFiberManualRecord,
      color: '#d32f2f',
      enabled: startEnabled,
      handler: handleStart,
      description: 'Start recording',
      shortcut: 'Space',
    },
    {
      label: 'Save',
      icon: MdSave,
      color: '#388e3c',
      enabled: finishEnabled,
      handler: handleFinish,
      description: 'Save recording',
      shortcut: 'Ctrl+Shift+X',
    },
    {
      label: 'Discard',
      icon: MdClose,
      color: '#757575',
      enabled: cancelEnabled,
      handler: handleCancel,
      description: 'Discard current recording',
      shortcut: 'Escape',
    },
  ];

  return (
    <div className={classBody}>
      <span className="text-lg font-semibold text-gray-500 whitespace-nowrap px-1 shrink-0">Record</span>
      <div className="w-px h-2/3 bg-gray-300 shrink-0"></div>
      {controlButtons.map(({ label, icon: Icon, color, enabled, handler, description, shortcut }) => {
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
            className="relative h-full shrink-0"
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
                style={{ fontSize: '1.1rem' }}
                color={isDisabled ? '#9ca3af' : color}
              />
              {label}
            </button>
          </Tooltip>
        );
      })}

      <div className="w-px h-2/3 bg-gray-400 shrink-0"></div>

      <div className="flex items-center gap-1 shrink-0 px-1">
        <span className="text-gray-600 font-semibold text-lg whitespace-nowrap">
          {phaseGuideMessages[phase] || ''}
        </span>
        {isBusy && (
          <span className="font-mono text-blue-500 text-sm">
            {spinnerFrames[spinnerIndex]}
          </span>
        )}
      </div>

      {isRecording && (
        <>
          <div className="w-px h-2/3 bg-gray-400 shrink-0"></div>
          <div className="flex items-center gap-1 shrink-0 px-1">
            <span className="text-gray-500 text-lg font-medium">
              {recordStatus.proceedTime}s
            </span>
          </div>
        </>
      )}

      <div className="w-px h-2/3 bg-gray-400 shrink-0"></div>

      <div className="flex items-center gap-1 shrink-0 px-1">
        <span className="text-gray-500 text-lg font-medium">EP</span>
        <span className="bg-gray-100 rounded px-1.5 py-0.5 text-lg font-bold">
          {recordStatus.currentEpisodeNumber}
        </span>
      </div>
    </div>
  );
}
