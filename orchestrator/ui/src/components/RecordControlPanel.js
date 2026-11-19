// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");

import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import clsx from 'clsx';
import { RecordPhase } from '../constants/taskPhases';

const phaseGuideMessages = {
  [RecordPhase.READY]: 'Ready',
  [RecordPhase.RECORDING]: 'Recording',
  [RecordPhase.SAVING]: 'Saving...',
  [RecordPhase.PAUSED]: 'Paused',
};

const spinnerFrames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];

export default function RecordControlPanel() {
  const recordStatus = useSelector((state) => state.tasks.recordStatus);
  const [spinnerIndex, setSpinnerIndex] = useState(0);

  const phase = recordStatus.recordPhase;
  const isRecording = phase === RecordPhase.RECORDING;
  const isSaving = phase === RecordPhase.SAVING;
  const isBusy = isRecording || isSaving;
  const subtaskCount = Number(recordStatus.subtaskCount || 0);
  const currentSubtaskIndex = Number(recordStatus.currentSubtaskIndex || 0);
  const hasSubtasks = subtaskCount > 0;

  useEffect(() => {
    setSpinnerIndex((prev) => (prev + 1) % spinnerFrames.length);
  }, [recordStatus]);

  const classBody = clsx(
    'bg-white/90',
    'backdrop-blur-sm',
    'rounded-md',
    'px-3',
    'py-1',
    'flex',
    'flex-row',
    'flex-wrap',
    'min-w-0',
    'max-w-full',
    'items-center',
    'gap-1.5',
    'border',
    'border-gray-100'
  );

  return (
    <div className={classBody}>
      <span className="text-sm font-semibold text-gray-700 whitespace-nowrap px-1 shrink-0">
        Record
      </span>
      <div className="w-px h-2/3 bg-gray-300 shrink-0"></div>

      <div className="flex items-center gap-1 shrink-0 px-1">
        <span className="text-gray-600 text-sm whitespace-nowrap">
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
            <span className="text-gray-500 text-sm font-medium">
              {recordStatus.proceedTime}s
            </span>
          </div>
        </>
      )}

      <div role="group" aria-label="Episode" className="flex items-stretch shrink-0 rounded border border-blue-400 bg-blue-50 text-blue-700 dark:text-blue-200 overflow-hidden">
        <span className="px-2 py-1 text-sm font-semibold self-center">Episode</span>
        <span className="min-w-[3rem] border-l border-blue-400 px-2 py-1 text-base leading-5 text-center font-bold tabular-nums">
          {recordStatus.currentEpisodeNumber}
        </span>
      </div>

      {hasSubtasks && (
          <div role="group" aria-label="Current subtask" className="flex items-stretch min-w-0 max-w-[min(260px,100%)] rounded border border-gray-400 bg-gray-50 overflow-hidden">
            <span className="shrink-0 bg-gray-100 px-2 py-1 text-gray-800 text-sm font-semibold whitespace-nowrap">
              ST {Math.min(currentSubtaskIndex + 1, subtaskCount)}/{subtaskCount}
            </span>
            {recordStatus.currentSubtaskInstruction && (
              <span title={recordStatus.currentSubtaskInstruction} className="min-w-0 border-l border-gray-400 px-2 py-1 text-sm text-gray-700 truncate">
                {recordStatus.currentSubtaskInstruction}
              </span>
            )}
          </div>
      )}
    </div>
  );
}
