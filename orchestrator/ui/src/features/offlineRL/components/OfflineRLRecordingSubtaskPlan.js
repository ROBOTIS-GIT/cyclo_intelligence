// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import {
  MdAdd,
  MdArrowForward,
  MdCheckCircle,
  MdRemove,
  MdRestartAlt,
} from 'react-icons/md';

export const MAX_OFFLINE_RL_SUBTASKS = 50;

const boundedCount = (value) => {
  const count = Number.parseInt(value, 10);
  if (!Number.isFinite(count)) return 0;
  return Math.max(0, Math.min(MAX_OFFLINE_RL_SUBTASKS, count));
};

export default function OfflineRLRecordingSubtaskPlan({
  count = 0,
  instructions = [],
  disabled = false,
  onCountChange = () => {},
  onInstructionChange = () => {},
  onReset = () => {},
  activeIndex = 0,
  savedIndices = [],
  recordingActive = false,
  advancing = false,
  onSaveAndNext = null,
}) {
  const plannedCount = boundedCount(count);
  const saved = new Set(savedIndices);
  const hasNext = activeIndex >= 0 && activeIndex < plannedCount - 1;

  const applyCount = (value) => {
    if (disabled) return;
    onCountChange(boundedCount(value));
  };

  const updateInstruction = (index, value) => {
    if (disabled) return;
    onInstructionChange(index, value);
  };

  const resetPlan = () => {
    if (disabled) return;
    onReset();
  };

  return (
    <section
      className="rounded-xl border border-[#e2dcd1] bg-[#fbfaf6] p-3"
      aria-label="Subtask plan"
      data-testid="offline-rl-recording-subtask-plan"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#81796d]">
            Subtask
          </div>
          <p className="mt-0.5 text-[9px] text-[#8f877b]">
            Plan ordered instructions before recording.
          </p>
        </div>

        <div className="flex items-center gap-1.5">
          <label
            htmlFor="offline-rl-subtask-count"
            className="text-[10px] font-semibold text-[#6e675c]"
          >
            Count
          </label>
          <div className="flex h-8 overflow-hidden rounded-lg border border-[#d9d2c5] bg-white">
            <button
              type="button"
              onClick={() => applyCount(plannedCount - 1)}
              disabled={disabled || plannedCount <= 0}
              aria-label="Decrease subtask count"
              className="grid w-8 place-items-center border-r border-[#e2dcd1] text-[#625b50] hover:bg-[#f1ede4] disabled:cursor-not-allowed disabled:opacity-35"
            >
              <MdRemove size={15} />
            </button>
            <input
              id="offline-rl-subtask-count"
              type="number"
              min="0"
              max={MAX_OFFLINE_RL_SUBTASKS}
              value={plannedCount}
              onChange={(event) => applyCount(event.target.value)}
              disabled={disabled}
              className="w-12 appearance-none bg-white px-1 text-center text-[10px] font-semibold text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
            />
            <button
              type="button"
              onClick={() => applyCount(plannedCount + 1)}
              disabled={disabled || plannedCount >= MAX_OFFLINE_RL_SUBTASKS}
              aria-label="Increase subtask count"
              className="grid w-8 place-items-center border-l border-[#e2dcd1] text-[#625b50] hover:bg-[#f1ede4] disabled:cursor-not-allowed disabled:opacity-35"
            >
              <MdAdd size={15} />
            </button>
          </div>
          <button
            type="button"
            onClick={resetPlan}
            disabled={disabled || plannedCount === 0}
            aria-label="Reset subtask plan"
            className="grid h-8 w-8 place-items-center rounded-lg border border-[#d9d2c5] bg-[#f5f2eb] text-[#625b50] hover:bg-[#ebe6dd] disabled:cursor-not-allowed disabled:opacity-35"
            title="Reset subtask plan"
          >
            <MdRestartAlt size={16} />
          </button>
        </div>
      </div>

      {plannedCount > 0 ? (
        <div className="mt-2.5 max-h-28 space-y-1.5 overflow-y-auto pr-1">
          {Array.from({ length: plannedCount }, (_, index) => (
            <label
              key={index}
              className={`grid grid-cols-[30px_minmax(0,1fr)] items-center gap-2 rounded-lg px-1 py-0.5 ${
                saved.has(index)
                  ? 'bg-[#e4ebe3]'
                  : recordingActive && index === activeIndex
                    ? 'bg-[#f1e2df]'
                    : ''
              }`}
            >
              <span className="text-center font-mono text-[9px] font-semibold text-[#8f877b]">
                {saved.has(index) ? (
                  <MdCheckCircle
                    className="mx-auto text-[#607563]"
                    size={14}
                    aria-label={`Subtask ${index + 1} saved`}
                  />
                ) : (
                  `#${String(index + 1).padStart(2, '0')}`
                )}
              </span>
              <input
                type="text"
                value={instructions[index] || ''}
                onChange={(event) => updateInstruction(index, event.target.value)}
                disabled={disabled}
                aria-label={`Subtask ${index + 1} instruction`}
                placeholder={`Subtask ${index + 1} instruction`}
                className="h-8 min-w-0 rounded-lg border border-[#d9d2c5] bg-white px-2.5 text-[10px] font-medium text-[#403b34] outline-none focus:border-[#9a9182] focus:ring-1 focus:ring-[#c7beb0] disabled:cursor-not-allowed disabled:bg-[#ece8df]"
              />
            </label>
          ))}
        </div>
      ) : (
        <div className="mt-2.5 rounded-lg border border-dashed border-[#ded8cc] bg-[#f6f3ec] px-3 py-2 text-center text-[9px] text-[#8f877b]">
          Set Count to add ordered subtask instructions.
        </div>
      )}

      {recordingActive && plannedCount > 0 && (
        <div className="mt-2 flex items-center justify-between gap-2 rounded-lg border border-[#e2dcd1] bg-[#f6f3ec] px-2.5 py-2">
          <span className="min-w-0 truncate text-[9px] font-semibold text-[#6e675c]">
            Subtask {activeIndex + 1} / {plannedCount}
          </span>
          {hasNext ? (
            <button
              type="button"
              onClick={onSaveAndNext}
              disabled={advancing || typeof onSaveAndNext !== 'function'}
              className="flex h-8 shrink-0 items-center justify-center gap-1 rounded-lg bg-[#69866f] px-3 text-[9px] font-semibold text-white hover:bg-[#5f7965] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {advancing ? 'Saving…' : 'Save & Next'}
              {!advancing && <MdArrowForward size={14} aria-hidden="true" />}
            </button>
          ) : (
            <span className="text-[9px] font-medium text-[#8f877b]">
              Finish with Success or Fail
            </span>
          )}
        </div>
      )}
    </section>
  );
}
