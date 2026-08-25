// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useMemo, useState } from 'react';
import clsx from 'clsx';
import { MdAcUnit, MdArrowDownward, MdWhatshot } from 'react-icons/md';

export const PI05_FINETUNE_MODES = [
  {
    id: 'full_finetune',
    label: 'Full fine-tune',
    trainableGroups: ['vision_encoder', 'paligemma_vlm', 'action_projections', 'action_expert'],
  },
  {
    id: 'freeze_vision',
    label: 'Frozen vision',
    trainableGroups: ['paligemma_vlm', 'action_projections', 'action_expert'],
  },
  {
    id: 'expert_only',
    label: 'Expert only',
    trainableGroups: ['action_projections', 'action_expert'],
  },
];

export const DEFAULT_PI05_FINETUNE_MODE = PI05_FINETUNE_MODES[0].id;

const PI05_GROUPS = [
  {
    id: 'vision_encoder',
    label: 'SigLIP vision encoder',
    detail: 'Camera images → visual tokens',
  },
  {
    id: 'paligemma_vlm',
    label: 'PaliGemma VLM',
    detail: 'Language + state prefix tokens',
  },
  {
    id: 'action_projections',
    label: 'Action/time projections',
    detail: 'Noisy action + time ↔ velocity output',
  },
  {
    id: 'action_expert',
    label: 'Gemma action expert',
    detail: 'AdaRMS flow matching → action chunk',
  },
];

export const getPI05TrainableGroups = (modeId) => (
  PI05_FINETUNE_MODES.find(({ id }) => id === modeId)?.trainableGroups ||
  PI05_FINETUNE_MODES[0].trainableGroups
);

function ArchitectureNode({ group, trainable }) {
  return (
    <div
      role="group"
      aria-label={`${group.label}: ${trainable ? 'Trainable' : 'Frozen'}`}
      className={clsx(
        'flex h-full min-h-[58px] w-full min-w-0 flex-col justify-center rounded-lg border px-3 py-2.5 text-left',
        trainable
          ? 'border-[#9faf9f] bg-[#edf3ec] text-[#344a38]'
          : 'border-[#d9d2c5] bg-[#f1eee7] text-[#7d7569]'
      )}
      data-trainable-group={group.id}
    >
      <span className="flex items-center justify-between gap-2">
        <span className="truncate text-[12px] font-semibold">{group.label}</span>
        <span
          className={clsx(
            'flex shrink-0 items-center gap-0.5 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em]',
            trainable
              ? 'bg-[#69866f] text-white'
              : 'border border-[#d3ccc0] bg-[#e7e2d9] text-[#80776a]'
          )}
        >
          {trainable ? (
            <><MdWhatshot size={10} aria-hidden="true" /> Fire · Trainable</>
          ) : (
            <><MdAcUnit size={10} aria-hidden="true" /> Frozen</>
          )}
        </span>
      </span>
      <span className="mt-1.5 block truncate text-[10px] opacity-75">{group.detail}</span>
    </div>
  );
}

/**
 * Preview of the only three freeze modes supported by PI05Config.
 * The mode is intentionally local: no unsupported training request is emitted.
 */
export default function PI05ArchitectureDiagram({ disabled = false }) {
  const [mode, setMode] = useState(DEFAULT_PI05_FINETUNE_MODE);
  const trainableGroups = useMemo(() => new Set(getPI05TrainableGroups(mode)), [mode]);

  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-xl border border-[#e0d9ce] bg-[#f5f1e9] p-3"
      data-testid="pi05-architecture-diagram"
    >
      <div className="mb-3 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">Pi0.5 architecture</div>
          <div className="text-[10px] text-[#8d8579]">Supported fine-tune modes · preview only</div>
        </div>
        <span className="rounded-full bg-[#e8ebef] px-2.5 py-1 text-[10px] font-semibold text-[#65707e]">
          Policy
        </span>
      </div>

      <div className="mb-3 flex shrink-0 flex-wrap gap-1.5" role="group" aria-label="Pi0.5 fine-tune mode">
        {PI05_FINETUNE_MODES.map(({ id, label }) => {
          const selected = mode === id;
          return (
            <button
              key={id}
              type="button"
              aria-pressed={selected}
              disabled={disabled}
              onClick={() => setMode(id)}
              className={clsx(
                'h-7 rounded-md border px-2.5 text-[9px] font-semibold transition-colors',
                selected
                  ? 'border-[#69866f] bg-[#69866f] text-white'
                  : 'border-[#d6cfc3] bg-white text-[#746d62]',
                disabled && 'cursor-not-allowed opacity-60'
              )}
            >
              {label}
            </button>
          );
        })}
      </div>

      <div className="grid min-h-0 flex-1 grid-rows-[minmax(58px,1fr)_24px_minmax(58px,1fr)_24px_minmax(58px,1fr)]">
        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <ArchitectureNode
            group={PI05_GROUPS[0]}
            trainable={trainableGroups.has(PI05_GROUPS[0].id)}
          />
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">+</span>
          <ArchitectureNode
            group={PI05_GROUPS[1]}
            trainable={trainableGroups.has(PI05_GROUPS[1].id)}
          />
        </div>

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <ArchitectureNode
          group={PI05_GROUPS[2]}
          trainable={trainableGroups.has(PI05_GROUPS[2].id)}
        />

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <ArchitectureNode
          group={PI05_GROUPS[3]}
          trainable={trainableGroups.has(PI05_GROUPS[3].id)}
        />
      </div>
    </div>
  );
}
