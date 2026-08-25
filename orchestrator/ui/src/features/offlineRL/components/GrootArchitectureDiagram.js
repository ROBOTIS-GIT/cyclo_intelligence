// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import clsx from 'clsx';
import { MdAcUnit, MdArrowDownward, MdLock, MdWhatshot } from 'react-icons/md';

export const GROOT_N17_DEFAULT_GROUPS = [
  {
    id: 'visual_encoder',
    label: 'Visual encoder',
    detail: 'Cosmos Qwen3-VL vision tower',
    trainable: false,
  },
  {
    id: 'language_model',
    label: 'Language model',
    detail: 'Cosmos Qwen3-VL language layers',
    trainable: false,
  },
  {
    id: 'vl_adapter',
    label: 'VL adapter',
    detail: 'VL norm + self-attention',
    trainable: true,
  },
  {
    id: 'state_action_projectors',
    label: 'State/action projectors',
    detail: 'Encoders + positional/action decoder',
    trainable: true,
  },
  {
    id: 'flow_matching_dit',
    label: 'Flow-matching DiT',
    detail: 'Denoising transformer → action chunk',
    trainable: true,
  },
];

function LockedArchitectureNode({ group }) {
  return (
    <button
      type="button"
      aria-pressed={group.trainable}
      aria-label={`${group.label}: ${group.trainable ? 'Trainable' : 'Frozen'}; locked`}
      disabled
      className={clsx(
        'flex h-full min-h-[58px] w-full min-w-0 cursor-not-allowed flex-col justify-center rounded-lg border px-3 py-2.5 text-left opacity-75',
        group.trainable
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
            group.trainable
              ? 'bg-[#69866f] text-white'
              : 'border border-[#d3ccc0] bg-[#e7e2d9] text-[#80776a]'
          )}
        >
          {group.trainable ? (
            <><MdWhatshot size={10} aria-hidden="true" /> Fire · Trainable</>
          ) : (
            <><MdAcUnit size={10} aria-hidden="true" /> Frozen</>
          )}
        </span>
      </span>
      <span className="mt-1.5 block truncate text-[10px] opacity-75">{group.detail}</span>
    </button>
  );
}

/** Presentation-only view of the official GR00T N1.7 fine-tuning defaults. */
export default function GrootArchitectureDiagram() {
  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-xl border border-[#e0d9ce] bg-[#f5f1e9] p-3"
      data-testid="groot-architecture-diagram"
    >
      <div className="mb-3 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">GR00T N1.7 architecture</div>
          <div className="text-[10px] text-[#8d8579]">Official fine-tuning defaults · view only</div>
        </div>
        <span className="flex items-center gap-1 rounded-full bg-[#ece8df] px-2.5 py-1 text-[10px] font-semibold text-[#766f64]">
          <MdLock size={10} aria-hidden="true" /> Locked
        </span>
      </div>

      <div className="grid min-h-0 flex-1 grid-rows-[minmax(58px,1fr)_24px_minmax(58px,1fr)_24px_minmax(58px,1fr)]">
        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <LockedArchitectureNode group={GROOT_N17_DEFAULT_GROUPS[0]} />
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">+</span>
          <LockedArchitectureNode group={GROOT_N17_DEFAULT_GROUPS[1]} />
        </div>

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <LockedArchitectureNode group={GROOT_N17_DEFAULT_GROUPS[2]} />
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">+</span>
          <LockedArchitectureNode group={GROOT_N17_DEFAULT_GROUPS[3]} />
        </div>

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <LockedArchitectureNode group={GROOT_N17_DEFAULT_GROUPS[4]} />
      </div>
    </div>
  );
}
