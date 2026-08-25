// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import { MdAcUnit, MdArrowDownward, MdWhatshot } from 'react-icons/md';

const StatusBadge = ({ trainable }) => (
  <span className={trainable
    ? 'flex shrink-0 items-center gap-0.5 rounded-full bg-[#69866f] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.04em] text-white'
    : 'flex shrink-0 items-center gap-0.5 rounded-full border border-[#d3ccc0] bg-[#e7e2d9] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.04em] text-[#756e63]'}
  >
    {trainable
      ? <><MdWhatshot size={10} aria-hidden="true" /> Fire · Trainable</>
      : <><MdAcUnit size={10} aria-hidden="true" /> Frozen</>}
  </span>
);

const ArchitectureNode = ({ label, detail, trainable = false }) => (
  <div
    aria-label={`${label}: ${trainable ? 'Fire; Trainable' : 'Frozen'}; fixed`}
    className={trainable
      ? 'flex h-full min-h-[58px] min-w-0 flex-col justify-center rounded-lg border border-[#9faf9f] bg-[#edf3ec] px-3 py-2.5'
      : 'flex h-full min-h-[58px] min-w-0 flex-col justify-center rounded-lg border border-[#d9d2c5] bg-[#f1eee7] px-3 py-2.5'}
  >
    <div className="flex min-w-0 items-center justify-between gap-2">
      <span className={trainable
        ? 'truncate text-[12px] font-semibold text-[#344a38]'
        : 'truncate text-[12px] font-semibold text-[#655e54]'}
      >
        {label}
      </span>
      <StatusBadge trainable={trainable} />
    </div>
    <div className={trainable
      ? 'mt-1.5 truncate text-[10px] text-[#607364]'
      : 'mt-1.5 truncate text-[10px] text-[#8b8377]'}
    >
      {detail}
    </div>
  </div>
);

/** Presentation of the exact Cyclo MultiTaskDiT Flow-Matching training boundary. */
export default function MultiTaskDiTArchitectureDiagram() {
  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-xl border border-[#e0d9ce] bg-[#f5f1e9] p-3"
      data-testid="multi-task-dit-architecture-diagram"
    >
      <div className="mb-3 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">
            Diffusion Transformer
          </div>
          <div className="text-[10px] text-[#8d8579]">
            MultiTaskDiT · Flow-Matching action policy
          </div>
        </div>
        <span className="rounded-full bg-[#e6ece6] px-2.5 py-1 text-[10px] font-semibold text-[#5f7664]">
          multi_task_dit
        </span>
      </div>

      <div
        className="grid min-h-0 flex-1 grid-rows-[minmax(58px,1fr)_22px_minmax(58px,1fr)_22px_minmax(58px,1fr)] gap-1"
        data-testid="multi-task-dit-architecture-flow"
      >
        <div className="grid min-h-0 grid-cols-2 gap-1.5">
          <ArchitectureNode
            label="Visual + language encoder"
            detail="3 cameras + task tokens"
          />
          <ArchitectureNode
            label="Robot-state encoder"
            detail="22D proprioceptive state"
          />
        </div>

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <ArchitectureNode
          label="Frozen observation conditioning"
          detail="Deterministic feature vector shared by actor and value head"
        />

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <ArchitectureNode
          label="Flow-Matching DiT action head"
          detail="Noise + time + conditioning → 16 × 22D velocity"
          trainable
        />
      </div>
    </div>
  );
}
