// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import { MdArrowDownward, MdWhatshot } from 'react-icons/md';

const TrainableNode = ({ label, detail }) => (
  <div
    aria-label={`${label}: Fire; Trainable; fixed`}
    className="flex h-full min-h-[54px] min-w-0 flex-col justify-center rounded-lg border border-[#9faf9f] bg-[#edf3ec] px-2.5 py-2"
  >
    <div className="flex min-w-0 items-center justify-between gap-1">
      <span className="truncate text-[11px] font-semibold text-[#344a38]">{label}</span>
      <span className="flex shrink-0 items-center gap-0.5 rounded-full bg-[#69866f] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.04em] text-white">
        <MdWhatshot size={10} aria-hidden="true" /> Fire
      </span>
    </div>
    <div className="mt-1.5 truncate text-[9px] text-[#607364]">{detail}</div>
  </div>
);

const FixedNode = ({ label, detail }) => (
  <div className="flex h-full min-h-[50px] min-w-0 flex-col justify-center rounded-lg border border-[#ddd6ca] bg-white px-2.5 py-2">
    <div className="truncate text-[11px] font-semibold text-[#514b42]">{label}</div>
    <div className="mt-1 truncate text-[9px] text-[#8b8377]">{detail}</div>
  </div>
);

/** Presentation-only graph of the MultiTaskDiT Flow-SDE PPO update. */
export default function FlowSDEPPOArchitectureDiagram({ backendReady = false }) {
  return (
    <div
      className="flex min-h-0 flex-1 flex-col"
      data-testid="flow-sde-ppo-architecture-diagram"
    >
      <div className="mb-3 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">
            Flow-SDE PPO
          </div>
          <div className="text-[10px] text-[#8d8579]">
            Online action-chunk policy optimization
          </div>
        </div>
        <span className={backendReady
          ? 'rounded-full bg-[#e6ece6] px-2.5 py-1 text-[10px] font-semibold text-[#5f7664]'
          : 'rounded-full bg-[#ece8df] px-2.5 py-1 text-[10px] font-semibold text-[#81796d]'}
        >
          {backendReady ? 'Backend ready' : 'Backend pending'}
        </span>
      </div>

      <div
        className="grid min-h-0 flex-1 grid-rows-[minmax(50px,1fr)_20px_minmax(50px,1fr)_20px_minmax(54px,1fr)] gap-1"
        data-testid="flow-sde-ppo-architecture-flow"
      >
        <FixedNode
          label="Flow-SDE rollout"
          detail="Store latent chain, log-probability, reward, done"
        />
        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={15} aria-hidden="true" />
        </div>
        <FixedNode
          label="GAE advantages"
          detail="Chunk reward + frozen conditioning value"
        />
        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={15} aria-hidden="true" />
        </div>
        <div className="grid min-h-0 grid-cols-2 gap-1.5">
          <TrainableNode label="DiT actor" detail="Clipped PPO surrogate" />
          <TrainableNode label="Value head" detail="Clipped value loss" />
        </div>
      </div>
    </div>
  );
}
