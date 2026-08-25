// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import { MdAcUnit, MdArrowDownward, MdWhatshot } from 'react-icons/md';

const TrainableBadge = () => (
  <span className="flex shrink-0 items-center gap-0.5 rounded-full bg-[#69866f] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.04em] text-white">
    <MdWhatshot size={10} aria-hidden="true" /> Fire · Trainable
  </span>
);

const FrozenPolyakBadge = () => (
  <span className="flex shrink-0 items-center gap-0.5 rounded-full border border-[#d3ccc0] bg-[#e7e2d9] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.04em] text-[#756e63]">
    <MdAcUnit size={10} aria-hidden="true" /> Frozen · Polyak
  </span>
);

const InputNode = ({ label, detail }) => (
  <div className="flex h-full min-h-[50px] min-w-0 flex-col justify-center rounded-lg border border-[#ddd6ca] bg-white px-2.5 py-2">
    <div className="truncate text-[11px] font-semibold text-[#514b42]">{label}</div>
    <div className="mt-1 truncate text-[9px] text-[#8b8377]">{detail}</div>
  </div>
);

const CriticNode = ({ label }) => (
  <div
    aria-label={`${label}: Fire; Trainable; fixed`}
    className="flex h-full min-h-[56px] min-w-0 flex-col justify-center rounded-lg border border-[#9faf9f] bg-[#edf3ec] px-2.5 py-2"
  >
    <div className="flex min-w-0 items-center justify-between gap-1">
      <span className="truncate text-[11px] font-semibold text-[#344a38]">{label}</span>
      <TrainableBadge />
    </div>
    <div className="mt-1.5 truncate text-[9px] text-[#607364]">
      Obs encoder · Chunk encoder · Q MLP
    </div>
  </div>
);

const TargetNode = ({ label, detail }) => (
  <div
    aria-label={`${label}: Frozen; Polyak target; fixed`}
    className="flex h-full min-h-[56px] min-w-0 flex-col justify-center rounded-lg border border-[#d9d2c5] bg-[#f1eee7] px-2.5 py-2"
  >
    <div className="flex min-w-0 items-center justify-between gap-1">
      <span className="truncate text-[11px] font-semibold text-[#655e54]">{label}</span>
      <FrozenPolyakBadge />
    </div>
    <div className="mt-1.5 truncate text-[9px] text-[#8b8377]">{detail}</div>
  </div>
);

/**
 * Presentation-only graph of the ACT executed-prefix TD3 contract.
 *
 * Critic and target trainability are deliberately fixed: only ACT policy
 * blocks are user-configurable in the neighboring policy graph.
 */
export default function TD3ArchitectureDiagram() {
  return (
    <div
      className="flex min-h-0 flex-1 flex-col"
      data-testid="td3-architecture-diagram"
    >
      <div className="mb-3 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">TD3 critic architecture</div>
          <div className="text-[10px] text-[#8d8579]">Independent chunk Q-functions · fixed contract</div>
        </div>
        <span className="rounded-full bg-[#e6ece6] px-2.5 py-1 text-[10px] font-semibold text-[#5f7664]">
          RL algorithm
        </span>
      </div>

      <div
        className="grid min-h-0 flex-1 grid-rows-[minmax(50px,1fr)_22px_minmax(56px,1fr)_auto_20px_minmax(56px,1fr)_auto] gap-1"
        data-testid="td3-architecture-flow"
      >
        <div className="grid min-h-0 grid-cols-2 gap-1.5">
          <InputNode label="Observation" detail="3 images + robot state" />
          <InputNode label="Executed action" detail="Action chunk + prefix mask" />
        </div>

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <div className="grid min-h-0 grid-cols-2 gap-1.5" aria-label="Independent twin critics">
          <CriticNode label="Q1 critic" />
          <CriticNode label="Q2 critic" />
        </div>

        <div className="rounded-lg border border-[#cdd8ce] bg-white px-2.5 py-2 text-[10px] text-[#536958]">
          <span className="font-semibold">Actor gradient</span>
          <span className="float-right font-semibold">ACT ← maximize Q1</span>
          <div className="mt-1 clear-both text-[9px] text-[#859087]">
            Delayed Q gradient alongside the ACT behavior-cloning anchor
          </div>
        </div>

        <div className="flex items-center gap-1 text-[9px] font-semibold uppercase tracking-[0.06em] text-[#999185]">
          <span className="h-px flex-1 bg-[#ddd6ca]" /> Target networks <span className="h-px flex-1 bg-[#ddd6ca]" />
        </div>

        <div className="grid min-h-0 grid-cols-2 gap-1.5">
          <TargetNode label="Target ACT" detail="Smoothed next action chunk" />
          <TargetNode label="Target Q1 / Q2" detail="Independent target critics" />
        </div>

        <div className="rounded-lg border border-[#d9d2c5] bg-white px-2.5 py-2 text-center text-[10px] font-semibold text-[#514b42]">
          min(target Q1, target Q2) → Bellman target
        </div>
      </div>
    </div>
  );
}
