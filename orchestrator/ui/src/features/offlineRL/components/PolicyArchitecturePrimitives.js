// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import clsx from 'clsx';
import {
  MdAcUnit,
  MdArrowDownward,
  MdWhatshot,
} from 'react-icons/md';

/** Shared status treatment for policy modules; node behavior stays model-owned. */
export function PolicyTrainabilityBadge({
  trainable = false,
  mixed = false,
  trainableLabel = 'Fire · Trainable',
  frozenLabel = 'Frozen',
  mixedLabel = 'Mixed',
  frozenBackgroundClassName = 'bg-white/60',
}) {
  const state = mixed ? 'mixed' : (trainable ? 'trainable' : 'frozen');

  return (
    <span
      className={clsx(
        'flex shrink-0 items-center gap-0.5 rounded-full px-2 py-0.5 text-[11px] font-bold uppercase tracking-[0.06em]',
        state === 'trainable'
          ? 'bg-[#4c7055] text-white'
          : state === 'mixed'
            ? 'border border-[#c9b986] bg-[#f8f0d7] text-[#79662f]'
            : ['border border-[#d3ccc0] text-[#625a4e]', frozenBackgroundClassName]
      )}
    >
      {state === 'trainable' ? (
        <><MdWhatshot size={10} aria-hidden="true" /> {trainableLabel}</>
      ) : state === 'mixed' ? (
        <><MdWhatshot size={10} aria-hidden="true" /> {mixedLabel}</>
      ) : (
        <><MdAcUnit size={10} aria-hidden="true" /> {frozenLabel}</>
      )}
    </span>
  );
}

/** Common visual/state input card used by policy diagrams. */
export function PolicyInputNode({ icon: Icon, label, detail, compact = false }) {
  return (
    <div
      className={clsx(
        'flex min-w-0 items-center rounded-xl border border-[#b9d3e2] bg-[#edf6fa] py-2 text-[#34586b]',
        compact ? 'gap-2 px-2.5' : 'gap-2.5 px-3'
      )}
    >
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[#d9ecf4] text-[#517b91]">
        <Icon size={15} aria-hidden="true" />
      </span>
      <span className="min-w-0">
        <span className="block truncate text-[14px] font-semibold" title={`${label}: ${detail}`}>{label}</span>
      </span>
    </div>
  );
}

/** Vertical connector shared by the compact policy-flow diagrams. */
export function PolicyFlowArrow(props) {
  return (
    <div className="flex items-center justify-center text-[#aaa295]" {...props}>
      <MdArrowDownward size={15} aria-hidden="true" />
    </div>
  );
}

/** Common action-chunk output card. */
export function PolicyOutputNode({ testId, badge, label = 'Action chunk' }) {
  return (
    <div
      className="flex items-center justify-between gap-3 rounded-xl border border-[#9faacf] bg-[#e9edfa] px-3 py-2 text-[#36456f]"
      data-testid={testId}
    >
      <span className="min-w-0">
        <span className="block text-[12px] font-bold uppercase tracking-[0.08em] text-[#69769e]">
          Policy output
        </span>
        <span className="block truncate text-[12px] font-semibold">{label}</span>
      </span>
      <span className="shrink-0 rounded-full bg-[#485984] px-2.5 py-1 text-[11px] font-semibold text-white">
        {badge}
      </span>
    </div>
  );
}
