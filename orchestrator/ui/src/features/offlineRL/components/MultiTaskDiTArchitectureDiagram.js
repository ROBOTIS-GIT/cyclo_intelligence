// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import {
  MdAcUnit,
  MdArrowDownward,
  MdCameraAlt,
  MdMemory,
  MdWhatshot,
} from 'react-icons/md';

const StatusBadge = ({ trainable, label = '' }) => (
  <span
    className={trainable
      ? 'flex shrink-0 items-center gap-0.5 rounded-full bg-[#69866f] px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em] text-white'
      : 'flex shrink-0 items-center gap-0.5 rounded-full border border-[#d3ccc0] bg-white/70 px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em] text-[#80776a]'}
  >
    {trainable
      ? <><MdWhatshot size={10} aria-hidden="true" /> {label || 'Fire · Trainable'}</>
      : <><MdAcUnit size={10} aria-hidden="true" /> {label || 'Frozen'}</>}
  </span>
);

const NODE_TONES = {
  encoder: {
    border: 'border-[#d9d2c5]',
    background: 'bg-[#f1eee7]',
    text: 'text-[#7d7569]',
    detail: 'text-[#958c80]',
    eyebrow: 'Observation encoder',
  },
  actor: {
    border: 'border-[#acc2ae]',
    background: 'bg-[#edf4ec]',
    text: 'text-[#38533d]',
    detail: 'text-[#667d69]',
    eyebrow: 'Action module',
  },
};

const ArchitectureNode = ({
  label,
  detail,
  tone = 'encoder',
  trainable = false,
  ariaLabel = '',
  statusLabel = '',
}) => {
  const colors = NODE_TONES[tone];

  return (
    <div
      aria-label={ariaLabel || `${label}: ${trainable ? 'Fire; Trainable' : 'Frozen'}; fixed`}
      className={`flex h-full min-h-[58px] min-w-0 flex-col justify-center rounded-xl border px-3 py-2.5 shadow-[0_1px_2px_rgba(56,50,42,0.04)] ${colors.border} ${colors.background} ${colors.text}`}
    >
      <span className={`mb-1 text-[8px] font-bold uppercase tracking-[0.12em] ${colors.detail}`}>
        {colors.eyebrow}
      </span>
      <span className="flex min-w-0 items-center justify-between gap-2">
        <span className="truncate text-[12px] font-semibold">{label}</span>
        <StatusBadge trainable={trainable} label={statusLabel} />
      </span>
      <span className={`mt-1 block truncate text-[10px] ${colors.detail}`}>
        {detail}
      </span>
    </div>
  );
};

function InputNode({ icon: Icon, label, detail }) {
  return (
    <div className="flex min-w-0 items-center gap-2.5 rounded-xl border border-[#b9d3e2] bg-[#edf6fa] px-3 py-2 text-[#34586b] shadow-[0_1px_2px_rgba(54,91,110,0.04)]">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[#d9ecf4] text-[#517b91]">
        <Icon size={15} aria-hidden="true" />
      </span>
      <span className="min-w-0">
        <span className="block truncate text-[10px] font-semibold">{label}</span>
        <span className="block truncate text-[8px] text-[#6f93a5]">{detail}</span>
      </span>
    </div>
  );
}

const FlowArrow = () => (
  <div className="flex items-center justify-center text-[#aaa295]">
    <MdArrowDownward size={15} aria-hidden="true" />
  </div>
);

/** Presentation of the exact Cyclo MultiTaskDiT Flow-Matching training boundary. */
export default function MultiTaskDiTArchitectureDiagram({ criticOnly = false }) {
  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5 shadow-[0_8px_24px_rgba(61,55,46,0.06)]"
      data-testid="multi-task-dit-architecture-diagram"
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">
            Diffusion Transformer Policy
          </div>
          <div className="text-[9px] text-[#8d8579]">
            MultiTaskDiT · Flow-Matching action policy
          </div>
        </div>
        <span className="rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
          {criticOnly ? 'Policy frozen' : 'Flow policy'}
        </span>
      </div>

      <div
        className="grid min-h-0 flex-1 grid-rows-[auto_18px_minmax(58px,1fr)_18px_minmax(58px,1fr)_18px_auto]"
        data-testid="multi-task-dit-architecture-flow"
      >
        <div className="grid grid-cols-2 gap-1.5" data-testid="multi-task-dit-policy-inputs">
          <InputNode
            icon={MdCameraAlt}
            label="3 camera images + task"
            detail="Head · Left wrist · Right wrist · language"
          />
          <InputNode
            icon={MdMemory}
            label="22D robot state"
            detail="Proprioceptive observation"
          />
        </div>

        <FlowArrow />

        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <ArchitectureNode
            label="Visual + task encoder"
            detail="Images and task tokens → features"
          />
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">+</span>
          <ArchitectureNode
            label="Robot-state encoder"
            detail="22D state → proprioceptive features"
          />
        </div>

        <FlowArrow />

        <ArchitectureNode
          label="Action Module"
          detail={criticOnly
            ? 'Frozen while the offline value critic is trained'
            : 'Frozen conditioning → trainable Flow-Matching DiT'}
          tone={criticOnly ? 'encoder' : 'actor'}
          trainable={!criticOnly}
          statusLabel={criticOnly ? 'Frozen' : 'DiT · Trainable'}
          ariaLabel={criticOnly
            ? 'Action Module: Frozen during value critic warm-up; fixed'
            : 'Action Module: conditioning Frozen; Flow-Matching DiT Fire; Trainable; fixed'}
        />

        <FlowArrow />

        <div
          className="flex items-center justify-between gap-3 rounded-xl border border-[#9faacf] bg-[#e9edfa] px-3 py-2 text-[#36456f] shadow-[0_2px_8px_rgba(54,69,111,0.08)]"
          data-testid="multi-task-dit-policy-output"
        >
          <span className="min-w-0">
            <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-[#69769e]">Policy output</span>
            <span className="block truncate text-[11px] font-semibold">Action chunk</span>
          </span>
          <span className="shrink-0 rounded-full bg-[#485984] px-2.5 py-1 text-[9px] font-semibold text-white">
            16 × 22D
          </span>
        </div>
      </div>
    </div>
  );
}
