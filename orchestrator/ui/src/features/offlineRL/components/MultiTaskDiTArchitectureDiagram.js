// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import {
  MdCameraAlt,
  MdMemory,
} from 'react-icons/md';
import {
  PolicyFlowArrow as FlowArrow,
  PolicyInputNode as InputNode,
  PolicyOutputNode,
  PolicyTrainabilityBadge,
} from './PolicyArchitecturePrimitives';

const NODE_TONES = {
  encoder: {
    border: 'border-[#d9d2c5]',
    background: 'bg-[#f1eee7]',
    text: 'text-[#7d7569]',
    detail: 'text-[#6e665a]',
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
      title={detail}
      className={`flex h-full min-h-[58px] min-w-0 flex-col justify-center rounded-xl border px-3 py-2.5 ${colors.border} ${colors.background} ${colors.text}`}
    >
      <span className="flex min-w-0 flex-wrap items-center justify-between gap-2">
        <span className="truncate text-[14px] font-semibold">{label}</span>
        <PolicyTrainabilityBadge
          trainable={trainable}
          trainableLabel={statusLabel || undefined}
          frozenLabel={statusLabel || undefined}
          frozenBackgroundClassName="bg-white/70"
        />
      </span>
    </div>
  );
};

/** Presentation of the exact Cyclo MultiTaskDiT Flow-Matching training boundary. */
export default function MultiTaskDiTArchitectureDiagram({ criticOnly = false }) {
  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5"
      data-testid="multi-task-dit-architecture-diagram"
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[14px] font-semibold text-[#39352e]">
            Diffusion Transformer Policy
          </div>
        </div>
        <span className="rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
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
          <span className="flex items-center justify-center text-[12px] font-semibold uppercase text-[#aaa295]">+</span>
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

        <PolicyOutputNode
          testId="multi-task-dit-policy-output"
          badge="16 × 22D"
        />
      </div>
    </div>
  );
}
