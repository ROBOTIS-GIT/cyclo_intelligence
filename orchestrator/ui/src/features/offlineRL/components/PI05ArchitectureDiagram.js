// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import {
  MdArrowDownward,
  MdCameraAlt,
  MdMemory,
} from 'react-icons/md';

const CORE_NODES = {
  visionLanguageEncoder: {
    eyebrow: 'Encoder',
    label: 'Vision-language encoder',
    detail: 'SigLIP + PaliGemma → multimodal tokens',
    className: 'border-[#c7bde6] bg-[#f0edfa] text-[#514672]',
    eyebrowClassName: 'text-[#756b94]',
  },
  actionConditioning: {
    eyebrow: 'Condition encoder',
    label: 'Action conditioning',
    detail: 'Robot state + noisy action + time',
    className: 'border-[#dfc6a4] bg-[#fbf2e5] text-[#6c4f2e]',
    eyebrowClassName: 'text-[#9a7650]',
  },
  actionModule: {
    eyebrow: 'Action model',
    label: 'Action Module',
    detail: 'Flow-matching velocity prediction',
    className: 'border-[#acc2ae] bg-[#edf4ec] text-[#38533d]',
    eyebrowClassName: 'text-[#667d69]',
  },
};

function InputNode({ icon: Icon, label, detail }) {
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-xl border border-[#b9d3e2] bg-[#edf6fa] px-2.5 py-2 text-[#34586b] shadow-[0_1px_2px_rgba(54,91,110,0.04)]">
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

function CoreNode({ node, testId }) {
  return (
    <div
      role="group"
      aria-label={node.label}
      className={`flex h-full min-h-[58px] w-full min-w-0 flex-col justify-center rounded-xl border px-3 py-2.5 text-left shadow-[0_1px_2px_rgba(56,50,42,0.04)] ${node.className}`}
      data-testid={testId}
    >
      <span className={`mb-1 text-[8px] font-bold uppercase tracking-[0.12em] ${node.eyebrowClassName}`}>
        {node.eyebrow}
      </span>
      <span className="truncate text-[12px] font-semibold">{node.label}</span>
      <span className={`mt-1 block truncate text-[10px] ${node.eyebrowClassName}`}>
        {node.detail}
      </span>
    </div>
  );
}

const FlowArrow = () => (
  <div className="flex items-center justify-center text-[#aaa295]">
    <MdArrowDownward size={15} aria-hidden="true" />
  </div>
);

/**
 * Static Pi0.5 policy topology.
 *
 * Trainability is deliberately not editable here. The current training request
 * has no Pi0.5 module-freezing contract, so displaying local Frozen / Fire
 * controls would imply that an unsubmitted choice affects the backend.
 */
export default function PI05ArchitectureDiagram() {
  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5 shadow-[0_8px_24px_rgba(61,55,46,0.06)]"
      data-testid="pi05-architecture-diagram"
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">Pi0.5 Policy</div>
          <div className="text-[9px] text-[#8d8579]">Reference topology · training controls pending integration</div>
        </div>
        <span className="rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
          VLA policy
        </span>
      </div>

      <div
        className="grid min-h-0 flex-1 grid-rows-[auto_18px_minmax(58px,1fr)_18px_minmax(58px,1fr)_18px_auto]"
        data-testid="pi05-architecture-flow"
      >
        <div className="grid grid-cols-2 gap-1.5" data-testid="pi05-policy-inputs">
          <InputNode
            icon={MdCameraAlt}
            label="Camera images + task instruction"
            detail="Multi-view RGB + language prompt"
          />
          <InputNode icon={MdMemory} label="Robot state" detail="Proprioception" />
        </div>

        <FlowArrow />

        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <CoreNode
            node={CORE_NODES.visionLanguageEncoder}
            testId="pi05-vlm-encoder-node"
          />
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">
            +
          </span>
          <CoreNode
            node={CORE_NODES.actionConditioning}
            testId="pi05-conditioning-node"
          />
        </div>

        <FlowArrow />

        <CoreNode node={CORE_NODES.actionModule} testId="pi05-action-module-node" />

        <FlowArrow />

        <div
          className="flex items-center justify-between gap-3 rounded-xl border border-[#9faacf] bg-[#e9edfa] px-3 py-2 text-[#36456f] shadow-[0_2px_8px_rgba(54,69,111,0.08)]"
          data-testid="pi05-policy-output"
        >
          <span className="min-w-0">
            <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-[#69769e]">Policy output</span>
            <span className="block truncate text-[11px] font-semibold">Action chunk</span>
          </span>
          <span className="shrink-0 rounded-full bg-[#485984] px-2.5 py-1 text-[9px] font-semibold text-white">
            Flow matched
          </span>
        </div>
      </div>
    </div>
  );
}
