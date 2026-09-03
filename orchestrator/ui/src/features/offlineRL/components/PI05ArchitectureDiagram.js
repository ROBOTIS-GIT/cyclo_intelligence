// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import clsx from 'clsx';
import {
  MdAcUnit,
  MdArrowDownward,
  MdCameraAlt,
  MdLock,
  MdMemory,
  MdWhatshot,
} from 'react-icons/md';

const CORE_NODES = {
  visionLanguageEncoder: {
    eyebrow: 'Encoder',
    label: 'Vision-language encoder',
    detail: 'SigLIP + PaliGemma → multimodal tokens',
    className: 'border-[#c7bde6] bg-[#f0edfa] text-[#514672]',
    eyebrowClassName: 'text-[#756b94]',
    trainable: false,
  },
  actionConditioning: {
    eyebrow: 'Condition encoder',
    label: 'Action conditioning',
    detail: 'Robot state + noisy action + time',
    className: 'border-[#dfc6a4] bg-[#fbf2e5] text-[#6c4f2e]',
    eyebrowClassName: 'text-[#9a7650]',
    trainable: true,
  },
  actionModule: {
    eyebrow: 'Action model',
    label: 'Action Module',
    detail: 'Flow-matching velocity prediction',
    className: 'border-[#acc2ae] bg-[#edf4ec] text-[#38533d]',
    eyebrowClassName: 'text-[#667d69]',
    trainable: true,
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

function LockedArchitectureNode({ node, testId }) {
  return (
    <button
      type="button"
      aria-pressed={node.trainable}
      aria-label={`${node.label}: ${node.trainable ? 'Trainable' : 'Frozen'}; locked`}
      disabled
      className={clsx(
        'flex h-full min-h-[58px] w-full min-w-0 flex-col justify-center rounded-xl border px-3 py-2.5 text-left shadow-[0_1px_2px_rgba(56,50,42,0.04)]',
        node.trainable
          ? [node.className, 'cursor-default opacity-100']
          : 'cursor-not-allowed border-[#d9d2c5] bg-[#f1eee7] text-[#7d7569] opacity-75'
      )}
      data-testid={testId}
      data-trainable-group={node.id}
    >
      <span
        className={clsx(
          'mb-1 text-[8px] font-bold uppercase tracking-[0.12em]',
          node.trainable ? node.eyebrowClassName : 'text-[#999084]'
        )}
      >
        {node.eyebrow}
      </span>
      <span className="flex items-center justify-between gap-2">
        <span className="truncate text-[12px] font-semibold">{node.label}</span>
        <span
          className={clsx(
            'flex shrink-0 items-center gap-0.5 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em]',
            node.trainable
              ? 'bg-[#69866f] text-white'
              : 'border border-[#d3ccc0] bg-white/60 text-[#80776a]'
          )}
        >
          {node.trainable ? (
            <><MdWhatshot size={10} aria-hidden="true" /> Fire · Trainable</>
          ) : (
            <><MdAcUnit size={10} aria-hidden="true" /> Frozen</>
          )}
        </span>
      </span>
      <span
        className={clsx(
          'mt-1 block truncate text-[10px]',
          node.trainable ? node.eyebrowClassName : 'text-[#958c80]'
        )}
      >
        {node.detail}
      </span>
    </button>
  );
}

const FlowArrow = () => (
  <div className="flex items-center justify-center text-[#aaa295]">
    <MdArrowDownward size={15} aria-hidden="true" />
  </div>
);

/**
 * Presentation-only Pi0.5 policy topology.
 *
 * The fine-tuning view shows the intended frozen VLM / trainable action-side
 * boundary as locked status, not interactive controls. RLT freezes the complete
 * base VLA while its external adapter is trained.
 */
export default function PI05ArchitectureDiagram({
  mode = 'finetune',
  allFrozen = false,
}) {
  const freezeBasePolicy = allFrozen || mode === 'rlt';
  const nodes = Object.fromEntries(Object.entries(CORE_NODES).map(([key, node]) => [
    key,
    {
      ...node,
      id: key,
      trainable: freezeBasePolicy ? false : node.trainable,
    },
  ]));

  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5 shadow-[0_8px_24px_rgba(61,55,46,0.06)]"
      data-testid="pi05-architecture-diagram"
      data-architecture-mode={freezeBasePolicy ? 'all-frozen' : 'finetune'}
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">Pi0.5 Policy</div>
          <div className="text-[9px] text-[#8d8579]">
            {freezeBasePolicy
              ? 'RLT base policy · all modules frozen'
              : 'Fine-tuning boundary · view only'}
          </div>
        </div>
        <span className="flex items-center gap-1 rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
          <MdLock size={10} aria-hidden="true" /> Locked policy
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
          <LockedArchitectureNode
            node={nodes.visionLanguageEncoder}
            testId="pi05-vlm-encoder-node"
          />
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">
            +
          </span>
          <LockedArchitectureNode
            node={nodes.actionConditioning}
            testId="pi05-conditioning-node"
          />
        </div>

        <FlowArrow />

        <LockedArchitectureNode node={nodes.actionModule} testId="pi05-action-module-node" />

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
