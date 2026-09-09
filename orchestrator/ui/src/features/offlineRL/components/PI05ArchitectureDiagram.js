// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import clsx from 'clsx';
import {
  MdCameraAlt,
  MdLock,
  MdMemory,
} from 'react-icons/md';
import {
  PolicyFlowArrow as FlowArrow,
  PolicyInputNode as InputNode,
  PolicyOutputNode,
  PolicyTrainabilityBadge,
} from './PolicyArchitecturePrimitives';

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

function LockedArchitectureNode({ node, testId }) {
  return (
    <button
      type="button"
      aria-pressed={node.trainable}
      aria-label={`${node.label}: ${node.trainable ? 'Trainable' : 'Frozen'}; locked`}
      disabled
      title={node.detail}
      className={clsx(
        'flex h-full min-h-[58px] w-full min-w-0 flex-col justify-center rounded-xl border px-3 py-2.5 text-left',
        node.trainable
          ? [node.className, 'cursor-default opacity-100']
          : 'cursor-not-allowed border-[#d9d2c5] bg-[#f1eee7] text-[#7d7569] opacity-75'
      )}
      data-testid={testId}
      data-trainable-group={node.id}
    >
      <span className="flex flex-wrap items-center justify-between gap-2">
        <span className="truncate text-[14px] font-semibold">{node.label}</span>
        <PolicyTrainabilityBadge trainable={node.trainable} />
      </span>
    </button>
  );
}

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
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5"
      data-testid="pi05-architecture-diagram"
      data-architecture-mode={freezeBasePolicy ? 'all-frozen' : 'finetune'}
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[14px] font-semibold text-[#39352e]">Pi0.5 Policy</div>
        </div>
        <span className="flex items-center gap-1 rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
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
            compact
          />
          <InputNode icon={MdMemory} label="Robot state" detail="Proprioception" compact />
        </div>

        <FlowArrow />

        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <LockedArchitectureNode
            node={nodes.visionLanguageEncoder}
            testId="pi05-vlm-encoder-node"
          />
          <span className="flex items-center justify-center text-[12px] font-semibold uppercase text-[#aaa295]">
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

        <PolicyOutputNode testId="pi05-policy-output" badge="Flow matched" />
      </div>
    </div>
  );
}
