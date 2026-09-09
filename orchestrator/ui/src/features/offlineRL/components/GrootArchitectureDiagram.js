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
    detail: 'State encoder + action projector',
    trainable: true,
  },
  {
    id: 'flow_matching_dit',
    label: 'Flow-matching DiT',
    detail: 'Conditioned denoising transformer',
    trainable: true,
  },
];

const GROUP_TONES = {
  visual_encoder: {
    border: 'border-[#c7bde6]',
    background: 'bg-[#f0edfa]',
    text: 'text-[#514672]',
    detail: 'text-[#756b94]',
    eyebrow: 'Vision backbone',
  },
  language_model: {
    border: 'border-[#c7bde6]',
    background: 'bg-[#f0edfa]',
    text: 'text-[#514672]',
    detail: 'text-[#756b94]',
    eyebrow: 'Language backbone',
  },
  vl_adapter: {
    border: 'border-[#acc2ae]',
    background: 'bg-[#edf4ec]',
    text: 'text-[#38533d]',
    detail: 'text-[#667d69]',
    eyebrow: 'Multimodal adapter',
  },
  state_action_projectors: {
    border: 'border-[#acc2ae]',
    background: 'bg-[#edf4ec]',
    text: 'text-[#38533d]',
    detail: 'text-[#667d69]',
    eyebrow: 'Robot interface',
  },
  flow_matching_dit: {
    border: 'border-[#d8c4a5]',
    background: 'bg-[#f8f1e6]',
    text: 'text-[#654f32]',
    detail: 'text-[#8b7659]',
    eyebrow: 'Action head',
  },
  action_module: {
    border: 'border-[#acc2ae]',
    background: 'bg-[#edf4ec]',
    text: 'text-[#38533d]',
    detail: 'text-[#667d69]',
    eyebrow: 'Action head',
  },
};

const GROOT_ACTION_MODULE_GROUPS = [
  'vl_adapter',
  'state_action_projectors',
  'flow_matching_dit',
];

function LockedArchitectureNode({ group, memberGroups = [group.id] }) {
  const tone = GROUP_TONES[group.id];

  return (
    <button
      type="button"
      aria-pressed={group.trainable}
      aria-label={`${group.label}: ${group.trainable ? 'Trainable' : 'Frozen'}; locked`}
      disabled
      title={group.detail}
      className={clsx(
        'flex h-full min-h-[58px] w-full min-w-0 flex-col justify-center rounded-xl border px-3 py-2.5 text-left',
        group.trainable
          ? [tone.border, tone.background, tone.text, 'cursor-default opacity-100']
          : 'cursor-not-allowed border-[#d9d2c5] bg-[#f1eee7] text-[#7d7569] opacity-75'
      )}
      data-trainable-group={group.id}
      data-member-groups={memberGroups.join(' ')}
    >
      <span className="flex flex-wrap items-center justify-between gap-2">
        <span className="truncate text-[14px] font-semibold">{group.label}</span>
        <PolicyTrainabilityBadge trainable={group.trainable} />
      </span>
    </button>
  );
}

/**
 * Presentation-only GR00T N1.7 topology.
 *
 * The default mode mirrors the current fine-tuning boundary. RLT keeps the
 * complete base VLA frozen while its external RL-token and action-MLP modules
 * are trained, so callers may select that view with `mode="rlt"` or the more
 * explicit `allFrozen` flag.
 */
export default function GrootArchitectureDiagram({
  mode = 'finetune',
  allFrozen = false,
}) {
  const freezeBasePolicy = allFrozen || mode === 'rlt';
  const groups = GROOT_N17_DEFAULT_GROUPS.map((group) => ({
    ...group,
    trainable: freezeBasePolicy ? false : group.trainable,
  }));
  const actionModuleGroups = groups.filter((group) => (
    GROOT_ACTION_MODULE_GROUPS.includes(group.id)
  ));
  const actionModule = {
    id: 'action_module',
    label: 'Action Module',
    detail: 'VL adapter + robot projectors + Flow-matching DiT',
    trainable: actionModuleGroups.every((group) => group.trainable),
  };

  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5"
      data-testid="groot-architecture-diagram"
      data-architecture-mode={freezeBasePolicy ? 'all-frozen' : 'finetune'}
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[14px] font-semibold text-[#39352e]">GR00T N1.7 Policy</div>
        </div>
        <span className="flex items-center gap-1 rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
          <MdLock size={10} aria-hidden="true" /> Locked policy
        </span>
      </div>

      <div
        className="grid min-h-0 flex-1 grid-rows-[auto_18px_minmax(58px,1fr)_18px_minmax(58px,1fr)_18px_auto]"
        data-testid="groot-architecture-flow"
      >
        <div className="grid grid-cols-2 gap-1.5" data-testid="groot-policy-inputs">
          <InputNode
            icon={MdCameraAlt}
            label="3 camera images + task instruction"
            detail="Head · left wrist · right wrist · language"
          />
          <InputNode
            icon={MdMemory}
            label="Robot state"
            detail="Proprioceptive observation"
          />
        </div>

        <FlowArrow />

        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <LockedArchitectureNode group={groups[0]} />
          <span className="flex items-center justify-center text-[12px] font-semibold uppercase text-[#aaa295]">+</span>
          <LockedArchitectureNode group={groups[1]} />
        </div>

        <FlowArrow />

        <LockedArchitectureNode
          group={actionModule}
          memberGroups={GROOT_ACTION_MODULE_GROUPS}
        />

        <FlowArrow />

        <PolicyOutputNode
          testId="groot-policy-output"
          badge="Chunked controls"
        />
      </div>
    </div>
  );
}
