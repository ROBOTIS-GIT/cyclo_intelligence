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
    border: 'border-[#d8c4a5]',
    background: 'bg-[#f8f1e6]',
    text: 'text-[#654f32]',
    detail: 'text-[#8b7659]',
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
      className={clsx(
        'flex h-full min-h-[58px] w-full min-w-0 cursor-not-allowed flex-col justify-center rounded-xl border px-3 py-2.5 text-left opacity-75 shadow-[0_1px_2px_rgba(56,50,42,0.04)]',
        group.trainable
          ? [tone.border, tone.background, tone.text]
          : 'border-[#d9d2c5] bg-[#f1eee7] text-[#7d7569]'
      )}
      data-trainable-group={group.id}
      data-member-groups={memberGroups.join(' ')}
    >
      <span
        className={clsx(
          'mb-1 text-[8px] font-bold uppercase tracking-[0.12em]',
          group.trainable ? tone.detail : 'text-[#999084]'
        )}
      >
        {tone.eyebrow}
      </span>
      <span className="flex items-center justify-between gap-2">
        <span className="truncate text-[12px] font-semibold">{group.label}</span>
        <span
          className={clsx(
            'flex shrink-0 items-center gap-0.5 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em]',
            group.trainable
              ? 'bg-[#69866f] text-white'
              : 'border border-[#d3ccc0] bg-white/60 text-[#80776a]'
          )}
        >
          {group.trainable ? (
            <><MdWhatshot size={10} aria-hidden="true" /> Fire · Trainable</>
          ) : (
            <><MdAcUnit size={10} aria-hidden="true" /> Frozen</>
          )}
        </span>
      </span>
      <span
        className={clsx(
          'mt-1 block truncate text-[10px]',
          group.trainable ? tone.detail : 'text-[#958c80]'
        )}
      >
        {group.detail}
      </span>
    </button>
  );
}

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
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5 shadow-[0_8px_24px_rgba(61,55,46,0.06)]"
      data-testid="groot-architecture-diagram"
      data-architecture-mode={freezeBasePolicy ? 'all-frozen' : 'finetune'}
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">GR00T N1.7 Policy</div>
          <div className="text-[9px] text-[#8d8579]">
            {freezeBasePolicy
              ? 'RLT base policy · all modules frozen'
              : 'Official fine-tuning defaults · view only'}
          </div>
        </div>
        <span className="flex items-center gap-1 rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
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
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">+</span>
          <LockedArchitectureNode group={groups[1]} />
        </div>

        <FlowArrow />

        <LockedArchitectureNode
          group={actionModule}
          memberGroups={GROOT_ACTION_MODULE_GROUPS}
        />

        <FlowArrow />

        <div
          className="flex items-center justify-between gap-3 rounded-xl border border-[#9faacf] bg-[#e9edfa] px-3 py-2 text-[#36456f] shadow-[0_2px_8px_rgba(54,69,111,0.08)]"
          data-testid="groot-policy-output"
        >
          <span className="min-w-0">
            <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-[#69769e]">Policy output</span>
            <span className="block truncate text-[11px] font-semibold">Action chunk</span>
          </span>
          <span className="shrink-0 rounded-full bg-[#485984] px-2.5 py-1 text-[9px] font-semibold text-white">
            Chunked controls
          </span>
        </div>
      </div>
    </div>
  );
}
