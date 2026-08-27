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
  MdMemory,
  MdWhatshot,
} from 'react-icons/md';

export const ACT_TRAINABLE_GROUPS = [
  {
    id: 'visual_backbone',
    label: 'Visual backbone',
    detail: '3 cameras → ResNet features',
  },
  {
    id: 'cvae_encoder',
    label: 'CVAE encoder',
    detail: 'State + target actions → latent',
  },
  {
    id: 'transformer_encoder',
    label: 'Transformer encoder',
    detail: 'Visual, state, and latent tokens',
  },
  {
    id: 'action_decoder',
    label: 'Action decoder',
    detail: 'Decoder queries → action chunk',
  },
];

export const DEFAULT_ACT_TRAINABLE_GROUPS = ACT_TRAINABLE_GROUPS.map(
  ({ id }) => id
);

const ACT_ACTION_MODULE_GROUPS = [
  'transformer_encoder',
  'action_decoder',
];

const ACT_ACTION_MODULE = {
  id: 'action_module',
  label: 'Action Module',
  detail: 'Transformer + decoder → action chunk',
};

const GROUP_TONES = {
  visual_backbone: {
    border: 'border-[#c7bde6]',
    background: 'bg-[#f0edfa]',
    text: 'text-[#514672]',
    detail: 'text-[#756b94]',
    eyebrow: 'Vision encoder',
  },
  cvae_encoder: {
    border: 'border-[#acc2ae]',
    background: 'bg-[#edf4ec]',
    text: 'text-[#38533d]',
    detail: 'text-[#667d69]',
    eyebrow: 'Latent encoder',
  },
  transformer_encoder: {
    border: 'border-[#acc2ae]',
    background: 'bg-[#edf4ec]',
    text: 'text-[#38533d]',
    detail: 'text-[#667d69]',
    eyebrow: 'Actor',
  },
  action_decoder: {
    border: 'border-[#acc2ae]',
    background: 'bg-[#edf4ec]',
    text: 'text-[#38533d]',
    detail: 'text-[#667d69]',
    eyebrow: 'Actor head',
  },
  action_module: {
    border: 'border-[#acc2ae]',
    background: 'bg-[#edf4ec]',
    text: 'text-[#38533d]',
    detail: 'text-[#667d69]',
    eyebrow: 'Actor',
  },
};

function ArchitectureNode({
  group,
  trainable,
  mixed = false,
  disabled,
  locked = false,
  onToggle,
}) {
  const status = mixed ? 'Mixed' : (trainable ? 'Trainable' : 'Frozen');
  const nextAction = locked
    ? 'locked for pure TD3'
    : (trainable && !mixed ? 'freeze' : 'make trainable');
  const tone = GROUP_TONES[group.id];

  return (
    <button
      type="button"
      aria-pressed={mixed ? 'mixed' : trainable}
      aria-label={`${group.label}: ${status}; ${nextAction}`}
      disabled={disabled || locked}
      onClick={onToggle}
      title={locked ? 'Not used by the pure TD3 actor objective' : undefined}
      className={clsx(
        'group flex h-full min-h-[58px] w-full min-w-0 flex-col justify-center rounded-xl border px-3 py-2.5 text-left shadow-[0_1px_2px_rgba(56,50,42,0.04)] transition-all',
        'hover:-translate-y-px hover:shadow-[0_4px_12px_rgba(56,50,42,0.08)] focus:outline-none focus:ring-2 focus:ring-[#9eaa9f] focus:ring-offset-1',
        (trainable || mixed)
          ? [tone.border, tone.background, tone.text]
          : 'border-[#d9d2c5] bg-[#f1eee7] text-[#7d7569]',
        (disabled || locked) && 'cursor-not-allowed opacity-60'
      )}
      data-trainable-group={group.id}
    >
      <span
        className={clsx(
          'mb-1 text-[8px] font-bold uppercase tracking-[0.12em]',
          (trainable || mixed) ? tone.detail : 'text-[#999084]'
        )}
      >
        {tone.eyebrow}
      </span>
      <span className="flex items-center justify-between gap-2">
        <span className="truncate text-[12px] font-semibold">{group.label}</span>
        <span
          className={clsx(
            'flex shrink-0 items-center gap-0.5 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em]',
            trainable
              ? 'bg-[#69866f] text-white'
              : mixed
                ? 'border border-[#c9b986] bg-[#f8f0d7] text-[#79662f]'
              : 'border border-[#d3ccc0] bg-white/60 text-[#80776a]'
          )}
        >
          {locked ? (
            <><MdAcUnit size={10} aria-hidden="true" /> Frozen · TD3</>
          ) : mixed ? (
            <><MdWhatshot size={10} aria-hidden="true" /> Mixed</>
          ) : trainable ? (
            <><MdWhatshot size={10} aria-hidden="true" /> Fire · Trainable</>
          ) : (
            <><MdAcUnit size={10} aria-hidden="true" /> Frozen</>
          )}
        </span>
      </span>
      <span
        className={clsx(
          'mt-1 block truncate text-[10px]',
          (trainable || mixed) ? tone.detail : 'text-[#958c80]'
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
 * Controlled, presentation-only ACT trainability graph.
 *
 * `trainableGroups` is the complete source of truth. A click reports the next
 * ordered group list through `onChange`; persistence and training submission
 * remain the responsibility of the parent controller.
 */
export default function ACTArchitectureDiagram({
  trainableGroups,
  onChange,
  disabled = false,
  lockedGroups = [],
}) {
  const selected = new Set(trainableGroups || []);
  const locked = new Set(lockedGroups || []);

  const toggleGroups = (groupIds) => {
    if (groupIds.some((groupId) => locked.has(groupId))) return;
    const nextSelected = new Set(selected);
    const allSelected = groupIds.every((groupId) => nextSelected.has(groupId));
    groupIds.forEach((groupId) => {
      if (allSelected) nextSelected.delete(groupId);
      else nextSelected.add(groupId);
    });
    onChange(
      ACT_TRAINABLE_GROUPS
        .map(({ id }) => id)
        .filter((id) => nextSelected.has(id))
    );
  };

  const actionModuleSelectedCount = ACT_ACTION_MODULE_GROUPS.filter(
    (groupId) => selected.has(groupId)
  ).length;
  const actionModuleTrainable = (
    actionModuleSelectedCount === ACT_ACTION_MODULE_GROUPS.length
  );
  const actionModuleMixed = (
    actionModuleSelectedCount > 0 && !actionModuleTrainable
  );

  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5 shadow-[0_8px_24px_rgba(61,55,46,0.06)]"
      data-testid="act-architecture-diagram"
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">ACT Policy</div>
          <div className="text-[9px] text-[#8d8579]">Select a module to switch Frozen / Trainable</div>
        </div>
        <span className="rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
          Actor policy
        </span>
      </div>

      <div
        className="grid min-h-0 flex-1 grid-rows-[auto_18px_minmax(58px,1fr)_18px_minmax(58px,1fr)_18px_auto]"
        data-testid="act-architecture-flow"
      >
        <div className="grid grid-cols-2 gap-1.5" data-testid="act-policy-inputs">
          <InputNode icon={MdCameraAlt} label="3 camera images" detail="Head · Left wrist · Right wrist" />
          <InputNode icon={MdMemory} label="Robot state" detail="Proprioceptive observation" />
        </div>

        <FlowArrow />

        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <ArchitectureNode
            group={ACT_TRAINABLE_GROUPS[0]}
            trainable={selected.has(ACT_TRAINABLE_GROUPS[0].id)}
            disabled={disabled}
            locked={locked.has(ACT_TRAINABLE_GROUPS[0].id)}
            onToggle={() => toggleGroups([ACT_TRAINABLE_GROUPS[0].id])}
          />
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">+</span>
          <ArchitectureNode
            group={ACT_TRAINABLE_GROUPS[1]}
            trainable={selected.has(ACT_TRAINABLE_GROUPS[1].id)}
            disabled={disabled}
            locked={locked.has(ACT_TRAINABLE_GROUPS[1].id)}
            onToggle={() => toggleGroups([ACT_TRAINABLE_GROUPS[1].id])}
          />
        </div>

        <FlowArrow />

        <ArchitectureNode
          group={ACT_ACTION_MODULE}
          trainable={actionModuleTrainable}
          mixed={actionModuleMixed}
          disabled={disabled}
          locked={ACT_ACTION_MODULE_GROUPS.some((groupId) => locked.has(groupId))}
          onToggle={() => toggleGroups(ACT_ACTION_MODULE_GROUPS)}
        />

        <FlowArrow />

        <div
          className="flex items-center justify-between gap-3 rounded-xl border border-[#9faacf] bg-[#e9edfa] px-3 py-2 text-[#36456f] shadow-[0_2px_8px_rgba(54,69,111,0.08)]"
          data-testid="act-policy-output"
        >
          <span className="min-w-0">
            <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-[#69769e]">Policy output</span>
            <span className="block truncate text-[11px] font-semibold">Action chunk</span>
          </span>
          <span className="shrink-0 rounded-full bg-[#485984] px-2.5 py-1 text-[9px] font-semibold text-white">
            30 steps
          </span>
        </div>
      </div>
    </div>
  );
}
