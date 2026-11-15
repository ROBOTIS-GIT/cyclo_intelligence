// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import clsx from 'clsx';
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
      title={locked ? 'Not used by the pure TD3 actor objective' : group.detail}
      className={clsx(
        'group flex h-full min-h-[58px] w-full min-w-0 flex-col justify-center rounded-xl border px-3 py-2.5 text-left transition-all',
        'hover:border-[#9eaa9f] focus:outline-none focus:ring-2 focus:ring-[#9eaa9f] focus:ring-offset-1',
        (trainable || mixed)
          ? [tone.border, tone.background, tone.text]
          : 'border-[#d9d2c5] bg-[#f1eee7] text-[#7d7569]',
        (disabled || locked) && 'cursor-not-allowed opacity-60'
      )}
      data-trainable-group={group.id}
    >
      <span className="flex flex-wrap items-center justify-between gap-2">
        <span className="truncate text-[14px] font-semibold">{group.label}</span>
        <PolicyTrainabilityBadge
          trainable={trainable && !locked}
          mixed={mixed && !locked}
          frozenLabel={locked ? 'Frozen · TD3' : 'Frozen'}
        />
      </span>
    </button>
  );
}

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
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#e0d9ce] bg-white p-3.5"
      data-testid="act-architecture-diagram"
    >
      <div className="mb-2.5 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[14px] font-semibold text-[#39352e]">ACT Policy</div>
        </div>
        <span className="rounded-full border border-[#d7ddea] bg-[#f2f4fa] px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.08em] text-[#5c6684]">
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
          <span className="flex items-center justify-center text-[12px] font-semibold uppercase text-[#aaa295]">+</span>
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

        <PolicyOutputNode testId="act-policy-output" badge="30 steps" />
      </div>
    </div>
  );
}
