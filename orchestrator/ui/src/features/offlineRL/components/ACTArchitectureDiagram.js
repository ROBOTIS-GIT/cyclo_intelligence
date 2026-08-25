// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import clsx from 'clsx';
import { MdAcUnit, MdArrowDownward, MdWhatshot } from 'react-icons/md';

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

function ArchitectureNode({ group, trainable, disabled, onToggle }) {
  const nextAction = trainable ? 'freeze' : 'make trainable';

  return (
    <button
      type="button"
      aria-pressed={trainable}
      aria-label={`${group.label}: ${trainable ? 'Trainable' : 'Frozen'}; ${nextAction}`}
      disabled={disabled}
      onClick={() => onToggle(group.id)}
      className={clsx(
        'flex h-full min-h-[58px] w-full min-w-0 flex-col justify-center rounded-lg border px-3 py-2.5 text-left transition-colors',
        'focus:outline-none focus:ring-2 focus:ring-[#9eaa9f] focus:ring-offset-1',
        trainable
          ? 'border-[#9faf9f] bg-[#edf3ec] text-[#344a38]'
          : 'border-[#d9d2c5] bg-[#f1eee7] text-[#7d7569]',
        disabled && 'cursor-not-allowed opacity-60'
      )}
      data-trainable-group={group.id}
    >
      <span className="flex items-center justify-between gap-2">
        <span className="truncate text-[12px] font-semibold">{group.label}</span>
        <span
          className={clsx(
            'flex shrink-0 items-center gap-0.5 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em]',
            trainable
              ? 'bg-[#69866f] text-white'
              : 'border border-[#d3ccc0] bg-[#e7e2d9] text-[#80776a]'
          )}
        >
          {trainable ? (
            <><MdWhatshot size={10} aria-hidden="true" /> Fire · Trainable</>
          ) : (
            <><MdAcUnit size={10} aria-hidden="true" /> Frozen</>
          )}
        </span>
      </span>
      <span className="mt-1.5 block truncate text-[10px] opacity-75">
        {group.detail}
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
}) {
  const selected = new Set(trainableGroups || []);

  const toggleGroup = (groupId) => {
    const nextSelected = new Set(selected);
    if (nextSelected.has(groupId)) nextSelected.delete(groupId);
    else nextSelected.add(groupId);
    onChange(
      ACT_TRAINABLE_GROUPS
        .map(({ id }) => id)
        .filter((id) => nextSelected.has(id))
    );
  };

  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-xl border border-[#e0d9ce] bg-[#f5f1e9] p-3"
      data-testid="act-architecture-diagram"
    >
      <div className="mb-3 flex shrink-0 items-center justify-between gap-2">
        <div>
          <div className="text-[13px] font-semibold text-[#39352e]">ACT architecture</div>
          <div className="text-[10px] text-[#8d8579]">Click a network block to freeze or train it</div>
        </div>
        <span className="rounded-full bg-[#e8ebef] px-2.5 py-1 text-[10px] font-semibold text-[#65707e]">
          Policy
        </span>
      </div>

      <div
        className="grid min-h-0 flex-1 grid-rows-[minmax(58px,1fr)_24px_minmax(58px,1fr)_24px_minmax(58px,1fr)]"
        data-testid="act-architecture-flow"
      >
        <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)] items-stretch gap-1.5">
          <ArchitectureNode
            group={ACT_TRAINABLE_GROUPS[0]}
            trainable={selected.has(ACT_TRAINABLE_GROUPS[0].id)}
            disabled={disabled}
            onToggle={toggleGroup}
          />
          <span className="flex items-center justify-center text-[10px] font-semibold uppercase text-[#aaa295]">+</span>
          <ArchitectureNode
            group={ACT_TRAINABLE_GROUPS[1]}
            trainable={selected.has(ACT_TRAINABLE_GROUPS[1].id)}
            disabled={disabled}
            onToggle={toggleGroup}
          />
        </div>

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <ArchitectureNode
          group={ACT_TRAINABLE_GROUPS[2]}
          trainable={selected.has(ACT_TRAINABLE_GROUPS[2].id)}
          disabled={disabled}
          onToggle={toggleGroup}
        />

        <div className="flex items-center justify-center text-[#aaa295]">
          <MdArrowDownward size={16} aria-hidden="true" />
        </div>

        <ArchitectureNode
          group={ACT_TRAINABLE_GROUPS[3]}
          trainable={selected.has(ACT_TRAINABLE_GROUPS[3].id)}
          disabled={disabled}
          onToggle={toggleGroup}
        />
      </div>
    </div>
  );
}
