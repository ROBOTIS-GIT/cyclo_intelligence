// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import clsx from 'clsx';
import {
  MdAcUnit,
  MdArrowDownward,
  MdWhatshot,
} from 'react-icons/md';

export const RLT_TRAINABLE_GROUPS = [
  {
    id: 'rl_token_encoder',
    label: 'RL Token Encoder',
    detail: 'Language features → 2,048D RL token',
  },
  {
    id: 'action_mlp',
    label: 'Action MLP',
    detail: 'RL token + reference action → action chunk',
  },
];

// PI RLT Stage 2 freezes the learned representation by default and updates
// only the lightweight action policy. Either block remains explicitly
// configurable from the diagram.
export const DEFAULT_RLT_TRAINABLE_GROUPS = ['action_mlp'];

function TrainableNode({ group, trainable, disabled, onToggle }) {
  const nextAction = trainable ? 'freeze' : 'make trainable';

  return (
    <button
      type="button"
      aria-pressed={trainable}
      aria-label={`${group.label}: ${trainable ? 'Trainable' : 'Frozen'}; ${nextAction}`}
      disabled={disabled}
      onClick={() => onToggle(group.id)}
      className={clsx(
        'flex min-h-[58px] w-full min-w-0 flex-col justify-center rounded-lg border px-3 py-2.5 text-left transition-colors',
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
            'flex shrink-0 items-center gap-0.5 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.05em]',
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

function RLTQCriticDiagram() {
  return (
    <div
      className="flex min-w-0 flex-col border-t border-[#e3ddd3] pt-3 lg:border-l lg:border-t-0 lg:pl-4 lg:pt-0"
      data-testid="rlt-q-critic-diagram"
    >
      <div className="flex shrink-0 items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-[13px] font-semibold text-[#39352e]">
            Q critic network
          </div>
          <div className="truncate text-[9px] text-[#8d8579]">
            Chunk-level value estimation · Stage 2
          </div>
        </div>
        <span className="shrink-0 rounded-full bg-[#ece9f2] px-2.5 py-1 text-[9px] font-semibold text-[#6f6780]">
          Twin Q
        </span>
      </div>

      <div
        className="mt-3 grid grid-rows-[44px_18px_auto_18px_44px] gap-1.5"
        aria-label="RLT independent twin Q critic flow"
      >
        <div className="flex min-h-[44px] items-center justify-center rounded-lg border border-[#d9d2c5] bg-white px-3 text-center text-[10px] font-semibold text-[#5c554c]">
          RL token + proprio + action chunk
        </div>

        <div className="flex items-center justify-center text-[#aaa295]" aria-hidden="true">
          <MdArrowDownward size={15} />
        </div>

        <div className="grid grid-cols-2 gap-2" aria-label="Independent twin Q critics">
          {['Q1 MLP', 'Q2 MLP'].map((label) => (
            <div
              key={label}
              className="flex min-h-[58px] min-w-0 flex-col items-center justify-center rounded-lg border border-[#afa8bd] bg-[#f2f0f6] px-2 text-center text-[#514b61]"
              aria-label={`${label}: Trainable`}
            >
              <span className="text-[11px] font-semibold">{label}</span>
              <span className="mt-1 rounded-full bg-[#746b86] px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.05em] text-white">
                Fire · Trainable
              </span>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-center text-[#aaa295]" aria-hidden="true">
          <MdArrowDownward size={15} />
        </div>

        <div className="flex min-h-[44px] items-center justify-center rounded-lg border border-[#afa8bd] bg-white px-3 text-center text-[10px] font-semibold text-[#625a72]">
          min(Q1, Q2) · Bellman target
        </div>
      </div>
    </div>
  );
}

/**
 * Presentation-only RLT Stage-2 boundary. GR00T/Pi0.5 remain frozen while the
 * lightweight actor controls stay configurable and the required twin-Q
 * training structure remains visible beside them.
 */
export default function RLTArchitectureDiagram({
  policyLabel,
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
      RLT_TRAINABLE_GROUPS
        .map(({ id }) => id)
        .filter((id) => nextSelected.has(id))
    );
  };

  return (
    <div
      className="grid min-w-0 grid-cols-1 gap-3 lg:grid-cols-2 lg:gap-4"
      data-testid="rlt-architecture-diagram"
    >
      <div
        className="flex min-w-0 flex-col"
        data-testid="rlt-action-policy-diagram"
        data-loop-policy-update-source="top-center"
      >
        <div className="flex shrink-0 items-center justify-between gap-2">
          <div className="min-w-0">
            <div className="truncate text-[13px] font-semibold text-[#39352e]">
              RLT action policy
            </div>
            <div className="truncate text-[9px] text-[#8d8579]">
              Frozen {policyLabel} backbone · configurable lightweight actor
            </div>
          </div>
          <span className="shrink-0 rounded-full bg-[#e6ece6] px-2.5 py-1 text-[9px] font-semibold text-[#5f7664]">
            RLT
          </span>
        </div>

        <div
          className="mt-3 grid grid-rows-[auto_18px_auto_18px_44px] gap-1.5"
          data-testid="rlt-architecture-flow"
        >
          <TrainableNode
            group={RLT_TRAINABLE_GROUPS[0]}
            trainable={selected.has(RLT_TRAINABLE_GROUPS[0].id)}
            disabled={disabled}
            onToggle={toggleGroup}
          />

          <div
            className="flex items-center justify-center text-[#aaa295]"
            role="img"
            aria-label="RL Token Encoder to Action MLP"
          >
            <MdArrowDownward size={15} aria-hidden="true" />
          </div>

          <TrainableNode
            group={RLT_TRAINABLE_GROUPS[1]}
            trainable={selected.has(RLT_TRAINABLE_GROUPS[1].id)}
            disabled={disabled}
            onToggle={toggleGroup}
          />

          <div
            className="flex items-center justify-center text-[#aaa295]"
            role="img"
            aria-label="Action MLP to 10 by 19 action chunk"
          >
            <MdArrowDownward size={15} aria-hidden="true" />
          </div>

          <div className="flex min-h-[44px] items-center justify-center rounded-lg border border-[#9faf9f] bg-white px-3 text-[11px] font-semibold text-[#47604c]">
            10 × 19 action chunk
          </div>
        </div>
      </div>

      <RLTQCriticDiagram />
    </div>
  );
}
