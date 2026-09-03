// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import {
  MdArrowBack,
  MdDns,
  MdModelTraining,
} from 'react-icons/md';
import PanelToggleGlyph from './PanelToggleGlyph';
import RobotLabIcon from './RobotLabIcon';

export const RL_FRAMEWORK_SECTIONS = [
  { id: 'environment', label: 'Environment', icon: RobotLabIcon },
  { id: 'replay', label: 'Replay Buffer', icon: MdDns },
  { id: 'training', label: 'Training', icon: MdModelTraining },
];

export default function RLFrameworkRail({
  activeSection = 'environment',
  openSections = null,
  collapsed = false,
  onBack = () => {},
  onSectionChange = () => {},
  onToggleCollapsed = () => {},
  sectionControls = {},
}) {
  const widthClassName = collapsed ? 'w-[68px]' : 'w-[220px]';
  const toggleLabel = collapsed
    ? 'Expand Playground menu'
    : 'Collapse Playground menu';

  return (
    <aside
      aria-label="Playground navigation"
      data-collapsed={String(collapsed)}
      className={[
        widthClassName,
        'flex h-full shrink-0 flex-col border-r border-[#ded8cc]',
        'bg-[#f3f0e8] px-2.5 py-3 text-[#3f3b33]',
        'transition-[width] duration-200 ease-out',
      ].join(' ')}
    >
      <div
        data-testid="rl-framework-rail-header"
        className={[
          'flex items-center gap-1.5 px-0.5',
          collapsed ? 'flex-col' : 'justify-between',
        ].join(' ')}
      >
        <button
          type="button"
          aria-label="Back to main navigation"
          title="Back to main navigation"
          onClick={onBack}
          className={[
            'grid h-9 w-9 shrink-0 place-items-center rounded-lg',
            'text-[#5f5b53] transition-colors hover:bg-[#e3e1db]',
            'hover:text-[#302d27] focus:outline-none focus:ring-2',
            'focus:ring-[#879b83] focus:ring-offset-2 focus:ring-offset-[#f3f0e8]',
          ].join(' ')}
        >
          <MdArrowBack size={19} aria-hidden="true" />
        </button>

        {!collapsed && (
          <div
            data-testid="rl-framework-title"
            className="flex h-9 min-w-0 flex-1 items-center gap-2 px-2.5 text-[#302d27]"
          >
            <RobotLabIcon size={18} className="shrink-0" aria-hidden="true" />
            <span className="truncate text-xs font-semibold tracking-[0.04em]">
              PLAYGROUND
            </span>
          </div>
        )}

        <button
          type="button"
          aria-label={toggleLabel}
          aria-expanded={!collapsed}
          title={toggleLabel}
          onClick={onToggleCollapsed}
          className={[
            'grid h-9 w-9 shrink-0 place-items-center rounded-lg',
            'border border-transparent bg-transparent text-[#6b655b]',
            'transition-colors hover:bg-[#e3e1db] hover:text-[#343129]',
            'focus:outline-none focus:ring-2 focus:ring-[#879b83]',
            'focus:ring-offset-2 focus:ring-offset-[#f3f0e8]',
          ].join(' ')}
        >
          <PanelToggleGlyph
            glyphTestId="rl-framework-toggle-glyph"
            accentTestId="rl-framework-toggle-accent"
          />
        </button>
      </div>

      <nav aria-label="Playground sections" className="mt-5 flex flex-col gap-1.5">
        {RL_FRAMEWORK_SECTIONS.map(({ id, label, icon: Icon }) => {
          const controlledPanelId = sectionControls[id];
          const isExpanded = controlledPanelId
            ? Boolean(openSections?.[id] ?? (activeSection === id))
            : undefined;
          const isActive = controlledPanelId
            ? isExpanded
            : activeSection === id;

          return (
            <button
              key={id}
              id={`rl-framework-section-${id}`}
              type="button"
              aria-label={label}
              aria-current={activeSection === id ? 'page' : undefined}
              aria-controls={controlledPanelId}
              aria-expanded={isExpanded}
              title={collapsed ? label : undefined}
              onClick={() => onSectionChange(id)}
              className={[
                'flex h-11 w-full items-center rounded-lg border text-left',
                'transition-colors focus:outline-none focus:ring-2',
                'focus:ring-[#879b83] focus:ring-offset-2',
                'focus:ring-offset-[#f3f0e8]',
                collapsed ? 'justify-center px-0' : 'gap-3 px-3',
                isActive
                  ? 'border-[#c8d3c5] bg-[#e2e9df] text-[#3f5c46] shadow-sm'
                  : [
                    'border-transparent text-[#6d675d]',
                    'hover:border-[#ded8cc] hover:bg-[#ebe7de] hover:text-[#343129]',
                  ].join(' '),
              ].join(' ')}
            >
              <Icon size={19} className="shrink-0" aria-hidden="true" />
              {!collapsed && (
                <span className="truncate text-sm font-medium">{label}</span>
              )}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
