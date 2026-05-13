// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import React from 'react';
import { Handle, Position } from '@xyflow/react';
import clsx from 'clsx';

const TYPE_ICONS = {
  Sequence: '→',
  Loop: '↻',
  Fallback: '?',
  Parallel: '⇉',
};

export default function BTControlNode({ id, data }) {
  const icon = TYPE_ICONS[data.nodeType] || '□';
  const isActive = data.isActive;
  const isSelected = data.isSelected;
  const collapsed = !!data.collapsed;
  const childCount = data.childCount ?? 0;
  const hasChildren = childCount > 0;

  return (
    <div
      className={clsx(
        'relative px-4 py-3 rounded-lg border-2 min-w-[160px] text-center shadow-sm cursor-pointer',
        // Selection ring is independent of active state. Active state on
        // a Control node bubbles up from any active descendant — we keep
        // the blue palette to preserve the Control identity and only add
        // animate-pulse so the user can see "something inside here is
        // running" (especially useful when the Control is collapsed).
        isSelected
          ? 'border-blue-600 bg-blue-100 ring-2 ring-blue-300'
          : 'border-blue-500 bg-blue-50',
        isActive && 'animate-pulse',
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-blue-500" />
      <div className="text-xs text-blue-600 font-semibold mb-1">
        {icon} {data.nodeType}
      </div>
      <div className="text-sm font-medium text-gray-800 truncate">
        {data.label}
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-blue-500" />

      {/* Collapse / expand toggle. stopPropagation so the click doesn't
          double as a node-select. Disabled when this Control node has no
          children to hide. */}
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          if (!hasChildren) return;
          data.onToggleCollapse?.(id);
        }}
        disabled={!hasChildren}
        title={
          !hasChildren
            ? 'No children to collapse'
            : collapsed
              ? 'Expand'
              : 'Collapse'
        }
        className={clsx(
          'absolute -right-2 -top-2 w-5 h-5 rounded-full border bg-white shadow text-xs leading-none flex items-center justify-center select-none',
          hasChildren
            ? 'border-blue-400 text-blue-600 hover:bg-blue-50 cursor-pointer'
            : 'border-gray-200 text-gray-300 cursor-not-allowed'
        )}
      >
        {collapsed ? '+' : '−'}
      </button>

      {collapsed && hasChildren && (
        <div className="absolute -bottom-2 right-1 text-[10px] text-gray-600 bg-white border border-gray-200 rounded px-1 leading-tight">
          {childCount} hidden
        </div>
      )}
    </div>
  );
}
