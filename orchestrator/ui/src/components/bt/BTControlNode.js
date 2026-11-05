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
        'relative px-4 py-3 rounded-xl border-2 min-w-[160px] text-center shadow-sm cursor-pointer',
        'border-[#1c1a17] bg-white',
        isSelected && 'ring-2 ring-[#1c1a17]/20',
        isActive && 'animate-pulse',
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-[#1c1a17]" />
      <div className="text-xs text-[#6f6a5d] font-semibold mb-1 font-mono">
        {icon} {data.nodeType}
      </div>
      <div className="text-sm font-semibold text-[#1c1a17] truncate">
        {data.label}
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-[#1c1a17]" />

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
            ? 'border-[#dcd7ca] text-[#6f6a5d] hover:bg-[#f4e5dc] hover:text-[#c96442] cursor-pointer'
            : 'border-[#e7e3d9] text-[#c3bcac] cursor-not-allowed'
        )}
      >
        {collapsed ? '+' : '−'}
      </button>

      {collapsed && hasChildren && (
        <div className="absolute -bottom-2 right-1 text-[10px] text-[#6f6a5d] bg-white border border-[#e7e3d9] rounded px-1 leading-tight">
          {childCount} hidden
        </div>
      )}
    </div>
  );
}
