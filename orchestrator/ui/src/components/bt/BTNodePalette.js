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

import { useMemo } from 'react';
import { MdRefresh } from 'react-icons/md';
import { useBTNodeCatalog } from '../../hooks/useBTNodeCatalog';

export const PALETTE_DRAG_MIME = 'application/bt-node-tag';

export default function BTNodePalette({ canUpdateCatalog = true }) {
  const { catalog, source, refreshCatalog } = useBTNodeCatalog();
  const isUpdating = source === 'loading';
  const isUpdateDisabled = !canUpdateCatalog || isUpdating;
  const updateTitle = 'Refresh available task steps';

  const grouped = useMemo(() => ({
    control: catalog.filter((n) => n.category === 'control'),
    action: catalog.filter((n) => n.category === 'action'),
  }), [catalog]);

  const handleDragStart = (event, tag) => {
    event.dataTransfer.setData(PALETTE_DRAG_MIME, tag);
    event.dataTransfer.setData('text/plain', tag);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="w-[180px] shrink-0 bg-[var(--mc-surface-2)] border-r border-[var(--mc-border)] flex flex-col">
      {canUpdateCatalog && <div className="px-3 py-3 border-b border-[var(--mc-border)]">
        <button
          type="button"
          onClick={() => refreshCatalog({ force: true })}
          disabled={isUpdateDisabled}
          className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded-[10px] text-sm font-semibold transition-colors ${
            isUpdateDisabled
              ? 'bg-[var(--mc-surface-hover)] text-[var(--mc-text-subtle)] cursor-not-allowed'
              : 'bg-[var(--mc-surface)] text-[var(--mc-text)] border border-[var(--mc-border-strong)] hover:bg-[var(--mc-surface-hover)]'
          }`}
          title={updateTitle}
          aria-label="Refresh task steps"
        >
          <MdRefresh size={17} />
          {isUpdating ? (
            <span>Refreshing...</span>
          ) : (
            <span className="leading-tight text-center">
              <span className="block">Refresh</span>
              <span className="block">Steps</span>
            </span>
          )}
        </button>
      </div>}

      <div className="flex-1 overflow-y-auto py-2">
        <Section
          title="Flow Controls"
          chipClass="border-[#1c1a17] text-[#1c1a17]"
          items={grouped.control}
          onDragStart={handleDragStart}
        />
        <Section
          title="Actions"
          chipClass="border-[#9db89f] text-[#4f7a52]"
          items={grouped.action}
          onDragStart={handleDragStart}
        />
      </div>
    </div>
  );
}

function Section({ title, items, chipClass, onDragStart }) {
  return (
    <div className="mb-2">
      <div className="px-3 py-1 text-[10px] font-mono uppercase tracking-[0.12em] text-[var(--mc-text-subtle)] font-semibold">
        {title}
      </div>
      <div className="px-2 space-y-1.5">
        {items.map((item) => (
          <div
            key={item.tag}
            draggable
            onDragStart={(e) => onDragStart(e, item.tag)}
            className={`px-2.5 py-1.5 border ${chipClass} bg-[var(--mc-surface)] rounded-lg text-xs font-medium cursor-grab active:cursor-grabbing hover:shadow-sm hover:-translate-y-0.5 transition-all select-none`}
            title={item.tag}
          >
            {item.tag}
          </div>
        ))}
      </div>
    </div>
  );
}
