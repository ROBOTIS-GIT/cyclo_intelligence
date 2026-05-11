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

import React, { useEffect, useRef } from 'react';
import { CATALOG_BY_CATEGORY, isControlTag } from '../../constants/btNodeCatalog';

const MENU_WIDTH = 220;

export default function BTNodeContextMenu({ x, y, parentNode, onAddChild, onClose }) {
  const ref = useRef(null);
  const parentTag = parentNode?.data?.nodeType;
  const parentIsControl = isControlTag(parentTag);

  // Close on outside click / ESC / scroll
  useEffect(() => {
    const handlePointer = (e) => {
      if (ref.current && !ref.current.contains(e.target)) onClose();
    };
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handlePointer);
    document.addEventListener('keydown', handleKey);
    window.addEventListener('scroll', onClose, true);
    return () => {
      document.removeEventListener('mousedown', handlePointer);
      document.removeEventListener('keydown', handleKey);
      window.removeEventListener('scroll', onClose, true);
    };
  }, [onClose]);

  // Clamp to viewport — flip left when there isn't room on the right.
  const left = Math.min(x, window.innerWidth - MENU_WIDTH - 8);
  const top = Math.min(y, window.innerHeight - 320);

  return (
    <div
      ref={ref}
      style={{ left, top, width: MENU_WIDTH }}
      className="fixed z-50 bg-white border border-gray-200 rounded-md shadow-xl py-1 text-sm"
      onContextMenu={(e) => e.preventDefault()}
    >
      <div className="px-3 py-1.5 text-xs text-gray-500 border-b border-gray-100">
        {parentNode?.data?.label || parentTag}
      </div>

      {!parentIsControl ? (
        <div className="px-3 py-2 text-xs text-gray-400 italic">
          Action nodes can't have children
        </div>
      ) : (
        <>
          <CategorySection
            title="Controls"
            items={CATALOG_BY_CATEGORY.control}
            onPick={(tag) => {
              onAddChild(parentNode.id, tag);
              onClose();
            }}
          />
          <div className="border-t border-gray-100 my-1" />
          <CategorySection
            title="Actions"
            items={CATALOG_BY_CATEGORY.action}
            onPick={(tag) => {
              onAddChild(parentNode.id, tag);
              onClose();
            }}
          />
        </>
      )}
    </div>
  );
}

function CategorySection({ title, items, onPick }) {
  return (
    <div className="py-1">
      <div className="px-3 py-1 text-[10px] uppercase tracking-wider text-gray-400 font-semibold">
        {title}
      </div>
      {items.map((item) => (
        <button
          key={item.tag}
          type="button"
          onClick={() => onPick(item.tag)}
          className="w-full text-left px-3 py-1.5 hover:bg-blue-50 text-gray-700 hover:text-blue-700 transition-colors"
        >
          {item.tag}
        </button>
      ))}
    </div>
  );
}
