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
import clsx from "clsx";
import { MdUploadFile, MdSave, MdUndo, MdRedo, MdAutoFixHigh, MdDeleteSweep } from "react-icons/md";

export default function ActionBtHeader({ view, history, commands, files }) {
  const {
    autonomyStudioVariant,
    title,
    subtitle,
    treeFileName,
    hasTree,
    canClearTree,
    clearTreeArmed,
  } = view;
  const { undoHistory, canUndo, redoHistory, canRedo } = history;
  const { handleAutoLayout, handleClearTree } = commands;
  const { setSaveFileName, setSaveConflict, setShowSaveDialog, setShowTreeList } = files;
  return (<>
      {/* Header */}
      <div className={clsx(
        'flex items-center justify-between px-6 border-b',
        autonomyStudioVariant
          ? 'h-14 border-[var(--mc-border)] bg-[var(--mc-surface-2)]'
          : 'py-4 border-black bg-white',
      )}>
        <div className="min-w-0">
          <h1 className={clsx(
            'font-bold truncate',
            autonomyStudioVariant
              ? 'text-[16px] tracking-tight text-[var(--mc-text)]'
              : 'text-xl text-gray-800',
          )}>
            {title}
          </h1>
          {subtitle && (
            <p className={clsx(
              'mt-0.5 truncate text-[10px] font-mono',
              autonomyStudioVariant
                ? 'text-[var(--mc-text-muted)]'
                : 'text-gray-500',
            )}>
              {subtitle}
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className={clsx(
            'max-w-[220px] truncate font-mono',
            autonomyStudioVariant
              ? 'text-[11px] text-[var(--mc-text-muted)]'
              : 'text-sm text-gray-500',
          )}>
            {treeFileName || 'No file loaded'}
          </span>
          <button
            onClick={undoHistory}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              autonomyStudioVariant
                ? canUndo
                  ? 'border border-[var(--mc-border-strong)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)] shadow-[var(--mc-shadow)] cursor-pointer'
                  : 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                : canUndo
                ? 'bg-gray-100 hover:bg-gray-200 text-gray-700 cursor-pointer'
                : 'bg-gray-100 text-gray-300 cursor-not-allowed'
            )}
          >
            <MdUndo size={18} />
          </button>
          <button
            onClick={redoHistory}
            disabled={!canRedo}
            title="Redo (Ctrl+Shift+Z)"
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              autonomyStudioVariant
                ? canRedo
                  ? 'border border-[var(--mc-border-strong)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)] shadow-[var(--mc-shadow)] cursor-pointer'
                  : 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                : canRedo
                ? 'bg-gray-100 hover:bg-gray-200 text-gray-700 cursor-pointer'
                : 'bg-gray-100 text-gray-300 cursor-not-allowed'
            )}
          >
            <MdRedo size={18} />
          </button>
          <button
            onClick={handleAutoLayout}
            disabled={!hasTree}
            title="Auto Layout"
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              autonomyStudioVariant
                ? hasTree
                  ? 'border border-[var(--mc-border-strong)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)] shadow-[var(--mc-shadow)] cursor-pointer'
                  : 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                : hasTree
                ? 'bg-gray-100 hover:bg-gray-200 text-gray-700 cursor-pointer'
                : 'bg-gray-100 text-gray-300 cursor-not-allowed'
            )}
          >
            <MdAutoFixHigh size={18} />
          </button>
          <button
            type="button"
            onClick={handleClearTree}
            disabled={!canClearTree}
            aria-label={clearTreeArmed ? 'Confirm clear current task' : 'Clear current task'}
            title={clearTreeArmed ? 'Click again to clear the current task' : 'Clear current task'}
            className={clsx(
              'flex items-center justify-center w-9 h-9 rounded-lg transition-colors duration-150',
              autonomyStudioVariant
                ? !canClearTree
                  ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                  : clearTreeArmed
                    ? 'border border-[var(--mc-danger)] bg-[var(--mc-danger)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] cursor-pointer'
                    : 'border border-[var(--mc-danger-border)] bg-[var(--mc-surface)] text-[var(--mc-danger)] shadow-[var(--mc-shadow)] hover:bg-[var(--mc-surface-hover)] cursor-pointer'
                : !canClearTree
                  ? 'bg-gray-100 text-gray-300 cursor-not-allowed'
                  : clearTreeArmed
                    ? 'bg-red-600 text-white cursor-pointer'
                    : 'bg-red-50 hover:bg-red-100 text-red-600 cursor-pointer',
            )}
          >
            <MdDeleteSweep size={18} aria-hidden="true" />
          </button>
          <button
            onClick={() => {
              setSaveFileName(treeFileName ? treeFileName.replace(/\.xml$/i, '') : '');
              setSaveConflict(null);
              setShowSaveDialog(true);
            }}
            disabled={!hasTree}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg',
              'text-sm font-medium transition-colors duration-150',
              autonomyStudioVariant
                ? hasTree
                  ? 'border border-[var(--mc-accent)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-accent-hover)] shadow-[var(--mc-shadow)] cursor-pointer'
                  : 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                : hasTree
                ? 'bg-blue-50 hover:bg-blue-100 text-blue-700 cursor-pointer'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            )}
          >
            <MdSave size={18} />
            Save Task As
          </button>
          <button
            onClick={() => setShowTreeList(true)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg cursor-pointer',
              autonomyStudioVariant
                ? 'border border-[var(--mc-border-strong)] bg-[var(--mc-surface)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)] shadow-[var(--mc-shadow)] text-sm font-medium'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium',
              'transition-colors duration-150'
            )}
          >
            <MdUploadFile size={18} />
            Open Task
          </button>
        </div>
      </div>


  </>);
}
