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
import TreeListModal from "./TreeListModal";

export default function ActionBtDialogs({ files, view }) {
  const {
    showTreeList,
    setShowTreeList,
    handleServerFileSelect,
    showSaveDialog,
    saveFileName,
    setSaveFileName,
    setSaveConflict,
    handleSaveAs,
    setShowSaveDialog,
    saveConflict,
  } = files;
  const { variant, autonomyStudioVariant } = view;
  return (<>
      {/* Tree List Modal */}
      <TreeListModal
        isOpen={showTreeList}
        onClose={() => setShowTreeList(false)}
        onSelect={handleServerFileSelect}
        variant={variant}
      />

      {/* Save As Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className={clsx(
            'rounded-2xl p-6 w-80',
            autonomyStudioVariant
              ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text)] shadow-[var(--mc-shadow)]'
              : 'bg-white shadow-xl',
          )}>
            <h2 className={clsx(
              'text-base font-semibold mb-4',
              autonomyStudioVariant ? 'text-[var(--mc-text)]' : 'text-gray-800',
            )}>
              Save Task As
            </h2>
            <div className={clsx(
              'flex items-center gap-1 border rounded-lg px-3 py-2 focus-within:ring-2',
              autonomyStudioVariant
                ? 'border-[var(--mc-border-strong)] bg-[var(--mc-surface-2)] focus-within:ring-[var(--mc-accent)]'
                : 'border-gray-300 focus-within:ring-blue-400',
            )}>
              <input
                autoFocus
                type="text"
                value={saveFileName}
                onChange={(e) => {
                  setSaveFileName(e.target.value);
                  setSaveConflict(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveAs();
                  if (e.key === 'Escape') {
                    setShowSaveDialog(false);
                    setSaveConflict(null);
                  }
                }}
                placeholder="filename"
                className={clsx(
                  'flex-1 min-w-0 text-sm outline-none bg-transparent',
                  autonomyStudioVariant && 'text-[var(--mc-text)] placeholder:text-[var(--mc-text-subtle)]',
                )}
              />
              <span className={clsx(
                'text-sm',
                autonomyStudioVariant ? 'text-[var(--mc-text-subtle)]' : 'text-gray-400',
              )}>
                .xml
              </span>
            </div>
            {saveConflict && (
              <div className={clsx(
                'mt-3 rounded-lg border px-3 py-2 text-sm',
                autonomyStudioVariant
                  ? 'border-[var(--mc-warning)] bg-[var(--mc-surface-2)] text-[var(--mc-warning)]'
                  : 'border-amber-200 bg-amber-50 text-amber-800',
              )}>
                <div className="font-medium">File already exists</div>
                <div className="mt-1">
                  Choose another name or overwrite {saveConflict.filename || 'this file'}.
                </div>
              </div>
            )}
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => {
                  setShowSaveDialog(false);
                  setSaveConflict(null);
                }}
                className={clsx(
                  'px-4 py-2 text-sm rounded-lg transition-colors',
                  autonomyStudioVariant
                    ? 'text-[var(--mc-text-muted)] hover:bg-[var(--mc-surface-hover)]'
                    : 'text-gray-600 hover:bg-gray-100',
                )}
              >
                Cancel
              </button>
              {saveConflict && (
                <button
                  onClick={() => handleSaveAs({ overwrite: true })}
                  className={clsx(
                    'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                    autonomyStudioVariant
                      ? 'border border-[var(--mc-danger-border)] bg-[var(--mc-surface)] text-[var(--mc-danger)] hover:bg-[var(--mc-surface-hover)]'
                      : 'bg-red-50 hover:bg-red-100 text-red-700',
                  )}
                >
                  Overwrite
                </button>
              )}
              <button
                onClick={handleSaveAs}
                disabled={!saveFileName.trim()}
                className={clsx(
                  'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                  autonomyStudioVariant
                    ? saveFileName.trim()
                      ? 'bg-[var(--mc-accent)] hover:bg-[var(--mc-accent-hover)] text-[var(--mc-accent-fg)]'
                      : 'bg-[var(--mc-surface-hover)] text-[var(--mc-text-subtle)] cursor-not-allowed'
                    : saveFileName.trim()
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                )}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

  </>);
}
