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

export default function MissionBtFileDialogs({ files, options }) {
  const {
    showLoadDialog,
    handleLoadXml,
    pendingLoadPath,
    availableFileOptions,
    setPendingLoadPath,
    setShowLoadDialog,
    fileAction,
    showSaveAsDialog,
    handleSaveXmlAs,
    saveAsName,
    setSaveAsName,
    setShowSaveAsDialog,
  } = files;
  const { defaultFilePath } = options;
  return (<>
        {showLoadDialog && (
          <div
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/45 p-6"
            role="dialog"
            aria-modal="true"
            aria-label="Waypoint Task files"
          >
            <form
              className="w-full max-w-md rounded-xl border border-[var(--mc-border)] bg-[var(--mc-surface)] p-4 shadow-xl"
              onSubmit={(event) => {
                event.preventDefault();
                void handleLoadXml(pendingLoadPath);
              }}
            >
              <div className="text-sm font-semibold">Open Waypoint Task</div>
              <div className="mt-1 text-xs text-[var(--mc-text-subtle)]">
                Opening another task changes the editor only. Choose Use for Run to change mission behavior.
              </div>
              <div className="mt-3 max-h-64 space-y-2 overflow-auto">
                {availableFileOptions.map((path) => (
                  <label
                    key={path}
                    className={clsx(
                      "flex cursor-pointer items-start gap-2 rounded-lg border p-2.5",
                      pendingLoadPath === path
                        ? "border-[var(--mc-accent-hover)] bg-[var(--mc-accent-soft)]"
                        : "border-[var(--mc-border)] bg-[var(--mc-surface-2)]",
                    )}
                  >
                    <input
                      type="radio"
                      name="local-bt-xml"
                      value={path}
                      checked={pendingLoadPath === path}
                      onChange={() => setPendingLoadPath(path)}
                      className="mt-0.5 accent-[var(--mc-accent-hover)]"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-1 text-xs font-semibold">
                        <span className="truncate">{path.split("/").pop()}</span>
                        {path === defaultFilePath && (
                          <span className="rounded bg-[var(--mc-accent-soft)] px-1 py-0.5 text-[9px] text-[var(--mc-accent-hover)]">
                            Run Task
                          </span>
                        )}
                      </span>
                      <span className="block truncate text-[10px] text-[var(--mc-text-subtle)]">
                        {path}
                      </span>
                    </span>
                  </label>
                ))}
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <button
                  type="button"
                  className="rounded-lg border border-[var(--mc-border)] px-3 py-2 text-xs"
                  onClick={() => setShowLoadDialog(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!pendingLoadPath || Boolean(fileAction)}
                  className="rounded-lg bg-[var(--mc-accent-hover)] px-3 py-2 text-xs font-semibold text-[var(--mc-accent-fg)] hover:brightness-90 disabled:opacity-50"
                >
                  {fileAction === "load" ? "Opening..." : "Open Selected"}
                </button>
              </div>
            </form>
          </div>
        )}
        {showSaveAsDialog && (
          <div
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/45 p-6"
            role="dialog"
            aria-modal="true"
            aria-label="Save Waypoint Task As"
          >
            <form
              className="w-full max-w-sm rounded-xl border border-[var(--mc-border)] bg-[var(--mc-surface)] p-4 shadow-xl"
              onSubmit={handleSaveXmlAs}
            >
              <div className="text-sm font-semibold">Save Waypoint Task As</div>
              <div className="mt-1 text-xs text-[var(--mc-text-subtle)]">
                A new task file is added to this waypoint. The task used for Run does not change.
              </div>
              <label className="mt-4 block text-xs font-medium" htmlFor="local-bt-save-as-name">
                New task file name
              </label>
              <input
                id="local-bt-save-as-name"
                aria-label="New task file name"
                value={saveAsName}
                onChange={(event) => setSaveAsName(event.target.value)}
                placeholder="alternate.xml"
                className="mt-1 w-full rounded-lg border border-[var(--mc-border)] bg-[var(--mc-surface-2)] px-3 py-2 text-sm outline-none focus:border-[var(--mc-accent-hover)]"
                autoFocus
              />
              <div className="mt-1 text-[10px] text-[var(--mc-text-subtle)]">
                .xml is added automatically. Use letters, numbers, dot, underscore or hyphen.
              </div>
              <div className="mt-4 flex justify-end gap-2">
                <button
                  type="button"
                  className="rounded-lg border border-[var(--mc-border)] px-3 py-2 text-xs"
                  onClick={() => setShowSaveAsDialog(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!saveAsName.trim() || Boolean(fileAction)}
                  className="rounded-lg bg-[var(--mc-accent-hover)] px-3 py-2 text-xs font-semibold text-[var(--mc-accent-fg)] hover:brightness-90 disabled:opacity-50"
                >
                  {fileAction === "save-as" ? "Saving..." : "Save Task As"}
                </button>
              </div>
            </form>
          </div>
        )}

  </>);
}
