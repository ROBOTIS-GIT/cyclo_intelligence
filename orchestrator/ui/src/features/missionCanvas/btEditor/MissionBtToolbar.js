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
import {
  MdAutoFixHigh,
  MdDeleteSweep,
  MdDriveFileRenameOutline,
  MdRedo,
  MdSave,
  MdStar,
  MdUndo,
  MdUploadFile,
} from "react-icons/md";

export default function MissionBtToolbar({ view, history, commands, graph, files, options }) {
  const { selectedNodeId, canClearTree, clearTreeArmed, hydratedPath } = view;
  const { undoHistory, canUndo, redoHistory, canRedo } = history;
  const { handleAutoLayout, handleClearTree } = commands;
  const { nodes, parseError } = graph;
  const {
    setPendingLoadPath,
    availableFileOptions,
    setShowLoadDialog,
    fileAction,
    handleSaveXml,
    setShowSaveAsDialog,
    handleSetDefaultXml,
  } = files;
  const {
    filePath,
    loading,
    fileActionsDisabled,
    onLoadXml,
    onSaveXml,
    onSaveXmlAs,
    defaultFilePath,
    onSetDefaultXml,
  } = options;
  return (<>
      <div
        aria-label="Waypoint Task file actions"
        className="absolute top-3 z-20 flex items-center gap-1 rounded-[10px] border border-[var(--mc-border)] bg-[var(--mc-surface)]/95 p-1 shadow-sm transition-[right]"
        style={{ right: selectedNodeId ? 332 : 12 }}
      >
          <button
            type="button"
            onClick={undoHistory}
            disabled={!canUndo}
            title="Undo"
            className={clsx(
              "h-8 w-8 rounded-lg flex items-center justify-center transition-colors",
              canUndo
                ? "bg-[var(--mc-surface-2)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)]"
                : "bg-[var(--mc-surface-2)] text-[var(--mc-text-subtle)] opacity-50",
            )}
          >
            <MdUndo size={18} />
          </button>
          <button
            type="button"
            onClick={redoHistory}
            disabled={!canRedo}
            title="Redo"
            className={clsx(
              "h-8 w-8 rounded-lg flex items-center justify-center transition-colors",
              canRedo
                ? "bg-[var(--mc-surface-2)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)]"
                : "bg-[var(--mc-surface-2)] text-[var(--mc-text-subtle)] opacity-50",
            )}
          >
            <MdRedo size={18} />
          </button>
          <button
            type="button"
            onClick={handleAutoLayout}
            disabled={!nodes.length}
            title="Auto layout"
            className={clsx(
              "h-8 w-8 rounded-lg flex items-center justify-center transition-colors",
              nodes.length
                ? "bg-[var(--mc-surface-2)] hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-muted)]"
                : "bg-[var(--mc-surface-2)] text-[var(--mc-text-subtle)] opacity-50",
            )}
          >
            <MdAutoFixHigh size={18} />
          </button>
          <button
            type="button"
            onClick={handleClearTree}
            disabled={!canClearTree}
            aria-label={clearTreeArmed
              ? "Confirm clear current waypoint task"
              : "Clear current waypoint task"}
            title={clearTreeArmed
              ? "Click again to clear the current waypoint task"
              : "Clear current waypoint task"}
            className={clsx(
              "h-8 w-8 rounded-lg flex items-center justify-center border transition-colors",
              !canClearTree
                ? "border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50"
                : clearTreeArmed
                  ? "border-[var(--mc-danger)] bg-[var(--mc-danger)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)]"
                  : "border-[var(--mc-danger-border)] bg-[var(--mc-surface)] text-[var(--mc-danger)] shadow-[var(--mc-shadow)] hover:bg-[var(--mc-surface-hover)]",
            )}
          >
            <MdDeleteSweep size={18} aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={() => {
              setPendingLoadPath(filePath || availableFileOptions[0] || "");
              setShowLoadDialog(true);
            }}
            disabled={loading || fileActionsDisabled || Boolean(fileAction) || !onLoadXml}
            title={filePath ? `Open ${filePath}` : "Open Task"}
            aria-label="Open Task"
            className={clsx(
              "h-8 flex items-center gap-1.5 rounded-lg bg-[var(--mc-surface-2)] px-2.5 text-xs font-medium text-[var(--mc-text-muted)] transition-colors hover:bg-[var(--mc-surface-hover)]",
              (loading || fileActionsDisabled || fileAction || !onLoadXml) && "cursor-not-allowed opacity-60",
            )}
          >
            <MdUploadFile size={18} />
            {fileAction === "load" ? "Opening..." : "Open Task"}
          </button>
          <button
            type="button"
            onClick={handleSaveXml}
            disabled={(
              loading ||
              fileActionsDisabled ||
              Boolean(fileAction) ||
              Boolean(parseError) ||
              hydratedPath !== filePath ||
              !onSaveXml
            )}
            title={filePath ? `Save Task to ${filePath}` : "Save Task"}
            aria-label="Save Task"
            className={clsx(
              "h-8 flex items-center gap-1.5 rounded-lg bg-[var(--mc-surface-2)] px-2.5 text-xs font-medium text-[var(--mc-text-muted)] transition-colors hover:bg-[var(--mc-surface-hover)]",
              (
                loading ||
                fileActionsDisabled ||
                fileAction ||
                parseError ||
                hydratedPath !== filePath ||
                !onSaveXml
              ) && "cursor-not-allowed opacity-60",
            )}
          >
            <MdSave size={18} />
            {fileAction === "save" ? "Saving..." : "Save"}
          </button>
          <button
            type="button"
            onClick={() => setShowSaveAsDialog(true)}
            disabled={(
              loading
              || fileActionsDisabled
              || Boolean(fileAction)
              || Boolean(parseError)
              || hydratedPath !== filePath
              || !onSaveXmlAs
            )}
            title="Save the current waypoint task as another file"
            aria-label="Save Task As"
            className={clsx(
              "h-8 flex items-center gap-1.5 rounded-lg bg-[var(--mc-surface-2)] px-2.5 text-xs font-medium text-[var(--mc-text-muted)] transition-colors hover:bg-[var(--mc-surface-hover)]",
              (
                loading
                || fileActionsDisabled
                || fileAction
                || parseError
                || hydratedPath !== filePath
                || !onSaveXmlAs
              ) && "cursor-not-allowed opacity-60",
            )}
          >
            <MdDriveFileRenameOutline size={18} />
            {fileAction === "save-as" ? "Saving..." : "Save Task As"}
          </button>
          <button
            type="button"
            onClick={handleSetDefaultXml}
            disabled={(
              loading
              || fileActionsDisabled
              || Boolean(fileAction)
              || !filePath
              || filePath === defaultFilePath
              || !onSetDefaultXml
            )}
            title={filePath === defaultFilePath
              ? "This task is already used when running the mission"
              : "Use this task when the mission runs"}
            aria-label="Use for Run"
            className={clsx(
              "h-8 flex items-center gap-1.5 rounded-lg bg-[var(--mc-surface-2)] px-2.5 text-xs font-medium text-[var(--mc-text-muted)] transition-colors hover:bg-[var(--mc-surface-hover)]",
              (
                loading
                || fileActionsDisabled
                || fileAction
                || !filePath
                || filePath === defaultFilePath
                || !onSetDefaultXml
              ) && "cursor-not-allowed opacity-60",
            )}
          >
            <MdStar size={18} />
            {fileAction === "set-default" ? "Updating..." : "Use for Run"}
          </button>
      </div>


  </>);
}
