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

import { DEFAULT_MISSION_NAME, isValidMissionName } from "../lib/missionNames";
import { ActionButton } from "./primitives";

export function SaveMapDialog({ open, value, busy, onChange, onCancel, onSubmit }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4" role="dialog" aria-modal="true" aria-labelledby="mission-save-map-title"
      style={{ backgroundColor: "rgba(28,26,23,0.45)", backdropFilter: "blur(3px)" }}>
      <form
        className="w-full max-w-sm grid gap-4 p-5"
        style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border)", borderRadius: 16, boxShadow: "var(--mc-shadow)" }}
        onSubmit={(event) => { event.preventDefault(); onSubmit(); }}
      >
        <div id="mission-save-map-title" className="text-[15px] font-bold">Save Map</div>
        <label className="grid gap-1.5 text-xs">
          <span style={{ color: "var(--mc-text-muted)" }}>Map name</span>
          <input
            autoFocus aria-label="Save map name" value={value} disabled={busy}
            onChange={(event) => onChange(event.currentTarget.value)}
            className="h-9 px-3 text-sm"
            style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface-2)", border: "1px solid var(--mc-border-strong)", borderRadius: 10 }}
          />
        </label>
        <div className="flex justify-end gap-2">
          <ActionButton disabled={busy} onClick={onCancel} variant="secondary">Cancel</ActionButton>
          <ActionButton disabled={busy || !value.trim()} type="submit">Save</ActionButton>
        </div>
      </form>
    </div>
  );
}

// Save/duplicate a mission under a typed name: prefilled for one-Enter saves,
// existing names offered as chips, overwrite called out inline.
export function SaveMissionDialog({
  open,
  title = "Save Mission",
  fieldLabel = "Mission name",
  inputAriaLabel = "Save mission name",
  submitLabel = "Save",
  value,
  existingNames = [],
  currentName = "",
  disallowExisting = false,
  hint = "",
  busy,
  onChange,
  onCancel,
  onSubmit,
}) {
  if (!open) return null;
  const trimmed = value.trim();
  const valid = isValidMissionName(trimmed);
  const isExisting = existingNames.includes(trimmed);
  const wouldOverwrite = valid && isExisting && trimmed !== currentName;
  const blocked = busy || !valid || (disallowExisting && isExisting);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4" role="dialog" aria-modal="true" aria-labelledby="mission-save-mission-title"
      style={{ backgroundColor: "rgba(28,26,23,0.45)", backdropFilter: "blur(3px)" }}>
      <form
        className="w-full max-w-sm grid gap-4 p-5"
        style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border)", borderRadius: 16, boxShadow: "var(--mc-shadow)" }}
        onSubmit={(event) => { event.preventDefault(); if (!blocked) onSubmit(); }}
      >
        <div id="mission-save-mission-title" className="text-[15px] font-bold">{title}</div>
        <label className="grid gap-1.5 text-xs">
          <span style={{ color: "var(--mc-text-muted)" }}>{fieldLabel}</span>
          <input
            autoFocus aria-label={inputAriaLabel} value={value} disabled={busy}
            onChange={(event) => onChange(event.currentTarget.value)}
            className="h-9 px-3 text-sm"
            style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface-2)", border: "1px solid var(--mc-border-strong)", borderRadius: 10 }}
          />
        </label>
        {existingNames.length > 0 && (
          <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
            {existingNames.map((name) => (
              <button
                key={name}
                type="button"
                disabled={busy}
                onClick={() => onChange(name)}
                className="px-2 py-0.5 text-[11px]"
                style={{
                  borderRadius: 8,
                  border: "1px solid var(--mc-border-strong)",
                  backgroundColor: trimmed === name ? "var(--mc-surface-hover)" : "var(--mc-surface-2)",
                  color: "var(--mc-text-muted)",
                }}
              >
                {name}
              </button>
            ))}
          </div>
        )}
        {trimmed && !valid && (
          <div className="text-[11px]" style={{ color: "var(--mc-text-subtle)" }}>
            Only letters, numbers, '.', '_' and '-'
          </div>
        )}
        {wouldOverwrite && !disallowExisting && (
          <div className="text-[11px]" style={{ color: "var(--mc-warning)" }}>
            A mission named "{trimmed}" already exists — saving will replace it.
          </div>
        )}
        {disallowExisting && isExisting && (
          <div className="text-[11px]" style={{ color: "var(--mc-warning)" }}>
            A mission named "{trimmed}" already exists.
          </div>
        )}
        {hint && (
          <div className="text-[11px]" style={{ color: "var(--mc-text-subtle)" }}>{hint}</div>
        )}
        <div className="flex justify-end gap-2">
          <ActionButton disabled={busy} onClick={onCancel} variant="secondary">Cancel</ActionButton>
          <ActionButton disabled={blocked} type="submit">
            {wouldOverwrite && !disallowExisting ? "Overwrite" : submitLabel}
          </ActionButton>
        </div>
      </form>
    </div>
  );
}

// Generic confirm with an optional middle action (e.g. Save & continue).
export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel,
  confirmVariant = "danger",
  altLabel = "",
  cancelLabel = "Cancel",
  hint = "",
  busy,
  onConfirm,
  onAlt,
  onCancel,
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center px-4" role="dialog" aria-modal="true" aria-labelledby="mission-confirm-title"
      style={{ backgroundColor: "rgba(28,26,23,0.45)", backdropFilter: "blur(3px)" }}>
      <div
        className="w-full max-w-sm grid gap-4 p-5"
        style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border)", borderRadius: 16, boxShadow: "var(--mc-shadow)" }}
      >
        <div id="mission-confirm-title" className="text-[15px] font-bold">{title}</div>
        <div className="text-[13px]" style={{ color: "var(--mc-text-muted)" }}>{body}</div>
        {hint && (
          <div className="text-[11px]" style={{ color: "var(--mc-text-subtle)" }}>{hint}</div>
        )}
        <div className="flex justify-end gap-2">
          <ActionButton disabled={busy} onClick={onCancel} variant="secondary">{cancelLabel}</ActionButton>
          {altLabel && (
            <ActionButton disabled={busy} onClick={onAlt} variant="secondary">{altLabel}</ActionButton>
          )}
          <ActionButton disabled={busy} onClick={onConfirm} variant={confirmVariant}>{confirmLabel}</ActionButton>
        </div>
      </div>
    </div>
  );
}

export function LoadMapDialog({
  open,
  files,
  selectedPath,
  missionNames = null,
  selectedMissionName = "",
  busy,
  catalogReady = true,
  title = "Load Map",
  fieldLabel = "Map file",
  selectAriaLabel = "Map file",
  missionSelectAriaLabel = "Mission file",
  onChange,
  onMissionChange,
  onCancel,
  onSubmit,
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4" role="dialog" aria-modal="true" aria-labelledby="mission-load-map-title"
      style={{ backgroundColor: "rgba(28,26,23,0.45)", backdropFilter: "blur(3px)" }}>
      <form
        className="w-full max-w-sm grid gap-4 p-5"
        style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border)", borderRadius: 16, boxShadow: "var(--mc-shadow)" }}
        onSubmit={(event) => { event.preventDefault(); onSubmit(); }}
      >
        <div id="mission-load-map-title" className="text-[15px] font-bold">{title}</div>
        {catalogReady ? (
          <>
            <label className="grid gap-1.5 text-xs">
              <span style={{ color: "var(--mc-text-muted)" }}>{fieldLabel}</span>
              <select
                aria-label={selectAriaLabel} value={selectedPath} disabled={busy || files.length === 0}
                onChange={(event) => onChange(event.currentTarget.value)}
                className="h-9 px-2.5 text-sm"
                style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface-2)", border: "1px solid var(--mc-border-strong)", borderRadius: 10 }}
              >
                {files.length === 0 ? (<option value="">No maps found</option>)
                  : files.map((file) => (<option key={file.path} value={file.path}>{file.name || file.path}</option>))}
              </select>
            </label>
            {missionNames !== null && (
              <label className="grid gap-1.5 text-xs">
                <span style={{ color: "var(--mc-text-muted)" }}>Mission</span>
                <select
                  aria-label={missionSelectAriaLabel}
                  value={selectedMissionName}
                  disabled={busy}
                  onChange={(event) => onMissionChange(event.currentTarget.value)}
                  className="h-9 px-2.5 text-sm"
                  style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface-2)", border: "1px solid var(--mc-border-strong)", borderRadius: 10 }}
                >
                  {missionNames.length === 0
                    // A map with no missions yet: loading the default name starts a
                    // fresh mission (created on first save).
                    ? (<option value={DEFAULT_MISSION_NAME}>{DEFAULT_MISSION_NAME} (new)</option>)
                    : missionNames.map((name) => (<option key={name} value={name}>{name}</option>))}
                </select>
              </label>
            )}
          </>
        ) : (
          <div role="status" className="h-9 flex items-center text-xs" style={{ color: "var(--mc-text-muted)" }}>
            Loading saved maps...
          </div>
        )}
        <div className="flex justify-end gap-2">
          <ActionButton disabled={busy} onClick={onCancel} variant="secondary">Cancel</ActionButton>
          <ActionButton
            disabled={
              busy
              || !catalogReady
              || !selectedPath
              || (missionNames !== null && !selectedMissionName)
            }
            type="submit"
            variant="secondary"
          >Load</ActionButton>
        </div>
      </form>
    </div>
  );
}

// Confirm popup for deleting a saved map — the warning spells out the
// cascade (areas + missions) before asking.
export function DeleteMapDialog({ file, missionCount, busy, onCancel, onConfirm }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="mission-delete-map-title"
      style={{ backgroundColor: "rgba(28,26,23,0.45)", backdropFilter: "blur(3px)" }}
    >
      <div
        className="w-full max-w-sm grid gap-4 p-5"
        style={{ color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border)", borderRadius: 16, boxShadow: "var(--mc-shadow)" }}
      >
        <div id="mission-delete-map-title" className="text-[15px] font-bold">Delete this map?</div>
        <div className="grid gap-1 text-[13px]">
          <span className="font-mono truncate" title={file.path}>{file.path}</span>
          <span style={{ color: "var(--mc-danger)" }}>
            {missionCount === null
              ? "This map, its areas, and its missions will be deleted permanently."
              : missionCount === 0
                ? "This map and its areas will be deleted permanently."
                : `This map, its areas, and ${missionCount} mission${missionCount === 1 ? "" : "s"} will be deleted permanently.`}
          </span>
        </div>
        <div className="flex justify-end gap-2">
          <ActionButton disabled={busy} onClick={onCancel} variant="secondary">No</ActionButton>
          <ActionButton disabled={busy} onClick={onConfirm} variant="danger">Yes</ActionButton>
        </div>
      </div>
    </div>
  );
}
