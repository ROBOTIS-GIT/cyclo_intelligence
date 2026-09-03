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

import { useEffect, useRef, useState } from "react";
import { MdDelete } from "react-icons/md";
import {
  ANNOTATION_ERASE_TOOL,
  ANNOTATION_EXTEND_TOOL,
  ANNOTATION_TOOL,
  BRUSH_SIZE_OPTIONS,
  EDIT_TOOLS,
  nextAutoAreaLabel,
} from "../../../../components/navigation/MapEditor";

// The Map Edit HUD groups the editor tools behind two icons: "Map Edit"
// (pixel tools + brush) and "Add Label" (area tools), each opening a
// text-button popover below — the Design HUD's waypoint-options idiom.
export const MAP_EDIT_PIXEL_TOOL_IDS = EDIT_TOOLS.map((tool) => tool.id);

export const MAP_EDIT_AREA_TOOLS = [ANNOTATION_TOOL, ANNOTATION_EXTEND_TOOL, ANNOTATION_ERASE_TOOL];

export const MAP_EDIT_AREA_TOOL_IDS = MAP_EDIT_AREA_TOOLS.map((tool) => tool.id);

// Area management inside the Add Label popover. The list shows for every
// area tool (Extend/Erase pick their target here; delete is two-step, rename
// on double-click); the name input only for Area, which creates by dragging
// a rectangle on the map.
export function MapAreaManager({ mapEditor, showNameInput = false }) {
  const [renamingId, setRenamingId] = useState("");
  const [renameDraft, setRenameDraft] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState("");
  const confirmTimerRef = useRef(null);
  useEffect(() => () => {
    if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
  }, []);
  const busy = mapEditor.busy;
  const armConfirm = (id) => {
    setConfirmDeleteId(id);
    if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    confirmTimerRef.current = setTimeout(() => setConfirmDeleteId(""), 4000);
  };
  const commitRename = (annotation) => {
    mapEditor.renameAnnotation(annotation.id, renameDraft);
    setRenamingId("");
    setRenameDraft("");
  };
  return (
    <div className="grid gap-2 w-full">
      {showNameInput && (
        <label className="grid gap-1">
          <span className="text-[10px] font-mono tracking-[0.12em]" style={{ color: "var(--mc-text-subtle)" }}>
            AREA NAME
          </span>
          <input
            aria-label="Area name"
            value={mapEditor.annotationLabel}
            placeholder={nextAutoAreaLabel(mapEditor.annotations)}
            disabled={busy}
            onChange={(event) => mapEditor.setAnnotationLabel(event.currentTarget.value)}
            className="h-8 w-full px-2.5 text-xs font-medium"
            style={{ borderRadius: 8, color: "var(--mc-text)", backgroundColor: "var(--mc-surface-2)", border: "1px solid var(--mc-border-strong)" }}
          />
        </label>
      )}
      <div className="grid gap-1.5 min-w-0 border-t pt-2" style={{ borderColor: "var(--mc-border)" }}>
        <div className="flex items-center justify-between px-0.5">
          <span className="text-[10px] font-mono tracking-[0.12em]" style={{ color: "var(--mc-text-subtle)" }}>
            LABELED AREAS
          </span>
          <span className="text-[10px] font-mono" style={{ color: "var(--mc-text-muted)" }}>
            {mapEditor.annotations.length}
          </span>
        </div>
        <div
          role="group"
          aria-label="Map areas"
          className="grid max-h-28 gap-2 content-start overflow-y-auto overscroll-contain pr-1"
          style={{ scrollbarGutter: "stable" }}
        >
          {mapEditor.annotations.map((annotation) => {
            const selected = annotation.id === mapEditor.selectedAnnotationId;
            const confirming = confirmDeleteId === annotation.id;
            return (
              <div key={annotation.id} className="flex h-8 items-center gap-1.5 min-w-0">
                {renamingId === annotation.id ? (
                  <input
                    autoFocus
                    aria-label={`Rename area ${annotation.label}`}
                    value={renameDraft}
                    disabled={busy}
                    onChange={(event) => setRenameDraft(event.currentTarget.value)}
                    onBlur={() => commitRename(annotation)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        commitRename(annotation);
                      }
                      if (event.key === "Escape") {
                        event.preventDefault();
                        setRenamingId("");
                        setRenameDraft("");
                      }
                    }}
                    className="h-8 flex-1 px-2 text-[13px] min-w-0"
                    style={{ borderRadius: 8, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text)" }}
                  />
                ) : (
                  <button
                    type="button"
                    aria-pressed={selected}
                    disabled={busy}
                    onClick={(event) => {
                      event.stopPropagation();
                      mapEditor.setSelectedAnnotationId(annotation.id);
                    }}
                    onDoubleClick={() => {
                      setRenamingId(annotation.id);
                      setRenameDraft(annotation.label);
                    }}
                    title={`${annotation.label} — double-click to rename`}
                    className="h-8 flex-1 px-2.5 min-w-0 inline-flex items-center gap-2 text-left text-[12.5px] font-semibold disabled:opacity-50"
                    style={{
                      borderRadius: 8,
                      border: `1px solid ${selected ? "var(--mc-accent)" : "var(--mc-border)"}`,
                      backgroundColor: selected ? "var(--mc-accent-soft)" : "var(--mc-surface-2)",
                      color: "var(--mc-text)",
                    }}
                  >
                    <span aria-hidden="true" className="shrink-0" style={{ width: 8, height: 8, borderRadius: 999, backgroundColor: annotation.color }} />
                    <span className="block truncate">{annotation.label}</span>
                  </button>
                )}
                <button
                  type="button"
                  disabled={busy}
                  aria-label={confirming ? `Confirm delete area ${annotation.label}` : `Delete area ${annotation.label}`}
                  title={confirming ? "Click again to delete" : `Delete ${annotation.label}`}
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    if (confirming) {
                      mapEditor.deleteAnnotationById(annotation.id);
                      setConfirmDeleteId("");
                      return;
                    }
                    armConfirm(annotation.id);
                  }}
                  className="h-8 w-8 shrink-0 inline-flex items-center justify-center active:translate-y-px disabled:opacity-50"
                  style={{
                    borderRadius: 8,
                    border: `1px solid ${confirming ? "var(--mc-danger)" : "var(--mc-border-strong)"}`,
                    backgroundColor: confirming ? "var(--mc-danger)" : "var(--mc-surface)",
                    color: confirming ? "var(--mc-accent-fg)" : "var(--mc-danger)",
                  }}
                >
                  <MdDelete size={15} />
                </button>
              </div>
            );
          })}
          {mapEditor.annotations.length === 0 && (
            <div className="text-[12px]" style={{ color: "var(--mc-text-muted)" }}>
              {showNameInput ? "Drag on the map to mark a region." : "No labeled areas."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Brush-size row shared by both tool popovers — pixel painting and the
// area extend/erase tools all stroke with the same brush.
export function MapEditBrushRow({ brushSize, setBrushSize, disabled = false }) {
  return (
    <div className="flex items-center gap-2">
      <span className="shrink-0 pl-1 pr-1 text-[10px] font-mono tracking-[0.12em]" style={{ color: "var(--mc-text-subtle)" }}>
        BRUSH
      </span>
      {BRUSH_SIZE_OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-label={`Brush size ${option.label}`}
          aria-pressed={brushSize === option.value}
          disabled={disabled}
          onClick={() => setBrushSize(option.value)}
          title={`Brush ${option.label}: ${option.value}px`}
          className="h-8 min-w-[34px] px-2 inline-flex items-center justify-center text-[11px] font-bold disabled:opacity-50"
          style={{
            borderRadius: 9,
            border: `1px solid ${brushSize === option.value ? "var(--mc-accent)" : "var(--mc-border-strong)"}`,
            backgroundColor: brushSize === option.value ? "var(--mc-accent-soft)" : "var(--mc-surface)",
            color: "var(--mc-text)",
          }}
        >
          {option.label === "Small" ? "S" : option.label === "Medium" ? "M" : option.label === "Large" ? "L" : "XL"}
        </button>
      ))}
    </div>
  );
}
