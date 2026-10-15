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
// Author: Howon Kim, Seongwoo Kim

"use client";

import { useState, useRef, useCallback, useEffect } from 'react';
import { ANNOTATION_TOOL, ANNOTATION_EXTEND_TOOL, ANNOTATION_ERASE_TOOL, EDIT_TOOLS, BRUSH_SIZE_OPTIONS } from './constants';
import { TOOL_ICONS, EditorSection, SegGroup, SegButton } from './editorControls';
import { nextAutoAreaLabel } from './annotationIdentity';
import { MdUndo, MdRedo } from 'react-icons/md';
export function MapEditorControls({
  files,
  selectedPath,
  setSelectedPath,
  tool,
  setTool,
  brushSize,
  setBrushSize,
  busy,
  image,
  dirty,
  canUndo = false,
  canRedo = false,
  undo,
  redo,
  save,
  enableAnnotations = false,
  annotations = [],
  annotationLabel = "Area",
  setAnnotationLabel = () => {},
  selectedAnnotationId = "",
  setSelectedAnnotationId = () => {},
  deleteAnnotationById = () => {},
  renameAnnotation = () => {}
}) {
  const [renamingId, setRenamingId] = useState("");
  const [renameDraft, setRenameDraft] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState("");
  const confirmTimerRef = useRef(null);
  const disarmConfirms = useCallback(() => {
    setConfirmDeleteId("");
  }, []);
  const armConfirmTimer = useCallback(() => {
    if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    confirmTimerRef.current = setTimeout(disarmConfirms, 4000);
  }, [disarmConfirms]);
  useEffect(() => () => {
    if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
  }, []);
  const iconButtonStyle = enabled => ({
    borderRadius: 9,
    border: "1px solid var(--mc-border-strong)",
    backgroundColor: "var(--mc-surface)",
    color: enabled ? "var(--mc-text)" : "var(--mc-text-subtle)"
  });
  const areaTools = [ANNOTATION_TOOL, ANNOTATION_EXTEND_TOOL, ANNOTATION_ERASE_TOOL];
  const commitRename = annotation => {
    renameAnnotation(annotation.id, renameDraft);
    setRenamingId("");
    setRenameDraft("");
  };
  const areasChips = enableAnnotations && (annotations.length > 0) && <div role="group" aria-label="Map areas" className="flex max-h-24 w-full flex-wrap items-center gap-1.5 overflow-y-auto">
            {annotations.map(annotation => {
      const selected = annotation.id === selectedAnnotationId;
      const confirming = confirmDeleteId === annotation.id;
      if (renamingId === annotation.id) {
        return <input key={annotation.id} autoFocus aria-label={`Rename area ${annotation.label}`} value={renameDraft} disabled={busy} onChange={event => setRenameDraft(event.currentTarget.value)} onBlur={() => commitRename(annotation)} onKeyDown={event => {
          if (event.key === "Enter") {
            event.preventDefault();
            commitRename(annotation);
          }
          if (event.key === "Escape") {
            event.preventDefault();
            setRenamingId("");
            setRenameDraft("");
          }
        }} className="h-8 w-32 px-3 text-[12px] font-semibold" style={{
          borderRadius: 9,
          color: "var(--mc-text)",
          backgroundColor: "var(--mc-surface)",
          border: "1px solid var(--mc-border-strong)"
        }} />;
      }
      return <span key={annotation.id} className="inline-flex h-8 items-center gap-1 pl-2.5 pr-1" style={{
        borderRadius: 999,
        border: `1px solid ${selected ? "var(--mc-text)" : "var(--mc-border-strong)"}`,
        backgroundColor: selected ? "var(--mc-surface-hover)" : "var(--mc-surface)"
      }}>
                        <button type="button" aria-pressed={selected} disabled={busy} onClick={() => setSelectedAnnotationId(annotation.id)} onDoubleClick={() => {
          setRenamingId(annotation.id);
          setRenameDraft(annotation.label);
        }} title={`${annotation.label} — double-click to rename`} className="inline-flex items-center gap-1.5 text-[12px] font-semibold disabled:opacity-50" style={{
          border: "none",
          backgroundColor: "transparent",
          color: "var(--mc-text)"
        }}>
                            <span aria-hidden="true" className="shrink-0" style={{
            width: 8,
            height: 8,
            borderRadius: 999,
            backgroundColor: annotation.color
          }} />
                            {annotation.label}
                        </button>
                        <button type="button" disabled={busy} aria-label={confirming ? `Confirm delete area ${annotation.label}` : `Delete area ${annotation.label}`} onClick={() => {
          if (confirming) {
            deleteAnnotationById(annotation.id);
            disarmConfirms();
            return;
          }
          setConfirmDeleteId(annotation.id);
          armConfirmTimer();
        }} className="inline-flex h-6 w-6 items-center justify-center text-[13px] leading-none disabled:opacity-50" style={{
          borderRadius: 999,
          border: "none",
          backgroundColor: confirming ? "var(--mc-danger)" : "transparent",
          color: confirming ? "var(--mc-accent-fg)" : "var(--mc-danger)"
        }}>
                            ×
                        </button>
                    </span>;
    })}
        </div>;
  return <div className="flex w-full flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2">
                <select aria-label="PGM map" value={selectedPath} disabled={busy || (files.length === 0)} onChange={event => setSelectedPath(event.currentTarget.value)} className="h-9 min-w-[220px] px-3 text-[12px] font-mono" style={{
        borderRadius: 10,
        color: "var(--mc-text)",
        backgroundColor: "var(--mc-surface)",
        border: "1px solid var(--mc-border-strong)"
      }}>
                    <option value="">Select a PGM</option>
                    {files.map(file => <option key={file.path} value={file.path}>{file.path}</option>)}
                </select>

                {/* Standalone (no SegGroup pill around it), so it carries its own
                    surface + border to read as a button rather than plain text. */}
                <button type="button" disabled={busy} aria-pressed={tool === "view"} onClick={() => setTool("view")} title="View" aria-label="View" className="h-9 px-3 inline-flex items-center gap-1.5 text-[12.5px] font-semibold transition-colors disabled:opacity-50" style={{
        borderRadius: 9,
        border: "1px solid var(--mc-border-strong)",
        backgroundColor: tool === "view" ? "var(--mc-text)" : "var(--mc-surface)",
        color: tool === "view" ? "var(--mc-bg)" : "var(--mc-text)"
      }}>
                    {TOOL_ICONS.view}View
                </button>

                <EditorSection label="MAP EDIT">
                    <SegGroup ariaLabel="Map edit tools">
                        {EDIT_TOOLS.map(editTool => <SegButton key={editTool.id} selected={tool === editTool.id} disabled={busy} onClick={() => setTool(editTool.id)} title={editTool.label} ariaLabel={editTool.label}>
                                {TOOL_ICONS[editTool.id]}{editTool.label}
                            </SegButton>)}
                    </SegGroup>
                </EditorSection>

                {enableAnnotations && <EditorSection label="AREA EDIT">
                        <SegGroup ariaLabel="Area edit tools">
                            {areaTools.map(editTool => <SegButton key={editTool.id} selected={tool === editTool.id} disabled={busy} onClick={() => setTool(editTool.id)} title={editTool.label} ariaLabel={editTool.label}>
                                    {TOOL_ICONS[editTool.id]}{editTool.label}
                                </SegButton>)}
                        </SegGroup>
                        <input aria-label="Area name" value={annotationLabel} placeholder={nextAutoAreaLabel(annotations)} disabled={busy} onChange={event => setAnnotationLabel(event.currentTarget.value)} className="h-8 w-28 px-3 text-[12px] font-semibold" style={{
          borderRadius: 9,
          color: "var(--mc-text)",
          backgroundColor: "var(--mc-surface)",
          border: "1px solid var(--mc-border-strong)"
        }} />
                    </EditorSection>}

                <EditorSection label="BRUSH">
                    <SegGroup ariaLabel="Brush size">
                        {BRUSH_SIZE_OPTIONS.map(option => <SegButton key={option.value} narrow selected={brushSize === option.value} disabled={busy} onClick={() => setBrushSize(option.value)} title={`Brush ${option.label}: ${option.value}px`} ariaLabel={`Brush size ${option.label}`}>
                                {option.label === "Small" ? "S" : option.label === "Medium" ? "M" : option.label === "Large" ? "L" : "XL"}
                            </SegButton>)}
                    </SegGroup>
                </EditorSection>

                <button type="button" disabled={busy || !canUndo} onClick={undo} title="Undo" aria-label="Undo" className="h-9 w-9 inline-flex items-center justify-center transition-all active:translate-y-px disabled:opacity-50 disabled:active:translate-y-0" style={iconButtonStyle(canUndo)}>
                    <MdUndo size={17} />
                </button>
                <button type="button" disabled={busy || !canRedo} onClick={redo} title="Redo" aria-label="Redo" className="h-9 w-9 inline-flex items-center justify-center transition-all active:translate-y-px disabled:opacity-50 disabled:active:translate-y-0" style={iconButtonStyle(canRedo)}>
                    <MdRedo size={17} />
                </button>
                <button type="button" disabled={busy || !dirty} onClick={save} className="h-9 px-4 text-[12.5px] font-semibold transition-all active:translate-y-px disabled:opacity-50 disabled:active:translate-y-0" style={{
        borderRadius: 9,
        border: "none",
        backgroundColor: "var(--mc-accent)",
        color: "var(--mc-accent-fg)",
        boxShadow: dirty ? "var(--mc-shadow)" : "none"
      }}>
                    Save
                </button>

                <span className="text-[11px] font-mono" style={{
        color: "var(--mc-text-subtle)"
      }}>
                    {image ? <>
                            {image.width} × {image.height}
                            {dirty && <span style={{
            color: "var(--mc-accent)"
          }}> · unsaved</span>}
                        </> : "Select a PGM"}
                </span>
            </div>

            {areasChips}
        </div>;
}
