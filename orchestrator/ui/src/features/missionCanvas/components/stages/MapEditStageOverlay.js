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

import {
  MdEdit,
  MdLabel,
  MdRedo,
  MdSave,
  MdUndo,
  MdVisibility,
} from "react-icons/md";
import {
  ANNOTATION_ERASE_TOOL,
  ANNOTATION_EXTEND_TOOL,
  ANNOTATION_TOOL,
  EDIT_TOOLS,
} from "../../../../components/navigation/MapEditor";
import { MapEditToolButton, WaypointOptionButton } from "../primitives";
import {
  MAP_EDIT_AREA_TOOLS,
  MAP_EDIT_AREA_TOOL_IDS,
  MAP_EDIT_PIXEL_TOOL_IDS,
  MapAreaManager,
  MapEditBrushRow,
} from "../mapEdit/MapEditTools";

// View / Map Edit / Add Label / Undo / Redo / Save share one glass HUD.
// The selected-map chip remains a separate top-right overlay in the same
// component so callers can replace both existing Map Edit stage fragments at
// once without moving any editor behavior into this presentation layer.
export default function MapEditStageOverlay({
  mapEditor,
  mapEditToolsOpen,
  labelToolsOpen,
  setMapEditToolsOpen,
  setLabelToolsOpen,
}) {
  return (
    <>
      <div
        className="absolute top-5 left-5 z-20 flex items-center gap-2 p-2"
        style={{ borderRadius: 14, backgroundColor: "color-mix(in srgb, var(--mc-surface) 88%, transparent)", border: "1px solid var(--mc-border)", boxShadow: "var(--mc-shadow)", backdropFilter: "blur(8px)" }}
      >
        <MapEditToolButton
          label="View"
          active={mapEditor.tool === "view"}
          disabled={mapEditor.busy || !mapEditor.image}
          onClick={() => {
            mapEditor.setTool("view");
            setMapEditToolsOpen(false);
            setLabelToolsOpen(false);
          }}
        >
          <MdVisibility size={17} aria-hidden="true" />
        </MapEditToolButton>
        <div className="relative">
          <MapEditToolButton
            label="Map Edit"
            active={mapEditToolsOpen || MAP_EDIT_PIXEL_TOOL_IDS.includes(mapEditor.tool)}
            disabled={mapEditor.busy || !mapEditor.image}
            onClick={() => {
              setMapEditToolsOpen((open) => !open);
              setLabelToolsOpen(false);
            }}
          >
            <MdEdit size={16} aria-hidden="true" />
          </MapEditToolButton>
          {mapEditToolsOpen && (
            <div
              className="absolute left-0 top-[calc(100%+6px)] grid gap-2 p-2"
              role="menu"
              aria-label="Map edit tools"
              style={{ borderRadius: 12, backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border-strong)", boxShadow: "var(--mc-shadow)" }}
            >
              <div className="flex items-center gap-2">
                {EDIT_TOOLS.map((editTool) => (
                  <WaypointOptionButton
                    key={editTool.id}
                    active={mapEditor.tool === editTool.id}
                    disabled={mapEditor.busy}
                    onClick={() => mapEditor.setTool(editTool.id)}
                  >
                    {editTool.label}
                  </WaypointOptionButton>
                ))}
              </div>
              <MapEditBrushRow brushSize={mapEditor.brushSize} setBrushSize={mapEditor.setBrushSize} disabled={mapEditor.busy} />
            </div>
          )}
        </div>
        <div className="relative">
          <MapEditToolButton
            label="Add Label"
            active={labelToolsOpen || MAP_EDIT_AREA_TOOL_IDS.includes(mapEditor.tool)}
            disabled={mapEditor.busy || !mapEditor.image}
            onClick={() => {
              setLabelToolsOpen((open) => !open);
              setMapEditToolsOpen(false);
            }}
          >
            <MdLabel size={16} aria-hidden="true" />
          </MapEditToolButton>
          {labelToolsOpen && (
            <div
              className="absolute left-0 top-[calc(100%+6px)] z-30 isolate grid w-80 max-w-[calc(100vw-2rem)] gap-2 p-2 pointer-events-auto"
              role="menu"
              aria-label="Map labeling tools"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => event.stopPropagation()}
              style={{ borderRadius: 12, backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border-strong)", boxShadow: "var(--mc-shadow)" }}
            >
              <div className="flex items-center gap-2">
                {MAP_EDIT_AREA_TOOLS.map((areaTool) => (
                  <WaypointOptionButton
                    key={areaTool.id}
                    active={mapEditor.tool === areaTool.id}
                    disabled={mapEditor.busy}
                    onClick={() => mapEditor.setTool(areaTool.id)}
                  >
                    {areaTool.label}
                  </WaypointOptionButton>
                ))}
              </div>
              {(mapEditor.tool === ANNOTATION_EXTEND_TOOL.id || mapEditor.tool === ANNOTATION_ERASE_TOOL.id) && (
                <MapEditBrushRow brushSize={mapEditor.brushSize} setBrushSize={mapEditor.setBrushSize} disabled={mapEditor.busy} />
              )}
              {MAP_EDIT_AREA_TOOL_IDS.includes(mapEditor.tool) && (
                <MapAreaManager mapEditor={mapEditor} showNameInput={mapEditor.tool === ANNOTATION_TOOL.id} />
              )}
            </div>
          )}
        </div>
        <span className="h-5 w-px shrink-0" style={{ backgroundColor: "var(--mc-border)" }} aria-hidden="true" />
        <button
          type="button"
          onClick={mapEditor.undo}
          disabled={mapEditor.busy || !mapEditor.canUndo}
          aria-label="Undo"
          title="Undo (Ctrl+Z)"
          className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
          style={{ borderRadius: 9, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text)" }}
        >
          <MdUndo size={17} aria-hidden="true" />
        </button>
        <button
          type="button"
          onClick={mapEditor.redo}
          disabled={mapEditor.busy || !mapEditor.canRedo}
          aria-label="Redo"
          title="Redo (Ctrl+Shift+Z)"
          className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
          style={{ borderRadius: 9, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text)" }}
        >
          <MdRedo size={17} aria-hidden="true" />
        </button>
        <span className="h-5 w-px shrink-0" style={{ backgroundColor: "var(--mc-border)" }} aria-hidden="true" />
        <button
          type="button"
          onClick={mapEditor.save}
          disabled={mapEditor.busy || !mapEditor.dirty}
          aria-label="Save"
          title="Save map changes"
          className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
          style={{ borderRadius: 9, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text)" }}
        >
          <MdSave size={17} aria-hidden="true" />
        </button>
      </div>

      {mapEditor.selectedPath && (
        <div
          className="absolute top-5 right-5 z-10 flex h-9 items-center gap-1.5 px-3.5 text-[11px] font-mono"
          style={{ borderRadius: 999, backgroundColor: "color-mix(in srgb, var(--mc-surface) 88%, transparent)", border: "1px solid var(--mc-border)", boxShadow: "var(--mc-shadow)", backdropFilter: "blur(8px)", color: "var(--mc-text-muted)" }}
        >
          <span className="max-w-[260px] truncate">{mapEditor.selectedPath}</span>
          {mapEditor.image && (
            <span className="shrink-0" style={{ color: "var(--mc-text-subtle)" }}>
              {mapEditor.image.width} × {mapEditor.image.height}
            </span>
          )}
          {mapEditor.dirty && <span className="shrink-0" style={{ color: "var(--mc-accent)" }}>· unsaved</span>}
        </div>
      )}
    </>
  );
}
