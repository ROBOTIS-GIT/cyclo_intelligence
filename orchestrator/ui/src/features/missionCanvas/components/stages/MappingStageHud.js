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

import { MdSave, MdStop } from "react-icons/md";
import MapDeleteControl from "../mapping/MapDeleteControl";

// Save Map / Stop float over the map as icon buttons while recording;
// Start Mapping remains a session-level action in the workspace header.
export default function MappingStageHud({
  busy,
  mappingRuntimeActive,
  showSaveMapDialog,
  savedMaps,
  protectedPaths,
  onOpenSaveMapDialog,
  onStopNavigation,
  onDeleteSavedMap,
  dialogHost,
}) {
  return (
    <div
      className="absolute top-5 left-5 z-10 flex items-center gap-2 p-2"
      style={{ borderRadius: 14, backgroundColor: "color-mix(in srgb, var(--mc-surface) 88%, transparent)", border: "1px solid var(--mc-border)", boxShadow: "var(--mc-shadow)", backdropFilter: "blur(8px)" }}
    >
      <button
        type="button"
        onClick={onOpenSaveMapDialog}
        disabled={!!busy || !mappingRuntimeActive}
        aria-label="Save Map"
        aria-pressed={(showSaveMapDialog || busy === "Save map") ? true : undefined}
        title="Save the mapped floor plan"
        className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
        style={{
          borderRadius: 9,
          border: `1px solid ${(showSaveMapDialog || busy === "Save map") ? "var(--mc-accent)" : "var(--mc-border-strong)"}`,
          backgroundColor: (showSaveMapDialog || busy === "Save map") ? "var(--mc-accent-soft)" : "var(--mc-surface)",
          color: "var(--mc-text)",
        }}
      >
        <MdSave size={17} aria-hidden="true" />
      </button>
      <button
        type="button"
        onClick={onStopNavigation}
        disabled={!!busy || !mappingRuntimeActive}
        aria-label="Stop"
        aria-pressed={busy === "Stop" ? true : undefined}
        title="Stop mapping"
        className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
        style={{
          borderRadius: 9,
          border: "1px solid var(--mc-danger-border)",
          backgroundColor: busy === "Stop" ? "var(--mc-danger)" : "var(--mc-surface)",
          color: busy === "Stop" ? "var(--mc-accent-fg)" : "var(--mc-danger)",
        }}
      >
        <MdStop size={18} aria-hidden="true" />
      </button>
      <span className="h-5 w-px shrink-0" style={{ backgroundColor: "var(--mc-border)" }} aria-hidden="true" />
      <MapDeleteControl
        files={savedMaps}
        disabled={!!busy}
        protectedPaths={protectedPaths}
        onDelete={onDeleteSavedMap}
        dialogHost={dialogHost}
      />
    </div>
  );
}
