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
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_MAP_EDIT,
  STAGE_RUN,
  WORKSPACE_STAGES,
} from "../../lib/stages";
import {
  MISSION_BORDER,
  MISSION_STAGE_EMPTY,
  MISSION_SURFACE,
  MISSION_TEXT,
  MISSION_TEXT_MUTED,
} from "../../lib/theme";
import { ActionButton } from "../primitives";

export default function StageHeader({
  btNodeBusy,
  busy,
  designMapBusy,
  mapEditorBusy,
  mappingRuntimeActive,
  missionRunnerActive,
  onBackToDesignMap,
  onOpenDesignMap,
  onOpenEditMap,
  onOpenRunMap,
  onStartMapping,
  runBtLayer,
  runCurrentIndex,
  runFamilyStage,
  runMapBusy,
  runRuntimeActive,
  runShutdownPending,
  runTotal,
  running,
  showDesignMapDialog,
  showEditMapDialog,
  showRunMapDialog,
  waypointBtLayer,
  workspaceStage,
}) {
  return (
    <header
      className="shrink-0 h-14 flex items-center justify-between gap-4 px-6 border-b"
      style={{ borderColor: MISSION_BORDER, backgroundColor: MISSION_SURFACE }}
    >
      {workspaceStage === STAGE_AUTHORING && waypointBtLayer ? (
        <>
          <div className="flex items-center gap-2.5 min-w-0 text-[14px]">
            <span className="font-bold tracking-tight">Design</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--mc-text-subtle)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m9 18 6-6-6-6" /></svg>
            <span className="font-semibold truncate" style={{ color: MISSION_TEXT_MUTED }}>{waypointBtLayer.spot.label || waypointBtLayer.spot.id}</span>
            <span className="text-[11px] font-mono shrink-0" style={{ color: "var(--mc-text-subtle)" }}>· Waypoint Task</span>
          </div>
          <ActionButton
            onClick={onBackToDesignMap}
            title="Return to the Design map"
            variant="secondary"
          >
            ← Back to Map
          </ActionButton>
        </>
      ) : workspaceStage === STAGE_RUN && runBtLayer ? (
        <>
          <div className="flex items-center gap-2.5 min-w-0 text-[14px]">
            <span className="font-bold tracking-tight">Run</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--mc-text-subtle)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m9 18 6-6-6-6" /></svg>
            <span className="font-semibold truncate" style={{ color: MISSION_TEXT_MUTED }}>{runBtLayer.spot.label || runBtLayer.spot.id}</span>
            <span className="text-[11px] font-mono shrink-0" style={{ color: "var(--mc-text-subtle)" }}>· Waypoint {runCurrentIndex + 1} / {runTotal}</span>
          </div>
          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-2 px-3 py-1.5" style={{ borderRadius: 999, backgroundColor: "color-mix(in srgb, var(--mc-success) 14%, transparent)", border: "1px solid color-mix(in srgb, var(--mc-success) 35%, transparent)" }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: "var(--mc-success)" }} />
              <span className="text-[12px] font-semibold" style={{ color: "var(--mc-success)" }}>Task running</span>
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="flex items-center gap-3 min-w-0">
            <span className="text-[16px] font-bold tracking-tight" style={{ color: MISSION_TEXT }}>
              {WORKSPACE_STAGES.find((stage) => stage.id === workspaceStage)?.label}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {workspaceStage === STAGE_MAPPING && (
              // Stop / Save Map live on the map canvas as the mapping HUD;
              // the header keeps only the session-level Start Mapping.
              <ActionButton
                active={busy === "Mapping" || mappingRuntimeActive}
                disabled={!!busy || mappingRuntimeActive || runRuntimeActive || runShutdownPending}
                onClick={onStartMapping}
                variant="secondary"
              >
                Start Mapping
              </ActionButton>
            )}
            {workspaceStage === STAGE_MAP_EDIT && (
              <ActionButton
                active={showEditMapDialog || mapEditorBusy}
                disabled={!!busy || mapEditorBusy}
                onClick={onOpenEditMap}
                variant="secondary"
              >
                Load Map
              </ActionButton>
            )}
            {workspaceStage === STAGE_AUTHORING && (
              <ActionButton
                active={showDesignMapDialog || designMapBusy}
                disabled={!!busy || designMapBusy}
                onClick={onOpenDesignMap}
                variant="secondary"
              >
                Load Map
              </ActionButton>
            )}
            {runFamilyStage && (
              <ActionButton
                active={showRunMapDialog || runMapBusy}
                disabled={!!busy || running || missionRunnerActive || !!btNodeBusy || runMapBusy || runShutdownPending}
                onClick={onOpenRunMap}
                variant="secondary"
              >
                Load Map
              </ActionButton>
            )}
            <div
              className="h-9 flex items-center gap-2 px-3 border shrink-0"
              style={{ borderRadius: 999, borderColor: MISSION_BORDER, backgroundColor: MISSION_STAGE_EMPTY }}
            >
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: running ? "var(--mc-success)" : "var(--mc-text-subtle)" }}
                title={running ? "Navigation running" : "Navigation idle"}
                aria-label={running ? "Navigation running" : "Navigation idle"}
              />
              <span className="text-[12px] font-semibold whitespace-nowrap" style={{ color: running ? "var(--mc-success)" : MISSION_TEXT_MUTED }}>Status: {running ? "running" : "idle"}</span>
            </div>
          </div>
        </>
      )}
    </header>
  );
}
