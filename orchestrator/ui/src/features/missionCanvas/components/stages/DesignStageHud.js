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
  MdAdd,
  MdAddLocationAlt,
  MdContentCopy,
  MdDelete,
  MdEdit,
  MdRedo,
  MdRoute,
  MdUndo,
} from "react-icons/md";
import { WaypointOptionButton } from "../primitives";

export default function DesignStageHud({
  busy,
  canRedoDesign,
  canUndoDesign,
  currentMapName,
  designHistoryLocked,
  designMapActive,
  designMapAvailable,
  designMapBusy,
  designMissionLoadError,
  interactionMode,
  mappingRuntimeActive,
  missionName,
  missionNames,
  missionRouteMode,
  missionRunnerActive,
  onCreateSpotAtRobot,
  onDeleteMission,
  onDuplicateMission,
  onMissionChange,
  onNewMission,
  onRedoDesign,
  onRenameMission,
  onSaveMission,
  onToggleMissionRouteMode,
  onToggleSpotMode,
  onToggleWaypointOptions,
  onUndoDesign,
  runRuntimeActive,
  runShutdownPending,
  showWaypointOptions,
}) {
  return (
    <div className="absolute top-5 left-5 z-10 flex flex-col items-start gap-2">
      {/* HUD toolbar — top-left (glass): Create Waypoint + Edit Route.
          z-20 keeps the waypoint options popover above the mission hub
          below (both blur → stacking contexts, so DOM order would
          otherwise paint the hub over the popover). */}
      <div
        className="relative z-20 flex items-center gap-2 p-2"
        style={{ borderRadius: 14, backgroundColor: "color-mix(in srgb, var(--mc-surface) 88%, transparent)", border: "1px solid var(--mc-border)", boxShadow: "var(--mc-shadow)", backdropFilter: "blur(8px)" }}
      >
        <div className="relative">
          <button
            type="button"
            onClick={onToggleWaypointOptions}
            disabled={!designMapAvailable || missionRouteMode}
            aria-label="Create Waypoint"
            aria-pressed={(showWaypointOptions || interactionMode === "spot") ? true : undefined}
            title={missionRouteMode ? "Turn off Edit Route first" : "Add a waypoint"}
            className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
            style={{ borderRadius: 9, border: "none", backgroundColor: (showWaypointOptions || interactionMode === "spot") ? "var(--mc-accent)" : "var(--mc-text)", color: (showWaypointOptions || interactionMode === "spot") ? "var(--mc-accent-fg)" : "var(--mc-bg)" }}
          >
            <MdAddLocationAlt size={17} aria-hidden="true" />
          </button>
          {showWaypointOptions && (
            <div className="absolute left-0 top-[calc(100%+6px)] flex items-center gap-2 p-2" role="menu" aria-label="Waypoint creation options" style={{ borderRadius: 12, backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border-strong)", boxShadow: "var(--mc-shadow)" }}>
              <WaypointOptionButton active={interactionMode === "spot"} disabled={!designMapAvailable || missionRouteMode} onClick={onToggleSpotMode}>On Map</WaypointOptionButton>
              <WaypointOptionButton
                active={interactionMode === "initial" || busy === "At Robot"}
                disabled={!!busy || !designMapAvailable || missionRouteMode || runShutdownPending || mappingRuntimeActive || runRuntimeActive || missionRunnerActive}
                onClick={onCreateSpotAtRobot}
              >
                At Robot
              </WaypointOptionButton>
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onToggleMissionRouteMode}
          disabled={!!busy || !designMapAvailable}
          aria-label="Edit On Map"
          aria-pressed={missionRouteMode ? true : undefined}
          title="Edit the mission route on the map"
          className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
          style={{ borderRadius: 9, border: `1px solid ${missionRouteMode ? "var(--mc-accent)" : "var(--mc-border-strong)"}`, backgroundColor: missionRouteMode ? "var(--mc-accent-soft)" : "var(--mc-surface)", color: "var(--mc-text)" }}
        >
          <MdRoute size={17} aria-hidden="true" />
        </button>
        <span className="h-5 w-px shrink-0" style={{ backgroundColor: "var(--mc-border)" }} aria-hidden="true" />
        <button
          type="button"
          onClick={onUndoDesign}
          disabled={designHistoryLocked || !canUndoDesign}
          aria-label="Undo"
          title="Undo (Ctrl+Z)"
          className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
          style={{ borderRadius: 9, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text)" }}
        >
          <MdUndo size={17} aria-hidden="true" />
        </button>
        <button
          type="button"
          onClick={onRedoDesign}
          disabled={designHistoryLocked || !canRedoDesign}
          aria-label="Redo"
          title="Redo (Ctrl+Shift+Z)"
          className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
          style={{ borderRadius: 9, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text)" }}
        >
          <MdRedo size={17} aria-hidden="true" />
        </button>
      </div>

      {/* Mission hub — create/save/rename/duplicate/delete, right under
          the authoring tools so mission management sits with editing. */}
      {designMapActive && (
        <div
          className="w-[210px] grid gap-1.5 p-2.5"
          style={{ borderRadius: 14, backgroundColor: "color-mix(in srgb, var(--mc-surface) 88%, transparent)", border: "1px solid var(--mc-border)", boxShadow: "var(--mc-shadow)", backdropFilter: "blur(8px)" }}
        >
          {/* The rail session card is gone; the loaded map is named here.
              Mono per the page's data-typography rule (names/paths). */}
          <div className="text-[11px] font-mono truncate" style={{ color: "var(--mc-text-muted)" }}>
            {currentMapName}
          </div>
          <label className="grid gap-1">
            {/* Mono styling stays on the caption span only, so the select
                inherits the page's Hanken Grotesk instead of fighting it. */}
            <span className="text-[10px] font-mono tracking-[0.12em]" style={{ color: "var(--mc-text-subtle)" }}>
              MISSION
            </span>
            <select
              aria-label="Active mission"
              value={missionName}
              disabled={designMapBusy || !!busy}
              onChange={(event) => onMissionChange(event.currentTarget.value)}
              className="w-full h-7 px-2 text-xs font-medium"
              style={{ borderRadius: 8, color: "var(--mc-text)", backgroundColor: "var(--mc-surface-2)", border: "1px solid var(--mc-border-strong)" }}
            >
              {(missionNames.includes(missionName)
                ? missionNames
                : [...missionNames, missionName]
              ).map((name) => (
                <option key={name} value={name}>
                  {missionNames.includes(name) ? name : `${name} (unsaved)`}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            aria-label="Save Mission"
            title={designMissionLoadError
              ? "Reload the mission before saving"
              : "Save Mission"}
            disabled={!!busy || designMapBusy || !!designMissionLoadError}
            onClick={onSaveMission}
            className="w-full h-7 text-[11px] font-semibold disabled:opacity-40"
            style={{ borderRadius: 8, border: "1px solid transparent", backgroundColor: "var(--mc-accent)", color: "var(--mc-accent-fg)" }}
          >
            Save
          </button>
          <div className="flex gap-1.5">
            {[
              {
                label: "New Mission",
                Icon: MdAdd,
                onClick: onNewMission,
                disabled: !!busy || designMapBusy,
              },
              { label: "Rename mission", Icon: MdEdit, onClick: onRenameMission },
              { label: "Duplicate mission", Icon: MdContentCopy, onClick: onDuplicateMission },
              { label: "Delete mission", Icon: MdDelete, onClick: onDeleteMission },
            ].map(({ label, Icon, onClick, disabled }) => (
              <button
                key={label}
                type="button"
                aria-label={label}
                title={label}
                disabled={disabled ?? (
                  !!busy ||
                  designMapBusy ||
                  missionRunnerActive ||
                  (label !== "New Mission" && !designMapAvailable) ||
                  !missionNames.includes(missionName)
                )}
                onClick={onClick}
                className="flex-1 flex items-center justify-center h-7 disabled:opacity-40"
                style={{ borderRadius: 8, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface-2)", color: "var(--mc-text-muted)" }}
              >
                <Icon size={13} />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
