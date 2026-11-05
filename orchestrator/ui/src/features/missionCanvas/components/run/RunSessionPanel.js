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

import { MISSION_TEXT, MISSION_TEXT_MUTED } from "../../lib/theme";
import { Panel, SessionRow } from "../primitives";

export const RUNNER_PHASE_LABEL = {
  "nav-sent": "Navigating",
  "awaiting-nav-result": "Navigating",
  arrived: "Arrived",
  "bt-loading": "Starting task",
  "bt-running": "Running task",
  "bt-done": "Waypoint done",
};

export const WAYPOINT_STATE_META = {
  pending: { mark: "○", color: "var(--mc-text-subtle)", note: "" },
  navigating: { mark: "◐", color: "var(--mc-success)", note: "Navigating" },
  "running-bt": { mark: "◑", color: "var(--mc-accent)", note: "Task" },
  done: { mark: "●", color: "var(--mc-success)", note: "" },
  skipped: { mark: "●", color: "var(--mc-text-subtle)", note: "Nav only" },
  failed: { mark: "✕", color: "var(--mc-danger)", note: "Failed" },
};

export default function RunSessionPanel({
  mapName,
  running,
  runner,
  poseReady,
  missionName,
  missionNames,
  missionSelectDisabled,
  onMissionChange,
}) {
  // Runner progress intentionally survives a cancellation internally so it can
  // describe the last outcome. Once the loaded Run snapshot is cleared,
  // however, none of that previous mission belongs in the session panel.
  const hasLoadedSnapshot = Boolean(mapName);
  const showProgress = hasLoadedSnapshot && runner.total > 0;
  const showReason = hasLoadedSnapshot
    && (runner.status === "failed" || runner.status === "cancelled")
    && runner.reason;
  const phaseLabel = RUNNER_PHASE_LABEL[runner.phase];
  return (
    <Panel title="Run Session" className="grid gap-2">
      <SessionRow label="Runtime" value={running ? "Running" : "Idle"} />
      <SessionRow label="Selected map" value={mapName || "Not selected"} />
      {/* Run state already reads from Progress / the reason box, so the
          mission row is just the selector — no separate status line. */}
      <label className="flex items-center justify-between gap-2 text-xs min-w-0">
        <span className="shrink-0" style={{ color: MISSION_TEXT_MUTED }}>Mission</span>
        <select
          aria-label="Active mission"
          value={missionName}
          disabled={missionSelectDisabled || missionNames.length === 0}
          onChange={(event) => onMissionChange(event.currentTarget.value)}
          className="min-w-0 max-w-[11rem] h-7 px-2 text-xs font-medium"
          style={{ borderRadius: 8, color: "var(--mc-text)", backgroundColor: "var(--mc-surface-2)", border: "1px solid var(--mc-border-strong)" }}
        >
          {missionNames.length === 0 && <option value="">Not selected</option>}
          {missionNames.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
      </label>
      {running && (
        <div className="flex items-center justify-between gap-2 text-xs min-w-0">
          <span style={{ color: MISSION_TEXT_MUTED }}>Localization</span>
          <span className="flex items-center gap-1.5 font-mono">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: poseReady ? "var(--mc-success)" : "var(--mc-warning)" }}
            />
            {poseReady ? "Ready" : "Set robot pose"}
          </span>
        </div>
      )}
      {showProgress && (
        <SessionRow
          label="Progress"
          value={
            runner.currentIndex >= 0
              ? `Waypoint ${runner.currentIndex + 1} / ${runner.total}${phaseLabel ? ` · ${phaseLabel}` : ""}`
              : `${runner.total} waypoint${runner.total === 1 ? "" : "s"}`
          }
        />
      )}
      {showReason && (
        <div
          className="text-xs rounded-md px-2.5 py-1.5"
          style={{
            color: runner.status === "failed" ? "var(--mc-danger)" : "var(--mc-warning)",
            backgroundColor: "var(--mc-surface-2)",
          }}
        >
          {runner.reason}
        </div>
      )}
      {showProgress && (
        <ol className="grid gap-1 mt-0.5 max-h-40 overflow-y-auto" aria-label="Mission waypoints">
          {runner.progress.map((entry, index) => {
            const meta = WAYPOINT_STATE_META[entry.state] || WAYPOINT_STATE_META.pending;
            const returnsToStart = (
              index === runner.progress.length - 1
              && index > 0
              && entry.id === runner.progress[0]?.id
            );
            return (
              <li key={`${entry.id}:${index}`} className="flex items-center gap-2 text-xs min-w-0">
                <span className="shrink-0 font-mono" style={{ color: meta.color, width: 14 }}>{meta.mark}</span>
                <span className="shrink-0 tabular-nums" style={{ color: MISSION_TEXT_MUTED, width: 18 }}>{index + 1}</span>
                <span className="truncate flex-1" style={{ color: MISSION_TEXT }}>
                  {returnsToStart ? `Return to ${entry.label}` : entry.label}
                </span>
                {meta.note && (
                  <span className="shrink-0 font-mono text-[10.5px]" style={{ color: meta.color }}>{meta.note}</span>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}
