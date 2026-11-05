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

import { MdMyLocation, MdPlayArrow, MdStop } from "react-icons/md";

import { TopicStatusPanel } from "../mapChrome";
import RunSessionPanel from "../run/RunSessionPanel";

export function RunStageHud({
  busy,
  btNodeBusy,
  runMapBusy,
  missionMapLoaded,
  runMapSnapshotInvalid,
  runShutdownPending,
  interactionMode,
  onLocalize,
  runPoseInitialized,
  missionRunnerActive,
  onRunMission,
  onStopNavigation,
  missionRunnerStopping,
  running,
}) {
  return (
    <div
      className="absolute top-5 left-5 z-10 flex items-center gap-2 p-2"
      style={{ borderRadius: 14, backgroundColor: "color-mix(in srgb, var(--mc-surface) 88%, transparent)", border: "1px solid var(--mc-border)", boxShadow: "var(--mc-shadow)", backdropFilter: "blur(8px)" }}
    >
      <button
        type="button"
        onClick={onLocalize}
        disabled={!!busy || runMapBusy || !missionMapLoaded || runMapSnapshotInvalid || runShutdownPending}
        aria-label="Localize"
        aria-pressed={(interactionMode === "initial" || busy === "Localize" || busy === "Set robot pose") ? true : undefined}
        title="Bring navigation up and set the robot pose"
        className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
        style={{
          borderRadius: 9,
          border: `1px solid ${(interactionMode === "initial" || busy === "Localize" || busy === "Set robot pose") ? "var(--mc-accent)" : "var(--mc-border-strong)"}`,
          backgroundColor: (interactionMode === "initial" || busy === "Localize" || busy === "Set robot pose") ? "var(--mc-accent-soft)" : "var(--mc-surface)",
          color: "var(--mc-text)",
        }}
      >
        <MdMyLocation size={17} aria-hidden="true" />
      </button>
      <button
        type="button"
        onClick={onRunMission}
        disabled={!!busy || !!btNodeBusy || runMapBusy || !missionMapLoaded || runMapSnapshotInvalid || !runPoseInitialized || missionRunnerActive || runShutdownPending}
        aria-label="Run Mission"
        aria-pressed={(busy === "Run mission" || missionRunnerActive) ? true : undefined}
        title="Run the mission route"
        className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
        style={{ borderRadius: 9, border: "none", backgroundColor: "var(--mc-success)", color: "var(--mc-accent-fg)" }}
      >
        <MdPlayArrow size={19} aria-hidden="true" />
      </button>
      <button
        type="button"
        onClick={onStopNavigation}
        disabled={!!busy || missionRunnerStopping || (!running && !missionRunnerActive && !runShutdownPending)}
        aria-label="Stop"
        aria-pressed={(busy === "Stop" || runShutdownPending) ? true : undefined}
        title="Stop the mission and navigation"
        className="h-8 w-8 inline-flex items-center justify-center disabled:opacity-45"
        style={{
          borderRadius: 9,
          border: "1px solid var(--mc-danger-border)",
          backgroundColor: (busy === "Stop" || runShutdownPending) ? "var(--mc-danger)" : "var(--mc-surface)",
          color: (busy === "Stop" || runShutdownPending) ? "var(--mc-accent-fg)" : "var(--mc-danger)",
        }}
      >
        <MdStop size={18} aria-hidden="true" />
      </button>
    </div>
  );
}

export function RunStageSidebar({
  mapName,
  running,
  runner,
  poseReady,
  missionName,
  missionNames,
  missionSelectDisabled,
  onMissionChange,
  topicRows,
}) {
  return (
    <>
      <RunSessionPanel
        mapName={mapName}
        running={running}
        runner={runner}
        poseReady={poseReady}
        missionName={missionName}
        missionNames={missionNames}
        missionSelectDisabled={missionSelectDisabled}
        onMissionChange={onMissionChange}
      />
      <TopicStatusPanel topicRows={topicRows} />
    </>
  );
}
