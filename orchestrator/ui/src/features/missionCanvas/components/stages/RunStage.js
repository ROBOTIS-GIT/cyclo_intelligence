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

import { TopicStatusPanel } from "../../../mapWorkspace/components/mapChrome";
import RuntimeControlsHud from "../../../mapWorkspace/components/RuntimeControlsHud";
import RunTaskParameters from "../run/RunTaskParameters";
import RunSessionPanel from "../run/RunSessionPanel";

export function RunStageHud({ busy, btNodeBusy, runMapBusy, missionMapLoaded, runMapSnapshotInvalid,
  runShutdownPending, interactionMode, onLocalize, runPoseInitialized, missionRunnerActive,
  onRunMission, onStopNavigation, missionRunnerStopping, running }) {
  return <RuntimeControlsHud
    localize={{
      onClick: onLocalize,
      disabled: !!busy || missionRunnerActive || runMapBusy || !missionMapLoaded || runMapSnapshotInvalid || runShutdownPending,
      pressed: interactionMode === "initial" || busy === "Localize" || busy === "Set robot pose",
    }}
    start={{
      onClick: onRunMission,
      disabled: !!busy || !!btNodeBusy || runMapBusy || !missionMapLoaded || runMapSnapshotInvalid || !runPoseInitialized || missionRunnerActive || runShutdownPending,
      pressed: busy === "Run mission" || missionRunnerActive,
      label: "Run Mission", title: "Run the mission route",
    }}
    stop={{
      onClick: onStopNavigation,
      disabled: !!busy || missionRunnerStopping || (!running && !missionRunnerActive && !runShutdownPending),
      pressed: busy === "Stop" || runShutdownPending,
      title: "Stop the mission and navigation",
    }}
  />;
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
  onTaskParametersSaved,
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
        taskParameters={<RunTaskParameters mapName={mapName} missionName={missionName} disabled={missionSelectDisabled} onSaved={onTaskParametersSaved} />}
      />
      <TopicStatusPanel topicRows={topicRows} />
    </>
  );
}
