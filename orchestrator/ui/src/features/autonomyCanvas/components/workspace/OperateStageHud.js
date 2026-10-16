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

import OperateModelPicker from "../OperateModelPicker";
import RuntimeControlsHud from "../../../mapWorkspace/components/RuntimeControlsHud";

export default function OperateStageHud({ observation, busy, btNodeBusy, runMapBusy, missionMapLoaded,
  runShutdownPending, interactionMode, onLocalize, runPoseInitialized, missionRunnerActive,
  onStopNavigation, missionRunnerStopping, running }) {
  return <RuntimeControlsHud
    leading={<OperateModelPicker choice={observation?.modelChoice} active={observation?.active} />}
    localize={{
      onClick: onLocalize,
      disabled: !!busy || missionRunnerActive || runMapBusy || !missionMapLoaded || runShutdownPending,
      pressed: interactionMode === "initial" || busy === "Localize" || busy === "Set robot pose",
    }}
    start={{
      onClick: observation?.start,
      disabled: !!busy || !!btNodeBusy || runMapBusy || !missionMapLoaded || !runPoseInitialized || !running
        || observation?.canStart === false || observation?.active || observation?.phase === "stopping" || missionRunnerActive || runShutdownPending,
      pressed: busy === "Run mission" || missionRunnerActive,
      label: "Run autonomy", title: "Start reasoning and mission execution",
    }}
    stop={{
      onClick: onStopNavigation,
      disabled: !!busy || missionRunnerStopping || observation?.phase === "stopping"
        || (!running && !missionRunnerActive && !runShutdownPending && !observation?.needsStop),
      pressed: busy === "Stop" || runShutdownPending,
      title: "Stop Qwen container and navigation",
    }}
  />;
}
