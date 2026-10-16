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

import { STAGE_LEARN } from "../lib/stages";


// Presentational mapping only; execution state stays in its owning controller.
export default function buildWorkspaceRail({ runtime, recordData, workspace }) {
  const {
    railFooter, workspaceKind, missionRunnerActive, collectWaypoints, mappingRuntimeActive, handleSelectStageTab
  } = runtime;
  const {
    collectEditorBusy, collectDraft
  } = recordData;
  const {
    dataStageBusy, busy, workspaceStage
  } = workspace;
  return {
    footer: railFooter,
    workspaceKind: workspaceKind,
    interactionLocked: collectEditorBusy || dataStageBusy,
    disabledStages: (collectDraft.active || collectDraft.recording || collectDraft.busy || missionRunnerActive || collectWaypoints.moving) ? [STAGE_LEARN] : [],
    busy: busy,
    mappingRuntimeActive: mappingRuntimeActive,
    onSelectStage: handleSelectStageTab,
    workspaceStage: workspaceStage
  };
}
