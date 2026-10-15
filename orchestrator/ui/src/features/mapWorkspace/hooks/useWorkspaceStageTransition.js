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

import { workspaceHasStage, STAGE_NAVIGATE, STAGE_RUN, STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN, normalizeWorkspaceStage } from "../lib/stages";

export default function useWorkspaceStageTransition({
  workspace, recordData, designEditing, design, workspaceKind, collectObservations, invalidateRunMapSelection, invalidateNavGoal, runRuntimeActive, missionRunnerActive, navGoalDriving, runCommand, stopActiveNavigationSession
}) {
  const {
    workspaceStage, dataStageBusy, busy, setMessage, invalidateRunPoseRequestRef, setMissionMapLoaded,
    setInteractionMode, setMapEditToolsOpen, setLabelToolsOpen, setWorkspaceStage,
    navigationRuntimeMode, runRuntimeOwned, runShutdownPending, setDataStageBusy,
  } = workspace;
  const { collectEditorBusy, collectDraft, qwenObservation, setRequestedPreviewModel } = recordData;
  const { cancelPendingDesignLocalization, resetMissionRouteEditing } = designEditing;
  const { clearPendingBehaviorPlacement, setShowWaypointOptions, setBtLayerSpotId } = design;
  const handleSelectStageTab = async (requestedStage) => {
    const stageId = normalizeWorkspaceStage(requestedStage);
    if (!workspaceHasStage(workspaceKind, stageId)) return;
    if (stageId === workspaceStage || collectEditorBusy || dataStageBusy || busy) return false;
    if (collectDraft.active || collectDraft.recording || collectDraft.busy) {
      setMessage("Save or discard the current episode before changing tabs.");
      return false;
    }
    if (workspaceStage === STAGE_COLLECT && (collectObservations.draft || collectObservations.planDirty)) {
      setMessage("Save or discard the observation changes before changing tabs.");
      return;
    }

    const applyStageSelection = () => {
      // Run and Navigate share the nav runtime AND the loaded map snapshot:
      // an idle switch between them may retain the loaded map. Active runtime
      // switches are stopped before this cleanup runs.
      const runFamilySwitch = (
        (stageId === STAGE_RUN || stageId === STAGE_NAVIGATE || [STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN].includes(stageId))
        && (workspaceStage === STAGE_RUN || workspaceStage === STAGE_NAVIGATE || [STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN].includes(workspaceStage))
      );
      invalidateRunMapSelection();
      invalidateRunPoseRequestRef.current();
      cancelPendingDesignLocalization();
      if (workspaceStage === STAGE_NAVIGATE) invalidateNavGoal();
      if (!runFamilySwitch) setMissionMapLoaded(false);
      setInteractionMode("view");
      clearPendingBehaviorPlacement();
      setShowWaypointOptions(false);
      resetMissionRouteEditing();
      setBtLayerSpotId("");
      setMapEditToolsOpen(false);
      setLabelToolsOpen(false);
      setWorkspaceStage(stageId);
    };

    const leavingRunFamilyStage = (
      workspaceStage === STAGE_RUN || workspaceStage === STAGE_NAVIGATE || [STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN].includes(workspaceStage)
    );
    const runFamilyRuntimeNeedsStop = (
      navigationRuntimeMode === "run"
      || runRuntimeActive
      || runRuntimeOwned
      || runShutdownPending
      || missionRunnerActive
      || navGoalDriving
    );

    setDataStageBusy(true);
    try {
      if (qwenObservation.needsStop && !(await qwenObservation.stop())) return false;
      if (leavingRunFamilyStage && runFamilyRuntimeNeedsStop) {
        const dataStages = [STAGE_COLLECT, STAGE_LEARN];
        const preserveDatasetMap = dataStages.includes(workspaceStage) && dataStages.includes(stageId);
        const clearRunSnapshot = !preserveDatasetMap && workspaceStage !== STAGE_NAVIGATE;
        let switched = false;
        await runCommand("Stop", async () => {
          const result = await stopActiveNavigationSession({ clearRunSnapshot });
          applyStageSelection();
          switched = true;
          return result;
        });
        return switched;
      }
      applyStageSelection();
      return true;
    } finally { setDataStageBusy(false); }
  };

  const handleTestModel = async (model) => {
    if (!model?.id) return;
    if (workspaceStage === STAGE_COLLECT || await handleSelectStageTab(STAGE_COLLECT)) {
      setRequestedPreviewModel({ ...model });
    }
  };

  return {
    handleSelectStageTab, handleTestModel,
  };
}
