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

import { getPgmFiles } from "../../../utils/navigationApi";
import { STAGE_COLLECT } from "../lib/stages";
import useRunMapMissionSelectionController from "./useRunMapMissionSelectionController";

export default function useWorkspaceMapSelection({
  workspace, design, runMissionSnapshot, loadRunMissionForMap, clearRunMissionSnapshot, invalidateRunMissionLoad, cancelAndClearRunMissionSnapshot, invalidateNavGoal, collectExecution, runtimeMapName, collectObservations
}) {
  const {
    workspaceStageRef, setWorkspaceStage, setInteractionMode, setMissionMapLoaded,
    invalidateRunPoseRequestRef, setRunPoseInitialized, missionMapLoaded, setMessage,
  } = workspace;
  const { mapName, missionName, fetchMissionNames } = design;
  const runMapMissionSelection = useRunMapMissionSelectionController({
    getCurrentStage: () => workspaceStageRef.current,
    getDefaults: () => ({ mapName, missionName }),
    inventory: {
      listMaps: getPgmFiles,
      listMissions: fetchMissionNames,
    },
    snapshot: {
      get: () => runMissionSnapshot,
      load: loadRunMissionForMap,
      clear: clearRunMissionSnapshot,
      invalidate: invalidateRunMissionLoad,
      cancelAndClear: cancelAndClearRunMissionSnapshot,
    },
    commits: {
      setStage: setWorkspaceStage,
      setInteractionMode,
      setMapLoaded: setMissionMapLoaded,
      resetPose: () => {
        invalidateRunPoseRequestRef.current();
        setRunPoseInitialized(false);
      },
      invalidateGoal: invalidateNavGoal,
      onMapLoaded: ({ stage, mapName: loadedMapName }) => {
        if (stage !== STAGE_COLLECT) return;
        collectExecution.resetIdle();
        if (missionMapLoaded && runtimeMapName === loadedMapName) collectObservations.resetSession();
      },
    },
    onMessage: setMessage,
  });

  const {
    open: showRunMapDialog,
    stage: runMapDialogStage,
    files: runMapFiles,
    missionNames: runMapDialogMissionNames,
    selectedPath: runMapPath,
    selectedMission: pendingRunMissionName,
    setSelectedMission: setPendingRunMissionName,
    busy: runMapBusy,
    openDialog: handleOpenRunMapDialog,
    changeMap: handleRunMapChange,
    cancel: cancelRunMapDialog,
    confirm: handleConfirmRunMap,
  } = runMapMissionSelection.dialog;

  const runMissionNames = runMapDialogMissionNames || [];

  const handleMissionChange = runMapMissionSelection.switchMission;

  const invalidateRunMapSelection = runMapMissionSelection.invalidate;

  return {
    showRunMapDialog, runMapDialogStage, runMapFiles, runMapPath, pendingRunMissionName,
    setPendingRunMissionName, runMapBusy, handleOpenRunMapDialog, handleRunMapChange,
    cancelRunMapDialog, handleConfirmRunMap, runMissionNames, handleMissionChange,
    invalidateRunMapSelection,
  };
}
