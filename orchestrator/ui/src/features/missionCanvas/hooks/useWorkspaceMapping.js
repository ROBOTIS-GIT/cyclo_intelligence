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

import { useMemo } from "react";
import { deletePgmMap, getPgmFiles, saveNavigationMap, startNavigation } from "../../../utils/navigationApi";
import { STAGE_MAPPING } from "../../mapWorkspace/lib/stages";
import { saveMissionSession } from "../../mapWorkspace/lib/session";
import useMappingLifecycleController from "./useMappingLifecycleController";

export default function useWorkspaceMapping({
  workspace, design, running, missionRunnerActive, runMapPath, clearLocalizationPoseCache, resetMappingPoseSync, stopActiveNavigationSession, runCommand
}) {
  const {
    runShutdownPending, workspaceStage, mapEditorReloadToken, setWorkspaceStage,
    setNavigationRuntimeMode, setRunRuntimeOwned, setRunShutdownPending, setInteractionMode,
    setMapEditorReloadToken, setMessage,
  } = workspace;
  const { designMapName, setDesignPoseInitialized, invalidateDesignMission, missionName, setMapName } = design;
  const protectedMapPaths = useMemo(() => (
    (running || missionRunnerActive || runShutdownPending) && runMapPath
      ? [runMapPath]
      : []
  ), [missionRunnerActive, runMapPath, runShutdownPending, running]);

  const mappingLifecycle = useMappingLifecycleController({
    active: workspaceStage === STAGE_MAPPING,
    inventoryRefreshToken: mapEditorReloadToken,
    getMapName: () => designMapName,
    runtime: {
      prepareStart: () => {
        setWorkspaceStage(STAGE_MAPPING);
        clearLocalizationPoseCache();
        resetMappingPoseSync();
      },
      start: (targetMapName) => startNavigation("map", targetMapName),
      commitStarted: (targetMapName) => {
        setNavigationRuntimeMode("mapping");
        setDesignPoseInitialized(false);
        setRunRuntimeOwned(false);
        setRunShutdownPending(false);
      },
      persistStarted: (targetMapName) => {
        saveMissionSession({
          mapName: targetMapName,
          workspaceStage: STAGE_MAPPING,
          navigationRuntimeMode: "mapping",
          designPoseInitialized: false,
          runRuntimeOwned: false,
          runShutdownPending: false,
          runShutdownRequestedAt: null,
        });
      },
      stop: () => stopActiveNavigationSession({ clearRunSnapshot: false }),
    },
    document: {
      save: saveNavigationMap,
      commitSavedMap: (targetMapName) => {
        invalidateDesignMission({ mapName: targetMapName, missionName });
        setMapName(targetMapName);
        setWorkspaceStage(STAGE_MAPPING);
        setInteractionMode("view");
        setMapEditorReloadToken((value) => value + 1);
      },
    },
    inventory: {
      list: getPgmFiles,
      remove: deletePgmMap,
      isProtected: (path) => protectedMapPaths.includes(path),
    },
    runCommand,
    onMessage: setMessage,
  });

  const {
    savedMaps,
    start: handleStartMapping,
    stop: handleStopMapping,
    removeSavedMap: handleDeleteSavedMap,
    saveDialog: mappingSaveDialog,
  } = mappingLifecycle;

  const {
    open: showSaveMapDialog,
    name: saveMapName,
    setName: setSaveMapName,
    openDialog: handleOpenSaveMapDialog,
    close: closeSaveMapDialog,
    confirm: handleConfirmSaveMap,
  } = mappingSaveDialog;

  return {
    protectedMapPaths, savedMaps, handleStartMapping, handleStopMapping, handleDeleteSavedMap,
    showSaveMapDialog, saveMapName, setSaveMapName, handleOpenSaveMapDialog, closeSaveMapDialog,
    handleConfirmSaveMap,
  };
}
