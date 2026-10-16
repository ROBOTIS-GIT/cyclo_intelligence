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

import { STAGE_NAVIGATE, STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN } from "../lib/stages";


// Presentational mapping only; execution state stays in its owning controller.
export default function buildWorkspaceDialogs({ mapping, workspace, designEditing, design, runtime }) {
  const {
    showSaveMapDialog, saveMapName, setSaveMapName, closeSaveMapDialog, handleConfirmSaveMap
  } = mapping;
  const {
    busy
  } = workspace;
  const {
    showSaveMissionDialog, saveMissionName, setSaveMissionName, closeSaveMissionDialog, handleConfirmSaveMission, showUnsavedDialog, unsavedSavePending, resolveUnsavedDialog, cancelUnsavedDialog, runGuardedDesignAction
  } = designEditing;
  const {
    designCatalog, missionName, showRenameMissionDialog, renameMissionName, setRenameMissionName, closeRenameMissionDialog, handleConfirmRenameMission, showDuplicateMissionDialog, duplicateMissionName, setDuplicateMissionName, closeDuplicateMissionDialog, handleConfirmDuplicateMission, showDeleteMissionDialog, handleConfirmDeleteMission, closeDeleteMissionDialog, showDesignMapDialog, designMapFiles, pendingDesignMapPath, designMissionNames, pendingDesignMissionName, designMapBusy, designDialogCatalogReady, handleDesignMapChange, setPendingDesignMissionName, cancelDesignMapDialog, handleConfirmDesignMap
  } = design;
  const {
    showRunMapDialog, runMapFiles, runMapPath, runMapDialogStage, runMissionNames, pendingRunMissionName, runMapBusy, handleRunMapChange, setPendingRunMissionName, cancelRunMapDialog, handleConfirmRunMap, showEditMapDialog, mapEditor, pendingEditMapPath, setPendingEditMapPath, cancelEditMapDialog, handleConfirmEditMap
  } = runtime;
  return {
    saveMap: {
          open: showSaveMapDialog,
          value: saveMapName,
          busy: !!busy,
          onChange: setSaveMapName,
          onCancel: closeSaveMapDialog,
          onSubmit: handleConfirmSaveMap,
        },
    saveMission: {
          open: showSaveMissionDialog,
          value: saveMissionName,
          existingNames: designCatalog.names,
          currentName: missionName,
          busy: !!busy,
          onChange: setSaveMissionName,
          onCancel: closeSaveMissionDialog,
          onSubmit: handleConfirmSaveMission,
        },
    renameMission: {
          open: showRenameMissionDialog,
          value: renameMissionName,
          existingNames: designCatalog.names.filter((name) => name !== missionName),
          busy: !!busy,
          onChange: setRenameMissionName,
          onCancel: closeRenameMissionDialog,
          onSubmit: handleConfirmRenameMission,
        },
    duplicateMission: {
          open: showDuplicateMissionDialog,
          value: duplicateMissionName,
          existingNames: designCatalog.names,
          busy: !!busy,
          onChange: setDuplicateMissionName,
          onCancel: closeDuplicateMissionDialog,
          onSubmit: handleConfirmDuplicateMission,
        },
    deleteMission: {
          open: showDeleteMissionDialog,
          body: `Delete mission "${missionName}"? This permanently removes its waypoints, route, and Waypoint Tasks.`,
          busy: !!busy,
          onConfirm: handleConfirmDeleteMission,
          onCancel: closeDeleteMissionDialog,
        },
    unsaved: {
          open: showUnsavedDialog,
          body: `"${missionName}" has unsaved changes.`,
          altLabel: designCatalog.names.includes(missionName) ? "Save & continue" : "",
          hint: designCatalog.names.includes(missionName)
            ? ""
            : "Use Save Mission to name this mission first.",
          busy: !!busy || unsavedSavePending,
          onConfirm: () => resolveUnsavedDialog("discard"),
          onAlt: () => resolveUnsavedDialog("save"),
          onCancel: cancelUnsavedDialog,
        },
    designLoad: {
          open: showDesignMapDialog && !showUnsavedDialog,
          files: designMapFiles,
          selectedPath: pendingDesignMapPath,
          missionNames: designMissionNames,
          selectedMissionName: pendingDesignMissionName,
          busy: designMapBusy,
          catalogReady: designDialogCatalogReady,
          onChange: handleDesignMapChange,
          onMissionChange: setPendingDesignMissionName,
          onCancel: cancelDesignMapDialog,
          onSubmit: () => runGuardedDesignAction(handleConfirmDesignMap),
        },
    runLoad: {
          open: showRunMapDialog,
          files: runMapFiles,
          selectedPath: runMapPath,
          missionNames: [STAGE_NAVIGATE, STAGE_OPERATE, STAGE_COLLECT].includes(runMapDialogStage) ? null : runMissionNames,
          selectedMissionName: pendingRunMissionName,
          busy: runMapBusy,
          navigationOnly: runMapDialogStage === STAGE_NAVIGATE,
          autonomyOnly: [STAGE_OPERATE, STAGE_COLLECT, STAGE_LEARN].includes(runMapDialogStage),
          onChange: handleRunMapChange,
          onMissionChange: setPendingRunMissionName,
          onCancel: cancelRunMapDialog,
          onSubmit: handleConfirmRunMap,
        },
    editLoad: {
          open: showEditMapDialog,
          files: mapEditor.files,
          selectedPath: pendingEditMapPath,
          busy: mapEditor.busy,
          onChange: setPendingEditMapPath,
          onCancel: cancelEditMapDialog,
          onSubmit: handleConfirmEditMap,
        }
  };
}
