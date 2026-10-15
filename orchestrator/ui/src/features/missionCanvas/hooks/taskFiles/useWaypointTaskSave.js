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

import { useCallback } from "react";
import { canonicalLocalBtPathForSpot } from "../../../mapWorkspace/lib/missionBtFiles";
import {
  DEFAULT_MAP_NAME,
  DEFAULT_MISSION_NAME,
  missionRequestName,
} from "../../../mapWorkspace/lib/missionNames";

export default function useWaypointTaskSave({ selection, ledger, operations, ui, document, history, api }) {
  const { selectedSpot, filePaths, selectedDirectory } = selection;
  const {
    hasActiveSave,
    getLiveBtFiles,
    recordBtEdit,
    hasPersistedLocalBtPath,
    getPersistedRevision,
    checkpointPersistedBtFile,
  } = ledger;
  const { isBusy, beginOperation, assertDocumentCurrent, finishOperation } = operations;
  const { busy } = ui;
  const { missionLoadError, saveMissionRef, missionName, mapName, captureDocumentLease } = document;
  const { captureHistory } = history;
  const { saveBtFile } = api;
  const saveXml = useCallback(async (path, content) => {
    if (!selectedSpot || !filePaths.includes(path)) {
      throw new Error("This XML does not belong to the selected waypoint");
    }
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (isBusy()) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busy) throw new Error(`${busy} is already in progress`);
    if (missionLoadError) {
      throw new Error("Reload the mission before saving its Waypoint Task");
    }
    if (getLiveBtFiles()[path] !== content) {
      captureHistory();
      recordBtEdit(path, content);
    }
    if (!hasPersistedLocalBtPath(path)) {
      const saveMission = saveMissionRef?.current;
      if (typeof saveMission !== "function") {
        throw new Error("Mission save is not ready yet");
      }
      await saveMission(missionName);
      const canonicalPath = canonicalLocalBtPathForSpot(
        selectedSpot,
        selectedDirectory,
      );
      const savedPath = hasPersistedLocalBtPath(path) ? path : canonicalPath;
      if (!hasPersistedLocalBtPath(savedPath)) {
        throw new Error("Failed to register this Waypoint Task. Reload the mission and retry.");
      }
      return {
        path: savedPath,
        content,
        exists: true,
        revision: getPersistedRevision(),
      };
    }
    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const documentLease = captureDocumentLease();
    const busyLabel = "Save Waypoint Task";
    const operation = beginOperation(busyLabel);
    try {
      const response = await saveBtFile(
        targetMapName,
        path,
        content,
        missionRequestName(targetMissionName),
        {
          waypointId: selectedSpot.id,
          expectedRevision: getPersistedRevision(),
        },
      );
      if (!assertDocumentCurrent(documentLease, targetMapName, targetMissionName)) {
        return response;
      }
      checkpointPersistedBtFile({
        path,
        content,
        revision: response?.revision,
      });
      return response;
    } finally {
      finishOperation(operation, busyLabel);
    }
  }, [
    assertDocumentCurrent,
    beginOperation,
    busy,
    captureDocumentLease,
    captureHistory,
    checkpointPersistedBtFile,
    filePaths,
    finishOperation,
    getLiveBtFiles,
    getPersistedRevision,
    hasActiveSave,
    hasPersistedLocalBtPath,
    isBusy,
    mapName,
    missionLoadError,
    missionName,
    recordBtEdit,
    saveBtFile,
    saveMissionRef,
    selectedDirectory,
    selectedSpot,
  ]);

  return saveXml;
}
