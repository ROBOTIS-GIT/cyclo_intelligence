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
import {
  canonicalLocalBtPathForSpot,
  localBtPathForSpot,
  localBtPathsForSpot,
  localBtSaveAsPath,
  withLocalBtLibrary,
} from "../../../mapWorkspace/lib/missionBtFiles";
import {
  DEFAULT_MAP_NAME,
  DEFAULT_MISSION_NAME,
  missionRequestName,
} from "../../../mapWorkspace/lib/missionNames";

export default function useWaypointTaskSaveAs({ ledger, operations, ui, document, selection, api, history }) {
  const {
    hasActiveSave,
    hasPersistedLocalBtPath,
    getLiveBtFiles,
    getPersistedLocalBtPaths,
    getPersistedRevision,
    checkpointPersistedBtFile,
    recordBtEdit,
  } = ledger;
  const { isBusy, beginOperation, assertDocumentCurrent, finishOperation } = operations;
  const { busy, setSpots } = ui;
  const {
    missionLoadError,
    missionStored,
    saveMissionRef,
    missionName,
    mapName,
    captureDocumentLease,
  } = document;
  const { selectedSpot, defaultPath, selectedDirectory, spots, setSelectedPathBySpotId } = selection;
  const { saveBtFile } = api;
  const { captureHistory } = history;
  const saveXmlAs = useCallback(async (_sourcePath, fileName, content) => {
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (isBusy()) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busy) throw new Error(`${busy} is already in progress`);
    if (missionLoadError) {
      throw new Error("Reload the mission before saving its Waypoint Task");
    }
    if (!selectedSpot) throw new Error("Select a waypoint first");
    if (!missionStored || !hasPersistedLocalBtPath(defaultPath)) {
      const saveMission = saveMissionRef?.current;
      if (typeof saveMission !== "function") {
        throw new Error("Mission save is not ready yet");
      }
      await saveMission(missionName);
      const canonicalPath = canonicalLocalBtPathForSpot(
        selectedSpot,
        selectedDirectory,
      );
      if (!hasPersistedLocalBtPath(canonicalPath)) {
        throw new Error("Failed to register this Waypoint Task. Reload the mission and retry.");
      }
    }

    const targetPath = localBtSaveAsPath(selectedSpot, fileName, selectedDirectory);
    const occupiedPaths = new Set([
      ...spots.flatMap((spot) => localBtPathsForSpot(spot)),
      ...Object.keys(getLiveBtFiles()),
      ...getPersistedLocalBtPaths(),
    ].map((path) => String(path).toLowerCase()));
    if (occupiedPaths.has(targetPath.toLowerCase())) {
      throw new Error(`A Waypoint Task named ${targetPath.split("/").pop()} already exists`);
    }

    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const targetSpotId = selectedSpot.id;
    const documentLease = captureDocumentLease();
    const busyLabel = "Save Waypoint Task as";
    const operation = beginOperation(busyLabel);
    try {
      const response = await saveBtFile(
        targetMapName,
        targetPath,
        content,
        missionRequestName(targetMissionName),
        {
          waypointId: targetSpotId,
          expectedRevision: getPersistedRevision(),
        },
      );
      if (!assertDocumentCurrent(documentLease, targetMapName, targetMissionName)) {
        throw new Error("Mission changed while the Waypoint Task was being saved");
      }

      captureHistory();
      checkpointPersistedBtFile({
        path: targetPath,
        content,
        revision: response?.revision,
        registerLocalPath: true,
        reconcile: false,
      });
      recordBtEdit(targetPath, content, { allowCreate: true });
      setSpots((current) => current.map((spot) => (
        spot.id === targetSpotId
          ? withLocalBtLibrary(
            spot,
            localBtPathForSpot(spot),
            [...localBtPathsForSpot(spot), targetPath],
          )
          : spot
      )));
      setSelectedPathBySpotId((current) => ({
        ...current,
        [targetSpotId]: targetPath,
      }));
      return { ...response, path: targetPath, selected: true };
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
    defaultPath,
    finishOperation,
    getLiveBtFiles,
    getPersistedLocalBtPaths,
    getPersistedRevision,
    hasActiveSave,
    hasPersistedLocalBtPath,
    isBusy,
    mapName,
    missionLoadError,
    missionName,
    missionStored,
    recordBtEdit,
    saveBtFile,
    saveMissionRef,
    selectedDirectory,
    selectedSpot,
    setSelectedPathBySpotId,
    setSpots,
    spots,
  ]);

  return saveXmlAs;
}
