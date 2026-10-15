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
import { localBtPathsForSpot, withLocalBtLibrary } from "../../../mapWorkspace/lib/missionBtFiles";
import {
  DEFAULT_MAP_NAME,
  DEFAULT_MISSION_NAME,
  missionRequestName,
} from "../../../mapWorkspace/lib/missionNames";

export default function useWaypointTaskDefault({ selection, ledger, operations, ui, document, api, history }) {
  const { selectedSpot, filePaths, defaultPath } = selection;
  const {
    hasActiveSave,
    hasPersistedLocalBtPath,
    getPersistedRevision,
    setPersistedRevision,
    reconcileDirty,
  } = ledger;
  const { isBusy, beginOperation, assertDocumentCurrent, finishOperation } = operations;
  const { busy, setSpots, onMessage } = ui;
  const { missionLoadError, missionStored, mapName, missionName, captureDocumentLease } = document;
  const { setDefaultBtFile } = api;
  const { resetHistory } = history;
  const setDefaultXml = useCallback(async (path) => {
    if (!selectedSpot || !filePaths.includes(path)) {
      throw new Error("This XML does not belong to the selected waypoint");
    }
    if (path === defaultPath) return undefined;
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (isBusy()) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busy) throw new Error(`${busy} is already in progress`);
    if (missionLoadError) {
      throw new Error("Reload the mission before changing its default Waypoint Task");
    }
    if (!missionStored || !hasPersistedLocalBtPath(path)) {
      throw new Error("Save Mission before changing its default Waypoint Task");
    }

    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const targetSpotId = selectedSpot.id;
    const documentLease = captureDocumentLease();
    const busyLabel = "Set default Waypoint Task";
    const operation = beginOperation(busyLabel);
    try {
      const response = await setDefaultBtFile(
        targetMapName,
        targetSpotId,
        path,
        missionRequestName(targetMissionName),
        { expectedRevision: getPersistedRevision() },
      );
      if (!assertDocumentCurrent(documentLease, targetMapName, targetMissionName)) {
        throw new Error("Mission changed while its default Waypoint Task was being updated");
      }
      setSpots((current) => current.map((spot) => (
        spot.id === targetSpotId
          ? withLocalBtLibrary(spot, path, localBtPathsForSpot(spot))
          : spot
      )));
      setPersistedRevision(response?.revision);
      resetHistory();
      reconcileDirty();
      onMessage(`${path.split("/").pop()} set as the default Waypoint Task`);
      return response;
    } finally {
      finishOperation(operation, busyLabel);
    }
  }, [
    assertDocumentCurrent,
    beginOperation,
    busy,
    captureDocumentLease,
    defaultPath,
    filePaths,
    finishOperation,
    getPersistedRevision,
    hasActiveSave,
    hasPersistedLocalBtPath,
    isBusy,
    mapName,
    missionLoadError,
    missionName,
    missionStored,
    onMessage,
    reconcileDirty,
    resetHistory,
    selectedSpot,
    setDefaultBtFile,
    setPersistedRevision,
    setSpots,
  ]);

  return setDefaultXml;
}
