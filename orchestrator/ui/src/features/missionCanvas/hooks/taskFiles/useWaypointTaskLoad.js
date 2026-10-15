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
  DEFAULT_MAP_NAME,
  DEFAULT_MISSION_NAME,
  missionRequestName,
} from "../../../mapWorkspace/lib/missionNames";

export default function useWaypointTaskLoad({ selection, ledger, operations, ui, document, api, history }) {
  const { selectedSpot, filePaths } = selection;
  const {
    hasActiveSave,
    hasPersistedLocalBtPath,
    getLiveBtFiles,
    getPersistedRevision,
    checkpointPersistedBtFile,
    recordBtEdit,
  } = ledger;
  const { isBusy, beginOperation, assertDocumentCurrent, finishOperation } = operations;
  const { busy } = ui;
  const { mapName, missionName, captureDocumentLease } = document;
  const { getBtFile, parseXml } = api;
  const { captureHistory } = history;
  const loadXml = useCallback(async (path) => {
    if (!selectedSpot || !filePaths.includes(path)) {
      throw new Error("This XML does not belong to the selected waypoint");
    }
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (isBusy()) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busy) throw new Error(`${busy} is already in progress`);
    if (!hasPersistedLocalBtPath(path)) {
      const content = getLiveBtFiles()[path];
      if (typeof content !== "string") {
        throw new Error("No Waypoint Task is available at this path");
      }
      return {
        path,
        content,
        exists: true,
        revision: getPersistedRevision(),
      };
    }
    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const documentLease = captureDocumentLease();
    const contentAtLoadStart = getLiveBtFiles()[path];
    const busyLabel = "Load Waypoint Task";
    const operation = beginOperation(busyLabel);
    try {
      const response = await getBtFile(
        targetMapName,
        path,
        missionRequestName(targetMissionName),
      );
      if (!assertDocumentCurrent(documentLease, targetMapName, targetMissionName)) {
        throw new Error("Mission changed while the Waypoint Task was loading");
      }
      if (
        Number.isInteger(response?.revision)
        && response.revision !== getPersistedRevision()
      ) {
        throw new Error("Mission changed in another session. Reload it before editing.");
      }
      if (!response?.exists || typeof response.content !== "string") {
        throw new Error(`No saved XML exists at ${path}`);
      }
      parseXml(response.content);
      const current = getLiveBtFiles();
      if (current[path] !== contentAtLoadStart) {
        throw new Error("Waypoint Task changed while its saved file was loading");
      }
      if (current[path] !== response.content) {
        captureHistory();
        checkpointPersistedBtFile({
          path,
          content: response.content,
          reconcile: false,
        });
        recordBtEdit(path, response.content);
      } else {
        checkpointPersistedBtFile({ path, content: response.content });
      }
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
    getBtFile,
    getLiveBtFiles,
    getPersistedRevision,
    hasActiveSave,
    hasPersistedLocalBtPath,
    isBusy,
    mapName,
    missionName,
    parseXml,
    recordBtEdit,
    selectedSpot,
  ]);

  return loadXml;
}
