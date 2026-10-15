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

import { useEffect, useState } from "react";
import { formatTaskDisplayMessage } from "../../../../utils/taskTerminology";
import { defaultLocalBtXml } from "../../../mapWorkspace/lib/missionBtFiles";
import {
  DEFAULT_MAP_NAME,
  DEFAULT_MISSION_NAME,
  missionRequestName,
} from "../../../mapWorkspace/lib/missionNames";

export default function useWaypointTaskHydration({ selection, ledger, operations, document, api, ui }) {
  const { selectedSpot, selectedPath } = selection;
  const { missionBtFiles, hasActiveSave, getPersistedRevision, replaceLiveBtFiles } = ledger;
  const {
    isBusy,
    hydrationExternallyBlocked,
    beginOperation,
    assertDocumentCurrent,
    finishOperation,
  } = operations;
  const { mapName, missionName, captureDocumentLease } = document;
  const { getBtFile } = api;
  const { onMessage } = ui;
  const [loadingPath, setLoadingPath] = useState("");
  // Hydration is display-only, but it still owns the same operation lease as
  // explicit file actions. A full save must not snapshot fallback XML while
  // the real file is still in flight.
  useEffect(() => {
    if (!selectedSpot || !selectedPath) return undefined;
    if (missionBtFiles?.[selectedPath] !== undefined) return undefined;
    if (
      hasActiveSave()
      || isBusy()
      || hydrationExternallyBlocked
    ) return undefined;
    let cancelled = false;
    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const documentLease = captureDocumentLease();
    const busyLabel = "Load Waypoint Task";
    const operation = beginOperation(busyLabel);
    setLoadingPath(selectedPath);
    getBtFile(
      targetMapName,
      selectedPath,
      missionRequestName(targetMissionName),
    )
      .then((response) => {
        if (
          cancelled
          || !assertDocumentCurrent(documentLease, targetMapName, targetMissionName)
        ) return;
        if (
          Number.isInteger(response?.revision)
          && response.revision !== getPersistedRevision()
        ) {
          onMessage("Mission changed in another session. Reload it before editing.");
          return;
        }
        replaceLiveBtFiles((current) => ({
          ...current,
          [selectedPath]: response?.exists && typeof response.content === "string"
            ? response.content
            : defaultLocalBtXml(selectedSpot),
        }), { bumpBtEpoch: false, reconcileDirty: false });
      })
      .catch((error) => {
        if (
          cancelled
          || !assertDocumentCurrent(documentLease, targetMapName, targetMissionName)
        ) return;
        onMessage(error instanceof Error
          ? `Failed to load ${selectedPath}: ${formatTaskDisplayMessage(error.message, "Waypoint Task")}`
          : `Failed to load ${selectedPath}`);
      })
      .finally(() => {
        if (!cancelled) setLoadingPath("");
        finishOperation(operation, busyLabel);
      });
    return () => {
      cancelled = true;
      setLoadingPath((current) => (current === selectedPath ? "" : current));
      finishOperation(operation, busyLabel);
    };
  }, [
    assertDocumentCurrent,
    beginOperation,
    captureDocumentLease,
    finishOperation,
    getBtFile,
    getPersistedRevision,
    hasActiveSave,
    hydrationExternallyBlocked,
    isBusy,
    mapName,
    missionBtFiles,
    missionName,
    onMessage,
    replaceLiveBtFiles,
    selectedPath,
    selectedSpot,
  ]);

  return loadingPath;
}
