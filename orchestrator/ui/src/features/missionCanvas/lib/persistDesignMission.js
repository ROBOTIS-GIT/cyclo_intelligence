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

import {
  deleteNavigationMissionBtFile,
  getNavigationMission,
  saveNavigationMission,
  saveNavigationMissionBtFile,
} from "../../../utils/navigationMissionsApi";
import {
  assembleMissionBtFilesForSave,
  buildGlobalMissionXml,
  canonicalLocalBtPathForSpot,
  canonicalLocalBtPathsForSpot,
  localBtDirectoriesForSpots,
  localBtPathsForSpot,
  migrateCanonicalLocalBtFileKeys,
  withLocalBtLibrary,
} from "./missionBtFiles";
import {
  filterMissionFlowEdges,
  missionStepSpotsFromMissionFlow,
  serializeMissionFlow,
  syncMissionFlowNodesWithSpots,
} from "./missionFlow";
import { missionRequestName } from "./missionNames";
import { missionWaypointsFromSpots } from "./missionSpots";
import { saveBehaviorNodesForMap } from "./designStore";

function historyObject(historyAtStart) {
  if (typeof historyAtStart === "string") return JSON.parse(historyAtStart);
  return historyAtStart && typeof historyAtStart === "object"
    ? { ...historyAtStart }
    : {};
}

function requireLedger(ledger) {
  const required = [
    "beginSave",
    "checkpointSaveUpload",
    "checkpointSaveManifest",
    "checkpointSaveCleanup",
    "reconcileSave",
    "abortSave",
    "setPersistedRevision",
  ];
  required.forEach((name) => {
    if (typeof ledger?.[name] !== "function") {
      throw new Error(`Design mission ledger is missing ${name}`);
    }
  });
  return ledger;
}

// Executes the durable portion of Save Mission. React/UI state intentionally
// stays with MissionCanvasWorkspace: this function returns the canonical
// document that was persisted and the ledger's newer-edit reconciliation.
//
// Each successful request is checkpointed before the next await. If a later
// request fails, abortSave releases only the transaction lock; the ledger keeps
// those revision/baseline checkpoints so a retry continues from server truth.
export default async function persistDesignMission({
  mapName,
  targetMissionName,
  targetKnown,
  visibleSpots = [],
  behaviorNodes = [],
  missionFlowNodes = [],
  missionFlowEdges = [],
  historyAtStart = {},
  ledger,
}, {
  getMission = getNavigationMission,
  saveBtFile = saveNavigationMissionBtFile,
  saveMission = saveNavigationMission,
  deleteBtFile = deleteNavigationMissionBtFile,
  saveBehaviorNodes = saveBehaviorNodesForMap,
} = {}) {
  const designLedger = requireLedger(ledger);
  let transaction = designLedger.beginSave();
  try {
    let savedManifestRevision = transaction.startingRevision;
    if (!targetKnown) {
      // Deleted mission names retain a tombstone revision outside the removed
      // directory. Read it before the first upload so revision zero cannot
      // accidentally resurrect a mission changed by another editor.
      const targetSnapshot = await getMission(
        mapName,
        missionRequestName(targetMissionName),
      );
      if (targetSnapshot?.exists) {
        throw new Error(
          `Mission ${targetMissionName} already exists. Reload it before saving.`,
        );
      }
      savedManifestRevision = Number.isInteger(targetSnapshot?.revision)
        ? targetSnapshot.revision
        : 0;
      designLedger.setPersistedRevision(savedManifestRevision);
    }

    saveBehaviorNodes(mapName, behaviorNodes);
    const canonicalDirectories = localBtDirectoriesForSpots(visibleSpots);
    const canonicalMissionSpots = visibleSpots.map((spot) => withLocalBtLibrary(
      spot,
      canonicalLocalBtPathForSpot(spot, canonicalDirectories.get(spot.id)),
      canonicalLocalBtPathsForSpot(spot, canonicalDirectories.get(spot.id)),
    ));
    const syncedMissionFlowNodes = syncMissionFlowNodesWithSpots(
      missionFlowNodes,
      canonicalMissionSpots,
    );
    const syncedMissionFlowEdges = filterMissionFlowEdges(
      missionFlowEdges,
      canonicalMissionSpots,
    );
    const routeMissionSpots = missionStepSpotsFromMissionFlow(
      canonicalMissionSpots,
      syncedMissionFlowNodes,
      syncedMissionFlowEdges,
    );
    const globalPath = "global.xml";
    const globalXml = buildGlobalMissionXml(routeMissionSpots);
    const { files: nextBtFiles, stalePaths } = assembleMissionBtFilesForSave(
      visibleSpots,
      transaction.btFiles,
      transaction.deletedBtPaths,
      globalPath,
      globalXml,
    );
    const savedHistorySnapshot = JSON.stringify({
      ...historyObject(historyAtStart),
      spots: canonicalMissionSpots,
      missionFlowNodes: syncedMissionFlowNodes,
      missionFlowEdges: syncedMissionFlowEdges,
      missionBtFiles: nextBtFiles,
      deletedMissionBtPaths: [],
      designDirty: false,
      nonBtDesignDirty: false,
    });

    const persistedLocalPathsAtSaveStart = new Set(
      transaction.persistedLocalBtPaths,
    );
    const waypointOwnerByPath = new Map();
    canonicalMissionSpots.forEach((spot) => {
      localBtPathsForSpot(spot).forEach((path) => {
        waypointOwnerByPath.set(path, spot.id);
      });
    });

    let uploadRevision = savedManifestRevision;
    // Sequential uploads form one explicit optimistic-concurrency chain.
    for (const [path, content] of Object.entries(nextBtFiles)) {
      const waypointId = waypointOwnerByPath.get(path);
      const options = waypointId && persistedLocalPathsAtSaveStart.has(path)
        ? { waypointId, expectedRevision: uploadRevision }
        : { expectedRevision: uploadRevision };
      const response = await saveBtFile(
        mapName,
        path,
        content,
        missionRequestName(targetMissionName),
        options,
      );
      if (Number.isInteger(response?.revision)) uploadRevision = response.revision;
      designLedger.checkpointSaveUpload(transaction, {
        path,
        content,
        revision: response?.revision,
      });
    }

    // Referenced XML files are durable before the manifest can point at them.
    const savedMission = await saveMission(mapName, {
      expected_revision: uploadRevision,
      global_bt: globalPath,
      waypoints: missionWaypointsFromSpots(canonicalMissionSpots),
      metadata: {
        source: "mission_canvas",
        behavior_node_count: behaviorNodes.length,
        mission_flow: serializeMissionFlow(
          syncedMissionFlowNodes,
          syncedMissionFlowEdges,
        ),
      },
    }, missionRequestName(targetMissionName));
    const manifestRevision = Number.isInteger(savedMission?.revision)
      ? savedMission.revision
      : uploadRevision + 1;
    designLedger.checkpointSaveManifest(transaction, {
      btFiles: nextBtFiles,
      localBtPaths: canonicalMissionSpots.flatMap((spot) => (
        localBtPathsForSpot(spot)
      )),
      revision: manifestRevision,
    });

    let cleanupRevision = manifestRevision;
    for (const path of stalePaths) {
      const deleted = await deleteBtFile(
        mapName,
        path,
        missionRequestName(targetMissionName),
        { expectedRevision: cleanupRevision },
      );
      if (Number.isInteger(deleted?.revision)) cleanupRevision = deleted.revision;
      designLedger.checkpointSaveCleanup(transaction, {
        revision: deleted?.revision,
      });
    }

    const saveResult = designLedger.reconcileSave(transaction, {
      stalePaths,
      migrateLiveBtFiles: (files) => migrateCanonicalLocalBtFileKeys(
        visibleSpots,
        files,
      ),
    });
    transaction = null;
    return {
      canonicalMissionSpots,
      syncedMissionFlowNodes,
      syncedMissionFlowEdges,
      savedHistorySnapshot,
      saveResult,
    };
  } catch (error) {
    if (transaction) designLedger.abortSave(transaction);
    throw error;
  }
}
