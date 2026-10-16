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

import { useCallback, useRef } from "react";
import {
  defaultLocalBtXml,
  initializeCreatedWaypointLocalBt,
} from "../../../mapWorkspace/lib/missionBtFiles";
import { nextWaypointLabel, spotPoseFromMapPose } from "../../../mapWorkspace/lib/missionSpots";

export default function useDesignWaypointCreation({ waypoints, persistence, selection, document, execution, api }) {
  const { spots, setSpots } = waypoints;
  const { ledger, markDirty } = persistence;
  const { setSelectedSpotId, clearBehaviorSelection, setShowWaypointOptions, setInteractionMode } = selection;
  const { mapName, getCurrentIdentity, captureDocumentLease, documentIsCurrent } = document;
  const { runCommand } = execution;
  const { createSpot, deleteSpot } = api;
  const createPendingRef = useRef(false);
  const commitCreatedWaypoint = useCallback((createdSpot) => {
    const initialized = initializeCreatedWaypointLocalBt(spots, createdSpot, [
      ...Object.keys(ledger.getLiveBtFiles()),
      ...ledger.getPersistedLocalBtPaths(),
      ...ledger.deletedPaths,
    ]);
    const files = ledger.getLiveBtFiles();
    const emptyXml = defaultLocalBtXml(initialized.spot);
    initialized.paths.forEach((path) => { files[path] = emptyXml; });
    markDirty();
    ledger.replaceLiveBtFiles(files, { bumpBtEpoch: true });
    setSpots((current) => [...current, initialized.spot]);
    setSelectedSpotId(initialized.spot.id);
    clearBehaviorSelection();
    return initialized.spot;
  }, [clearBehaviorSelection, ledger, markDirty, setSelectedSpotId, setSpots, spots]);

  const createWaypoint = useCallback(async ({
    commandLabel,
    waypointLabel,
    resolvePose,
    finalize,
    resultMessage,
  }) => {
    if (createPendingRef.current) return undefined;
    createPendingRef.current = true;
    setShowWaypointOptions(false);
    setInteractionMode("view");
    const label = typeof waypointLabel === "string" && waypointLabel.trim()
      ? waypointLabel
      : nextWaypointLabel(spots);
    const targetMapName = mapName;
    const identity = { ...getCurrentIdentity() };
    const lease = captureDocumentLease();
    try {
      return await runCommand(commandLabel, async () => {
        try {
          const { x, y, yaw } = await resolvePose();
          const created = await createSpot({
            map_name: targetMapName,
            label,
            pose: spotPoseFromMapPose(x, y, yaw),
            metadata: { source: "mission_canvas", coordinate_space: "map" },
          });
          if (!documentIsCurrent(lease, identity)) {
            try { await deleteSpot(created.id, targetMapName); } catch { /* best effort */ }
            throw new Error("Map or mission changed while the waypoint was being created");
          }
          const initialized = commitCreatedWaypoint(created);
          return resultMessage
            ? resultMessage(initialized)
            : `Created ${initialized.label}`;
        } finally {
          if (finalize) {
            await finalize({ documentCurrent: documentIsCurrent(lease, identity) });
          }
        }
      });
    } finally {
      createPendingRef.current = false;
    }
  }, [
    captureDocumentLease, commitCreatedWaypoint, createSpot, deleteSpot,
    documentIsCurrent, getCurrentIdentity, mapName, runCommand,
    setInteractionMode, setShowWaypointOptions, spots,
  ]);

  const createOnMap = useCallback((x, y, yaw) => createWaypoint({
    commandLabel: "Create Waypoint",
    resolvePose: async () => ({ x, y, yaw }),
  }), [createWaypoint]);

  const createFromMission = useCallback((waypoint, sourceMapName) => createWaypoint({
    commandLabel: "Import Waypoint",
    waypointLabel: waypoint?.label,
    resolvePose: async () => {
      const pose = waypoint?.pose;
      if (sourceMapName !== mapName || (pose?.frame_id && pose.frame_id !== "map") ||
          ![pose?.x, pose?.y, pose?.yaw].every(Number.isFinite)) {
        throw new Error("Select a valid waypoint from the current map.");
      }
      return { x: pose.x, y: pose.y, yaw: pose.yaw };
    },
  }), [createWaypoint, mapName]);

  const createAtRobot = useCallback((options) => createWaypoint({
    ...options,
    commandLabel: "At Robot",
  }), [createWaypoint]);
  return { createOnMap, createFromMission, createAtRobot };
}
