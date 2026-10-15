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

import { useCallback, useEffect } from "react";
import { STAGE_COLLECT } from "../lib/stages";
import { missionStepSpotsFromMissionFlow } from "../lib/missionFlow";

export default function useRunMissionStart({
  workspace, collectMission, recordNavigationMoving, runMapBusy, runMapSnapshotInvalid, runMissionName, runtimeMapName, runMapName, runMissionSnapshot, loadRunMissionForMap, recordMissionStartRef, runMissionId, runMissionRevision, missionRunner
}) {
  const {
    workspaceStage, setMessage, missionMapLoaded, runPoseInitialized, runStartRequestRef,
    setPendingRunStart, pendingRunStart,
  } = workspace;
  const handleRunMission = useCallback(async () => {
    if (workspaceStage === STAGE_COLLECT && (!collectMission.name)) {
      setMessage("Select a mission for the current observation first");
      return false;
    }
    if (recordNavigationMoving) {
      setMessage("Stop waypoint navigation before starting a route");
      return false;
    }
    if (
      runMapBusy
      || !missionMapLoaded
      || (runMapSnapshotInvalid && workspaceStage !== STAGE_COLLECT)
    ) {
      setMessage("Wait for the selected mission to finish loading");
      return false;
    }
    if (!runPoseInitialized) {
      setMessage("Localize the robot first");
      return false;
    }
    const targetMissionName = workspaceStage === STAGE_COLLECT
      ? collectMission.name
      : runMissionName;
    const targetMapName = runtimeMapName || runMapName;
    const request = runStartRequestRef.current + 1;
    runStartRequestRef.current = request;
    const expectedMissionId = (
      runMissionSnapshot.mapName === targetMapName
      && runMissionSnapshot.missionName === targetMissionName
    ) ? runMissionSnapshot.missionId : undefined;
    try {
      const result = await loadRunMissionForMap(targetMapName, targetMissionName, {
        expectedMissionId,
        requireExisting: workspaceStage === STAGE_COLLECT,
        returnSnapshot: true,
      });
      if (request !== runStartRequestRef.current || result?.stale) return false;
      const fresh = result?.snapshot;
      const executionSpots = fresh
        ? missionStepSpotsFromMissionFlow(
            fresh.spots,
            fresh.flowNodes,
            fresh.flowEdges,
          )
        : [];
      if (!fresh || !executionSpots.length) {
        setMessage("No route to run — connect waypoints in Design first");
        return false;
      }
      setPendingRunStart({
        request,
        mapName: fresh.mapName,
        missionName: fresh.missionName,
        missionId: fresh.missionId,
        revision: fresh.revision,
      });
      return true;
    } catch (error) {
      if (request === runStartRequestRef.current) {
        setMessage(error instanceof Error ? error.message : "Unable to refresh the mission");
      }
      return false;
    }
  }, [workspaceStage, collectMission.name, recordNavigationMoving, runMapBusy, missionMapLoaded, runMapSnapshotInvalid, runPoseInitialized, runMissionName, runtimeMapName, runMapName, runStartRequestRef, runMissionSnapshot.mapName, runMissionSnapshot.missionName, runMissionSnapshot.missionId, setMessage, loadRunMissionForMap, setPendingRunStart]);

  recordMissionStartRef.current = handleRunMission;

  useEffect(() => {
    if (!pendingRunStart) return;
    if (
      runMapSnapshotInvalid
      || runMapName !== pendingRunStart.mapName
      || runMissionName !== pendingRunStart.missionName
      || runMissionId !== pendingRunStart.missionId
      || runMissionRevision !== pendingRunStart.revision
    ) return;
    setPendingRunStart(null);
    if (!missionRunner.start()) {
      setMessage("Mission runner could not start. Check its route and navigation state.");
    }
  }, [
    missionRunner, pendingRunStart, runMapName, runMapSnapshotInvalid, runMissionId, runMissionName,
    runMissionRevision, setMessage, setPendingRunStart,
  ]);

  return {
    handleRunMission,
  };
}
