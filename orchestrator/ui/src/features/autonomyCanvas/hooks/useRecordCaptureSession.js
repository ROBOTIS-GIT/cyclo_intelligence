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

import { missionsAtObservation } from "./useRecordObservationSpots";

// Small adapter from actual execution to the recorder's immutable segment fields.
export default function useRecordCaptureSession({ execution, context, manager, mission, selectedMissionName, mapName }) {
  const cycle = execution.cycle;
  const type = execution.executionType;
  const failedRunner = cycle.cyclePhase === "failed" ? cycle
    : execution.navigation?.status === "failed" ? execution.navigation : null;
  const waitingReason = !context.ready && failedRunner
    ? failedRunner.reason || "Execution failed." : "";
  const sourceId = type === "mission" ? cycle.pinnedSpot?.id : execution.observationSpotId;
  const returnId = cycle.pinnedSpot?.id;
  const missionId = cycle.pinnedMissionId || mission.missionId;
  const catalog = manager.configuration?.missions || [];
  const missionName = catalog.find(item => item.mission_id === missionId)?.mission_name || mission.missionName;
  const plan = manager.configuration?.plan;
  const planReady = manager.ready && !manager.route?.routeMode && !manager.planDirty && !manager.draft && !manager.working;
  const plannedIds = (plan?.observations || []).map(point => point.spot_id);
  const selectedId = plannedIds.includes(execution.observationSpotId) ? execution.observationSpotId : null;
  const selectedMission = missionsAtObservation(manager.configuration, selectedId)
    .find(item => item.mission_name === selectedMissionName);
  const target = { planRevision: manager.planRevision, observationSpotId: selectedId,
    patrolSpotIds: plannedIds, missionId: selectedMission?.mission_id,
    missionName: selectedMission?.mission_name, returnSpotId: selectedId };
  return { type, ready: context.ready, waitingReason, planRevision: manager.planRevision, mapName,
    captureTargets: {
      observation: { ...target, type: "observation", ready: !!(planReady && selectedId) },
      patrol: { ...target, type: "patrol", ready: !!(planReady && plannedIds.length) },
      mission: { ...target, type: "mission", ready: !!(planReady && selectedId && selectedMission) },
    },
    missionId, missionName, observationSpotId: sourceId,
    patrolSpotIds: execution.returning ? (returnId ? [returnId] : [])
      : (manager.configuration?.plan?.observations || []).map(point => point.spot_id),
    returnSpotId: returnId, returnSpotName: cycle.pinnedSpot?.label,
    returning: execution.returning,
  };
}
