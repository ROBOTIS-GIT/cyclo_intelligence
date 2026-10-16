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

import { useMemo } from "react";
import { operateDestination } from "./operateDestination";
import { observationSpots } from "../lib/observationSpots";
import { missionRouteOverlay } from "../lib/missionRouteOverlay";

// Only runtime execution highlights a waypoint; a retained Qwen result never does.
export function autonomyMapView(state = {}) {
  const plan = state.config?.draft || state.config?.plan;
  const points = plan?.observations || [];
  const spots = observationSpots({ ...state.config, plan });
  const execution = state.execution;
  const active = execution ? execution.lifecycle === "active" && execution.owns_execution
    : !!state.active && !["stopped", "stopping", "error", "interrupted"].includes(state.phase);
  const running = active && (execution ? execution.control === "running" : !state.paused);
  const returning = execution?.activity?.startsWith("return_") || state.phase === "returning"
    || state.runner?.phase?.startsWith("return-");
  const inMission = !returning && state.mode === "mission";
  const targetId = returning ? state.resume_checkpoint?.return_spot_id || state.observation?.spot_id
    : execution?.target_spot_id || execution?.observation_spot_id || state.observation?.spot_id;
  // Render the execution snapshot while the map configuration is still loading.
  if (active && state.observation?.pose
    && !spots.some(spot => spot.id === state.observation.spot_id)) {
    spots.push({ ...state.observation, id: state.observation.spot_id });
  }
  const sourceId = state.resume_checkpoint?.return_spot_id || state.observation?.spot_id;
  const source = spots.find(spot => spot.id === sourceId);
  const order = plan?.mode === "patrol"
    ? points.filter(point => spots.some(spot => spot.id === point.spot_id))
      .map((point, index) => ({ id: point.spot_id, order: index + 1 })) : [];
  const navigationPhase = ["navigation", "patrol-navigation", "return-navigation"].includes(state.runner?.phase)
    || (["positioning", "returning"].includes(state.phase) && !state.runner?.phase?.endsWith("posture"));
  const navigationActive = running && (execution
    ? ["patrol_navigation", "mission_navigation", "return_navigation"].includes(execution.activity)
    : navigationPhase);
  return {
    spots, activeWaypointId: running && !inMission ? targetId : null,
    navigationActive,
    navigationDestination: operateDestination(state, spots, navigationActive),
    navigationKey: navigationActive
      ? [state.session_id, execution?.activity || state.runner?.phase, inMission ? state.runner?.activeSpotId : targetId].join(":") : null,
    route: { closed: order.length > 1 && plan?.route_closed !== false, order },
    missionOverlay: active && inMission ? missionRouteOverlay({
      id: state.mission?.mission_id, source, executionSpots: state.mission_route,
      activeWaypointId: running ? state.runner?.activeSpotId : null,
    }) : null,
  };
}

// ROS pose updates rerender the workspace, but only these fields change the
// observation/mission geometry. Keep its references stable for map caches.
export function useAutonomyMapView(state = {}) {
  const { active, phase, paused, mode, observation, session_id, mission_route } = state;
  const plan = state.config?.draft || state.config?.plan;
  const spots = state.config?.spots;
  const hasExecution = Boolean(state.execution);
  const { lifecycle, owns_execution, control, activity, target_spot_id, observation_spot_id } = state.execution || {};
  const runnerPhase = state.runner?.phase;
  const activeSpotId = state.runner?.activeSpotId;
  const returnSpotId = state.resume_checkpoint?.return_spot_id;
  const missionId = state.mission?.mission_id;
  return useMemo(() => autonomyMapView({
    active, phase, paused, mode, observation, session_id, mission_route,
    config: { plan, spots },
    execution: hasExecution ? { lifecycle, owns_execution, control, activity, target_spot_id, observation_spot_id } : null,
    runner: { phase: runnerPhase, activeSpotId },
    resume_checkpoint: { return_spot_id: returnSpotId },
    mission: { mission_id: missionId },
  }), [
    active, phase, paused, mode, observation, session_id, mission_route, plan, spots,
    hasExecution, lifecycle, owns_execution, control, activity, target_spot_id, observation_spot_id,
    runnerPhase, activeSpotId, returnSpotId, missionId,
  ]);
}
