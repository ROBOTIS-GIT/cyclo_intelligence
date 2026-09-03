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
  filterMissionFlowEdges,
  missionStepSpotsFromMissionFlow,
  orderedSpotsFromMissionFlow,
} from "./missionFlow";
import { spotForMapDisplay } from "./missionSpots";

// Build one immutable view of a mission document for a specific map. Design
// and Run call this independently so a stage switch cannot make authoring
// handlers accidentally consume the read-only Run snapshot (or vice versa).
export function deriveMissionRouteView({
  spots = [],
  flowNodes = [],
  flowEdges = [],
  map = null,
  routeSourceId = "",
} = {}) {
  const visibleSpots = spots.map((spot) => spotForMapDisplay(spot, map));
  const routeEdges = filterMissionFlowEdges(flowEdges, visibleSpots);
  const orderedSpots = routeEdges.length > 0
    ? orderedSpotsFromMissionFlow(visibleSpots, flowNodes, routeEdges)
    : [];
  const executionSpots = routeEdges.length > 0
    ? missionStepSpotsFromMissionFlow(visibleSpots, flowNodes, routeEdges)
    : [];
  const closed = (
    executionSpots.length > orderedSpots.length
    && executionSpots.length > 2
    && executionSpots[0]?.id === executionSpots[executionSpots.length - 1]?.id
  );
  const order = orderedSpots.map((spot, index) => ({
    id: spot.id,
    order: index + 1,
  }));
  const sourceSpot = routeSourceId
    ? visibleSpots.find((spot) => spot.id === routeSourceId)
    : null;
  const treeSpots = orderedSpots.length > 0
    ? orderedSpots
    : sourceSpot ? [sourceSpot] : [];

  return {
    visibleSpots,
    routeEdges,
    orderedSpots,
    executionSpots,
    treeSpots,
    closed,
    order,
  };
}
