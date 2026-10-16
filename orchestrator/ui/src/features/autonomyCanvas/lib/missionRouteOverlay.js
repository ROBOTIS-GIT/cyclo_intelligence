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

// Record and Operate share the same observation/child-mission map geometry.
export const OBSERVATION_WAYPOINT_COLORS = { idle: 0x8d7899 };

export function missionRouteOverlay({ id, source, executionSpots = [], order, activeWaypointId }) {
  if (!executionSpots.length) return null;
  const spots = [...new Map(executionSpots.map(spot => [spot.id, spot])).values()];
  return {
    id, spots,
    pathSpots: source ? [source, ...executionSpots, source] : executionSpots,
    order: order || spots.map((spot, index) => ({ id: spot.id, order: index + 1 })),
    activeWaypointId,
  };
}
