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

import { deriveMissionRouteView } from "./missionRouteView";

const spots = ["a", "b", "c"].map((id) => ({
  id,
  label: id.toUpperCase(),
  pose: { frame_id: "map", x: 0, y: 0, yaw: 0 },
  metadata: { coordinate_space: "map" },
}));
const flowNodes = spots.map((spot, index) => ({
  id: spot.id,
  position: { x: index * 100, y: 0 },
}));

test("derives an open route and its display order", () => {
  const view = deriveMissionRouteView({
    spots,
    flowNodes,
    flowEdges: [
      { id: "a-b", source: "a", target: "b" },
      { id: "b-c", source: "b", target: "c" },
    ],
  });

  expect(view.orderedSpots.map((spot) => spot.id)).toEqual(["a", "b", "c"]);
  expect(view.executionSpots.map((spot) => spot.id)).toEqual(["a", "b", "c"]);
  expect(view.order).toEqual([
    { id: "a", order: 1 },
    { id: "b", order: 2 },
    { id: "c", order: 3 },
  ]);
  expect(view.closed).toBe(false);
});

test("keeps the repeated starting waypoint only in a closed route execution", () => {
  const view = deriveMissionRouteView({
    spots,
    flowNodes,
    flowEdges: [
      { id: "a-b", source: "a", target: "b" },
      { id: "b-c", source: "b", target: "c" },
      { id: "c-a", source: "c", target: "a" },
    ],
  });

  expect(view.orderedSpots.map((spot) => spot.id)).toEqual(["a", "b", "c"]);
  expect(view.executionSpots.map((spot) => spot.id)).toEqual(["a", "b", "c", "a"]);
  expect(view.closed).toBe(true);
});

test("uses the selected source as the one-node authoring route", () => {
  const view = deriveMissionRouteView({
    spots,
    flowNodes,
    flowEdges: [],
    routeSourceId: "b",
  });

  expect(view.orderedSpots).toEqual([]);
  expect(view.treeSpots.map((spot) => spot.id)).toEqual(["b"]);
});

test("filters edges that reference a different mission document", () => {
  const view = deriveMissionRouteView({
    spots,
    flowNodes,
    flowEdges: [
      { id: "a-b", source: "a", target: "b" },
      { id: "b-foreign", source: "b", target: "foreign" },
    ],
  });

  expect(view.routeEdges).toEqual([{ id: "a-b", source: "a", target: "b" }]);
  expect(view.orderedSpots.map((spot) => spot.id)).toEqual(["a", "b"]);
});

test("converts legacy waypoint cells independently for each map", () => {
  const legacySpot = {
    id: "legacy",
    pose: { frame_id: "map", x: 50, y: 40, yaw: 0 },
    metadata: {},
  };
  const grid = (resolution, originX) => ({
    info: {
      width: 100,
      height: 100,
      resolution,
      origin: {
        position: { x: originX, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
      },
    },
  });

  const designView = deriveMissionRouteView({
    spots: [legacySpot],
    map: grid(0.05, 0),
  });
  const runView = deriveMissionRouteView({
    spots: [legacySpot],
    map: grid(0.1, 10),
  });

  expect(designView.visibleSpots[0].pose).toMatchObject({ x: 2.5, y: 2 });
  expect(runView.visibleSpots[0].pose).toMatchObject({ x: 15, y: 4 });
});
