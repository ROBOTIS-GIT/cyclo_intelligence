// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { useState } from "react";
import { act, renderHook } from "@testing-library/react";
import useMissionRouteEditor from "./useMissionRouteEditor";
import { missionFlowEdgesForRouteOrder } from "../lib/missionFlow";

const SPOTS = [
  { id: "a", label: "A", pose: { x: 0, y: 0, yaw: 0 } },
  { id: "b", label: "B", pose: { x: 1, y: 0, yaw: 0 } },
  { id: "c", label: "C", pose: { x: 2, y: 0, yaw: 0 } },
];

const FLOW_NODES = SPOTS.map((spot, index) => ({
  id: spot.id,
  position: { x: index * 100, y: 0 },
  data: { label: spot.label },
}));

function edgePairs(edges) {
  return edges.map(({ source, target }) => `${source}->${target}`);
}

function renderRouteEditor({
  initialEdges = [],
  initialBusy = false,
  initialDocumentReady = true,
} = {}) {
  const markDirty = jest.fn();
  const onMessage = jest.fn();
  const onPrepareEditMode = jest.fn();
  const onSelectSpot = jest.fn();
  const view = renderHook(({ busy, documentReady }) => {
    const [flowNodes, setFlowNodes] = useState(FLOW_NODES);
    const [flowEdges, setFlowEdges] = useState(initialEdges);
    const editor = useMissionRouteEditor({
      spots: SPOTS,
      flowNodes,
      flowEdges,
      setFlowNodes,
      setFlowEdges,
      busy,
      documentReady,
      markDirty,
      onMessage,
      onPrepareEditMode,
      onSelectSpot,
    });
    return { ...editor, flowNodes, flowEdges };
  }, {
    initialProps: {
      busy: initialBusy,
      documentReady: initialDocumentReady,
    },
  });
  return {
    markDirty,
    onMessage,
    onPrepareEditMode,
    onSelectSpot,
    view,
  };
}

test("edits a route only in explicit mode and preserves a closed loop on append", () => {
  const utils = renderRouteEditor();
  let changed;

  act(() => {
    changed = utils.view.result.current.appendSpot("a");
  });
  expect(changed).toBe(false);
  expect(utils.markDirty).not.toHaveBeenCalled();

  act(() => {
    utils.view.result.current.toggleMode();
  });
  expect(utils.view.result.current.routeMode).toBe(true);
  expect(utils.onPrepareEditMode).toHaveBeenCalledTimes(1);

  act(() => {
    utils.view.result.current.handleSpotClick("a");
  });
  expect(utils.view.result.current.routeSourceId).toBe("a");
  expect(utils.view.result.current.flowEdges).toEqual([]);
  expect(utils.markDirty).not.toHaveBeenCalled();

  act(() => {
    utils.view.result.current.handleSpotClick("b");
  });
  expect(edgePairs(utils.view.result.current.flowEdges)).toEqual(["a->b"]);
  expect(utils.view.result.current.routeSourceId).toBe("b");

  act(() => {
    utils.view.result.current.handleMapClick();
  });
  expect(utils.view.result.current.routeSourceId).toBe("");
  act(() => {
    utils.view.result.current.handleSpotClick("b");
  });
  act(() => {
    utils.view.result.current.handleSpotClick("a");
  });
  expect(edgePairs(utils.view.result.current.flowEdges)).toEqual([
    "a->b",
    "b->a",
  ]);
  expect(utils.view.result.current.routeView.closed).toBe(true);
  expect(utils.view.result.current.routeSourceId).toBe("");

  act(() => {
    utils.view.result.current.handleSpotClick("c");
  });
  expect(edgePairs(utils.view.result.current.flowEdges)).toEqual([
    "a->b",
    "b->c",
    "c->a",
  ]);
  expect(utils.view.result.current.routeView.closed).toBe(true);
  expect(utils.markDirty).toHaveBeenCalledTimes(3);
  expect(utils.onSelectSpot).toHaveBeenLastCalledWith("c");
});

test("reorders, removes, opens and clears a closed route", () => {
  const utils = renderRouteEditor({
    initialEdges: missionFlowEdgesForRouteOrder(["a", "b", "c"], true),
  });
  act(() => {
    utils.view.result.current.toggleMode();
  });
  act(() => {
    utils.view.result.current.moveSpot("b", 1);
  });
  expect(edgePairs(utils.view.result.current.flowEdges)).toEqual([
    "a->c",
    "c->b",
    "b->a",
  ]);
  expect(utils.view.result.current.routeView.closed).toBe(true);

  act(() => {
    utils.view.result.current.removeSpot("c");
  });
  expect(edgePairs(utils.view.result.current.flowEdges)).toEqual([
    "a->b",
    "b->a",
  ]);
  expect(utils.view.result.current.routeView.closed).toBe(true);

  act(() => {
    utils.view.result.current.openLoop();
  });
  expect(edgePairs(utils.view.result.current.flowEdges)).toEqual(["a->b"]);
  expect(utils.view.result.current.routeView.closed).toBe(false);
  expect(utils.view.result.current.routeSourceId).toBe("b");

  act(() => {
    utils.view.result.current.clearRoute();
  });
  expect(utils.view.result.current.flowEdges).toEqual([]);
  expect(utils.view.result.current.routeSourceId).toBe("");
  expect(utils.markDirty).toHaveBeenCalledTimes(4);
  expect(utils.onMessage.mock.calls.map(([message]) => message)).toEqual(
    expect.arrayContaining([
      "C removed from route",
      "Loop opened",
      "Route cleared",
    ]),
  );
});

test("validates route order and blocks mutations while busy or locked", () => {
  const utils = renderRouteEditor({ initialDocumentReady: false });
  let changed;

  act(() => {
    changed = utils.view.result.current.toggleMode();
  });
  expect(changed).toBe(false);
  expect(utils.onMessage).toHaveBeenLastCalledWith(
    "Load a mission before editing mission route",
  );

  utils.view.rerender({ busy: false, documentReady: true });
  act(() => {
    utils.view.result.current.toggleMode();
  });
  expect(utils.view.result.current.routeMode).toBe(true);

  expect(utils.view.result.current.tryAcquireMutationLock()).toBe(true);
  expect(utils.view.result.current.tryAcquireMutationLock()).toBe(false);
  act(() => {
    changed = utils.view.result.current.setRouteOrder(["a", "b"]);
  });
  expect(changed).toBe(false);
  utils.view.result.current.releaseMutationLock();

  act(() => {
    changed = utils.view.result.current.setRouteOrder(["a", "a", "missing", "b"]);
  });
  expect(changed).toBe(true);
  expect(edgePairs(utils.view.result.current.flowEdges)).toEqual(["a->b"]);
  expect(utils.markDirty).toHaveBeenCalledTimes(1);
  act(() => {
    changed = utils.view.result.current.setRouteOrder(["a", "b"]);
  });
  expect(changed).toBe(false);
  expect(utils.markDirty).toHaveBeenCalledTimes(1);

  utils.view.rerender({ busy: true, documentReady: true });
  act(() => {
    changed = utils.view.result.current.moveSpot("a", 1);
  });
  expect(changed).toBe(false);
  expect(utils.view.result.current.tryAcquireMutationLock()).toBe(false);

  act(() => {
    utils.view.result.current.resetEditing();
  });
  expect(utils.view.result.current.routeMode).toBe(false);
  expect(utils.view.result.current.routeSourceId).toBe("");
});
