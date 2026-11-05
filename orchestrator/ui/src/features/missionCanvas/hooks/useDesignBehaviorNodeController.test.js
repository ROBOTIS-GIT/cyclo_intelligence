// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useDesignBehaviorNodeController from "./useDesignBehaviorNodeController";

const node = (id, map = "map-a", yaw = 0) => ({
  id,
  map_name: map,
  tag: "Sequence",
  label: "Sequence",
  category: "control",
  pose: { frame_id: "map", x: 0, y: 0, yaw },
  metadata: { source: "mission_canvas" },
});

function renderController() {
  let generation = 1;
  let identity = { mapName: "map-a", missionName: "mission-a" };
  const markDirty = jest.fn();
  const onMessage = jest.fn();
  const onPrepareSelect = jest.fn();
  const onPlaced = jest.fn();
  const view = renderHook((props) => useDesignBehaviorNodeController({
    designMapName: props.designMapName,
    runtimeMapName: props.runtimeMapName,
    markDirty,
    onMessage,
    captureDocumentLease: () => ({ generation }),
    isDocumentLeaseCurrent: (lease) => lease.generation === generation,
    getCurrentIdentity: () => ({ ...identity }),
    onPrepareSelect,
    onPlaced,
  }), { initialProps: { designMapName: "map-a", runtimeMapName: "map-b" } });
  return {
    view,
    markDirty,
    onMessage,
    onPrepareSelect,
    onPlaced,
    advanceDocument: () => { generation += 1; },
    setIdentity: (next) => { identity = next; },
  };
}

test("filters the map-scoped cache for Design and Run", () => {
  const { view } = renderController();
  act(() => view.result.current.mergeMapPatch("map-a", [node("Sequence_1")]));
  act(() => view.result.current.mergeMapPatch("map-b", [node("Sequence_2", "map-b")]));
  expect(view.result.current.designNodes.map(({ id }) => id)).toEqual(["Sequence_1"]);
  expect(view.result.current.runNodes.map(({ id }) => id)).toEqual(["Sequence_2"]);
});

test("map patch replaces only its target map", () => {
  const { view } = renderController();
  act(() => {
    view.result.current.mergeMapPatch("map-a", [node("Sequence_1")]);
    view.result.current.mergeMapPatch("map-b", [node("Sequence_2", "map-b")]);
  });
  act(() => view.result.current.mergeMapPatch("map-a", [node("Sequence_3")]));
  expect(view.result.current.nodes.map(({ id }) => id).sort()).toEqual([
    "Sequence_2", "Sequence_3",
  ]);
});

test("history restore never lowers the node serial", () => {
  const { view } = renderController();
  act(() => view.result.current.restoreHistorySlice({
    behaviorNodes: [node("behavior_8_sequence")],
    selectedBehaviorNodeId: "behavior_8_sequence",
  }));
  act(() => view.result.current.restoreHistorySlice({ behaviorNodes: [] }));
  act(() => view.result.current.beginPlacement("Sequence"));
  let placed;
  act(() => { placed = view.result.current.placePendingAtPose(1, 2, 0); });
  expect(placed.id).not.toBe("behavior_8_sequence");
  expect(Number(placed.id.match(/^behavior_(\d+)_/)?.[1])).toBeGreaterThan(8);
});

test("selection coordinates with the shell and rejects unknown nodes", () => {
  const { view, onPrepareSelect } = renderController();
  act(() => view.result.current.mergeMapPatch("map-a", [node("Sequence_1")]));
  act(() => view.result.current.selectNode("missing"));
  expect(onPrepareSelect).not.toHaveBeenCalled();
  act(() => view.result.current.selectNode("Sequence_1"));
  expect(onPrepareSelect).toHaveBeenCalledWith("Sequence_1");
  expect(view.result.current.selectedNodeId).toBe("Sequence_1");
});

test("placement creates the legacy fields and clears pending state", () => {
  const { view, markDirty, onPlaced } = renderController();
  act(() => view.result.current.beginPlacement("Sequence"));
  act(() => view.result.current.placePendingAtPose(3, 4, 0.5));
  expect(view.result.current.designNodes[0]).toMatchObject({
    map_name: "map-a",
    tag: "Sequence",
    label: "Sequence",
    category: "control",
    pose: { frame_id: "map", x: 3, y: 4, yaw: 0.5 },
    metadata: { source: "mission_canvas" },
  });
  expect(view.result.current.pendingTag).toBe("");
  expect(markDirty).toHaveBeenCalledTimes(1);
  expect(onPlaced).toHaveBeenCalledTimes(1);
});

test("move preserves yaw and unknown move is a clean no-op", () => {
  const { view, markDirty, onMessage } = renderController();
  act(() => view.result.current.mergeMapPatch("map-a", [node("Sequence_1", "map-a", 1.2)]));
  act(() => view.result.current.moveNode("missing", 1, 1));
  expect(markDirty).not.toHaveBeenCalled();
  expect(onMessage).not.toHaveBeenCalled();
  act(() => view.result.current.moveNode("Sequence_1", 5, 6));
  expect(view.result.current.designNodes[0].pose).toMatchObject({ x: 5, y: 6, yaw: 1.2 });
});

test("delete clears only a matching selection and ignores deleted IDs", () => {
  const { view, markDirty } = renderController();
  const one = node("Sequence_1");
  const two = node("Sequence_2");
  act(() => view.result.current.mergeMapPatch("map-a", [one, two]));
  act(() => view.result.current.selectNode(one.id));
  act(() => view.result.current.deleteNode(two));
  expect(view.result.current.selectedNodeId).toBe(one.id);
  act(() => view.result.current.deleteNode(one));
  expect(view.result.current.selectedNodeId).toBe("");
  const dirtyCount = markDirty.mock.calls.length;
  act(() => view.result.current.deleteNode(one));
  expect(markDirty).toHaveBeenCalledTimes(dirtyCount);
});

test("callbacks captured by an old document cannot mutate the new document", () => {
  const {
    view, markDirty, onMessage, advanceDocument, setIdentity,
  } = renderController();
  act(() => view.result.current.mergeMapPatch("map-a", [node("Sequence_1")]));
  const staleMove = view.result.current.moveNode;
  const staleDelete = view.result.current.deleteNode;
  const staleSelect = view.result.current.selectNode;
  act(() => view.result.current.beginPlacement("Sequence"));
  const stalePlace = view.result.current.placePendingAtPose;
  advanceDocument();
  setIdentity({ mapName: "map-a", missionName: "mission-b" });
  act(() => staleMove("Sequence_1", 9, 9, 0));
  act(() => staleDelete(node("Sequence_1")));
  act(() => staleSelect("Sequence_1"));
  act(() => stalePlace(9, 9, 0));
  expect(view.result.current.designNodes[0].pose.x).toBe(0);
  expect(view.result.current.selectedNodeId).toBe("");
  expect(markDirty).not.toHaveBeenCalled();
  expect(onMessage).not.toHaveBeenCalled();
});

test("history restore keeps valid selection and clears a missing selection", () => {
  const { view } = renderController();
  act(() => view.result.current.restoreHistorySlice({
    behaviorNodes: [node("Sequence_1")],
    selectedBehaviorNodeId: "Sequence_1",
  }));
  expect(view.result.current.selectedNodeId).toBe("Sequence_1");
  act(() => view.result.current.restoreHistorySlice({
    behaviorNodes: [node("Sequence_1")],
    selectedBehaviorNodeId: "missing",
  }));
  expect(view.result.current.selectedNodeId).toBe("");
});

test("move and delete are scoped by map when IDs collide", () => {
  const { view } = renderController();
  const sharedA = node("shared", "map-a", 0.2);
  const sharedB = node("shared", "map-b", 0.8);
  act(() => {
    view.result.current.mergeMapPatch("map-a", [sharedA]);
    view.result.current.mergeMapPatch("map-b", [sharedB]);
  });
  act(() => view.result.current.moveNode("shared", 5, 6));
  expect(view.result.current.designNodes[0].pose.x).toBe(5);
  expect(view.result.current.runNodes[0].pose.x).toBe(0);
  act(() => view.result.current.deleteNode(sharedB));
  expect(view.result.current.designNodes).toHaveLength(1);
  act(() => view.result.current.deleteNode(view.result.current.designNodes[0]));
  expect(view.result.current.designNodes).toEqual([]);
  expect(view.result.current.runNodes).toEqual([sharedB]);
});
