// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useDesignDocumentContentController from "./useDesignDocumentContentController";

function setup(initialSpots = []) {
  const order = [];
  let spots = initialSpots;
  const port = (name, implementation) => jest.fn((...args) => {
    order.push(name);
    return implementation?.(...args);
  });
  const ports = {
    applySpots: port("applySpots", (next) => { spots = next; }),
    updateSpots: port("updateSpots", (updater) => { spots = updater(spots); }),
    setFlowNodes: port("setFlowNodes"),
    setFlowEdges: port("setFlowEdges"),
    mergeBehaviorMapPatch: port("mergeBehaviorMapPatch"),
    commitLedgerSnapshot: port("commitLedgerSnapshot"),
    resetLedgerNewDocument: port("resetLedgerNewDocument"),
    resetHistory: port("resetHistory"),
    resetLocalBtSelections: port("resetLocalBtSelections"),
    clearBehaviorSelection: port("clearBehaviorSelection"),
    clearPendingBehaviorPlacement: port("clearPendingBehaviorPlacement"),
    resetRouteEditing: port("resetRouteEditing"),
    setEditingSpotId: port("setEditingSpotId"),
    setEditingSpotLabel: port("setEditingSpotLabel"),
    setLoadError: port("setLoadError"),
  };
  const view = renderHook(() => useDesignDocumentContentController(ports));
  return { view, ports, order, getSpots: () => spots };
}

test("commits every loaded document slice atomically in the existing order", () => {
  const { view, ports, order } = setup();
  const snapshot = {
    identity: { mapName: "factory", missionName: "inspection" },
    spots: [{ id: "wp-1" }],
    behaviorNodesPatch: [{ id: "Sequence_1" }],
    flowNodes: [{ id: "wp-1" }],
    flowEdges: [{ id: "edge-1" }],
    btFiles: { "global.xml": "<root/>" },
    baseline: { revision: 4 },
  };
  let result;
  act(() => { result = view.result.current.commitLoadedSnapshot(snapshot); });

  expect(order).toEqual([
    "mergeBehaviorMapPatch",
    "applySpots",
    "setFlowNodes",
    "setFlowEdges",
    "commitLedgerSnapshot",
    "resetHistory",
    "resetLocalBtSelections",
    "clearBehaviorSelection",
    "clearPendingBehaviorPlacement",
  ]);
  expect(ports.mergeBehaviorMapPatch).toHaveBeenCalledWith(
    "factory",
    snapshot.behaviorNodesPatch,
  );
  expect(ports.commitLedgerSnapshot).toHaveBeenCalledWith({
    btFiles: snapshot.btFiles,
    baseline: snapshot.baseline,
  });
  expect(result).toEqual(expect.objectContaining({
    spots: snapshot.spots,
    flowNodes: snapshot.flowNodes,
    flowEdges: snapshot.flowEdges,
  }));
});

test("normalizes invalid snapshot slices without applying an invalid behavior patch", () => {
  const { view, ports } = setup([{ id: "old" }]);
  act(() => view.result.current.commitLoadedSnapshot({
    identity: {},
    spots: "bad",
    behaviorNodesPatch: {},
    flowNodes: null,
    flowEdges: "bad",
    btFiles: [],
    baseline: null,
  }));
  expect(ports.mergeBehaviorMapPatch).not.toHaveBeenCalled();
  expect(ports.applySpots).toHaveBeenCalledWith([]);
  expect(ports.setFlowNodes).toHaveBeenCalledWith([]);
  expect(ports.setFlowEdges).toHaveBeenCalledWith([]);
  expect(ports.commitLedgerSnapshot).toHaveBeenCalledWith({ btFiles: {}, baseline: {} });
});

test("uses the default map for a valid legacy behavior patch without identity", () => {
  const { view, ports } = setup();
  act(() => view.result.current.commitLoadedSnapshot({ behaviorNodesPatch: [] }));
  expect(ports.mergeBehaviorMapPatch).toHaveBeenCalledWith("map", []);
});

test("resets a new document and all transient authoring state in order", () => {
  const { view, ports, order } = setup([{ id: "old" }]);
  const btFiles = { "global.xml": "<new/>" };
  act(() => view.result.current.resetNewDocument({ btFiles }));
  expect(order).toEqual([
    "applySpots",
    "setFlowNodes",
    "setFlowEdges",
    "resetLocalBtSelections",
    "resetLedgerNewDocument",
    "resetHistory",
    "clearBehaviorSelection",
    "clearPendingBehaviorPlacement",
    "resetRouteEditing",
    "setEditingSpotId",
    "setEditingSpotLabel",
    "setLoadError",
  ]);
  expect(ports.resetLedgerNewDocument).toHaveBeenCalledWith({ btFiles });
  expect(ports.setEditingSpotId).toHaveBeenCalledWith("");
  expect(ports.setEditingSpotLabel).toHaveBeenCalledWith("");
  expect(ports.setLoadError).toHaveBeenCalledWith("");
});

test("merges saved canonical BT libraries into the latest live spots", () => {
  const current = [
    {
      id: "wp-1",
      label: "renamed while saving",
      pose: { x: 9, y: 8, yaw: 0.5 },
      linked_bt_tree: "legacy.xml",
      metadata: { custom: "keep", local_bt: "legacy.xml" },
    },
    { id: "wp-2", label: "newer waypoint", metadata: { custom: "new" } },
  ];
  const canonical = [{
    id: "wp-1",
    label: "saved label",
    linked_bt_tree: "locals/waypoint_1/main.xml",
    local_bt_files: [
      "locals/waypoint_1/main.xml",
      "locals/waypoint_1/recovery.xml",
    ],
    metadata: {
      local_bt: "locals/waypoint_1/main.xml",
      local_bt_files: [
        "locals/waypoint_1/main.xml",
        "locals/waypoint_1/recovery.xml",
      ],
    },
  }];
  const { view, ports, getSpots } = setup(current);
  act(() => view.result.current.applySavedCanonicalSpots(canonical));

  expect(ports.updateSpots).toHaveBeenCalledTimes(1);
  expect(getSpots()[0]).toEqual(expect.objectContaining({
    id: "wp-1",
    label: "renamed while saving",
    pose: current[0].pose,
    linked_bt_tree: "locals/waypoint_1/main.xml",
    local_bt_files: [
      "locals/waypoint_1/main.xml",
      "locals/waypoint_1/recovery.xml",
    ],
    metadata: expect.objectContaining({ custom: "keep" }),
  }));
  expect(getSpots()[1].label).toBe("newer waypoint");
  expect(getSpots()[1].linked_bt_tree).toMatch(/^locals\//);
  expect(getSpots()[1].metadata.custom).toBe("new");
});

test("rejects an invalid canonical result before touching live spots", () => {
  const { view, ports } = setup([{ id: "wp-1" }]);
  expect(() => view.result.current.applySavedCanonicalSpots(null)).toThrow(
    "Saved canonical spots must be an array",
  );
  expect(ports.updateSpots).not.toHaveBeenCalled();
});
