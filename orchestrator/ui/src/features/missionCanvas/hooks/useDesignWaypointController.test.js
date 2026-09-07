// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import { useState } from "react";
import useDesignWaypointController from "./useDesignWaypointController";

const SPOT = {
  id: "wp-1",
  map_name: "map-a",
  label: "Waypoint 1",
  pose: { frame_id: "map", x: 0, y: 0, yaw: 0 },
  linked_bt_tree: "locals/waypoint_1/main.xml",
  local_bt_files: ["locals/waypoint_1/main.xml"],
};

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function renderController(
  api = {},
  identity = { mapName: "map-a", missionName: "m1" },
  routeView = { treeSpots: [], closed: false },
) {
  const markDirty = jest.fn();
  const releaseLock = jest.fn();
  const deleteSpot = api.deleteSpot || jest.fn(async () => ({}));
  const setFlowEdges = jest.fn();
  const setRouteSource = jest.fn();
  const view = renderHook(() => {
    const [spots, setSpots] = useState([SPOT]);
    const [editingSpotId, setEditingSpotId] = useState("");
    const [editingSpotLabel, setEditingSpotLabel] = useState("");
    const controller = useDesignWaypointController({
      spots,
      setSpots,
      editingSpotId,
      editingSpotLabel,
      setEditingSpotId,
      setEditingSpotLabel,
      mapName: "map-a",
      captureDocumentLease: () => "lease",
      isDocumentLeaseCurrent: () => true,
      getCurrentIdentity: () => ({ ...identity }),
      runCommand: async (_label, command) => command(),
      markDirty,
      ledger: {
        deletedPaths: [],
        getLiveBtFiles: () => ({}),
        getPersistedLocalBtPaths: () => new Set(),
        replaceLiveBtFiles: jest.fn(),
        replaceDeletedBtPaths: jest.fn(),
      },
      routeView,
      setFlowEdges,
      tryAcquireRouteMutationLock: () => true,
      releaseRouteMutationLock: releaseLock,
      setRouteSourceAfterExternalMutation: setRouteSource,
      forgetTaskSelection: jest.fn(),
      setSelectedSpotId: jest.fn(),
      clearBehaviorSelection: jest.fn(),
      setTaskLayerSpotId: jest.fn(),
      setInteractionMode: jest.fn(),
      setShowWaypointOptions: jest.fn(),
      setBusy: jest.fn(),
      onMessage: jest.fn(),
      api: { ...api, deleteSpot },
    });
    return { controller, spots, setEditingSpotLabel };
  });
  return { view, markDirty, releaseLock, deleteSpot, setFlowEdges, setRouteSource };
}

test("serializes duplicate create gestures and cleans a stale created spot", async () => {
  const request = deferred();
  const createSpot = jest.fn(() => request.promise);
  const identity = { mapName: "map-a", missionName: "m1" };
  const { view, deleteSpot } = renderController({ createSpot }, identity);

  let first;
  await act(async () => {
    first = view.result.current.controller.createOnMap(1, 2, 0);
    await view.result.current.controller.createOnMap(3, 4, 0);
    identity.missionName = "m2";
    request.resolve({ ...SPOT, id: "new-spot" });
    try { await first; } catch { /* expected stale document */ }
  });

  expect(createSpot).toHaveBeenCalledTimes(1);
  expect(deleteSpot).toHaveBeenCalledWith("new-spot", "map-a");
  expect(view.result.current.spots).toEqual([SPOT]);
});

test("latest move wins when requests finish out of order", async () => {
  const first = deferred();
  const second = deferred();
  const updateSpot = jest.fn()
    .mockImplementationOnce(() => first.promise)
    .mockImplementationOnce(() => second.promise);
  const { view } = renderController({ updateSpot });

  let moveOne;
  let moveTwo;
  await act(async () => {
    moveOne = view.result.current.controller.moveWaypoint("wp-1", 1, 1, 0);
    moveTwo = view.result.current.controller.moveWaypoint("wp-1", 2, 2, 0);
    first.reject(new Error("old failure"));
    await moveOne;
    second.resolve({ ...SPOT, label: "stale-server-label", pose: { ...SPOT.pose, x: 2, y: 2 } });
    await moveTwo;
  });

  expect(view.result.current.spots[0].pose.x).toBe(2);
  expect(view.result.current.spots[0].label).toBe("Waypoint 1");
});

test("move queue rolls back to the last confirmed pose", async () => {
  const first = deferred();
  const second = deferred();
  const updateSpot = jest.fn()
    .mockImplementationOnce(() => first.promise)
    .mockImplementationOnce(() => second.promise);
  const { view } = renderController({ updateSpot });
  let moveOne;
  let moveTwo;
  await act(async () => {
    moveOne = view.result.current.controller.moveWaypoint("wp-1", 1, 1, 0);
    moveTwo = view.result.current.controller.moveWaypoint("wp-1", 2, 2, 0);
    first.resolve({ ...SPOT, pose: { ...SPOT.pose, x: 1, y: 1 } });
    await moveOne;
    second.reject(new Error("latest failed"));
    await moveTwo;
  });
  expect(view.result.current.spots[0].pose.x).toBe(1);

  const bothFailFirst = deferred();
  const bothFailSecond = deferred();
  const failedView = renderController({
    updateSpot: jest.fn()
      .mockImplementationOnce(() => bothFailFirst.promise)
      .mockImplementationOnce(() => bothFailSecond.promise),
  }).view;
  await act(async () => {
    const one = failedView.result.current.controller.moveWaypoint("wp-1", 3, 3, 0);
    const two = failedView.result.current.controller.moveWaypoint("wp-1", 4, 4, 0);
    bothFailFirst.reject(new Error("first failed"));
    await one;
    bothFailSecond.reject(new Error("second failed"));
    await two;
  });
  expect(failedView.result.current.spots[0].pose.x).toBe(0);
});

test("rename queue rolls back to the last confirmed label", async () => {
  const first = deferred();
  const second = deferred();
  const { view } = renderController({
    updateSpot: jest.fn()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise),
  });
  act(() => {
    view.result.current.setEditingSpotLabel("First");
  });
  let renameOne;
  act(() => { renameOne = view.result.current.controller.commitRename(SPOT); });
  act(() => { view.result.current.setEditingSpotLabel("Second"); });
  let renameTwo;
  await act(async () => {
    renameTwo = view.result.current.controller.commitRename(view.result.current.spots[0]);
    first.resolve({ ...SPOT, label: "First" });
    await renameOne;
    second.reject(new Error("latest failed"));
    await renameTwo;
  });
  expect(view.result.current.spots[0].label).toBe("First");

  const failOne = deferred();
  const failTwo = deferred();
  const failedView = renderController({
    updateSpot: jest.fn()
      .mockImplementationOnce(() => failOne.promise)
      .mockImplementationOnce(() => failTwo.promise),
  }).view;
  act(() => failedView.result.current.setEditingSpotLabel("Bad One"));
  let badOne;
  act(() => { badOne = failedView.result.current.controller.commitRename(SPOT); });
  act(() => failedView.result.current.setEditingSpotLabel("Bad Two"));
  let badTwo;
  act(() => {
    badTwo = failedView.result.current.controller.commitRename(
      failedView.result.current.spots[0],
    );
  });
  await act(async () => {
    failOne.reject(new Error("first failed"));
    await badOne;
    failTwo.reject(new Error("second failed"));
    await badTwo;
  });
  expect(failedView.result.current.spots[0].label).toBe("Waypoint 1");
});

test("move and rename responses preserve each other's fields and BT identity", async () => {
  const move = deferred();
  const rename = deferred();
  const updateSpot = jest.fn()
    .mockImplementationOnce(() => move.promise)
    .mockImplementationOnce(() => rename.promise);
  const { view } = renderController({ updateSpot });

  let moving;
  await act(async () => {
    moving = view.result.current.controller.moveWaypoint("wp-1", 5, 6, 0);
  });
  act(() => {
    view.result.current.controller.startRename(view.result.current.spots[0]);
    view.result.current.setEditingSpotLabel("Inspection");
  });
  let renaming;
  await act(async () => {
    renaming = view.result.current.controller.commitRename(view.result.current.spots[0]);
    rename.resolve({ ...SPOT, label: "Inspection", pose: SPOT.pose });
    await renaming;
    move.resolve({ ...SPOT, pose: { ...SPOT.pose, x: 5, y: 6 } });
    await moving;
  });

  expect(view.result.current.spots[0]).toMatchObject({
    label: "Inspection",
    pose: { x: 5, y: 6 },
    linked_bt_tree: SPOT.linked_bt_tree,
  });
});

test("failed delete leaves the document untouched and releases the route lock", async () => {
  const { view, releaseLock } = renderController({
    deleteSpot: jest.fn(async () => { throw new Error("delete failed"); }),
  });
  await act(async () => {
    await view.result.current.controller.deleteWaypoint(SPOT);
  });
  expect(view.result.current.spots).toEqual([SPOT]);
  expect(releaseLock).toHaveBeenCalledTimes(1);
});

test("deleting from a closed route preserves the loop when two waypoints remain", async () => {
  const routeView = {
    treeSpots: [SPOT, { id: "wp-2" }, { id: "wp-3" }],
    closed: true,
  };
  const { view, setFlowEdges, setRouteSource } = renderController({}, undefined, routeView);
  await act(async () => {
    await view.result.current.controller.deleteWaypoint(SPOT);
  });
  expect(setFlowEdges).toHaveBeenCalledWith(expect.arrayContaining([
    expect.objectContaining({ source: "wp-2", target: "wp-3" }),
    expect.objectContaining({ source: "wp-3", target: "wp-2" }),
  ]));
  expect(setRouteSource).toHaveBeenCalledWith("");
});

test("finalizes a failed create and releases the lock for retry", async () => {
  const finalize = jest.fn();
  const createSpot = jest.fn()
    .mockRejectedValueOnce(new Error("create failed"))
    .mockResolvedValueOnce({ ...SPOT, id: "wp-new", label: "Waypoint 2" });
  const { view } = renderController({ createSpot });
  await act(async () => {
    try {
      await view.result.current.controller.createAtRobot({
        resolvePose: async () => ({ x: 1, y: 1, yaw: 0 }),
        finalize,
      });
    } catch { /* expected */ }
  });
  await act(async () => {
    await view.result.current.controller.createAtRobot({
      resolvePose: async () => ({ x: 2, y: 2, yaw: 0 }),
      finalize,
    });
  });
  expect(createSpot).toHaveBeenCalledTimes(2);
  expect(finalize).toHaveBeenCalledTimes(2);
});
