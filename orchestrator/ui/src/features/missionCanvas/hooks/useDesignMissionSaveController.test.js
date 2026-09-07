// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useDesignMissionSaveController from "./useDesignMissionSaveController";

function baseSnapshot() {
  return {
    identity: { mapName: "map-a", mapPath: "/maps/map-a.pgm", missionName: "mission-a" },
    catalog: { mapName: "map-a", names: ["mission-a"] },
    content: {
      visibleSpots: [{ id: "wp-1", pose: { x: 1 } }],
      behaviorNodes: [{ id: "Sequence_1" }],
      missionFlowNodes: [{ id: "wp-1" }],
      missionFlowEdges: [{ id: "edge-1" }],
    },
    historyAtStart: { spots: [{ id: "wp-1" }] },
    loadError: "",
  };
}

function saveOutput(hasNewerEdits = false) {
  return {
    canonicalMissionSpots: [{ id: "wp-1", linked_bt_tree: "locals/wp_1/main.xml" }],
    savedHistorySnapshot: "saved-history",
    saveResult: { hasNewerEdits },
  };
}

function setup(overrides = {}) {
  const snapshot = overrides.snapshot || baseSnapshot();
  const persist = overrides.persist || jest.fn(async () => saveOutput(false));
  const runCommand = overrides.runCommand || jest.fn(async (_label, action) => action());
  const ports = {
    getSnapshot: jest.fn(() => snapshot),
    ledger: { beginSave: jest.fn() },
    content: { applySavedCanonicalSpots: jest.fn() },
    history: { reset: jest.fn(), rebase: jest.fn() },
    loader: {
      captureLease: jest.fn(() => ({ generation: 1 })),
      isCurrent: jest.fn(() => true),
      invalidate: jest.fn(),
    },
    catalog: { record: jest.fn(), refresh: jest.fn(async () => []) },
    getIdentity: jest.fn(() => ({ ...snapshot.identity })),
    setIdentity: jest.fn(),
    runCommand,
    onMessage: jest.fn(),
    isWaypointFileBusy: jest.fn(() => false),
    hasActiveSave: jest.fn(() => false),
    persist,
    ...overrides,
  };
  delete ports.snapshot;
  const view = renderHook(() => useDesignMissionSaveController(ports));
  return { view, ports, snapshot, persist, runCommand };
}

test.each([
  ["waypoint file", "isWaypointFileBusy", "waypoint-file-busy"],
  ["active save", "hasActiveSave", "save-active"],
])("preflights %s busy state before taking a snapshot", async (_label, query, reason) => {
  const override = { [query]: jest.fn(() => true) };
  const { view, ports, persist } = setup(override);
  let result;
  await act(async () => { result = await view.result.current.saveMission("mission-a"); });
  expect(result).toEqual({ skipped: true, reason });
  expect(ports.getSnapshot).not.toHaveBeenCalled();
  expect(persist).not.toHaveBeenCalled();
});

test("rejects an empty mission name before taking a snapshot", async () => {
  const { view, ports, runCommand, persist } = setup();
  let result;
  await act(async () => { result = await view.result.current.saveMission("   "); });
  expect(result).toEqual({ skipped: true, reason: "mission-name" });
  expect(ports.getSnapshot).not.toHaveBeenCalled();
  expect(runCommand).not.toHaveBeenCalled();
  expect(persist).not.toHaveBeenCalled();
});

test("preflights a failed document load without entering the save command", async () => {
  const snapshot = { ...baseSnapshot(), loadError: "BT revision conflict" };
  const { view, ports, runCommand, persist } = setup({ snapshot });
  await act(async () => view.result.current.saveMission("mission-a"));
  expect(runCommand).not.toHaveBeenCalled();
  expect(persist).not.toHaveBeenCalled();
  expect(ports.onMessage).toHaveBeenCalledWith(
    "Waypoint Task files did not finish loading. Reload the mission before saving.",
  );
});

test("captures content and history when Save is requested, not when its command starts", async () => {
  let startCommand;
  const commandGate = new Promise((resolve) => { startCommand = resolve; });
  let commandAction;
  const runCommand = jest.fn(async (_label, action) => {
    commandAction = action;
    await commandGate;
    return action();
  });
  const snapshot = baseSnapshot();
  const { view, persist } = setup({ snapshot, runCommand });
  let running;
  act(() => { running = view.result.current.saveMission("mission-a"); });
  expect(commandAction).toEqual(expect.any(Function));
  snapshot.content.visibleSpots[0].pose.x = 99;
  snapshot.historyAtStart.spots[0].id = "mutated";
  await act(async () => { startCommand(); await running; });
  expect(persist).toHaveBeenCalledWith(expect.objectContaining({
    visibleSpots: [{ id: "wp-1", pose: { x: 1 } }],
    historyAtStart: { spots: [{ id: "wp-1" }] },
  }));
});

test("a clean save applies canonical spots and starts a new history boundary", async () => {
  const { view, ports } = setup();
  let message;
  await act(async () => { message = await view.result.current.saveMission("mission-a"); });
  expect(ports.content.applySavedCanonicalSpots).toHaveBeenCalledWith(
    saveOutput(false).canonicalMissionSpots,
  );
  expect(ports.history.reset).toHaveBeenCalledTimes(1);
  expect(ports.history.rebase).not.toHaveBeenCalled();
  expect(ports.loader.invalidate).not.toHaveBeenCalled();
  expect(message).toBe("Saved mission-a for map-a");
});

test("a save with newer edits rebases history instead of deleting their undo entry", async () => {
  const persist = jest.fn(async () => saveOutput(true));
  const { view, ports } = setup({ persist });
  await act(async () => view.result.current.saveMission("mission-a"));
  expect(ports.history.rebase).toHaveBeenCalledWith("saved-history");
  expect(ports.history.reset).not.toHaveBeenCalled();
});

test("Save As invalidates the loader identity and commits the new mission identity", async () => {
  const { view, ports, persist } = setup();
  await act(async () => view.result.current.saveMission("mission-copy"));
  expect(persist).toHaveBeenCalledWith(expect.objectContaining({
    targetMissionName: "mission-copy",
    targetKnown: false,
  }));
  expect(ports.loader.invalidate).toHaveBeenCalledWith({
    mapName: "map-a", missionName: "mission-copy",
  });
  expect(ports.setIdentity).toHaveBeenCalledWith({
    mapName: "map-a", mapPath: "/maps/map-a.pgm", missionName: "mission-copy",
  });
});

test.each(["lease", "identity"])(
  "a changed source %s blocks UI post-commit after a durable save",
  async (staleKind) => {
    const loader = {
      captureLease: jest.fn(() => ({ generation: 1 })),
      isCurrent: staleKind === "lease"
        ? jest.fn().mockReturnValueOnce(true).mockReturnValue(false)
        : jest.fn(() => true),
      invalidate: jest.fn(),
    };
    const sourceIdentity = { mapName: "map-a", missionName: "mission-a" };
    const getIdentity = staleKind === "identity"
      ? jest.fn()
        .mockReturnValueOnce(sourceIdentity)
        .mockReturnValue({ mapName: "map-a", missionName: "mission-b" })
      : jest.fn(() => sourceIdentity);
    const { view, ports, persist } = setup({ loader, getIdentity });
    await act(async () => view.result.current.saveMission("mission-a"));
    expect(persist).toHaveBeenCalledTimes(1);
    expect(ports.content.applySavedCanonicalSpots).not.toHaveBeenCalled();
    expect(ports.history.reset).not.toHaveBeenCalled();
    expect(ports.setIdentity).not.toHaveBeenCalled();
    expect(ports.catalog.record).not.toHaveBeenCalled();
  },
);

test("catalog refresh failure keeps the durable optimistic catalog record", async () => {
  const catalog = {
    record: jest.fn(),
    refresh: jest.fn(async () => { throw new Error("catalog offline"); }),
  };
  const { view, ports } = setup({ catalog });
  await act(async () => view.result.current.saveMission("mission-a"));
  expect(ports.catalog.record).toHaveBeenCalledWith("map-a", "mission-a");
  expect(ports.content.applySavedCanonicalSpots).toHaveBeenCalledTimes(1);
  expect(ports.history.reset).toHaveBeenCalledTimes(1);
});

test.each(["rejected", "partial"])(
  "%s persistence never enters UI post-commit",
  async (failure) => {
    const ledger = { checkpointSaveUpload: jest.fn() };
    const persist = jest.fn(async () => {
      if (failure === "partial") ledger.checkpointSaveUpload();
      throw new Error("save failed");
    });
    const runCommand = jest.fn(async (_label, action) => {
      try { return await action(); } catch { return undefined; }
    });
    const { view, ports } = setup({ persist, runCommand, ledger });
    await act(async () => view.result.current.saveMission("mission-a"));
    expect(ledger.checkpointSaveUpload).toHaveBeenCalledTimes(failure === "partial" ? 1 : 0);
    expect(ports.content.applySavedCanonicalSpots).not.toHaveBeenCalled();
    expect(ports.history.reset).not.toHaveBeenCalled();
    expect(ports.history.rebase).not.toHaveBeenCalled();
    expect(ports.setIdentity).not.toHaveBeenCalled();
    expect(ports.catalog.record).not.toHaveBeenCalled();
  },
);
