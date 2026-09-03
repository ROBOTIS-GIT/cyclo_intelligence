// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { StrictMode } from "react";
import { act, renderHook } from "@testing-library/react";
import { STAGE_NAVIGATE, STAGE_RUN } from "../lib/stages";
import useRunMapMissionSelectionController from "./useRunMapMissionSelectionController";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup(overrides = {}, renderOptions = {}) {
  let currentStage = STAGE_RUN;
  let snapshotState = {
    mapName: "factory", missionName: "mission-a",
    catalog: { mapName: "factory", names: ["mission-a"] }, invalid: false,
  };
  const ports = {
    getCurrentStage: () => currentStage,
    getDefaults: () => ({
      mapName: "factory", missionName: "mission-a",
      selectedPath: "factory.pgm", selectedMission: "mission-a",
    }),
    inventory: {
      listMaps: jest.fn(async () => ({ files: [{ path: "factory.pgm" }] })),
      listMissions: jest.fn(async () => ["mission-a", "mission-b"]),
    },
    snapshot: {
      get: () => snapshotState,
      load: jest.fn(async (mapName, missionName) => {
        snapshotState = { ...snapshotState, mapName, missionName, invalid: false };
        return { exists: true };
      }),
      clear: jest.fn((identity = {}) => {
        snapshotState = { ...snapshotState, ...identity, invalid: true };
      }),
      invalidate: jest.fn(),
      cancelAndClear: jest.fn((identity = {}) => {
        snapshotState = { ...snapshotState, ...identity, invalid: true };
      }),
    },
    commits: {
      setStage: jest.fn((value) => { currentStage = value; }),
      setInteractionMode: jest.fn(), setMapLoaded: jest.fn(),
      resetPose: jest.fn(), invalidateGoal: jest.fn(),
    },
    onMessage: jest.fn(),
    ...overrides,
  };
  const view = renderHook(() => useRunMapMissionSelectionController(ports), renderOptions);
  return { view, ports, setStage: (value) => { currentStage = value; }, getSnapshot: () => snapshotState };
}

test("only the latest picker inventory request commits", async () => {
  const first = deferred();
  const second = deferred();
  const { view } = setup({
    inventory: {
      listMaps: jest.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise),
      listMissions: jest.fn(async () => []),
    },
  });
  let requestA;
  act(() => { requestA = view.result.current.dialog.openDialog(); });
  let requestB;
  act(() => { requestB = view.result.current.dialog.openDialog(); });
  await act(async () => { second.resolve({ files: [{ path: "new.pgm" }] }); await requestB; });
  await act(async () => { first.resolve({ files: [{ path: "old.pgm" }] }); await requestA; });
  expect(view.result.current.dialog.files).toEqual([{ path: "new.pgm" }]);
});

test("cancel invalidates a pending request", async () => {
  const pending = deferred();
  const { view } = setup({
    inventory: { listMaps: jest.fn(() => pending.promise), listMissions: jest.fn() },
  });
  let request;
  act(() => { request = view.result.current.dialog.openDialog(); });
  act(() => view.result.current.dialog.cancel());
  await act(async () => { pending.resolve({ files: [{ path: "late.pgm" }] }); await request; });
  expect(view.result.current.dialog.open).toBe(false);
  expect(view.result.current.dialog.files).toEqual([]);
  expect(view.result.current.dialog.busy).toBe(false);
});

test("Navigate never requests mission inventory", async () => {
  const { view, ports, setStage } = setup();
  setStage(STAGE_NAVIGATE);
  await act(async () => view.result.current.dialog.openDialog());
  expect(ports.inventory.listMissions).not.toHaveBeenCalled();
  expect(view.result.current.dialog.missionNames).toBeNull();
});

test("Run commits map loaded only after the durable snapshot load", async () => {
  const pending = deferred();
  const { view, ports } = setup();
  ports.snapshot.load.mockReturnValue(pending.promise);
  await act(async () => view.result.current.dialog.openDialog());
  let confirm;
  act(() => { confirm = view.result.current.dialog.confirm(); });
  expect(ports.commits.setMapLoaded).toHaveBeenLastCalledWith(false);
  expect(ports.snapshot.cancelAndClear).toHaveBeenCalled();
  expect(ports.commits.setMapLoaded).not.toHaveBeenCalledWith(true);
  await act(async () => { pending.resolve({ exists: true }); await confirm; });
  expect(ports.commits.setMapLoaded).toHaveBeenLastCalledWith(true);
});

test("same-map Navigate preserves a valid Run snapshot", async () => {
  const { view, ports, setStage } = setup();
  setStage(STAGE_NAVIGATE);
  await act(async () => view.result.current.dialog.openDialog());
  await act(async () => view.result.current.dialog.confirm());
  expect(ports.snapshot.cancelAndClear).not.toHaveBeenCalled();
  expect(ports.commits.resetPose).not.toHaveBeenCalled();
  expect(ports.commits.setMapLoaded).toHaveBeenCalledWith(true);
});

test("different-map Navigate clears the mission snapshot and pose", async () => {
  const { view, ports, setStage } = setup();
  setStage(STAGE_NAVIGATE);
  ports.inventory.listMaps.mockResolvedValue({ files: [{ path: "other.pgm" }] });
  await act(async () => view.result.current.dialog.openDialog());
  await act(async () => view.result.current.dialog.confirm());
  expect(ports.snapshot.cancelAndClear).toHaveBeenCalledWith({ mapName: "other" });
  expect(ports.commits.resetPose).toHaveBeenCalled();
});

test("a failed mission switch preserves the previous snapshot", async () => {
  const { view, ports, getSnapshot } = setup();
  ports.snapshot.load.mockRejectedValue(new Error("load failed"));
  const before = getSnapshot();
  await act(async () => view.result.current.switchMission("mission-b"));
  expect(getSnapshot()).toBe(before);
  expect(ports.snapshot.cancelAndClear).not.toHaveBeenCalled();
  expect(ports.onMessage).toHaveBeenCalledWith("load failed");
});

test("a stale Run load cannot commit map loaded", async () => {
  const pending = deferred();
  const { view, ports } = setup();
  ports.snapshot.load.mockReturnValue(pending.promise);
  await act(async () => view.result.current.dialog.openDialog());
  let confirm;
  act(() => { confirm = view.result.current.dialog.confirm(); });
  act(() => view.result.current.dialog.cancel());
  await act(async () => { pending.resolve({ exists: true }); await confirm; });
  expect(ports.commits.setMapLoaded).not.toHaveBeenCalledWith(true);
});

test("StrictMode still permits the current open inventory request to commit", async () => {
  const inventory = {
    listMaps: jest.fn(async () => ({ files: [{ path: "strict.pgm" }] })),
    listMissions: jest.fn(async () => ["strict-mission"]),
  };
  const { view, ports } = setup(
    { inventory },
    { wrapper: StrictMode },
  );

  await act(async () => view.result.current.dialog.openDialog());

  expect(view.result.current.dialog).toMatchObject({
    open: true,
    busy: false,
    files: [{ path: "strict.pgm" }],
    missionNames: ["strict-mission"],
    selectedPath: "strict.pgm",
    selectedMission: "strict-mission",
  });
  expect(ports.commits.setStage).toHaveBeenCalledWith(STAGE_RUN);
  expect(ports.onMessage).toHaveBeenCalledWith("Loading saved missions");
});

test("unmount ignores a late map inventory result without messages or commits", async () => {
  const maps = deferred();
  const inventory = {
    listMaps: jest.fn(() => maps.promise),
    listMissions: jest.fn(async () => ["unused"]),
  };
  const { view, ports } = setup({ inventory });
  let request;
  act(() => { request = view.result.current.dialog.openDialog(); });
  ports.onMessage.mockClear();
  Object.values(ports.commits).forEach((commit) => commit.mockClear());

  view.unmount();
  await act(async () => {
    maps.resolve({ files: [] });
    await request;
  });

  expect(ports.onMessage).not.toHaveBeenCalled();
  Object.values(ports.commits).forEach((commit) => expect(commit).not.toHaveBeenCalled());
  expect(inventory.listMissions).not.toHaveBeenCalled();
});

test("unmount ignores a late mission inventory result without messages or commits", async () => {
  const missions = deferred();
  const inventory = {
    listMaps: jest.fn(async () => ({ files: [{ path: "factory.pgm" }] })),
    listMissions: jest.fn(() => missions.promise),
  };
  const { view, ports } = setup({ inventory });
  let request;
  act(() => { request = view.result.current.dialog.openDialog(); });
  await act(async () => { await Promise.resolve(); });
  expect(inventory.listMissions).toHaveBeenCalledWith("factory");
  ports.onMessage.mockClear();
  Object.values(ports.commits).forEach((commit) => commit.mockClear());

  view.unmount();
  await act(async () => {
    missions.resolve([]);
    await request;
  });

  expect(ports.onMessage).not.toHaveBeenCalled();
  Object.values(ports.commits).forEach((commit) => expect(commit).not.toHaveBeenCalled());
});
