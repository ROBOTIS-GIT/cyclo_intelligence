// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useDesignMissionCatalogController from "./useDesignMissionCatalogController";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup(api = {}, overrides = {}) {
  let identity = { mapName: "map-a", missionName: "mission-a" };
  const invalidateDocument = jest.fn();
  const onDeleted = jest.fn();
  const onRenamed = jest.fn((missionName) => { identity = { ...identity, missionName }; });
  const onConfirmSelection = jest.fn(async () => {});
  const runCommand = jest.fn(async (_label, action) => {
    try { return await action(); } catch { return undefined; }
  });
  const resolvedApi = {
    listMaps: jest.fn(async () => ({ files: [{ path: "/maps/map-a.pgm" }] })),
    listMissions: jest.fn(async () => ({ missions: ["mission-a"] })),
    renameMission: jest.fn(async () => ({ revision: 8 })),
    duplicateMission: jest.fn(async () => ({})),
    deleteMission: jest.fn(async () => ({})),
    ...api,
  };
  const view = renderHook(() => useDesignMissionCatalogController({
    currentMapName: identity.mapName,
    currentMapPath: "/maps/map-a.pgm",
    currentMissionName: identity.missionName,
    getPersistedRevision: () => 7,
    setPersistedRevision: jest.fn(),
    invalidateDocument,
    runCommand,
    onMessage: jest.fn(),
    onConfirmSelection,
    onRenamed,
    onDeleted,
    api: resolvedApi,
    ...overrides,
  }));
  return {
    view, api: resolvedApi, invalidateDocument, onDeleted, onRenamed, onConfirmSelection,
    setIdentity: (next) => { identity = next; view.rerender(); },
  };
}

test("ignores an open request after cancel and a newer reopen", async () => {
  const firstMaps = deferred();
  const secondMaps = deferred();
  const listMaps = jest.fn()
    .mockImplementationOnce(() => firstMaps.promise)
    .mockImplementationOnce(() => secondMaps.promise);
  const { view } = setup({ listMaps });
  act(() => view.result.current.picker.openPicker());
  act(() => view.result.current.picker.cancelPicker());
  act(() => view.result.current.picker.openPicker());
  await act(async () => {
    secondMaps.resolve({ files: [{ path: "/maps/map-b.pgm" }] });
    await Promise.resolve(); await Promise.resolve();
  });
  expect(view.result.current.picker.pendingMapPath).toBe("/maps/map-b.pgm");
  await act(async () => {
    firstMaps.resolve({ files: [{ path: "/maps/stale.pgm" }] });
    await Promise.resolve(); await Promise.resolve();
  });
  expect(view.result.current.picker.pendingMapPath).toBe("/maps/map-b.pgm");
});

test("only the newest A to B mission inventory writes picker state", async () => {
  const a = deferred();
  const b = deferred();
  const listMissions = jest.fn((mapName) => (mapName === "a" ? a.promise : b.promise));
  const { view } = setup({ listMissions });
  act(() => view.result.current.picker.changePendingMap("/maps/a.pgm"));
  act(() => view.result.current.picker.changePendingMap("/maps/b.pgm"));
  await act(async () => { b.resolve({ missions: ["b-one"] }); await b.promise; });
  expect(view.result.current.picker.missionNames).toEqual(["b-one"]);
  await act(async () => { a.resolve({ missions: ["a-stale"] }); await a.promise; });
  expect(view.result.current.picker.missionNames).toEqual(["b-one"]);
});

test("double confirm is locked while the first document request is active", async () => {
  const pending = deferred();
  const onConfirmSelection = jest.fn(() => pending.promise);
  const { view } = setup({}, { onConfirmSelection });
  await act(async () => {
    view.result.current.picker.openPicker();
    await Promise.resolve(); await Promise.resolve();
  });
  await act(async () => {
    const first = view.result.current.picker.confirmSelection();
    const second = view.result.current.picker.confirmSelection();
    expect(onConfirmSelection).toHaveBeenCalledTimes(1);
    pending.resolve();
    await Promise.all([first, second]);
  });
});

test("delete rejection keeps the lease, catalog and callback untouched and permits retry", async () => {
  const deleteMission = jest.fn()
    .mockRejectedValueOnce(new Error("conflict"))
    .mockResolvedValueOnce({});
  const { view, invalidateDocument, onDeleted } = setup({ deleteMission });
  act(() => view.result.current.recordSavedMission("map-a", "mission-a"));
  act(() => view.result.current.deletion.openDialog());
  await act(async () => { view.result.current.deletion.confirm(); await Promise.resolve(); });
  expect(invalidateDocument).not.toHaveBeenCalled();
  expect(onDeleted).not.toHaveBeenCalled();
  expect(view.result.current.catalog.names).toContain("mission-a");
  act(() => view.result.current.deletion.openDialog());
  await act(async () => { view.result.current.deletion.confirm(); await Promise.resolve(); await Promise.resolve(); });
  expect(deleteMission).toHaveBeenCalledTimes(2);
  expect(invalidateDocument).toHaveBeenCalledTimes(1);
  expect(onDeleted).toHaveBeenCalledTimes(1);
});

test("late refresh cannot overwrite a new identity catalog", async () => {
  const refresh = deferred();
  const listMissions = jest.fn(() => refresh.promise);
  const { view, setIdentity } = setup({ listMissions });
  let running;
  act(() => { running = view.result.current.refreshCatalog("map-a"); });
  act(() => setIdentity({ mapName: "map-b", missionName: "mission-b" }));
  act(() => view.result.current.recordSavedMission("map-b", "mission-b"));
  await act(async () => { refresh.resolve({ missions: ["stale"] }); await running; });
  expect(view.result.current.catalog).toEqual({ mapName: "map-b", names: ["mission-b"] });
});

test("rename keeps document content in place and updates identity only after API success", async () => {
  const rename = deferred();
  const { view, api, invalidateDocument, onRenamed } = setup({
    renameMission: jest.fn(() => rename.promise),
    listMissions: jest.fn(async () => ({ missions: ["renamed"] })),
  });
  act(() => view.result.current.recordSavedMission("map-a", "mission-a"));
  act(() => view.result.current.rename.openDialog());
  act(() => view.result.current.rename.setName("renamed"));
  act(() => view.result.current.rename.confirm());
  expect(invalidateDocument).not.toHaveBeenCalled();
  await act(async () => { rename.resolve({ revision: 8 }); await rename.promise; await Promise.resolve(); });
  expect(api.renameMission).toHaveBeenCalledWith("map-a", "mission-a", "renamed", {
    expectedRevision: 7,
  });
  expect(invalidateDocument).toHaveBeenCalledWith({ mapName: "map-a", missionName: "renamed" });
  expect(onRenamed).toHaveBeenCalledWith("renamed");
});

test("duplicate leaves the current identity unchanged and retains optimistic entry on refresh failure", async () => {
  const { view, api, invalidateDocument, onRenamed } = setup({
    listMissions: jest.fn(async () => { throw new Error("refresh failed"); }),
  });
  act(() => view.result.current.recordSavedMission("map-a", "mission-a"));
  act(() => view.result.current.duplicate.openDialog());
  act(() => view.result.current.duplicate.setName("copy"));
  await act(async () => { view.result.current.duplicate.confirm(); await Promise.resolve(); await Promise.resolve(); });
  expect(api.duplicateMission).toHaveBeenCalledWith("map-a", "mission-a", "copy", {
    expectedRevision: 7,
  });
  expect(invalidateDocument).not.toHaveBeenCalled();
  expect(onRenamed).not.toHaveBeenCalled();
  expect(view.result.current.catalog.names).toContain("copy");
});

test("delete completion is discarded when identity changes while catalog refresh is pending", async () => {
  const refresh = deferred();
  const { view, invalidateDocument, onDeleted, setIdentity } = setup({
    deleteMission: jest.fn(async () => ({})),
    listMissions: jest.fn(() => refresh.promise),
  });
  act(() => view.result.current.recordSavedMission("map-a", "mission-a"));
  act(() => view.result.current.deletion.openDialog());
  act(() => view.result.current.deletion.confirm());
  await act(async () => { await Promise.resolve(); });
  act(() => setIdentity({ mapName: "map-a", missionName: "mission-b" }));
  await act(async () => {
    refresh.resolve({ missions: ["mission-b"] });
    await refresh.promise;
    await Promise.resolve();
  });
  expect(invalidateDocument).not.toHaveBeenCalled();
  expect(onDeleted).not.toHaveBeenCalled();
});

test("an invalidated confirm lease stays stale and cannot be submitted again", async () => {
  const pending = deferred();
  let capturedLease;
  const onConfirmSelection = jest.fn((selection) => {
    capturedLease = selection.isCurrent;
    return pending.promise;
  });
  const { view } = setup({}, { onConfirmSelection });
  await act(async () => {
    view.result.current.picker.openPicker();
    await Promise.resolve(); await Promise.resolve();
  });
  await act(async () => {
    const first = view.result.current.picker.confirmSelection();
    expect(capturedLease()).toBe(true);
    view.result.current.invalidateRequests();
    expect(capturedLease()).toBe(false);
    const second = view.result.current.picker.confirmSelection();
    expect(onConfirmSelection).toHaveBeenCalledTimes(1);
    pending.resolve();
    await Promise.all([first, second]);
  });
});
