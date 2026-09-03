// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useDesignMissionDocumentLifecycleController from "./useDesignMissionDocumentLifecycleController";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup({ load } = {}) {
  let identity = { mapName: "map-a", mapPath: "/maps/map-a.pgm", missionName: "mission-a" };
  let requestGeneration = 0;
  let loaderGeneration = 0;
  const order = [];
  const call = (name, implementation) => jest.fn((...args) => {
    order.push(name);
    return implementation?.(...args);
  });
  const loader = {
    load: jest.fn(load || (async () => {
      loaderGeneration += 1;
      return {
        exists: true,
        lease: { generation: loaderGeneration },
        snapshot: { spots: [{ id: "wp-1" }] },
      };
    })),
    isCurrent: jest.fn((lease) => lease?.generation === loaderGeneration),
    invalidate: call("loader.invalidate", () => { loaderGeneration += 1; }),
  };
  const requests = {
    begin: call("requests.begin", () => {
      requestGeneration += 1;
      const generation = requestGeneration;
      return { isCurrent: () => generation === requestGeneration };
    }),
    finish: call("requests.finish"),
    invalidate: call("requests.invalidate", () => { requestGeneration += 1; }),
  };
  const ports = {
    getIdentity: () => ({ ...identity }),
    setIdentity: call("setIdentity", (next) => { identity = { ...next }; }),
    loader,
    requests,
    content: {
      commitLoadedSnapshot: call("content.commitLoadedSnapshot"),
      resetNewDocument: call("content.resetNewDocument"),
    },
    getCatalogNames: () => ["untitled", "untitled-2"],
    setPendingMissionName: call("setPendingMissionName"),
    clearDirty: call("clearDirty"),
    setLoadError: call("setLoadError"),
    onPrepareChange: call("onPrepareChange"),
    onMessage: call("onMessage"),
  };
  const view = renderHook(() => useDesignMissionDocumentLifecycleController(ports));
  return {
    view, ports, loader, requests, order,
    getIdentity: () => identity,
    setIdentityDirect: (next) => { identity = next; },
    setLoaderGeneration: (next) => { loaderGeneration = next; },
  };
}

test("picker selection eagerly prepares identity then commits one current snapshot", async () => {
  const { view, ports, order, getIdentity } = setup();
  await act(async () => view.result.current.confirmPickerSelection({
    mapName: "map-b",
    mapPath: "/maps/map-b.pgm",
    missionName: "mission-b",
    isCurrent: () => true,
  }));
  expect(getIdentity()).toEqual({
    mapName: "map-b", mapPath: "/maps/map-b.pgm", missionName: "mission-b",
  });
  expect(order.slice(0, 5)).toEqual([
    "setIdentity", "setPendingMissionName", "onPrepareChange", "setLoadError", "clearDirty",
  ]);
  expect(ports.content.commitLoadedSnapshot).toHaveBeenCalledTimes(1);
  expect(ports.onMessage).toHaveBeenLastCalledWith("Loaded mission mission-b for map-b");
});

test("A to B out-of-order switch commits only B", async () => {
  const a = deferred();
  const b = deferred();
  let callIndex = 0;
  const controller = setup({ load: jest.fn(() => (++callIndex === 1 ? a.promise : b.promise)) });
  const { view, ports, setLoaderGeneration } = controller;
  let runA;
  let runB;
  act(() => { runA = view.result.current.switchMission("mission-b"); });
  act(() => { runB = view.result.current.switchMission("mission-c"); });
  setLoaderGeneration(2);
  await act(async () => {
    b.resolve({ exists: true, lease: { generation: 2 }, snapshot: { id: "B" } });
    await runB;
  });
  expect(ports.content.commitLoadedSnapshot).toHaveBeenCalledWith({ id: "B" });
  await act(async () => {
    a.resolve({ exists: true, lease: { generation: 1 }, snapshot: { id: "A" } });
    await runA;
  });
  expect(ports.content.commitLoadedSnapshot).toHaveBeenCalledTimes(1);
});

test.each(["outer", "lease", "identity"])(
  "rejects a snapshot with stale %s ownership",
  async (staleKind) => {
    const response = deferred();
    const controller = setup({ load: jest.fn(() => response.promise) });
    const { view, ports, setIdentityDirect, setLoaderGeneration } = controller;
    const outer = { current: true };
    let running;
    act(() => {
      running = view.result.current.confirmPickerSelection({
        mapName: "map-b",
        mapPath: "/maps/map-b.pgm",
        missionName: "mission-b",
        isCurrent: () => outer.current,
      });
    });
    if (staleKind === "outer") outer.current = false;
    if (staleKind === "lease") setLoaderGeneration(2);
    else setLoaderGeneration(1);
    if (staleKind === "identity") {
      setIdentityDirect({ mapName: "map-c", missionName: "mission-c" });
    }
    await act(async () => {
      response.resolve({ exists: true, lease: { generation: 1 }, snapshot: { id: "stale" } });
      await running;
    });
    expect(ports.content.commitLoadedSnapshot).not.toHaveBeenCalled();
  },
);

test("a current load rejection preserves eager identity and reports reload guidance", async () => {
  const { view, ports, getIdentity } = setup({
    load: jest.fn(async () => { throw new Error("revision conflict"); }),
  });
  await act(async () => view.result.current.confirmPickerSelection({
    mapName: "map-b",
    mapPath: "/maps/map-b.pgm",
    missionName: "mission-b",
    isCurrent: () => true,
  }));
  expect(getIdentity().missionName).toBe("mission-b");
  expect(ports.content.commitLoadedSnapshot).not.toHaveBeenCalled();
  expect(ports.setLoadError).toHaveBeenLastCalledWith("revision conflict");
  expect(ports.onMessage).toHaveBeenLastCalledWith(
    "revision conflict. Reload the mission before saving.",
  );
});

test("New invalidates both request systems before resetting the document", () => {
  const { view, ports, order, getIdentity } = setup();
  let result;
  act(() => { result = view.result.current.newDocument({ btFiles: { "global.xml": "<new/>" } }); });
  expect(result.missionName).toBe("untitled-3");
  expect(getIdentity().missionName).toBe("untitled-3");
  expect(order).toEqual([
    "requests.invalidate",
    "loader.invalidate",
    "setIdentity",
    "setPendingMissionName",
    "onPrepareChange",
    "content.resetNewDocument",
    "onMessage",
  ]);
  expect(ports.content.resetNewDocument).toHaveBeenCalledWith({
    btFiles: { "global.xml": "<new/>" },
  });
});

test("delete continuation clears dirty then switches to the first remaining mission", async () => {
  const { view, ports, order } = setup();
  await act(async () => view.result.current.continueAfterDelete({
    remainingNames: ["mission-b", "mission-c"],
  }));
  expect(order[0]).toBe("clearDirty");
  expect(ports.setPendingMissionName).toHaveBeenCalledWith("mission-b");
  expect(ports.content.commitLoadedSnapshot).toHaveBeenCalledTimes(1);
});

test("delete continuation starts a supplied new document when none remain", async () => {
  const { view, ports, order } = setup();
  await act(async () => view.result.current.continueAfterDelete({
    remainingNames: [],
    newDocument: { missionName: "untitled", btFiles: { "global.xml": "<root/>" } },
  }));
  expect(order[0]).toBe("clearDirty");
  expect(ports.content.resetNewDocument).toHaveBeenCalledWith({
    btFiles: { "global.xml": "<root/>" },
  });
});
