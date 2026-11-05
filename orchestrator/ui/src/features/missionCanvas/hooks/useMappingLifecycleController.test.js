// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useMappingLifecycleController from "./useMappingLifecycleController";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup(overrides = {}) {
  const order = [];
  const call = (name, implementation) => jest.fn((...args) => {
    order.push(name);
    return implementation?.(...args);
  });
  const ports = {
    getMapName: () => " factory ",
    runtime: {
      prepareStart: call("runtime.prepareStart"),
      start: call("runtime.start", async () => ({ ok: true })),
      persistStarted: call("runtime.persistStarted"),
      commitStarted: call("runtime.commitStarted"),
      stop: call("runtime.stop", async () => ({ ok: true })),
      commitStopped: call("runtime.commitStopped"),
    },
    document: {
      save: call("document.save", async () => ({ saved: true })),
      commitSavedMap: call("document.commitSavedMap"),
    },
    inventory: {
      list: call("inventory.list", async () => ({ files: [] })),
      remove: call("inventory.remove", async () => ({})),
    },
    runCommand: call("runCommand", async (_label, action) => action()),
    onMessage: call("onMessage"),
    ...overrides,
  };
  const view = renderHook(() => useMappingLifecycleController(ports));
  return { view, ports, order };
}

test("starts Mapping with prepare, backend, and runtime/session commit order", async () => {
  const { view, ports, order } = setup();
  await act(async () => view.result.current.start());
  expect(order).toEqual([
    "runCommand", "runtime.prepareStart", "runtime.start",
    "runtime.persistStarted", "runtime.commitStarted",
  ]);
  expect(ports.runCommand).toHaveBeenCalledWith("Mapping", expect.any(Function));
  expect(ports.runtime.prepareStart).toHaveBeenCalledWith("factory");
  expect(ports.runtime.start).toHaveBeenCalledWith("factory");
  expect(ports.runtime.persistStarted).toHaveBeenCalledWith("factory", { ok: true });
  expect(ports.runtime.commitStarted).toHaveBeenCalledWith("factory", { ok: true });
  expect(view.result.current.operation).toBe("");
});

test("a synchronous operation lock rejects a double Start and releases for retry", async () => {
  const pending = deferred();
  const runtime = {
    prepareStart: jest.fn(),
    start: jest.fn(() => pending.promise),
    commitStarted: jest.fn(),
    stop: jest.fn(async () => ({})),
  };
  const { view, ports } = setup({ runtime });
  let first;
  act(() => { first = view.result.current.start(); });
  let second;
  await act(async () => { second = await view.result.current.start(); });
  expect(second).toEqual({ skipped: true, reason: "operation-active" });
  expect(runtime.start).toHaveBeenCalledTimes(1);
  expect(ports.onMessage).toHaveBeenCalledWith("Mapping is already in progress");
  await act(async () => { pending.resolve({ ok: true }); await first; });
  await act(async () => view.result.current.start());
  expect(runtime.start).toHaveBeenCalledTimes(2);
});

test("a rejected Start releases the lock and permits retry", async () => {
  const runtime = {
    prepareStart: jest.fn(),
    start: jest.fn()
      .mockRejectedValueOnce(new Error("start failed"))
      .mockResolvedValueOnce({ ok: true }),
    commitStarted: jest.fn(),
    stop: jest.fn(async () => ({})),
  };
  const runCommand = jest.fn(async (_label, action) => {
    try { return await action(); } catch { return undefined; }
  });
  const { view } = setup({ runtime, runCommand });
  await act(async () => view.result.current.start());
  expect(view.result.current.operation).toBe("");
  await act(async () => view.result.current.start());
  expect(runtime.start).toHaveBeenCalledTimes(2);
  expect(runtime.commitStarted).toHaveBeenCalledTimes(1);
});

test("Stop delegates shared shutdown and commits its final status/session result", async () => {
  const { view, ports, order } = setup();
  await act(async () => view.result.current.stop());
  expect(order).toEqual(["runCommand", "runtime.stop", "runtime.commitStopped"]);
  expect(ports.runCommand).toHaveBeenCalledWith("Stop", expect.any(Function));
  expect(ports.runtime.commitStopped).toHaveBeenCalledWith({ ok: true });
});

test("Save validates the name and commits only after the backend succeeds", async () => {
  const { view, ports, order } = setup();
  let missing;
  await act(async () => { missing = await view.result.current.save("  "); });
  expect(missing).toEqual({ skipped: true, reason: "map-name" });
  expect(ports.document.save).not.toHaveBeenCalled();
  expect(ports.onMessage).toHaveBeenCalledWith("Map name required");

  order.splice(0);
  await act(async () => view.result.current.save(" saved-map "));
  expect(order).toEqual(["runCommand", "document.save", "document.commitSavedMap"]);
  expect(ports.document.commitSavedMap).toHaveBeenCalledWith("saved-map", { saved: true });
});

test.each(["start", "stop", "save"])(
  "a late %s result after unmount skips UI commit but preserves durable start state",
  async (operation) => {
    const pending = deferred();
    const runtime = {
      prepareStart: jest.fn(),
      start: jest.fn(() => pending.promise),
      persistStarted: jest.fn(),
      commitStarted: jest.fn(),
      stop: jest.fn(() => pending.promise),
      commitStopped: jest.fn(),
    };
    const document = {
      save: jest.fn(() => pending.promise),
      commitSavedMap: jest.fn(),
    };
    const { view } = setup({ runtime, document });
    let running;
    act(() => {
      running = operation === "save"
        ? view.result.current.save("map-a")
        : view.result.current[operation]();
    });
    view.unmount();
    await act(async () => { pending.resolve({ ok: true }); await running; });
    expect(runtime.commitStarted).not.toHaveBeenCalled();
    expect(runtime.persistStarted).toHaveBeenCalledTimes(operation === "start" ? 1 : 0);
    expect(runtime.commitStopped).not.toHaveBeenCalled();
    expect(document.commitSavedMap).not.toHaveBeenCalled();
  },
);

test("only the newest inventory refresh may replace the saved-map list", async () => {
  const first = deferred();
  const second = deferred();
  const inventory = {
    list: jest.fn()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise),
    remove: jest.fn(async () => ({})),
  };
  const { view } = setup({ inventory });
  let runFirst;
  let runSecond;
  act(() => { runFirst = view.result.current.refreshInventory(); });
  act(() => { runSecond = view.result.current.refreshInventory(); });
  await act(async () => {
    second.resolve({ files: [{ path: "new.pgm" }] });
    await runSecond;
  });
  expect(view.result.current.savedMaps).toEqual([{ path: "new.pgm" }]);
  await act(async () => {
    first.resolve({ files: [{ path: "stale.pgm" }] });
    await runFirst;
  });
  expect(view.result.current.savedMaps).toEqual([{ path: "new.pgm" }]);
  expect(view.result.current.inventoryLoading).toBe(false);
});

test("inventory failure clears files and delete removes only the confirmed path", async () => {
  const inventory = {
    list: jest.fn()
      .mockResolvedValueOnce({ files: [{ path: "a.pgm" }, { path: "b.pgm" }] })
      .mockRejectedValueOnce(new Error("offline")),
    remove: jest.fn(async () => ({})),
  };
  const { view, ports } = setup({ inventory });
  await act(async () => view.result.current.refreshInventory());
  expect(view.result.current.savedMaps).toHaveLength(2);
  await act(async () => view.result.current.removeSavedMap("a.pgm"));
  expect(view.result.current.savedMaps).toEqual([{ path: "b.pgm" }]);
  expect(ports.onMessage).toHaveBeenLastCalledWith("Deleted map a.pgm");
  await act(async () => view.result.current.refreshInventory());
  expect(view.result.current.savedMaps).toEqual([]);
});

test("active inventory tokens refresh automatically and an inactive stage ignores late results", async () => {
  const first = deferred();
  const second = deferred();
  const inventory = {
    list: jest.fn()
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise),
    remove: jest.fn(async () => ({})),
  };
  const { view, ports } = setup({ active: true, inventoryRefreshToken: 0, inventory });
  expect(inventory.list).toHaveBeenCalledTimes(1);
  await act(async () => {
    first.resolve({ files: [{ path: "first.pgm" }] });
    await first.promise;
  });
  expect(view.result.current.savedMaps).toEqual([{ path: "first.pgm" }]);

  ports.inventoryRefreshToken = 1;
  view.rerender();
  expect(inventory.list).toHaveBeenCalledTimes(2);
  ports.active = false;
  view.rerender();
  await act(async () => {
    second.resolve({ files: [{ path: "late.pgm" }] });
    await second.promise;
  });
  expect(view.result.current.savedMaps).toEqual([{ path: "first.pgm" }]);
  expect(view.result.current.inventoryLoading).toBe(false);
});

test("a pending inventory list cannot resurrect a map deleted after it started", async () => {
  const staleList = deferred();
  const inventory = {
    list: jest.fn()
      .mockResolvedValueOnce({ files: [{ path: "keep.pgm" }, { path: "delete.pgm" }] })
      .mockImplementationOnce(() => staleList.promise),
    remove: jest.fn(async () => ({})),
  };
  const { view, ports } = setup({ active: true, inventoryRefreshToken: 0, inventory });
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
  expect(view.result.current.savedMaps).toEqual([
    { path: "keep.pgm" }, { path: "delete.pgm" },
  ]);

  ports.inventoryRefreshToken = 1;
  view.rerender();
  await act(async () => view.result.current.removeSavedMap("delete.pgm"));
  expect(view.result.current.savedMaps).toEqual([{ path: "keep.pgm" }]);
  await act(async () => {
    staleList.resolve({ files: [{ path: "keep.pgm" }, { path: "delete.pgm" }] });
    await staleList.promise;
  });
  expect(view.result.current.savedMaps).toEqual([{ path: "keep.pgm" }]);
});

test("a failed delete preserves inventory, reports the error, and permits retry", async () => {
  const inventory = {
    list: jest.fn(async () => ({ files: [{ path: "retry.pgm" }] })),
    remove: jest.fn()
      .mockRejectedValueOnce(new Error("delete failed"))
      .mockResolvedValueOnce({}),
  };
  const { view, ports } = setup({ active: true, inventoryRefreshToken: 0, inventory });
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });

  let failed;
  await act(async () => { failed = await view.result.current.removeSavedMap("retry.pgm"); });
  expect(failed).toEqual({ removed: false, error: expect.any(Error) });
  expect(view.result.current.savedMaps).toEqual([{ path: "retry.pgm" }]);
  expect(ports.onMessage).toHaveBeenLastCalledWith("delete failed");

  await act(async () => view.result.current.removeSavedMap("retry.pgm"));
  expect(inventory.remove).toHaveBeenCalledTimes(2);
  expect(view.result.current.savedMaps).toEqual([]);
});

test("a protected map never reaches the delete API", async () => {
  const inventory = {
    list: jest.fn(async () => ({ files: [{ path: "active.pgm" }] })),
    remove: jest.fn(async () => ({})),
    isProtected: jest.fn(() => true),
  };
  const { view, ports } = setup({ inventory });
  let result;
  await act(async () => { result = await view.result.current.removeSavedMap("active.pgm"); });
  expect(result).toEqual({ skipped: true, reason: "protected" });
  expect(inventory.remove).not.toHaveBeenCalled();
  expect(ports.onMessage).toHaveBeenCalledWith("Stop navigation before deleting this map");
});

test("the Save dialog uses the current map name, survives failure, and closes on success", async () => {
  const document = {
    save: jest.fn()
      .mockRejectedValueOnce(new Error("save failed"))
      .mockResolvedValueOnce({ saved: true }),
    commitSavedMap: jest.fn(),
  };
  const runCommand = jest.fn(async (_label, action) => {
    try { return await action(); } catch { return undefined; }
  });
  const { view } = setup({ document, runCommand });

  act(() => view.result.current.saveDialog.openDialog());
  expect(view.result.current.saveDialog).toMatchObject({ open: true, name: "factory" });
  await act(async () => view.result.current.saveDialog.confirm());
  expect(view.result.current.saveDialog.open).toBe(true);
  expect(view.result.current.saveDialog.name).toBe("factory");
  expect(document.commitSavedMap).not.toHaveBeenCalled();

  await act(async () => view.result.current.saveDialog.confirm());
  expect(document.save).toHaveBeenCalledTimes(2);
  expect(document.commitSavedMap).toHaveBeenCalledWith("factory", { saved: true });
  expect(view.result.current.saveDialog.open).toBe(false);
});
