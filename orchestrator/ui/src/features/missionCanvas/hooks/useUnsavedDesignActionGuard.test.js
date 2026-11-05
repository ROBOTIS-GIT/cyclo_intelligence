// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import { StrictMode } from "react";
import useUnsavedDesignActionGuard from "./useUnsavedDesignActionGuard";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup({ dirty = false, save = jest.fn(async () => {}) } = {}) {
  let isDirty = dirty;
  let documentKey = "map-a/mission-a";
  const clearDirty = jest.fn(() => { isDirty = false; });
  const view = renderHook(() => useUnsavedDesignActionGuard({
    isDirty: () => isDirty,
    clearDirty,
    save,
    documentKey,
  }));
  return {
    view,
    save,
    clearDirty,
    setDirty: (next) => { isDirty = next; },
    setDocumentKey: (next) => { documentKey = next; view.rerender(); },
  };
}

test("runs a clean action immediately without opening the dialog", () => {
  const { view } = setup();
  const action = jest.fn();
  let result;
  act(() => { result = view.result.current.runGuardedAction(action); });
  expect(result).toBe(true);
  expect(action).toHaveBeenCalledTimes(1);
  expect(view.result.current.open).toBe(false);
});

test("keeps a dirty action pending until discard clears dirty state", async () => {
  const { view, clearDirty } = setup({ dirty: true });
  const action = jest.fn();
  act(() => view.result.current.runGuardedAction(action));
  expect(view.result.current.open).toBe(true);
  expect(action).not.toHaveBeenCalled();
  await act(async () => view.result.current.resolve("discard"));
  expect(clearDirty).toHaveBeenCalledTimes(1);
  expect(action).toHaveBeenCalledTimes(1);
  expect(view.result.current.open).toBe(false);
});

test("runs the pending action only after save leaves the document clean", async () => {
  const { view, save, setDirty } = setup({ dirty: true });
  save.mockImplementation(async () => { setDirty(false); });
  const action = jest.fn();
  act(() => view.result.current.runGuardedAction(action));
  await act(async () => view.result.current.resolve("save"));
  expect(save).toHaveBeenCalledTimes(1);
  expect(action).toHaveBeenCalledTimes(1);
  expect(view.result.current.open).toBe(false);
  expect(view.result.current.saving).toBe(false);
});

test.each(["reject", "dirty"])(
  "keeps the prompt and action pending after a %s save",
  async (failure) => {
    const save = failure === "reject"
      ? jest.fn(async () => { throw new Error("save failed"); })
      : jest.fn(async () => {});
    const { view } = setup({ dirty: true, save });
    const action = jest.fn();
    act(() => view.result.current.runGuardedAction(action));
    await act(async () => view.result.current.resolve("save"));
    expect(action).not.toHaveBeenCalled();
    expect(view.result.current.open).toBe(true);
    expect(view.result.current.saving).toBe(false);
  },
);

test("cancel drops the pending action", async () => {
  const { view, clearDirty } = setup({ dirty: true });
  const action = jest.fn();
  act(() => view.result.current.runGuardedAction(action));
  act(() => view.result.current.cancel());
  await act(async () => view.result.current.resolve("discard"));
  expect(action).not.toHaveBeenCalled();
  expect(clearDirty).not.toHaveBeenCalled();
  expect(view.result.current.open).toBe(false);
});

test("a late save cannot close or run a newer pending action", async () => {
  const pendingSave = deferred();
  const { view, setDirty } = setup({ dirty: true, save: jest.fn(() => pendingSave.promise) });
  const firstAction = jest.fn();
  const newerAction = jest.fn();
  act(() => view.result.current.runGuardedAction(firstAction));
  let saveRun;
  act(() => { saveRun = view.result.current.resolve("save"); });
  act(() => view.result.current.runGuardedAction(newerAction));
  setDirty(false);
  await act(async () => { pendingSave.resolve(); await saveRun; });
  expect(firstAction).not.toHaveBeenCalled();
  expect(newerAction).not.toHaveBeenCalled();
  expect(view.result.current.open).toBe(true);
  await act(async () => view.result.current.resolve("discard"));
  expect(newerAction).toHaveBeenCalledTimes(1);
});

test("cancel and unmount invalidate an in-flight save completion", async () => {
  const pendingSave = deferred();
  const { view, setDirty } = setup({ dirty: true, save: jest.fn(() => pendingSave.promise) });
  const action = jest.fn();
  act(() => view.result.current.runGuardedAction(action));
  let saveRun;
  act(() => { saveRun = view.result.current.resolve("save"); });
  act(() => view.result.current.cancel());
  setDirty(false);
  await act(async () => { pendingSave.resolve(); await saveRun; });
  expect(action).not.toHaveBeenCalled();
  expect(view.result.current.open).toBe(false);

  const secondSave = deferred();
  const second = setup({ dirty: true, save: jest.fn(() => secondSave.promise) });
  const secondAction = jest.fn();
  act(() => second.view.result.current.runGuardedAction(secondAction));
  let secondRun;
  act(() => { secondRun = second.view.result.current.resolve("save"); });
  second.view.unmount();
  second.setDirty(false);
  await act(async () => { secondSave.resolve(); await secondRun; });
  expect(secondAction).not.toHaveBeenCalled();
});

test("a document identity change invalidates its pending action and captured save", async () => {
  const save = jest.fn(async () => {});
  const { view, setDocumentKey } = setup({ dirty: true, save });
  const action = jest.fn();
  act(() => view.result.current.runGuardedAction(action));
  act(() => setDocumentKey("map-b/mission-b"));
  expect(view.result.current.open).toBe(false);
  await act(async () => view.result.current.resolve("save"));
  expect(save).not.toHaveBeenCalled();
  expect(action).not.toHaveBeenCalled();
});

test("accepts a saved action after the StrictMode effect replay", async () => {
  let dirty = true;
  const action = jest.fn();
  const view = renderHook(() => useUnsavedDesignActionGuard({
    isDirty: () => dirty,
    clearDirty: () => { dirty = false; },
    save: async () => { dirty = false; },
  }), { wrapper: StrictMode });
  act(() => view.result.current.runGuardedAction(action));
  await act(async () => view.result.current.resolve("save"));
  expect(action).toHaveBeenCalledTimes(1);
});
