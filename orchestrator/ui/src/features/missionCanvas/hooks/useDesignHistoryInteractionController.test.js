// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useDesignHistoryInteractionController from "./useDesignHistoryInteractionController";

function setup(overrides = {}) {
  const ports = {
    active: true, documentReady: true, busy: "", mapBusy: false,
    taskLayerOpen: false, canUndo: true, canRedo: true,
    undo: jest.fn(), redo: jest.fn(), onMessage: jest.fn(),
    ...overrides,
  };
  const view = renderHook((props) => useDesignHistoryInteractionController(props), {
    initialProps: ports,
  });
  return { view, ports };
}

test.each([
  [{ active: false }, true],
  [{ documentReady: false }, true],
  [{ busy: "Save" }, true],
  [{ mapBusy: true }, true],
  [{ taskLayerOpen: true }, true],
  [{}, false],
])("derives the exact history lock policy", (patch, expected) => {
  const { view } = setup(patch);
  expect(view.result.current.locked).toBe(expected);
});

test("actions honor lock and capability guards with exact success messages", () => {
  const { view, ports } = setup();
  expect(view.result.current.undoAction()).toBe(true);
  expect(view.result.current.redoAction()).toBe(true);
  expect(ports.undo).toHaveBeenCalledTimes(1);
  expect(ports.redo).toHaveBeenCalledTimes(1);
  expect(ports.onMessage).toHaveBeenNthCalledWith(1, "Undid design change");
  expect(ports.onMessage).toHaveBeenNthCalledWith(2, "Redid design change");

  view.rerender({ ...ports, busy: "Save", canUndo: true, canRedo: true });
  expect(view.result.current.undoAction()).toBe(false);
  expect(view.result.current.redoAction()).toBe(false);
  view.rerender({ ...ports, busy: "", canUndo: false, canRedo: false });
  expect(view.result.current.undoAction()).toBe(false);
  expect(view.result.current.redoAction()).toBe(false);
  expect(ports.undo).toHaveBeenCalledTimes(1);
  expect(ports.redo).toHaveBeenCalledTimes(1);
});

test.each([
  [{ key: "z", ctrlKey: true }, "undo"],
  [{ key: "Z", metaKey: true }, "undo"],
  [{ key: "z", ctrlKey: true, shiftKey: true }, "redo"],
  [{ key: "y", metaKey: true }, "redo"],
])("recognized shortcut dispatches %s and prevents default", (init, expected) => {
  const { ports } = setup();
  const event = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init });
  act(() => document.dispatchEvent(event));
  expect(event.defaultPrevented).toBe(true);
  expect(ports[expected]).toHaveBeenCalledTimes(1);
});

test("shift+y and unmodified keys are not recognized", () => {
  const { ports } = setup();
  const shiftY = new KeyboardEvent("keydown", {
    key: "y", ctrlKey: true, shiftKey: true, bubbles: true, cancelable: true,
  });
  const plainZ = new KeyboardEvent("keydown", { key: "z", bubbles: true, cancelable: true });
  act(() => {
    document.dispatchEvent(shiftY);
    document.dispatchEvent(plainZ);
  });
  expect(shiftY.defaultPrevented).toBe(false);
  expect(plainZ.defaultPrevented).toBe(false);
  expect(ports.undo).not.toHaveBeenCalled();
  expect(ports.redo).not.toHaveBeenCalled();
});

test("text inputs ignore shortcuts", () => {
  const { ports } = setup();
  const input = document.createElement("input");
  document.body.appendChild(input);
  const event = new KeyboardEvent("keydown", {
    key: "z", ctrlKey: true, bubbles: true, cancelable: true,
  });
  act(() => input.dispatchEvent(event));
  expect(event.defaultPrevented).toBe(false);
  expect(ports.undo).not.toHaveBeenCalled();
  input.remove();
});

test("busy and readiness remain live action guards while listener stays mounted", () => {
  const { view, ports } = setup();
  const undoAction = view.result.current.undoAction;
  view.rerender({ ...ports, busy: "Save", documentReady: true });
  const event = new KeyboardEvent("keydown", {
    key: "z", ctrlKey: true, bubbles: true, cancelable: true,
  });
  act(() => document.dispatchEvent(event));
  expect(event.defaultPrevented).toBe(true);
  expect(ports.undo).not.toHaveBeenCalled();
  expect(view.result.current.undoAction).toBe(undoAction);
});

test("inactive, task layer, and unmount remove the key listener", () => {
  const { view, ports } = setup({ active: false });
  const dispatchUndo = () => {
    const event = new KeyboardEvent("keydown", {
      key: "z", ctrlKey: true, bubbles: true, cancelable: true,
    });
    act(() => document.dispatchEvent(event));
    return event;
  };
  expect(dispatchUndo().defaultPrevented).toBe(false);
  view.rerender({ ...ports, active: true, taskLayerOpen: true });
  expect(dispatchUndo().defaultPrevented).toBe(false);
  view.rerender({ ...ports, active: true, taskLayerOpen: false });
  expect(dispatchUndo().defaultPrevented).toBe(true);
  view.unmount();
  expect(dispatchUndo().defaultPrevented).toBe(false);
  expect(ports.undo).toHaveBeenCalledTimes(1);
});
