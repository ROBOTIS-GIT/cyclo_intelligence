// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useMapEditStageCoordinator from "./useMapEditStageCoordinator";

function setup(overrides = {}) {
  const props = {
    active: true,
    currentMapName: "factory",
    currentMissionName: "inspection",
    editor: {
      files: [{ path: "first.pgm" }, { path: "second.pgm" }],
      selectedPath: "",
      setSelectedPath: jest.fn(),
      busy: false,
      undo: jest.fn(),
      redo: jest.fn(),
    },
    onSelectedMapIdentity: jest.fn(),
    ...overrides,
  };
  const view = renderHook(() => useMapEditStageCoordinator(props));
  return { view, props };
}

test("picker preselects the loaded path, then falls back to the first file", () => {
  const { view, props } = setup();
  act(() => view.result.current.picker.openPicker());
  expect(view.result.current.picker).toMatchObject({
    open: true,
    pendingPath: "first.pgm",
  });
  act(() => view.result.current.picker.cancelPicker());

  props.editor.selectedPath = "second.pgm";
  view.rerender();
  act(() => view.result.current.picker.openPicker());
  expect(view.result.current.picker.pendingPath).toBe("second.pgm");
});

test("cancel preserves the editor selection while confirm commits the pending path and closes", () => {
  const { view, props } = setup();
  act(() => view.result.current.picker.openPicker());
  act(() => view.result.current.picker.setPendingPath("second.pgm"));
  act(() => view.result.current.picker.cancelPicker());
  expect(props.editor.setSelectedPath).not.toHaveBeenCalled();
  expect(view.result.current.picker.open).toBe(false);

  act(() => view.result.current.picker.openPicker());
  act(() => view.result.current.picker.setPendingPath("second.pgm"));
  act(() => view.result.current.picker.confirmPicker());
  expect(props.editor.setSelectedPath).toHaveBeenCalledTimes(1);
  expect(props.editor.setSelectedPath).toHaveBeenCalledWith("second.pgm");
  expect(view.result.current.picker.open).toBe(false);
});

test("only an active changed selected path publishes its map and mission identity", () => {
  const { view, props } = setup();
  expect(props.onSelectedMapIdentity).not.toHaveBeenCalled();

  props.editor.selectedPath = "warehouse.pgm";
  view.rerender();
  expect(props.onSelectedMapIdentity).toHaveBeenCalledWith({
    mapName: "warehouse",
    missionName: "inspection",
  });
  props.onSelectedMapIdentity.mockClear();

  props.currentMapName = "warehouse";
  view.rerender();
  expect(props.onSelectedMapIdentity).not.toHaveBeenCalled();
  props.active = false;
  props.editor.selectedPath = "other.pgm";
  view.rerender();
  expect(props.onSelectedMapIdentity).not.toHaveBeenCalled();
});

test("shortcuts dispatch undo and redo only while active, idle, and outside the picker", () => {
  const { view, props } = setup();
  const dispatch = (key, options = {}, target = document) => {
    const event = new KeyboardEvent("keydown", {
      key, ctrlKey: true, cancelable: true, bubbles: true, ...options,
    });
    act(() => target.dispatchEvent(event));
    return event;
  };

  expect(dispatch("z").defaultPrevented).toBe(true);
  expect(props.editor.undo).toHaveBeenCalledTimes(1);
  expect(dispatch("z", { shiftKey: true }).defaultPrevented).toBe(true);
  expect(dispatch("y").defaultPrevented).toBe(true);
  expect(props.editor.redo).toHaveBeenCalledTimes(2);

  props.editor.busy = true;
  view.rerender();
  dispatch("z");
  expect(props.editor.undo).toHaveBeenCalledTimes(1);
  props.editor.busy = false;
  view.rerender();
  act(() => view.result.current.picker.openPicker());
  dispatch("z");
  expect(props.editor.undo).toHaveBeenCalledTimes(1);
  act(() => view.result.current.picker.cancelPicker());

  const input = document.createElement("input");
  document.body.appendChild(input);
  dispatch("z", {}, input);
  input.remove();
  expect(props.editor.undo).toHaveBeenCalledTimes(1);

  props.active = false;
  view.rerender();
  dispatch("z");
  expect(props.editor.undo).toHaveBeenCalledTimes(1);
});

test("unmount removes the keyboard listener", () => {
  const { view, props } = setup();
  view.unmount();
  act(() => document.dispatchEvent(new KeyboardEvent("keydown", {
    key: "z", ctrlKey: true, bubbles: true,
  })));
  expect(props.editor.undo).not.toHaveBeenCalled();
});
