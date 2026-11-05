// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useDesignMissionSaveDialog, {
  confirmDesignMissionSave,
} from "./useDesignMissionSaveDialog";

function renderDialog(overrides = {}) {
  const initialProps = {
    missionName: "mission-a",
    existingNames: [],
    saveMission: jest.fn(),
    ...overrides,
  };
  return renderHook((props) => useDesignMissionSaveDialog(props), { initialProps });
}

test("saves an existing mission directly without opening the dialog", () => {
  const saveMission = jest.fn();
  const view = renderDialog({
    existingNames: ["mission-a", "mission-b"],
    saveMission,
  });

  act(() => view.result.current.requestSave());

  expect(saveMission).toHaveBeenCalledWith("mission-a");
  expect(view.result.current.open).toBe(false);
  expect(view.result.current.name).toBe("");
});

test("prefills the current name and opens for a new mission", () => {
  const view = renderDialog({ missionName: "new-mission" });

  act(() => view.result.current.requestSave());

  expect(view.result.current.open).toBe(true);
  expect(view.result.current.name).toBe("new-mission");
});

test.each(["", "   ", "bad/name"])(
  "keeps the dialog open and does not save invalid name %p",
  (invalidName) => {
    const saveMission = jest.fn();
    const view = renderDialog({ saveMission });
    act(() => view.result.current.requestSave());
    act(() => view.result.current.setName(invalidName));

    act(() => view.result.current.confirm());

    expect(view.result.current.open).toBe(true);
    expect(saveMission).not.toHaveBeenCalled();
  },
);

test("confirm helper closes before requesting a trimmed valid save", () => {
  const events = [];
  const close = jest.fn(() => { events.push("close"); });
  const saveMission = jest.fn((name) => { events.push(["save", name]); });

  const confirmed = confirmDesignMissionSave({
    name: "  saved-name  ",
    close,
    saveMission,
  });

  expect(confirmed).toBe(true);
  expect(events).toEqual(["close", ["save", "saved-name"]]);
});

test("a valid confirm saves and leaves the dialog closed", () => {
  const saveMission = jest.fn();
  const view = renderDialog({ saveMission });
  act(() => view.result.current.requestSave());
  act(() => view.result.current.setName("  saved-name  "));
  act(() => view.result.current.confirm());

  expect(saveMission).toHaveBeenCalledWith("saved-name");
  expect(view.result.current.open).toBe(false);
});

test("setName and confirm in the same batch use the new name", () => {
  const saveMission = jest.fn();
  const view = renderDialog({ saveMission });
  act(() => view.result.current.requestSave());

  act(() => {
    view.result.current.setName("same-batch-name");
    view.result.current.confirm();
  });

  expect(saveMission).toHaveBeenCalledWith("same-batch-name");
  expect(view.result.current.open).toBe(false);
});

test("cancel closes only and preserves the entered name", () => {
  const view = renderDialog();
  act(() => view.result.current.requestSave());
  act(() => view.result.current.setName("keep-this-name"));

  act(() => view.result.current.cancel());

  expect(view.result.current.open).toBe(false);
  expect(view.result.current.name).toBe("keep-this-name");
});

test("stable actions use the latest mission identity, catalog, and save callback after rerender", () => {
  const firstSave = jest.fn();
  const secondSave = jest.fn();
  const view = renderDialog({
    missionName: "first",
    existingNames: [],
    saveMission: firstSave,
  });
  const actions = {
    requestSave: view.result.current.requestSave,
    cancel: view.result.current.cancel,
    confirm: view.result.current.confirm,
  };

  view.rerender({
    missionName: "second",
    existingNames: ["second"],
    saveMission: secondSave,
  });

  expect(view.result.current.requestSave).toBe(actions.requestSave);
  expect(view.result.current.cancel).toBe(actions.cancel);
  expect(view.result.current.confirm).toBe(actions.confirm);
  act(() => actions.requestSave());
  expect(firstSave).not.toHaveBeenCalled();
  expect(secondSave).toHaveBeenCalledWith("second");
  expect(view.result.current.open).toBe(false);
});

test("a missing catalog is treated as an empty catalog", () => {
  const saveMission = jest.fn();
  const view = renderDialog({ existingNames: undefined, saveMission });

  act(() => view.result.current.requestSave());

  expect(saveMission).not.toHaveBeenCalled();
  expect(view.result.current.open).toBe(true);
  expect(view.result.current.name).toBe("mission-a");
});

test("confirm uses the latest save callback after rerender", () => {
  const firstSave = jest.fn();
  const secondSave = jest.fn();
  const view = renderDialog({ saveMission: firstSave });
  act(() => view.result.current.requestSave());
  act(() => view.result.current.setName("target"));
  const confirm = view.result.current.confirm;

  view.rerender({
    missionName: "changed",
    existingNames: [],
    saveMission: secondSave,
  });
  act(() => confirm());

  expect(firstSave).not.toHaveBeenCalled();
  expect(secondSave).toHaveBeenCalledWith("target");
});
