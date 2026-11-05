// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { render } from "@testing-library/react";
import MissionCanvasDialogLayer from "./MissionCanvasDialogLayer";

const mockRendered = {
  saveMap: [],
  saveMission: [],
  confirm: [],
  loadMap: [],
};

jest.mock("../dialogs", () => ({
  SaveMapDialog: (props) => { mockRendered.saveMap.push(props); return null; },
  SaveMissionDialog: (props) => { mockRendered.saveMission.push(props); return null; },
  ConfirmDialog: (props) => { mockRendered.confirm.push(props); return null; },
  LoadMapDialog: (props) => { mockRendered.loadMap.push(props); return null; },
}));

const callback = () => {};

function configs(overrides = {}) {
  return {
    saveMap: { open: false, onCancel: callback, onSubmit: callback },
    saveMission: { open: false, onCancel: callback, onSubmit: callback },
    renameMission: { open: false, onCancel: callback, onSubmit: callback },
    duplicateMission: { open: false, onCancel: callback, onSubmit: callback },
    deleteMission: { open: false, onCancel: callback, onConfirm: callback },
    unsaved: { open: false, onCancel: callback, onConfirm: callback },
    designLoad: { open: false, onCancel: callback, onSubmit: callback },
    runLoad: { open: false, onCancel: callback, onSubmit: callback },
    editLoad: { open: false, onCancel: callback, onSubmit: callback },
    ...overrides,
  };
}

beforeEach(() => {
  Object.values(mockRendered).forEach((calls) => calls.splice(0));
});

test("forwards the hidden Design-load config beside an open unsaved dialog", () => {
  const props = configs({
    unsaved: { open: true, onCancel: callback },
    designLoad: { open: false, onCancel: callback },
  });
  render(<MissionCanvasDialogLayer {...props} />);

  expect(mockRendered.confirm[1]).toMatchObject({ open: true, title: "Unsaved changes" });
  expect(mockRendered.loadMap[0]).toMatchObject({ open: false, title: "Load Map" });
});

test("passes a null mission inventory to the Navigate map picker", () => {
  const props = configs({
    runLoad: {
      open: true,
      missionNames: null,
      navigationOnly: true,
      onCancel: callback,
      onSubmit: callback,
    },
  });
  render(<MissionCanvasDialogLayer {...props} />);

  expect(mockRendered.loadMap[1]).toMatchObject({
    open: true,
    missionNames: null,
    selectAriaLabel: "Navigation map file",
  });
});

test("forwards dialog configuration and callback identities unchanged", () => {
  const onSave = jest.fn();
  const onCancel = jest.fn();
  const onRunChange = jest.fn();
  const props = configs({
    saveMission: {
      open: true,
      busy: true,
      catalogReady: false,
      inputAriaLabel: "Mission name",
      onCancel,
      onSubmit: onSave,
    },
    runLoad: {
      open: true,
      busy: true,
      catalogReady: true,
      navigationOnly: false,
      onChange: onRunChange,
      onCancel,
      onSubmit: onSave,
    },
  });
  render(<MissionCanvasDialogLayer {...props} />);

  expect(mockRendered.saveMission[0]).toMatchObject({ ...props.saveMission, disallowExisting: true });
  expect(mockRendered.loadMap[1]).toMatchObject({
    ...props.runLoad,
    title: "Load Map",
    fieldLabel: "Map",
    selectAriaLabel: "Run mission map file",
    missionSelectAriaLabel: "Run mission file",
  });
  mockRendered.saveMission[0].onSubmit();
  mockRendered.loadMap[1].onCancel();
  mockRendered.loadMap[1].onChange("factory.pgm");
  expect(onSave).toHaveBeenCalledTimes(1);
  expect(onCancel).toHaveBeenCalledTimes(1);
  expect(onRunChange).toHaveBeenCalledWith("factory.pgm");
});

test("forwards every dialog slot in its stable render order", () => {
  const props = configs({
    saveMap: { slot: "save-map" },
    saveMission: { slot: "save-mission" },
    renameMission: { slot: "rename-mission" },
    duplicateMission: { slot: "duplicate-mission" },
    deleteMission: { slot: "delete-mission" },
    unsaved: { slot: "unsaved" },
    designLoad: { slot: "design-load" },
    runLoad: { slot: "run-load" },
    editLoad: { slot: "edit-load" },
  });
  render(<MissionCanvasDialogLayer {...props} />);

  expect(mockRendered.saveMap.map(({ slot }) => slot)).toEqual(["save-map"]);
  expect(mockRendered.saveMission.map(({ slot }) => slot)).toEqual([
    "save-mission",
    "rename-mission",
    "duplicate-mission",
  ]);
  expect(mockRendered.confirm.map(({ slot }) => slot)).toEqual([
    "delete-mission",
    "unsaved",
  ]);
  expect(mockRendered.loadMap.map(({ slot }) => slot)).toEqual([
    "design-load",
    "run-load",
    "edit-load",
  ]);
});
