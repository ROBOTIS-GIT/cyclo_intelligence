// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import { act, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import useWaypointTaskFileController from "./useWaypointTaskFileController";

const PATH = "locals/waypoint_1/main.xml";
const ALT_PATH = "locals/waypoint_1/inspect.xml";
const XML = "<root BTCPP_format=\"4\"><BehaviorTree ID=\"MainTree\"><Sequence /></BehaviorTree></root>";

function createLedger({
  activeSave = false,
  initialFiles = { [PATH]: XML, [ALT_PATH]: XML },
} = {}) {
  let files = { ...initialFiles };
  let revision = 4;
  const persisted = new Set([PATH, ALT_PATH]);
  return {
    get missionBtFiles() { return files; },
    recordBtEdit: jest.fn((path, content) => { files = { ...files, [path]: content }; }),
    replaceLiveBtFiles: jest.fn((next) => { files = typeof next === "function" ? next(files) : next; }),
    checkpointPersistedBtFile: jest.fn(({ path, content, revision: nextRevision }) => {
      files = { ...files, [path]: content };
      persisted.add(path);
      if (Number.isInteger(nextRevision)) revision = nextRevision;
    }),
    setPersistedRevision: jest.fn((next) => { if (Number.isInteger(next)) revision = next; }),
    reconcileDirty: jest.fn(),
    getLiveBtFiles: () => files,
    getPersistedRevision: () => revision,
    getPersistedLocalBtPaths: () => new Set(persisted),
    hasPersistedLocalBtPath: (path) => persisted.has(path),
    hasActiveSave: () => (
      typeof activeSave === "function" ? activeSave() : activeSave
    ),
  };
}

function renderController({
  ledger = createLedger(),
  api = {},
  captureDocumentLease = () => "lease",
  isDocumentLeaseCurrent = () => true,
  initialBusy = "",
} = {}) {
  const spot = {
    id: "wp-1",
    label: "Waypoint 1",
    local_bt: PATH,
    local_bt_files: [PATH, ALT_PATH],
  };
  const resetHistory = jest.fn();
  const captureHistory = jest.fn();
  const onMessage = jest.fn();
  const view = renderHook(() => {
    const [spots, setSpots] = useState([spot]);
    const [busy, setBusy] = useState(initialBusy);
    const controller = useWaypointTaskFileController({
      spots,
      selectedSpotId: spot.id,
      mapName: "warehouse",
      missionName: "Mission1",
      missionStored: true,
      busy,
      ledger,
      captureDocumentLease,
      isDocumentLeaseCurrent,
      captureHistory,
      resetHistory,
      saveMissionRef: { current: jest.fn() },
      setSpots,
      setBusy,
      onMessage,
      api: {
        getBtFile: jest.fn(async () => ({ exists: true, content: XML, revision: 4 })),
        saveBtFile: jest.fn(async () => ({ revision: 5 })),
        setDefaultBtFile: jest.fn(async () => ({ revision: 6 })),
        parseXml: jest.fn(),
        ...api,
      },
    });
    return { controller, spots, busy, setBusy };
  });
  return { view, ledger, resetHistory, captureHistory, onMessage };
}

test("selects a waypoint task file without changing the default", () => {
  const { view } = renderController();
  expect(view.result.current.controller.selectedPath).toBe(PATH);

  act(() => view.result.current.controller.selectXml(ALT_PATH));

  expect(view.result.current.controller.selectedPath).toBe(ALT_PATH);
  expect(view.result.current.controller.defaultPath).toBe(PATH);
});

test("saves XML through the revision-aware endpoint and checkpoints it", async () => {
  const saveBtFile = jest.fn(async () => ({ revision: 5 }));
  const { view, ledger } = renderController({ api: { saveBtFile } });
  const nextXml = XML.replace("Sequence", "Fallback");

  await act(async () => {
    await view.result.current.controller.saveXml(PATH, nextXml);
  });

  expect(saveBtFile).toHaveBeenCalledWith(
    "warehouse",
    PATH,
    nextXml,
    "Mission1",
    { waypointId: "wp-1", expectedRevision: 4 },
  );
  expect(ledger.checkpointPersistedBtFile).toHaveBeenCalledWith({
    path: PATH,
    content: nextXml,
    revision: 5,
  });
  expect(view.result.current.busy).toBe("");
});

test("default transition updates the spot and creates a history boundary", async () => {
  const setDefaultBtFile = jest.fn(async () => ({ revision: 6 }));
  const {
    view,
    resetHistory,
    ledger,
  } = renderController({ api: { setDefaultBtFile } });

  await act(async () => {
    await view.result.current.controller.setDefaultXml(ALT_PATH);
  });

  expect(setDefaultBtFile).toHaveBeenCalledWith(
    "warehouse",
    "wp-1",
    ALT_PATH,
    "Mission1",
    { expectedRevision: 4 },
  );
  expect(view.result.current.spots[0].linked_bt_tree).toBe(ALT_PATH);
  expect(resetHistory).toHaveBeenCalledTimes(1);
  expect(ledger.reconcileDirty).toHaveBeenCalledTimes(1);
});

test("rejects local file work while a full mission save owns the ledger", async () => {
  const { view } = renderController({ ledger: createLedger({ activeSave: true }) });
  let error;
  await act(async () => {
    try {
      await view.result.current.controller.loadXml(PATH);
    } catch (caught) {
      error = caught;
    }
  });
  expect(error).toEqual(new Error("A mission save is already in progress"));
});

test("exposes fallback hydration as busy until the XML request settles", async () => {
  let resolveRequest;
  const pending = new Promise((resolve) => { resolveRequest = resolve; });
  const ledger = createLedger({ initialFiles: {} });
  const { view } = renderController({
    ledger,
    api: { getBtFile: jest.fn(() => pending) },
  });

  expect(view.result.current.controller.isBusy()).toBe(true);
  expect(view.result.current.busy).toBe("Load Waypoint Task");

  await act(async () => {
    resolveRequest({ exists: true, content: XML, revision: 4 });
    await pending;
  });
  expect(view.result.current.controller.isBusy()).toBe(false);
});

test("ignores fallback hydration after the document lease becomes stale", async () => {
  let resolveRequest;
  let leaseCurrent = true;
  const pending = new Promise((resolve) => { resolveRequest = resolve; });
  const ledger = createLedger({ initialFiles: {} });
  renderController({
    ledger,
    isDocumentLeaseCurrent: () => leaseCurrent,
    api: { getBtFile: jest.fn(() => pending) },
  });

  leaseCurrent = false;
  await act(async () => {
    resolveRequest({ exists: true, content: XML, revision: 4 });
    await pending;
  });
  expect(ledger.getLiveBtFiles()[PATH]).toBeUndefined();
});

test("hydrates the selected file with its persisted XML", async () => {
  const hydratedXml = XML.replace("Sequence", "Fallback");
  const ledger = createLedger({ initialFiles: {} });
  const { view } = renderController({
    ledger,
    api: {
      getBtFile: jest.fn(async () => ({
        exists: true,
        content: hydratedXml,
        revision: 4,
      })),
    },
  });

  await waitFor(() => {
    expect(ledger.getLiveBtFiles()[PATH]).toBe(hydratedXml);
  });
  expect(view.result.current.controller.isBusy()).toBe(false);
});

test("rejects hydration from a different persisted revision", async () => {
  const ledger = createLedger({ initialFiles: {} });
  const { onMessage } = renderController({
    ledger,
    api: {
      getBtFile: jest.fn(async () => ({
        exists: true,
        content: XML,
        revision: 5,
      })),
    },
  });

  await waitFor(() => {
    expect(onMessage).toHaveBeenCalledWith(
      "Mission changed in another session. Reload it before editing.",
    );
  });
  expect(ledger.getLiveBtFiles()[PATH]).toBeUndefined();
});

test("retries hydration after an external full-save block is released", async () => {
  let saveActive = true;
  const getBtFile = jest.fn(async () => ({ exists: true, content: XML, revision: 4 }));
  const ledger = createLedger({
    activeSave: () => saveActive,
    initialFiles: {},
  });
  const { view } = renderController({
    ledger,
    initialBusy: "Save mission",
    api: { getBtFile },
  });
  expect(getBtFile).not.toHaveBeenCalled();

  saveActive = false;
  act(() => view.result.current.setBusy(""));

  await waitFor(() => expect(getBtFile).toHaveBeenCalledTimes(1));
  expect(ledger.getLiveBtFiles()[PATH]).toBe(XML);
});
