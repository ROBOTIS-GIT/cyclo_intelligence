// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { act, renderHook } from "@testing-library/react";
import useDesignMissionDocumentLedger from "./useDesignMissionDocumentLedger";

const GLOBAL_PATH = "global.xml";
const LOCAL_PATH = "locals/waypoint_1/main.xml";
const SECOND_LOCAL_PATH = "locals/waypoint_2/main.xml";
const OLD_DELETED_PATH = "locals/deleted-old/main.xml";
const NEW_DELETED_PATH = "locals/deleted-new/main.xml";
const OLD_FILES = {
  [GLOBAL_PATH]: "global-old",
  [LOCAL_PATH]: "local-old",
};

function renderLedger(options = {}) {
  return renderHook(() => useDesignMissionDocumentLedger(options));
}

function commitPersisted(view, {
  files = OLD_FILES,
  localPaths = [LOCAL_PATH],
  revision = 7,
} = {}) {
  act(() => {
    view.result.current.commitSnapshot({
      btFiles: files,
      baseline: {
        persistedBtFiles: files,
        persistedLocalBtPaths: localPaths,
        revision,
      },
    });
  });
}

test("atomically commits a loaded BT snapshot and its persisted baseline", () => {
  const view = renderLedger({
    initialMissionBtFiles: { stale: "xml" },
    initialNonBtDirty: true,
    initialDeletedMissionBtPaths: ["locals/stale/main.xml"],
  });

  commitPersisted(view);

  expect(view.result.current.missionBtFiles).toEqual(OLD_FILES);
  expect(view.result.current.deletedMissionBtPaths).toEqual([]);
  expect(view.result.current.designDirty).toBe(false);
  expect(view.result.current.getPersistedBtFiles()).toEqual(OLD_FILES);
  expect(view.result.current.getPersistedLocalBtPaths()).toEqual(
    new Set([LOCAL_PATH]),
  );
  expect(view.result.current.getPersistedRevision()).toBe(7);
  expect(view.result.current.getEpochs()).toEqual({ bt: 0, nonBt: 0 });
});

test("records a BT edit synchronously and restores its history slice", () => {
  const view = renderLedger();
  commitPersisted(view);
  const before = view.result.current.getHistorySlice();

  let recorded;
  let immediateFiles;
  act(() => {
    recorded = view.result.current.recordBtEdit(LOCAL_PATH, "local-new");
    immediateFiles = view.result.current.getLiveBtFiles();
  });

  expect(recorded).toBe(true);
  expect(immediateFiles[LOCAL_PATH]).toBe("local-new");
  expect(view.result.current.designDirty).toBe(true);
  expect(view.result.current.getDirtyLocalBtPaths()).toEqual(
    new Set([LOCAL_PATH]),
  );
  expect(view.result.current.getEpochs()).toEqual({ bt: 1, nonBt: 0 });

  act(() => {
    view.result.current.restoreHistorySlice(before);
  });

  expect(view.result.current.missionBtFiles).toEqual(OLD_FILES);
  expect(view.result.current.designDirty).toBe(false);
  expect(view.result.current.getDirtyLocalBtPaths()).toEqual(new Set());
  // Restoring history is itself a new document epoch. A save that was already
  // in flight must treat the restored XML as a newer edit.
  expect(view.result.current.getEpochs()).toEqual({ bt: 2, nonBt: 1 });
});

test("history restores deleted paths and the non-BT dirty component", () => {
  const view = renderLedger();
  commitPersisted(view);

  act(() => {
    view.result.current.markNonBtDirty();
    view.result.current.replaceDeletedBtPaths(["locals/deleted/main.xml"]);
  });
  const dirtySlice = view.result.current.getHistorySlice();

  act(() => {
    view.result.current.clearDirty();
    view.result.current.replaceDeletedBtPaths([]);
    view.result.current.restoreHistorySlice(dirtySlice);
  });

  expect(view.result.current.deletedMissionBtPaths).toEqual([
    "locals/deleted/main.xml",
  ]);
  expect(view.result.current.designDirty).toBe(true);
  expect(view.result.current.isDirty()).toBe(true);
});

test("clearDirty preserves BT dirtiness against the persisted baseline", () => {
  const view = renderLedger();
  commitPersisted(view);

  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "local-new");
    view.result.current.markNonBtDirty();
  });

  let result;
  act(() => {
    result = view.result.current.clearDirty();
  });

  expect(result).toEqual({
    dirty: true,
    dirtyLocalBtPaths: new Set([LOCAL_PATH]),
  });
  expect(view.result.current.designDirty).toBe(true);
  expect(view.result.current.isDirty()).toBe(true);
  expect(view.result.current.getDirtyLocalBtPaths()).toEqual(
    new Set([LOCAL_PATH]),
  );
});

test("standalone BT checkpoints update revision, local paths and dirty state", () => {
  const view = renderLedger();
  commitPersisted(view, { localPaths: [] });

  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "local-new");
  });
  expect(view.result.current.designDirty).toBe(true);

  act(() => {
    view.result.current.checkpointPersistedBtFile({
      path: LOCAL_PATH,
      content: "local-new",
      revision: 8,
      registerLocalPath: true,
    });
  });

  expect(view.result.current.designDirty).toBe(false);
  expect(view.result.current.getPersistedRevision()).toBe(8);
  expect(view.result.current.hasPersistedLocalBtPath(LOCAL_PATH)).toBe(true);
  expect(view.result.current.getPersistedBtFiles()[LOCAL_PATH]).toBe("local-new");
});

test("an aborted save releases its lock but retains partial upload checkpoints", () => {
  const view = renderLedger();
  commitPersisted(view);
  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "local-new");
  });

  let firstSave;
  act(() => {
    firstSave = view.result.current.beginSave();
    view.result.current.checkpointSaveUpload(firstSave, {
      path: GLOBAL_PATH,
      content: "global-new",
      revision: 8,
    });
    expect(view.result.current.abortSave(firstSave)).toBe(true);
  });

  expect(view.result.current.hasActiveSave()).toBe(false);
  expect(view.result.current.getPersistedRevision()).toBe(8);
  expect(view.result.current.getPersistedBtFiles()[GLOBAL_PATH]).toBe("global-new");
  expect(view.result.current.designDirty).toBe(true);

  let retry;
  act(() => {
    retry = view.result.current.beginSave();
  });
  expect(retry.startingRevision).toBe(8);
  act(() => {
    view.result.current.abortSave(retry);
  });
});

test("beginSave snapshots BT files, deleted paths and persisted local paths", () => {
  const view = renderLedger();
  commitPersisted(view);
  act(() => {
    view.result.current.replaceDeletedBtPaths([OLD_DELETED_PATH]);
  });

  let transaction;
  act(() => {
    transaction = view.result.current.beginSave();
    view.result.current.replaceLiveBtFiles((files) => ({
      ...files,
      [LOCAL_PATH]: "local-after-begin",
      [SECOND_LOCAL_PATH]: "second-after-begin",
    }));
    view.result.current.replaceDeletedBtPaths([NEW_DELETED_PATH]);
    view.result.current.registerPersistedLocalBtPath(SECOND_LOCAL_PATH);
  });

  expect(transaction.btFiles).toEqual(OLD_FILES);
  expect(transaction.deletedBtPaths).toEqual([OLD_DELETED_PATH]);
  expect(transaction.persistedLocalBtPaths).toEqual([LOCAL_PATH]);
  expect(Object.isFrozen(transaction.btFiles)).toBe(true);
  expect(Object.isFrozen(transaction.deletedBtPaths)).toBe(true);
  expect(Object.isFrozen(transaction.persistedLocalBtPaths)).toBe(true);
  expect(view.result.current.getLiveBtFiles()).toEqual({
    ...OLD_FILES,
    [LOCAL_PATH]: "local-after-begin",
    [SECOND_LOCAL_PATH]: "second-after-begin",
  });
  expect(view.result.current.deletedMissionBtPaths).toEqual([
    NEW_DELETED_PATH,
  ]);
  expect(view.result.current.getPersistedLocalBtPaths()).toEqual(
    new Set([LOCAL_PATH, SECOND_LOCAL_PATH]),
  );

  act(() => {
    view.result.current.abortSave(transaction);
  });
});

test("reconcile preserves a BT edit made after save began", () => {
  const view = renderLedger();
  commitPersisted(view);
  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "save-snapshot");
    view.result.current.replaceDeletedBtPaths([
      "locals/old/main.xml",
      "locals/newer/main.xml",
    ]);
  });

  let transaction;
  act(() => {
    transaction = view.result.current.beginSave();
    view.result.current.recordBtEdit(LOCAL_PATH, "newer-editor-value");
    view.result.current.checkpointSaveManifest(transaction, {
      btFiles: {
        [GLOBAL_PATH]: "global-saved",
        [LOCAL_PATH]: "save-snapshot",
      },
      localBtPaths: [LOCAL_PATH],
      revision: 10,
    });
    view.result.current.checkpointSaveCleanup(transaction, { revision: 11 });
  });

  let result;
  act(() => {
    result = view.result.current.reconcileSave(transaction, {
      stalePaths: ["locals/old/main.xml"],
      migrateLiveBtFiles: (files) => ({ ...files, migrated: "yes" }),
    });
  });

  expect(result).toMatchObject({
    hasNewerBtEdits: true,
    hasNewerNonBtEdits: false,
    hasNewerEdits: true,
    revision: 11,
  });
  expect(view.result.current.missionBtFiles).toEqual(expect.objectContaining({
    [LOCAL_PATH]: "newer-editor-value",
    migrated: "yes",
  }));
  expect(view.result.current.getPersistedBtFiles()[LOCAL_PATH]).toBe(
    "save-snapshot",
  );
  expect(view.result.current.deletedMissionBtPaths).toEqual([
    "locals/newer/main.xml",
  ]);
  expect(view.result.current.designDirty).toBe(true);
});

test("reconcile preserves a history restoration made after save began", () => {
  const view = renderLedger();
  commitPersisted(view);
  const beforeSaveEdit = view.result.current.getHistorySlice();
  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "save-snapshot");
  });

  let transaction;
  act(() => {
    transaction = view.result.current.beginSave();
    view.result.current.restoreHistorySlice(beforeSaveEdit);
    view.result.current.checkpointSaveManifest(transaction, {
      btFiles: {
        ...OLD_FILES,
        [LOCAL_PATH]: "save-snapshot",
      },
      localBtPaths: [LOCAL_PATH],
      revision: 8,
    });
  });

  let result;
  act(() => {
    result = view.result.current.reconcileSave(transaction);
  });

  expect(result).toMatchObject({
    hasNewerBtEdits: true,
    hasNewerNonBtEdits: true,
    hasNewerEdits: true,
  });
  expect(view.result.current.getLiveBtFiles()[LOCAL_PATH]).toBe("local-old");
  expect(view.result.current.getPersistedBtFiles()[LOCAL_PATH]).toBe(
    "save-snapshot",
  );
  expect(view.result.current.getDirtyLocalBtPaths()).toEqual(
    new Set([LOCAL_PATH]),
  );
});

test("reconcile preserves a replaceLiveBtFiles edit made after save began", () => {
  const view = renderLedger();
  commitPersisted(view);
  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "save-snapshot");
  });

  let transaction;
  act(() => {
    transaction = view.result.current.beginSave();
    view.result.current.replaceLiveBtFiles((files) => ({
      ...files,
      [LOCAL_PATH]: "replacement-after-begin",
    }));
    view.result.current.checkpointSaveManifest(transaction, {
      btFiles: {
        ...OLD_FILES,
        [LOCAL_PATH]: "save-snapshot",
      },
      localBtPaths: [LOCAL_PATH],
      revision: 8,
    });
  });

  let result;
  act(() => {
    result = view.result.current.reconcileSave(transaction);
  });

  expect(result).toMatchObject({
    hasNewerBtEdits: true,
    hasNewerNonBtEdits: false,
    hasNewerEdits: true,
  });
  expect(view.result.current.getLiveBtFiles()[LOCAL_PATH]).toBe(
    "replacement-after-begin",
  );
  expect(view.result.current.getDirtyLocalBtPaths()).toEqual(
    new Set([LOCAL_PATH]),
  );
});

test("reconcile reports a newer non-BT edit independently of BT content", () => {
  const view = renderLedger();
  commitPersisted(view);
  let transaction;

  act(() => {
    transaction = view.result.current.beginSave();
    view.result.current.markNonBtDirty();
    view.result.current.checkpointSaveManifest(transaction, {
      btFiles: OLD_FILES,
      localBtPaths: [LOCAL_PATH],
      revision: 8,
    });
  });

  let result;
  act(() => {
    result = view.result.current.reconcileSave(transaction);
  });

  expect(result).toMatchObject({
    hasNewerBtEdits: false,
    hasNewerNonBtEdits: true,
    hasNewerEdits: true,
  });
  expect(view.result.current.designDirty).toBe(true);
});

test("a save with no newer edit adopts the manifest baseline and becomes clean", () => {
  const view = renderLedger();
  commitPersisted(view);
  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "local-saved");
  });
  let transaction;

  act(() => {
    transaction = view.result.current.beginSave();
    view.result.current.checkpointSaveManifest(transaction, {
      btFiles: {
        [GLOBAL_PATH]: "global-saved",
        [LOCAL_PATH]: "local-saved",
      },
      localBtPaths: [LOCAL_PATH],
      revision: 9,
    });
  });

  let result;
  act(() => {
    result = view.result.current.reconcileSave(transaction);
  });

  expect(result).toMatchObject({
    hasNewerBtEdits: false,
    hasNewerNonBtEdits: false,
    hasNewerEdits: false,
  });
  expect(view.result.current.missionBtFiles[LOCAL_PATH]).toBe("local-saved");
  expect(view.result.current.designDirty).toBe(false);
  expect(view.result.current.getDirtyLocalBtPaths()).toEqual(new Set());
});

test("save transactions reject overlap and require a manifest before reconcile", () => {
  const view = renderLedger();
  commitPersisted(view);
  let transaction;

  act(() => {
    transaction = view.result.current.beginSave();
  });
  expect(() => view.result.current.beginSave()).toThrow(
    "A mission save is already in progress",
  );
  expect(() => view.result.current.reconcileSave(transaction)).toThrow(
    "Mission manifest was not checkpointed",
  );
  expect(view.result.current.hasActiveSave()).toBe(true);

  act(() => {
    view.result.current.abortSave(transaction);
  });
  expect(() => view.result.current.checkpointSaveCleanup(transaction, {
    revision: 99,
  })).toThrow("Save transaction is no longer active");
});

test("save transactions enforce upload, manifest and cleanup ordering", () => {
  const view = renderLedger();
  commitPersisted(view);
  let transaction;

  act(() => {
    transaction = view.result.current.beginSave();
  });
  expect(() => view.result.current.checkpointSaveCleanup(transaction, {
    revision: 8,
  })).toThrow("Mission manifest must be checkpointed before cleanup");

  act(() => {
    view.result.current.checkpointSaveManifest(transaction, {
      btFiles: OLD_FILES,
      localBtPaths: [LOCAL_PATH],
      revision: 8,
    });
  });

  expect(() => view.result.current.checkpointSaveUpload(transaction, {
    path: LOCAL_PATH,
    content: "too-late",
    revision: 9,
  })).toThrow("Cannot upload BT files after the mission manifest");
  expect(() => view.result.current.checkpointSaveManifest(transaction, {
    btFiles: OLD_FILES,
    localBtPaths: [LOCAL_PATH],
    revision: 9,
  })).toThrow("Mission manifest was already checkpointed");
  expect(view.result.current.hasActiveSave()).toBe(true);

  act(() => {
    view.result.current.abortSave(transaction);
  });
});

test("commitSnapshot invalidates an old transaction without releasing a newer one", () => {
  const view = renderLedger();
  commitPersisted(view);
  let oldTransaction;

  act(() => {
    oldTransaction = view.result.current.beginSave();
  });
  commitPersisted(view, { revision: 12 });
  expect(view.result.current.hasActiveSave()).toBe(false);
  expect(() => view.result.current.checkpointSaveUpload(oldTransaction, {
    path: LOCAL_PATH,
    content: "stale-upload",
    revision: 13,
  })).toThrow("Save transaction is no longer active");

  let newTransaction;
  act(() => {
    newTransaction = view.result.current.beginSave();
  });
  expect(newTransaction.id).toBeGreaterThan(oldTransaction.id);
  expect(view.result.current.abortSave(oldTransaction)).toBe(false);
  expect(view.result.current.hasActiveSave()).toBe(true);
  expect(() => view.result.current.checkpointSaveManifest(oldTransaction, {
    btFiles: OLD_FILES,
    localBtPaths: [LOCAL_PATH],
    revision: 13,
  })).toThrow("Save transaction is no longer active");

  act(() => {
    view.result.current.abortSave(newTransaction);
  });
  expect(view.result.current.hasActiveSave()).toBe(false);
});

test("resetNewDocument clears every persisted checkpoint and epoch", () => {
  const view = renderLedger();
  commitPersisted(view);
  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "edited");
    view.result.current.markNonBtDirty();
    view.result.current.replaceDeletedBtPaths([LOCAL_PATH]);
    view.result.current.resetNewDocument({
      btFiles: { [GLOBAL_PATH]: "empty-global" },
    });
  });

  expect(view.result.current.missionBtFiles).toEqual({
    [GLOBAL_PATH]: "empty-global",
  });
  expect(view.result.current.deletedMissionBtPaths).toEqual([]);
  expect(view.result.current.getPersistedBtFiles()).toEqual({});
  expect(view.result.current.getPersistedLocalBtPaths()).toEqual(new Set());
  expect(view.result.current.getPersistedRevision()).toBe(0);
  expect(view.result.current.getEpochs()).toEqual({ bt: 0, nonBt: 0 });
  expect(view.result.current.designDirty).toBe(false);
});
