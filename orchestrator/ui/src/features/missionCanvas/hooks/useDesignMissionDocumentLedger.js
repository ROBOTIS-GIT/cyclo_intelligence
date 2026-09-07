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

import { useCallback, useRef, useState } from "react";
import { changedLocalBtPaths } from "../lib/missionBtFiles";

function resolveValue(valueOrUpdater, current) {
  return typeof valueOrUpdater === "function"
    ? valueOrUpdater(current)
    : valueOrUpdater;
}

function normalizedFiles(files) {
  return files && typeof files === "object" && !Array.isArray(files)
    ? { ...files }
    : {};
}

function normalizedPaths(paths) {
  return [...new Set(
    Array.from(paths || [])
      .map((path) => String(path || "").trim())
      .filter(Boolean),
  )];
}

function normalizedRevision(value) {
  return Number.isInteger(value) ? value : 0;
}

function initialLedgerState(options) {
  const baseline = options.initialBaseline || {};
  const missionBtFiles = normalizedFiles(options.initialMissionBtFiles);
  const persistedMissionBtFiles = normalizedFiles(baseline.persistedBtFiles);
  const nonBtDirty = options.initialNonBtDirty === true;
  const dirtyLocalBtPaths = changedLocalBtPaths(
    missionBtFiles,
    persistedMissionBtFiles,
  );
  return {
    missionBtFiles,
    persistedMissionBtFiles,
    persistedLocalBtPaths: normalizedPaths(baseline.persistedLocalBtPaths),
    persistedMissionRevision: normalizedRevision(baseline.revision),
    deletedMissionBtPaths: normalizedPaths(options.initialDeletedMissionBtPaths),
    dirtyLocalBtPaths,
    nonBtDirty,
    dirty: nonBtDirty || dirtyLocalBtPaths.size > 0,
  };
}

// Design's map/waypoint/route state deliberately remains in the workspace.
// This hook owns only the mutable persistence ledger shared by the editor,
// history and the mission-save pipeline. Its query methods return snapshots;
// callers never receive the refs that establish event-synchronous ordering.
export default function useDesignMissionDocumentLedger(options = {}) {
  const initialStateRef = useRef(null);
  if (initialStateRef.current === null) {
    initialStateRef.current = initialLedgerState(options);
  }
  const initialState = initialStateRef.current;

  const [missionBtFiles, setMissionBtFilesState] = useState(
    initialState.missionBtFiles,
  );
  const [deletedMissionBtPaths, setDeletedMissionBtPathsState] = useState(
    initialState.deletedMissionBtPaths,
  );
  const [designDirty, setDesignDirtyState] = useState(initialState.dirty);

  const missionBtFilesRef = useRef(initialState.missionBtFiles);
  const persistedMissionBtFilesRef = useRef(initialState.persistedMissionBtFiles);
  const persistedLocalBtPathsRef = useRef(
    new Set(initialState.persistedLocalBtPaths),
  );
  const persistedMissionRevisionRef = useRef(
    initialState.persistedMissionRevision,
  );
  const dirtyLocalBtPathsRef = useRef(initialState.dirtyLocalBtPaths);
  const deletedMissionBtPathsRef = useRef(initialState.deletedMissionBtPaths);
  const designDirtyRef = useRef(initialState.dirty);
  const nonBtDesignDirtyRef = useRef(initialState.nonBtDirty);
  const designBtEpochRef = useRef(0);
  const designNonBtEpochRef = useRef(0);
  const saveTransactionSerialRef = useRef(0);
  const activeSaveRef = useRef(null);

  const replaceLiveBtFiles = useCallback((valueOrUpdater, {
    bumpBtEpoch = true,
    reconcileDirty = true,
  } = {}) => {
    const next = normalizedFiles(resolveValue(
      valueOrUpdater,
      missionBtFilesRef.current,
    ));
    missionBtFilesRef.current = next;
    setMissionBtFilesState(next);
    if (bumpBtEpoch) designBtEpochRef.current += 1;
    if (reconcileDirty) {
      const dirtyPaths = changedLocalBtPaths(
        next,
        persistedMissionBtFilesRef.current,
      );
      const dirty = nonBtDesignDirtyRef.current || dirtyPaths.size > 0;
      dirtyLocalBtPathsRef.current = dirtyPaths;
      designDirtyRef.current = dirty;
      setDesignDirtyState(dirty);
    }
    return next;
  }, []);

  const replaceDeletedBtPaths = useCallback((valueOrUpdater, {
    bumpNonBtEpoch = true,
    markDirty = true,
  } = {}) => {
    const next = normalizedPaths(resolveValue(
      valueOrUpdater,
      deletedMissionBtPathsRef.current,
    ));
    deletedMissionBtPathsRef.current = next;
    setDeletedMissionBtPathsState(next);
    if (bumpNonBtEpoch) designNonBtEpochRef.current += 1;
    if (markDirty) {
      nonBtDesignDirtyRef.current = true;
      designDirtyRef.current = true;
      setDesignDirtyState(true);
    }
    return next;
  }, []);

  const reconcileDirty = useCallback(() => {
    const dirtyPaths = changedLocalBtPaths(
      missionBtFilesRef.current,
      persistedMissionBtFilesRef.current,
    );
    const dirty = nonBtDesignDirtyRef.current || dirtyPaths.size > 0;
    dirtyLocalBtPathsRef.current = dirtyPaths;
    designDirtyRef.current = dirty;
    setDesignDirtyState(dirty);
    return {
      dirty,
      dirtyLocalBtPaths: new Set(dirtyPaths),
    };
  }, []);

  const recordBtEdit = useCallback((path, content, {
    allowCreate = false,
  } = {}) => {
    const normalizedPath = String(path || "").trim();
    if (!normalizedPath) return false;
    const current = missionBtFilesRef.current;
    if (!allowCreate && current[normalizedPath] === undefined) return false;
    if (current[normalizedPath] === content) return false;
    const next = { ...current, [normalizedPath]: content };
    missionBtFilesRef.current = next;
    designBtEpochRef.current += 1;
    const dirtyPaths = changedLocalBtPaths(
      next,
      persistedMissionBtFilesRef.current,
    );
    const dirty = nonBtDesignDirtyRef.current || dirtyPaths.size > 0;
    dirtyLocalBtPathsRef.current = dirtyPaths;
    designDirtyRef.current = dirty;
    setMissionBtFilesState(next);
    setDesignDirtyState(dirty);
    return true;
  }, []);

  const markNonBtDirty = useCallback(() => {
    designNonBtEpochRef.current += 1;
    nonBtDesignDirtyRef.current = true;
    designDirtyRef.current = true;
    setDesignDirtyState(true);
  }, []);

  const clearDirty = useCallback(() => {
    nonBtDesignDirtyRef.current = false;
    return reconcileDirty();
  }, [reconcileDirty]);

  const setPersistedRevision = useCallback((revision) => {
    if (Number.isInteger(revision)) {
      persistedMissionRevisionRef.current = revision;
    }
    return persistedMissionRevisionRef.current;
  }, []);

  const registerPersistedLocalBtPath = useCallback((path) => {
    const normalizedPath = String(path || "").trim();
    if (!normalizedPath) return false;
    persistedLocalBtPathsRef.current = new Set([
      ...persistedLocalBtPathsRef.current,
      normalizedPath,
    ]);
    return true;
  }, []);

  const replacePersistedLocalBtPaths = useCallback((paths) => {
    persistedLocalBtPathsRef.current = new Set(normalizedPaths(paths));
    return new Set(persistedLocalBtPathsRef.current);
  }, []);

  // Used by standalone Local Task saves. Full mission saves use the same
  // transition through checkpointSaveUpload with dirty reconciliation held
  // until the manifest and orphan cleanup have completed.
  const checkpointPersistedBtFile = useCallback(({
    path,
    content,
    revision,
    registerLocalPath = false,
    reconcile = true,
  }) => {
    const normalizedPath = String(path || "").trim();
    if (!normalizedPath) throw new Error("BT file path is required");
    persistedMissionBtFilesRef.current = {
      ...persistedMissionBtFilesRef.current,
      [normalizedPath]: content,
    };
    if (Number.isInteger(revision)) {
      persistedMissionRevisionRef.current = revision;
    }
    if (registerLocalPath) {
      persistedLocalBtPathsRef.current = new Set([
        ...persistedLocalBtPathsRef.current,
        normalizedPath,
      ]);
    }
    return reconcile ? reconcileDirty() : {
      dirty: designDirtyRef.current,
      dirtyLocalBtPaths: new Set(dirtyLocalBtPathsRef.current),
    };
  }, [reconcileDirty]);

  const replacePersistedBaseline = useCallback(({
    btFiles = {},
    localBtPaths = [],
    revision = 0,
  }, { reconcile = true } = {}) => {
    persistedMissionBtFilesRef.current = normalizedFiles(btFiles);
    persistedLocalBtPathsRef.current = new Set(normalizedPaths(localBtPaths));
    persistedMissionRevisionRef.current = normalizedRevision(revision);
    return reconcile ? reconcileDirty() : {
      dirty: designDirtyRef.current,
      dirtyLocalBtPaths: new Set(dirtyLocalBtPathsRef.current),
    };
  }, [reconcileDirty]);

  const commitSnapshot = useCallback((snapshot = {}) => {
    const baseline = snapshot.baseline || {};
    const nextFiles = normalizedFiles(snapshot.btFiles);
    activeSaveRef.current = null;
    missionBtFilesRef.current = nextFiles;
    persistedMissionBtFilesRef.current = normalizedFiles(
      baseline.persistedBtFiles,
    );
    persistedLocalBtPathsRef.current = new Set(normalizedPaths(
      baseline.persistedLocalBtPaths,
    ));
    persistedMissionRevisionRef.current = normalizedRevision(baseline.revision);
    dirtyLocalBtPathsRef.current = new Set();
    deletedMissionBtPathsRef.current = [];
    nonBtDesignDirtyRef.current = false;
    designBtEpochRef.current = 0;
    designNonBtEpochRef.current = 0;
    designDirtyRef.current = false;
    setMissionBtFilesState(nextFiles);
    setDeletedMissionBtPathsState([]);
    setDesignDirtyState(false);
    return {
      missionBtFiles: nextFiles,
      persistedMissionRevision: persistedMissionRevisionRef.current,
    };
  }, []);

  const resetNewDocument = useCallback(({ btFiles = {} } = {}) => {
    return commitSnapshot({
      btFiles,
      baseline: {
        persistedBtFiles: {},
        persistedLocalBtPaths: [],
        revision: 0,
      },
    });
  }, [commitSnapshot]);

  const getHistorySlice = useCallback(() => ({
    missionBtFiles: { ...missionBtFilesRef.current },
    deletedMissionBtPaths: [...deletedMissionBtPathsRef.current],
    designDirty: designDirtyRef.current,
    nonBtDesignDirty: nonBtDesignDirtyRef.current,
  }), []);

  const restoreHistorySlice = useCallback((slice = {}) => {
    const restoredFiles = normalizedFiles(slice.missionBtFiles);
    const restoredNonBtDirty = slice.nonBtDesignDirty === undefined
      ? slice.designDirty === true
      : slice.nonBtDesignDirty === true;
    const restoredDirtyPaths = changedLocalBtPaths(
      restoredFiles,
      persistedMissionBtFilesRef.current,
    );
    const restoredDirty = restoredNonBtDirty || restoredDirtyPaths.size > 0;
    missionBtFilesRef.current = restoredFiles;
    deletedMissionBtPathsRef.current = normalizedPaths(
      slice.deletedMissionBtPaths,
    );
    dirtyLocalBtPathsRef.current = restoredDirtyPaths;
    nonBtDesignDirtyRef.current = restoredNonBtDirty;
    designBtEpochRef.current += 1;
    designNonBtEpochRef.current += 1;
    designDirtyRef.current = restoredDirty;
    setMissionBtFilesState(restoredFiles);
    setDeletedMissionBtPathsState(deletedMissionBtPathsRef.current);
    setDesignDirtyState(restoredDirty);
    return {
      dirty: restoredDirty,
      dirtyLocalBtPaths: new Set(restoredDirtyPaths),
    };
  }, []);

  const assertActiveSave = useCallback((transaction) => {
    const active = activeSaveRef.current;
    if (!active || active.transaction !== transaction) {
      throw new Error("Save transaction is no longer active");
    }
    return active;
  }, []);

  const beginSave = useCallback(() => {
    if (activeSaveRef.current) {
      throw new Error("A mission save is already in progress");
    }
    saveTransactionSerialRef.current += 1;
    const transaction = Object.freeze({
      id: saveTransactionSerialRef.current,
      startingRevision: persistedMissionRevisionRef.current,
      btFiles: Object.freeze({ ...missionBtFilesRef.current }),
      deletedBtPaths: Object.freeze([...deletedMissionBtPathsRef.current]),
      persistedLocalBtPaths: Object.freeze([
        ...persistedLocalBtPathsRef.current,
      ]),
      btEpoch: designBtEpochRef.current,
      nonBtEpoch: designNonBtEpochRef.current,
    });
    activeSaveRef.current = {
      transaction,
      manifestCheckpointed: false,
    };
    return transaction;
  }, []);

  const checkpointSaveUpload = useCallback((transaction, {
    path,
    content,
    revision,
  }) => {
    const active = assertActiveSave(transaction);
    if (active.manifestCheckpointed) {
      throw new Error("Cannot upload BT files after the mission manifest");
    }
    return checkpointPersistedBtFile({
      path,
      content,
      revision,
      reconcile: false,
    });
  }, [assertActiveSave, checkpointPersistedBtFile]);

  const checkpointSaveManifest = useCallback((transaction, {
    btFiles,
    localBtPaths,
    revision,
  }) => {
    const active = assertActiveSave(transaction);
    if (active.manifestCheckpointed) {
      throw new Error("Mission manifest was already checkpointed");
    }
    replacePersistedBaseline({
      btFiles,
      localBtPaths,
      revision,
    }, { reconcile: false });
    active.manifestCheckpointed = true;
    return persistedMissionRevisionRef.current;
  }, [assertActiveSave, replacePersistedBaseline]);

  const checkpointSaveCleanup = useCallback((transaction, { revision } = {}) => {
    const active = assertActiveSave(transaction);
    if (!active.manifestCheckpointed) {
      throw new Error("Mission manifest must be checkpointed before cleanup");
    }
    return setPersistedRevision(revision);
  }, [assertActiveSave, setPersistedRevision]);

  // Failure releases only the operation lock. Durable upload/manifest
  // checkpoints intentionally remain so a retry begins at the server's latest
  // revision instead of replaying revision zero or the pre-save generation.
  const abortSave = useCallback((transaction) => {
    const active = activeSaveRef.current;
    if (!active || active.transaction !== transaction) return false;
    activeSaveRef.current = null;
    return true;
  }, []);

  const reconcileSave = useCallback((transaction, {
    stalePaths = [],
    migrateLiveBtFiles,
  } = {}) => {
    const active = assertActiveSave(transaction);
    if (!active.manifestCheckpointed) {
      throw new Error("Mission manifest was not checkpointed");
    }
    const hasNewerBtEdits = designBtEpochRef.current !== transaction.btEpoch;
    const hasNewerNonBtEdits = (
      designNonBtEpochRef.current !== transaction.nonBtEpoch
    );
    let currentFiles = missionBtFilesRef.current;
    if (
      (hasNewerBtEdits || hasNewerNonBtEdits)
      && typeof migrateLiveBtFiles === "function"
    ) {
      currentFiles = normalizedFiles(migrateLiveBtFiles(currentFiles));
    } else if (!hasNewerBtEdits) {
      currentFiles = { ...persistedMissionBtFilesRef.current };
    }
    const dirtyPaths = hasNewerBtEdits
      ? changedLocalBtPaths(currentFiles, persistedMissionBtFilesRef.current)
      : new Set();
    const stale = new Set(normalizedPaths(stalePaths));
    const nextDeletedPaths = deletedMissionBtPathsRef.current.filter(
      (path) => !stale.has(path),
    );
    const dirty = hasNewerNonBtEdits || dirtyPaths.size > 0;

    missionBtFilesRef.current = currentFiles;
    dirtyLocalBtPathsRef.current = dirtyPaths;
    deletedMissionBtPathsRef.current = nextDeletedPaths;
    nonBtDesignDirtyRef.current = hasNewerNonBtEdits;
    designDirtyRef.current = dirty;
    activeSaveRef.current = null;
    setMissionBtFilesState(currentFiles);
    setDeletedMissionBtPathsState(nextDeletedPaths);
    setDesignDirtyState(dirty);
    return {
      hasNewerBtEdits,
      hasNewerNonBtEdits,
      hasNewerEdits: dirty,
      dirtyLocalBtPaths: new Set(dirtyPaths),
      revision: persistedMissionRevisionRef.current,
    };
  }, [assertActiveSave]);

  const getLiveBtFiles = useCallback(
    () => ({ ...missionBtFilesRef.current }),
    [],
  );
  const getPersistedBtFiles = useCallback(
    () => ({ ...persistedMissionBtFilesRef.current }),
    [],
  );
  const getPersistedRevision = useCallback(
    () => persistedMissionRevisionRef.current,
    [],
  );
  const getPersistedLocalBtPaths = useCallback(
    () => new Set(persistedLocalBtPathsRef.current),
    [],
  );
  const hasPersistedLocalBtPath = useCallback(
    (path) => persistedLocalBtPathsRef.current.has(path),
    [],
  );
  const getDirtyLocalBtPaths = useCallback(
    () => new Set(dirtyLocalBtPathsRef.current),
    [],
  );
  const getEpochs = useCallback(() => ({
    bt: designBtEpochRef.current,
    nonBt: designNonBtEpochRef.current,
  }), []);
  const isDirty = useCallback(() => designDirtyRef.current, []);
  const hasActiveSave = useCallback(() => Boolean(activeSaveRef.current), []);

  return {
    missionBtFiles,
    deletedMissionBtPaths,
    designDirty,
    recordBtEdit,
    replaceLiveBtFiles,
    replaceDeletedBtPaths,
    markNonBtDirty,
    clearDirty,
    reconcileDirty,
    commitSnapshot,
    resetNewDocument,
    getHistorySlice,
    restoreHistorySlice,
    checkpointPersistedBtFile,
    replacePersistedBaseline,
    setPersistedRevision,
    registerPersistedLocalBtPath,
    replacePersistedLocalBtPaths,
    beginSave,
    checkpointSaveUpload,
    checkpointSaveManifest,
    checkpointSaveCleanup,
    reconcileSave,
    abortSave,
    getLiveBtFiles,
    getPersistedBtFiles,
    getPersistedRevision,
    getPersistedLocalBtPaths,
    hasPersistedLocalBtPath,
    getDirtyLocalBtPaths,
    getEpochs,
    isDirty,
    hasActiveSave,
  };
}
