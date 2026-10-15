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

import { changedLocalBtPaths } from '../../../mapWorkspace/lib/missionBtFiles';
import { normalizedFiles, normalizedPaths } from './ledgerValues';

export default function createLedgerSave(state, publish, { checkpointPersistedBtFile, replacePersistedBaseline, setPersistedRevision }) {
  const { setMissionBtFilesState, setDeletedMissionBtPathsState, setDesignDirtyState } = publish;

  const assertActiveSave = (transaction) => {
    const active = state.activeSave;
    if (!active || active.transaction !== transaction) {
      throw new Error("Save transaction is no longer active");
    }
    return active;
  };

  const beginSave = () => {
    if (state.activeSave) {
      throw new Error("A mission save is already in progress");
    }
    state.saveTransactionSerial += 1;
    const transaction = Object.freeze({
      id: state.saveTransactionSerial,
      startingRevision: state.persistedMissionRevision,
      btFiles: Object.freeze({ ...state.missionBtFiles }),
      deletedBtPaths: Object.freeze([...state.deletedMissionBtPaths]),
      persistedLocalBtPaths: Object.freeze([
        ...state.persistedLocalBtPaths,
      ]),
      btEpoch: state.designBtEpoch,
      nonBtEpoch: state.designNonBtEpoch,
    });
    state.activeSave = {
      transaction,
      manifestCheckpointed: false,
    };
    return transaction;
  };

  const checkpointSaveUpload = (transaction, {
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
  };

  const checkpointSaveManifest = (transaction, {
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
    return state.persistedMissionRevision;
  };

  const checkpointSaveCleanup = (transaction, { revision } = {}) => {
    const active = assertActiveSave(transaction);
    if (!active.manifestCheckpointed) {
      throw new Error("Mission manifest must be checkpointed before cleanup");
    }
    return setPersistedRevision(revision);
  };

  const abortSave = (transaction) => {
    const active = state.activeSave;
    if (!active || active.transaction !== transaction) return false;
    state.activeSave = null;
    return true;
  };

  const reconcileSave = (transaction, {
    stalePaths = [],
    migrateLiveBtFiles,
  } = {}) => {
    const active = assertActiveSave(transaction);
    if (!active.manifestCheckpointed) {
      throw new Error("Mission manifest was not checkpointed");
    }
    const hasNewerBtEdits = state.designBtEpoch !== transaction.btEpoch;
    const hasNewerNonBtEdits = (
      state.designNonBtEpoch !== transaction.nonBtEpoch
    );
    let currentFiles = state.missionBtFiles;
    if (
      (hasNewerBtEdits || hasNewerNonBtEdits)
      && typeof migrateLiveBtFiles === "function"
    ) {
      currentFiles = normalizedFiles(migrateLiveBtFiles(currentFiles));
    } else if (!hasNewerBtEdits) {
      currentFiles = { ...state.persistedMissionBtFiles };
    }
    const dirtyPaths = hasNewerBtEdits
      ? changedLocalBtPaths(currentFiles, state.persistedMissionBtFiles)
      : new Set();
    const stale = new Set(normalizedPaths(stalePaths));
    const nextDeletedPaths = state.deletedMissionBtPaths.filter(
      (path) => !stale.has(path),
    );
    const dirty = hasNewerNonBtEdits || dirtyPaths.size > 0;

    state.missionBtFiles = currentFiles;
    state.dirtyLocalBtPaths = dirtyPaths;
    state.deletedMissionBtPaths = nextDeletedPaths;
    state.nonBtDesignDirty = hasNewerNonBtEdits;
    state.designDirty = dirty;
    state.activeSave = null;
    setMissionBtFilesState(currentFiles);
    setDeletedMissionBtPathsState(nextDeletedPaths);
    setDesignDirtyState(dirty);
    return {
      hasNewerBtEdits,
      hasNewerNonBtEdits,
      hasNewerEdits: dirty,
      dirtyLocalBtPaths: new Set(dirtyPaths),
      revision: state.persistedMissionRevision,
    };
  };
  return {
    beginSave,
    checkpointSaveUpload,
    checkpointSaveManifest,
    checkpointSaveCleanup,
    abortSave,
    reconcileSave,
  };
}
