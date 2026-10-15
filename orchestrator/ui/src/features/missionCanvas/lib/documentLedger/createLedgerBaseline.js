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

import { normalizedFiles, normalizedPaths, normalizedRevision } from './ledgerValues';

export default function createLedgerBaseline(state, { reconcileDirty }) {
  const setPersistedRevision = (revision) => {
    if (Number.isInteger(revision)) {
      state.persistedMissionRevision = revision;
    }
    return state.persistedMissionRevision;
  };

  const registerPersistedLocalBtPath = (path) => {
    const normalizedPath = String(path || "").trim();
    if (!normalizedPath) return false;
    state.persistedLocalBtPaths = new Set([
      ...state.persistedLocalBtPaths,
      normalizedPath,
    ]);
    return true;
  };

  const replacePersistedLocalBtPaths = (paths) => {
    state.persistedLocalBtPaths = new Set(normalizedPaths(paths));
    return new Set(state.persistedLocalBtPaths);
  };

  const checkpointPersistedBtFile = ({
    path,
    content,
    revision,
    registerLocalPath = false,
    reconcile = true,
  }) => {
    const normalizedPath = String(path || "").trim();
    if (!normalizedPath) throw new Error("BT file path is required");
    state.persistedMissionBtFiles = {
      ...state.persistedMissionBtFiles,
      [normalizedPath]: content,
    };
    if (Number.isInteger(revision)) {
      state.persistedMissionRevision = revision;
    }
    if (registerLocalPath) {
      state.persistedLocalBtPaths = new Set([
        ...state.persistedLocalBtPaths,
        normalizedPath,
      ]);
    }
    return reconcile ? reconcileDirty() : {
      dirty: state.designDirty,
      dirtyLocalBtPaths: new Set(state.dirtyLocalBtPaths),
    };
  };

  const replacePersistedBaseline = ({
    btFiles = {},
    localBtPaths = [],
    revision = 0,
  }, { reconcile = true } = {}) => {
    state.persistedMissionBtFiles = normalizedFiles(btFiles);
    state.persistedLocalBtPaths = new Set(normalizedPaths(localBtPaths));
    state.persistedMissionRevision = normalizedRevision(revision);
    return reconcile ? reconcileDirty() : {
      dirty: state.designDirty,
      dirtyLocalBtPaths: new Set(state.dirtyLocalBtPaths),
    };
  };
  return {
    setPersistedRevision,
    registerPersistedLocalBtPath,
    replacePersistedLocalBtPaths,
    checkpointPersistedBtFile,
    replacePersistedBaseline,
  };
}
