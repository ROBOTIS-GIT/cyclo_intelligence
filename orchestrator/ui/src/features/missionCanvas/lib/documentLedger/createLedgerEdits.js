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
import { normalizedFiles, normalizedPaths, resolveValue } from './ledgerValues';

export default function createLedgerEdits(state, publish) {
  const { setMissionBtFilesState, setDeletedMissionBtPathsState, setDesignDirtyState } = publish;

  const replaceLiveBtFiles = (valueOrUpdater, {
    bumpBtEpoch = true,
    reconcileDirty = true,
  } = {}) => {
    const next = normalizedFiles(resolveValue(
      valueOrUpdater,
      state.missionBtFiles,
    ));
    state.missionBtFiles = next;
    setMissionBtFilesState(next);
    if (bumpBtEpoch) state.designBtEpoch += 1;
    if (reconcileDirty) {
      const dirtyPaths = changedLocalBtPaths(
        next,
        state.persistedMissionBtFiles,
      );
      const dirty = state.nonBtDesignDirty || dirtyPaths.size > 0;
      state.dirtyLocalBtPaths = dirtyPaths;
      state.designDirty = dirty;
      setDesignDirtyState(dirty);
    }
    return next;
  };

  const replaceDeletedBtPaths = (valueOrUpdater, {
    bumpNonBtEpoch = true,
    markDirty = true,
  } = {}) => {
    const next = normalizedPaths(resolveValue(
      valueOrUpdater,
      state.deletedMissionBtPaths,
    ));
    state.deletedMissionBtPaths = next;
    setDeletedMissionBtPathsState(next);
    if (bumpNonBtEpoch) state.designNonBtEpoch += 1;
    if (markDirty) {
      state.nonBtDesignDirty = true;
      state.designDirty = true;
      setDesignDirtyState(true);
    }
    return next;
  };

  const reconcileDirty = () => {
    const dirtyPaths = changedLocalBtPaths(
      state.missionBtFiles,
      state.persistedMissionBtFiles,
    );
    const dirty = state.nonBtDesignDirty || dirtyPaths.size > 0;
    state.dirtyLocalBtPaths = dirtyPaths;
    state.designDirty = dirty;
    setDesignDirtyState(dirty);
    return {
      dirty,
      dirtyLocalBtPaths: new Set(dirtyPaths),
    };
  };

  const recordBtEdit = (path, content, {
    allowCreate = false,
  } = {}) => {
    const normalizedPath = String(path || "").trim();
    if (!normalizedPath) return false;
    const current = state.missionBtFiles;
    if (!allowCreate && current[normalizedPath] === undefined) return false;
    if (current[normalizedPath] === content) return false;
    const next = { ...current, [normalizedPath]: content };
    state.missionBtFiles = next;
    state.designBtEpoch += 1;
    const dirtyPaths = changedLocalBtPaths(
      next,
      state.persistedMissionBtFiles,
    );
    const dirty = state.nonBtDesignDirty || dirtyPaths.size > 0;
    state.dirtyLocalBtPaths = dirtyPaths;
    state.designDirty = dirty;
    setMissionBtFilesState(next);
    setDesignDirtyState(dirty);
    return true;
  };

  const markNonBtDirty = () => {
    state.designNonBtEpoch += 1;
    state.nonBtDesignDirty = true;
    state.designDirty = true;
    setDesignDirtyState(true);
  };

  const clearDirty = () => {
    state.nonBtDesignDirty = false;
    return reconcileDirty();
  };
  return {
    replaceLiveBtFiles,
    replaceDeletedBtPaths,
    reconcileDirty,
    recordBtEdit,
    markNonBtDirty,
    clearDirty,
  };
}
