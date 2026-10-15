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
import { normalizedFiles, normalizedPaths, normalizedRevision } from './ledgerValues';

export default function createLedgerHistory(state, publish) {
  const { setMissionBtFilesState, setDeletedMissionBtPathsState, setDesignDirtyState } = publish;

  const commitSnapshot = (snapshot = {}) => {
    const baseline = snapshot.baseline || {};
    const nextFiles = normalizedFiles(snapshot.btFiles);
    state.activeSave = null;
    state.missionBtFiles = nextFiles;
    state.persistedMissionBtFiles = normalizedFiles(
      baseline.persistedBtFiles,
    );
    state.persistedLocalBtPaths = new Set(normalizedPaths(
      baseline.persistedLocalBtPaths,
    ));
    state.persistedMissionRevision = normalizedRevision(baseline.revision);
    state.dirtyLocalBtPaths = new Set();
    state.deletedMissionBtPaths = [];
    state.nonBtDesignDirty = false;
    state.designBtEpoch = 0;
    state.designNonBtEpoch = 0;
    state.designDirty = false;
    setMissionBtFilesState(nextFiles);
    setDeletedMissionBtPathsState([]);
    setDesignDirtyState(false);
    return {
      missionBtFiles: nextFiles,
      persistedMissionRevision: state.persistedMissionRevision,
    };
  };

  const resetNewDocument = ({ btFiles = {} } = {}) => {
    return commitSnapshot({
      btFiles,
      baseline: {
        persistedBtFiles: {},
        persistedLocalBtPaths: [],
        revision: 0,
      },
    });
  };

  const getHistorySlice = () => ({
    missionBtFiles: { ...state.missionBtFiles },
    deletedMissionBtPaths: [...state.deletedMissionBtPaths],
    designDirty: state.designDirty,
    nonBtDesignDirty: state.nonBtDesignDirty,
  });

  const restoreHistorySlice = (slice = {}) => {
    const restoredFiles = normalizedFiles(slice.missionBtFiles);
    const restoredNonBtDirty = slice.nonBtDesignDirty === undefined
      ? slice.designDirty === true
      : slice.nonBtDesignDirty === true;
    const restoredDirtyPaths = changedLocalBtPaths(
      restoredFiles,
      state.persistedMissionBtFiles,
    );
    const restoredDirty = restoredNonBtDirty || restoredDirtyPaths.size > 0;
    state.missionBtFiles = restoredFiles;
    state.deletedMissionBtPaths = normalizedPaths(
      slice.deletedMissionBtPaths,
    );
    state.dirtyLocalBtPaths = restoredDirtyPaths;
    state.nonBtDesignDirty = restoredNonBtDirty;
    state.designBtEpoch += 1;
    state.designNonBtEpoch += 1;
    state.designDirty = restoredDirty;
    setMissionBtFilesState(restoredFiles);
    setDeletedMissionBtPathsState(state.deletedMissionBtPaths);
    setDesignDirtyState(restoredDirty);
    return {
      dirty: restoredDirty,
      dirtyLocalBtPaths: new Set(restoredDirtyPaths),
    };
  };
  return { commitSnapshot, resetNewDocument, getHistorySlice, restoreHistorySlice };
}
