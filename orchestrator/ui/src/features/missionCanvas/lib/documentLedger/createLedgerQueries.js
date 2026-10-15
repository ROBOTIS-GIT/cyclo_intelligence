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

export default function createLedgerQueries(state) {
  const getLiveBtFiles = () => ({ ...state.missionBtFiles });

  const getPersistedBtFiles = () => ({ ...state.persistedMissionBtFiles });

  const getPersistedRevision = () => state.persistedMissionRevision;

  const getPersistedLocalBtPaths = () => new Set(state.persistedLocalBtPaths);

  const hasPersistedLocalBtPath = (path) => state.persistedLocalBtPaths.has(path);

  const getDirtyLocalBtPaths = () => new Set(state.dirtyLocalBtPaths);

  const getEpochs = () => ({
    bt: state.designBtEpoch,
    nonBt: state.designNonBtEpoch,
  });

  const isDirty = () => state.designDirty;

  const hasActiveSave = () => Boolean(state.activeSave);
  return {
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
