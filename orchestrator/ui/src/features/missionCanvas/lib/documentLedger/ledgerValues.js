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

export function resolveValue(valueOrUpdater, current) {
  return typeof valueOrUpdater === "function"
    ? valueOrUpdater(current)
    : valueOrUpdater;
}

export function normalizedFiles(files) {
  return files && typeof files === "object" && !Array.isArray(files)
    ? { ...files }
    : {};
}

export function normalizedPaths(paths) {
  return [...new Set(
    Array.from(paths || [])
      .map((path) => String(path || "").trim())
      .filter(Boolean),
  )];
}

export function normalizedRevision(value) {
  return Number.isInteger(value) ? value : 0;
}

export function initialLedgerState(options) {
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
