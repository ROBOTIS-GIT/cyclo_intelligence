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

function requiredFunction(value, name) {
  if (typeof value !== "function") {
    throw new Error(`Waypoint Task controller requires ${name}`);
  }
  return value;
}

export default function waypointTaskLedger(ledger) {
  const missionBtFiles = ledger?.missionBtFiles;
  const recordBtEdit = requiredFunction(ledger?.recordBtEdit, "ledger.recordBtEdit");
  const replaceLiveBtFiles = requiredFunction(
    ledger?.replaceLiveBtFiles,
    "ledger.replaceLiveBtFiles",
  );
  const checkpointPersistedBtFile = requiredFunction(
    ledger?.checkpointPersistedBtFile,
    "ledger.checkpointPersistedBtFile",
  );
  const setPersistedRevision = requiredFunction(
    ledger?.setPersistedRevision,
    "ledger.setPersistedRevision",
  );
  const reconcileDirty = requiredFunction(
    ledger?.reconcileDirty,
    "ledger.reconcileDirty",
  );
  const getLiveBtFiles = requiredFunction(
    ledger?.getLiveBtFiles,
    "ledger.getLiveBtFiles",
  );
  const getPersistedRevision = requiredFunction(
    ledger?.getPersistedRevision,
    "ledger.getPersistedRevision",
  );
  const getPersistedLocalBtPaths = requiredFunction(
    ledger?.getPersistedLocalBtPaths,
    "ledger.getPersistedLocalBtPaths",
  );
  const hasPersistedLocalBtPath = requiredFunction(
    ledger?.hasPersistedLocalBtPath,
    "ledger.hasPersistedLocalBtPath",
  );
  const hasActiveSave = requiredFunction(
    ledger?.hasActiveSave,
    "ledger.hasActiveSave",
  );

  return {
    missionBtFiles,
    recordBtEdit,
    replaceLiveBtFiles,
    checkpointPersistedBtFile,
    setPersistedRevision,
    reconcileDirty,
    getLiveBtFiles,
    getPersistedRevision,
    getPersistedLocalBtPaths,
    hasPersistedLocalBtPath,
    hasActiveSave,
  };
}
