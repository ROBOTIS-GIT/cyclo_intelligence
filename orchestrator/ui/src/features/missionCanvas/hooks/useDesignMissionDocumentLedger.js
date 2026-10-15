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

import { useRef, useState } from 'react';
import { initialLedgerState } from '../lib/documentLedger/ledgerValues';
import createDocumentLedger from '../lib/documentLedger/createDocumentLedger';

// React publishes the three UI snapshots. Ordering, dirty epochs and the save
// transaction remain in one stable store across renders and document changes.
export default function useDesignMissionDocumentLedger(options = {}) {
  const initialStateRef = useRef(null);
  if (initialStateRef.current === null) initialStateRef.current = initialLedgerState(options);
  const initialState = initialStateRef.current;
  const [missionBtFiles, setMissionBtFilesState] = useState(initialState.missionBtFiles);
  const [deletedMissionBtPaths, setDeletedMissionBtPathsState] = useState(initialState.deletedMissionBtPaths);
  const [designDirty, setDesignDirtyState] = useState(initialState.dirty);
  const storeRef = useRef(null);
  if (storeRef.current === null) {
    storeRef.current = createDocumentLedger(initialState, {
      setMissionBtFilesState, setDeletedMissionBtPathsState, setDesignDirtyState,
    });
  }
  return { ...storeRef.current, missionBtFiles, deletedMissionBtPaths, designDirty };
}
