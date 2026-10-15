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

import { useCallback, useRef } from "react";

export default function useWaypointTaskOperation({ ui, document, ledger }) {
  const { busy, setBusy } = ui;
  const { mapName, missionName, isDocumentLeaseCurrent } = document;
  const { hasActiveSave } = ledger;
  const operationRef = useRef(0);
  const operationBusyRef = useRef(false);
  const busyRef = useRef(busy);
  const mapNameRef = useRef(mapName);
  const missionNameRef = useRef(missionName);
  mapNameRef.current = mapName;
  missionNameRef.current = missionName;
  busyRef.current = busy;
  const assertDocumentCurrent = useCallback((lease, targetMapName, targetMissionName) => {
    if (
      !isDocumentLeaseCurrent(lease)
      || mapNameRef.current !== targetMapName
      || missionNameRef.current !== targetMissionName
    ) {
      return false;
    }
    return true;
  }, [isDocumentLeaseCurrent]);

  const beginOperation = useCallback((label) => {
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (operationBusyRef.current) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busyRef.current) throw new Error(`${busyRef.current} is already in progress`);
    const operation = operationRef.current + 1;
    operationRef.current = operation;
    operationBusyRef.current = true;
    setBusy(label);
    return operation;
  }, [hasActiveSave, setBusy]);

  const finishOperation = useCallback((operation, label) => {
    if (operationRef.current !== operation) return;
    operationBusyRef.current = false;
    setBusy((current) => (current === label ? "" : current));
  }, [setBusy]);

  const isBusy = useCallback(() => operationBusyRef.current, []);

  return { assertDocumentCurrent, beginOperation, finishOperation, isBusy };
}
