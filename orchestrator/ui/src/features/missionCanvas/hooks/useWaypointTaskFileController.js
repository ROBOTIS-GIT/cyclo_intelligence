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

import { useCallback } from 'react';
import { parseBTXml } from '../../../utils/btTreeParser';
import { getNavigationMissionBtFile, saveNavigationMissionBtFile,
  setNavigationMissionDefaultBtFile } from '../../../utils/navigationMissionsApi';
import { DEFAULT_MAP_NAME, DEFAULT_MISSION_NAME } from '../../mapWorkspace/lib/missionNames';
import waypointTaskLedger from '../lib/waypointTaskLedger';
import useWaypointTaskOperation from './taskFiles/useWaypointTaskOperation';
import useWaypointTaskSelection from './taskFiles/useWaypointTaskSelection';
import useWaypointTaskLoad from './taskFiles/useWaypointTaskLoad';
import useWaypointTaskSave from './taskFiles/useWaypointTaskSave';
import useWaypointTaskSaveAs from './taskFiles/useWaypointTaskSaveAs';
import useWaypointTaskDefault from './taskFiles/useWaypointTaskDefault';
import useWaypointTaskHydration from './taskFiles/useWaypointTaskHydration';
export default function useWaypointTaskFileController({
  spots = [],
  selectedSpotId = "",
  mapName = DEFAULT_MAP_NAME,
  missionName = DEFAULT_MISSION_NAME,
  missionStored = false,
  busy = "",
  operationsDisabled = false,
  missionLoadError = "",
  ledger,
  captureDocumentLease,
  isDocumentLeaseCurrent,
  captureHistory,
  resetHistory,
  saveMissionRef,
  setSpots,
  setBusy,
  onMessage,
  api = {},
}) {
  const document = { mapName, missionName, missionStored, captureDocumentLease,
    isDocumentLeaseCurrent, missionLoadError, saveMissionRef };
  const ui = { busy, setBusy, setSpots, onMessage, operationsDisabled };
  const history = { captureHistory, resetHistory };
  const services = {
    getBtFile: api.getBtFile || getNavigationMissionBtFile,
    saveBtFile: api.saveBtFile || saveNavigationMissionBtFile,
    setDefaultBtFile: api.setDefaultBtFile || setNavigationMissionDefaultBtFile,
    parseXml: api.parseXml || parseBTXml,
  };
  const ledgerPorts = waypointTaskLedger(ledger);
  const selection = { spots, selectedSpotId,
    ...useWaypointTaskSelection({ selection: { spots, selectedSpotId } }) };
  const operation = useWaypointTaskOperation({ document, ledger: ledgerPorts, ui });
  // The operation gate is shared by hydration, standalone files and full save.
  // Its own busy label must not cancel the hydration that acquired the gate.
  const operations = { ...operation,
    hydrationExternallyBlocked: Boolean(busy) && !operation.isBusy() };
  const loadXml = useWaypointTaskLoad({
    selection,
    ledger: ledgerPorts,
    operations,
    ui,
    document,
    api: services,
    history,
  });
  const saveXml = useWaypointTaskSave({
    selection,
    ledger: ledgerPorts,
    operations,
    ui,
    document,
    history,
    api: services,
  });
  const saveXmlAs = useWaypointTaskSaveAs({
    ledger: ledgerPorts,
    operations,
    ui,
    document,
    selection,
    api: services,
    history,
  });
  const setDefaultXml = useWaypointTaskDefault({
    selection,
    ledger: ledgerPorts,
    operations,
    ui,
    document,
    api: services,
    history,
  });
  const loadingPath = useWaypointTaskHydration({
    selection,
    ledger: ledgerPorts,
    operations,
    document,
    api: services,
    ui,
  });
  const { getLiveBtFiles, recordBtEdit } = ledgerPorts;
  const onEditorXmlChange = useCallback((path, nextXml) => {
    if (!path) return;
    const current = getLiveBtFiles();
    // Until the server fetch resolves this path, the editor only holds the
    // fallback tree. A hydration emission must never claim that slot.
    if (current[path] === undefined || current[path] === nextXml) return;
    captureHistory();
    recordBtEdit(path, nextXml);
  }, [captureHistory, getLiveBtFiles, recordBtEdit]);

  return {
    selectedSpot: selection.selectedSpot,
    selectedDirectory: selection.selectedDirectory,
    defaultPath: selection.defaultPath,
    filePaths: selection.filePaths,
    selectedPath: selection.selectedPath,
    loadingPath,
    fileActionsDisabled: Boolean(busy) || operationsDisabled || Boolean(missionLoadError),
    loadXml, saveXml, saveXmlAs, setDefaultXml, onEditorXmlChange,
    selectXml: selection.selectXml,
    resetSelectedPaths: selection.resetSelectedPaths,
    forgetSpotSelection: selection.forgetSpotSelection,
    isBusy: operation.isBusy,
  };
}
