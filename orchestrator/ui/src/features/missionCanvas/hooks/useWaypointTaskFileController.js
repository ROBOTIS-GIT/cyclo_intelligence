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

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { parseBTXml } from "../../../utils/btTreeParser";
import {
  getNavigationMissionBtFile,
  saveNavigationMissionBtFile,
  setNavigationMissionDefaultBtFile,
} from "../../../utils/navigationMissionsApi";
import { formatTaskDisplayMessage } from "../../../utils/taskTerminology";
import {
  canonicalLocalBtPathForSpot,
  defaultLocalBtXml,
  localBtDirectoriesForSpots,
  localBtDirectoryForSpot,
  localBtPathForSpot,
  localBtPathsForSpot,
  localBtSaveAsPath,
  withLocalBtLibrary,
} from "../lib/missionBtFiles";
import {
  DEFAULT_MAP_NAME,
  DEFAULT_MISSION_NAME,
  missionRequestName,
} from "../lib/missionNames";

function taskDisplayMessage(value) {
  return formatTaskDisplayMessage(value, "Waypoint Task");
}

function requiredFunction(value, name) {
  if (typeof value !== "function") {
    throw new Error(`Waypoint Task controller requires ${name}`);
  }
  return value;
}

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
  const getBtFile = api.getBtFile || getNavigationMissionBtFile;
  const saveBtFile = api.saveBtFile || saveNavigationMissionBtFile;
  const setDefaultBtFile = api.setDefaultBtFile
    || setNavigationMissionDefaultBtFile;
  const parseXml = api.parseXml || parseBTXml;
  const operationRef = useRef(0);
  const operationBusyRef = useRef(false);
  const busyRef = useRef(busy);
  const mapNameRef = useRef(mapName);
  const missionNameRef = useRef(missionName);
  mapNameRef.current = mapName;
  missionNameRef.current = missionName;
  busyRef.current = busy;
  const [loadingPath, setLoadingPath] = useState("");
  const [selectedPathBySpotId, setSelectedPathBySpotId] = useState({});

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

  const selectedSpot = useMemo(
    () => spots.find((spot) => spot.id === selectedSpotId) || null,
    [selectedSpotId, spots],
  );
  const directoryBySpotId = useMemo(
    () => localBtDirectoriesForSpots(spots),
    [spots],
  );
  const selectedDirectory = selectedSpot
    ? directoryBySpotId.get(selectedSpot.id)
      || localBtDirectoryForSpot(selectedSpot)
    : "";
  const defaultPath = selectedSpot ? localBtPathForSpot(selectedSpot) : "";
  const filePaths = useMemo(
    () => (selectedSpot ? localBtPathsForSpot(selectedSpot) : []),
    [selectedSpot],
  );
  const requestedPath = selectedSpot
    ? selectedPathBySpotId[selectedSpot.id]
    : "";
  const selectedPath = filePaths.includes(requestedPath)
    ? requestedPath
    : defaultPath;
  const fileActionsDisabled = (
    Boolean(busy)
    || operationsDisabled
    || Boolean(missionLoadError)
  );
  // Ignore the busy label owned by this controller. Depending on `busy`
  // directly would tear down hydration immediately after beginOperation sets
  // its own label, while this value still tracks external busy release.
  const hydrationExternallyBlocked = Boolean(busy) && !operationBusyRef.current;

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

  const loadXml = useCallback(async (path) => {
    if (!selectedSpot || !filePaths.includes(path)) {
      throw new Error("This XML does not belong to the selected waypoint");
    }
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (operationBusyRef.current) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busy) throw new Error(`${busy} is already in progress`);
    if (!hasPersistedLocalBtPath(path)) {
      const content = getLiveBtFiles()[path];
      if (typeof content !== "string") {
        throw new Error("No Waypoint Task is available at this path");
      }
      return {
        path,
        content,
        exists: true,
        revision: getPersistedRevision(),
      };
    }
    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const documentLease = captureDocumentLease();
    const contentAtLoadStart = getLiveBtFiles()[path];
    const busyLabel = "Load Waypoint Task";
    const operation = beginOperation(busyLabel);
    try {
      const response = await getBtFile(
        targetMapName,
        path,
        missionRequestName(targetMissionName),
      );
      if (!assertDocumentCurrent(documentLease, targetMapName, targetMissionName)) {
        throw new Error("Mission changed while the Waypoint Task was loading");
      }
      if (
        Number.isInteger(response?.revision)
        && response.revision !== getPersistedRevision()
      ) {
        throw new Error("Mission changed in another session. Reload it before editing.");
      }
      if (!response?.exists || typeof response.content !== "string") {
        throw new Error(`No saved XML exists at ${path}`);
      }
      parseXml(response.content);
      const current = getLiveBtFiles();
      if (current[path] !== contentAtLoadStart) {
        throw new Error("Waypoint Task changed while its saved file was loading");
      }
      if (current[path] !== response.content) {
        captureHistory();
        checkpointPersistedBtFile({
          path,
          content: response.content,
          reconcile: false,
        });
        recordBtEdit(path, response.content);
      } else {
        checkpointPersistedBtFile({ path, content: response.content });
      }
      return response;
    } finally {
      finishOperation(operation, busyLabel);
    }
  }, [
    assertDocumentCurrent,
    beginOperation,
    busy,
    captureDocumentLease,
    captureHistory,
    checkpointPersistedBtFile,
    filePaths,
    finishOperation,
    getBtFile,
    getLiveBtFiles,
    getPersistedRevision,
    hasActiveSave,
    hasPersistedLocalBtPath,
    mapName,
    missionName,
    parseXml,
    recordBtEdit,
    selectedSpot,
  ]);

  const saveXml = useCallback(async (path, content) => {
    if (!selectedSpot || !filePaths.includes(path)) {
      throw new Error("This XML does not belong to the selected waypoint");
    }
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (operationBusyRef.current) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busy) throw new Error(`${busy} is already in progress`);
    if (missionLoadError) {
      throw new Error("Reload the mission before saving its Waypoint Task");
    }
    if (getLiveBtFiles()[path] !== content) {
      captureHistory();
      recordBtEdit(path, content);
    }
    if (!hasPersistedLocalBtPath(path)) {
      const saveMission = saveMissionRef?.current;
      if (typeof saveMission !== "function") {
        throw new Error("Mission save is not ready yet");
      }
      await saveMission(missionName);
      const canonicalPath = canonicalLocalBtPathForSpot(
        selectedSpot,
        selectedDirectory,
      );
      const savedPath = hasPersistedLocalBtPath(path) ? path : canonicalPath;
      if (!hasPersistedLocalBtPath(savedPath)) {
        throw new Error("Failed to register this Waypoint Task. Reload the mission and retry.");
      }
      return {
        path: savedPath,
        content,
        exists: true,
        revision: getPersistedRevision(),
      };
    }
    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const documentLease = captureDocumentLease();
    const busyLabel = "Save Waypoint Task";
    const operation = beginOperation(busyLabel);
    try {
      const response = await saveBtFile(
        targetMapName,
        path,
        content,
        missionRequestName(targetMissionName),
        {
          waypointId: selectedSpot.id,
          expectedRevision: getPersistedRevision(),
        },
      );
      if (!assertDocumentCurrent(documentLease, targetMapName, targetMissionName)) {
        return response;
      }
      checkpointPersistedBtFile({
        path,
        content,
        revision: response?.revision,
      });
      return response;
    } finally {
      finishOperation(operation, busyLabel);
    }
  }, [
    assertDocumentCurrent,
    beginOperation,
    busy,
    captureDocumentLease,
    captureHistory,
    checkpointPersistedBtFile,
    filePaths,
    finishOperation,
    getLiveBtFiles,
    getPersistedRevision,
    hasActiveSave,
    hasPersistedLocalBtPath,
    mapName,
    missionLoadError,
    missionName,
    recordBtEdit,
    saveBtFile,
    saveMissionRef,
    selectedDirectory,
    selectedSpot,
  ]);

  const selectXml = useCallback((path) => {
    if (!selectedSpot || !filePaths.includes(path)) {
      throw new Error("This XML does not belong to the selected waypoint");
    }
    setSelectedPathBySpotId((current) => ({
      ...current,
      [selectedSpot.id]: path,
    }));
  }, [filePaths, selectedSpot]);

  const saveXmlAs = useCallback(async (_sourcePath, fileName, content) => {
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (operationBusyRef.current) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busy) throw new Error(`${busy} is already in progress`);
    if (missionLoadError) {
      throw new Error("Reload the mission before saving its Waypoint Task");
    }
    if (!selectedSpot) throw new Error("Select a waypoint first");
    if (!missionStored || !hasPersistedLocalBtPath(defaultPath)) {
      const saveMission = saveMissionRef?.current;
      if (typeof saveMission !== "function") {
        throw new Error("Mission save is not ready yet");
      }
      await saveMission(missionName);
      const canonicalPath = canonicalLocalBtPathForSpot(
        selectedSpot,
        selectedDirectory,
      );
      if (!hasPersistedLocalBtPath(canonicalPath)) {
        throw new Error("Failed to register this Waypoint Task. Reload the mission and retry.");
      }
    }

    const targetPath = localBtSaveAsPath(selectedSpot, fileName, selectedDirectory);
    const occupiedPaths = new Set([
      ...spots.flatMap((spot) => localBtPathsForSpot(spot)),
      ...Object.keys(getLiveBtFiles()),
      ...getPersistedLocalBtPaths(),
    ].map((path) => String(path).toLowerCase()));
    if (occupiedPaths.has(targetPath.toLowerCase())) {
      throw new Error(`A Waypoint Task named ${targetPath.split("/").pop()} already exists`);
    }

    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const targetSpotId = selectedSpot.id;
    const documentLease = captureDocumentLease();
    const busyLabel = "Save Waypoint Task as";
    const operation = beginOperation(busyLabel);
    try {
      const response = await saveBtFile(
        targetMapName,
        targetPath,
        content,
        missionRequestName(targetMissionName),
        {
          waypointId: targetSpotId,
          expectedRevision: getPersistedRevision(),
        },
      );
      if (!assertDocumentCurrent(documentLease, targetMapName, targetMissionName)) {
        throw new Error("Mission changed while the Waypoint Task was being saved");
      }

      captureHistory();
      checkpointPersistedBtFile({
        path: targetPath,
        content,
        revision: response?.revision,
        registerLocalPath: true,
        reconcile: false,
      });
      recordBtEdit(targetPath, content, { allowCreate: true });
      setSpots((current) => current.map((spot) => (
        spot.id === targetSpotId
          ? withLocalBtLibrary(
            spot,
            localBtPathForSpot(spot),
            [...localBtPathsForSpot(spot), targetPath],
          )
          : spot
      )));
      setSelectedPathBySpotId((current) => ({
        ...current,
        [targetSpotId]: targetPath,
      }));
      return { ...response, path: targetPath, selected: true };
    } finally {
      finishOperation(operation, busyLabel);
    }
  }, [
    assertDocumentCurrent,
    beginOperation,
    busy,
    captureDocumentLease,
    captureHistory,
    checkpointPersistedBtFile,
    defaultPath,
    finishOperation,
    getLiveBtFiles,
    getPersistedLocalBtPaths,
    getPersistedRevision,
    hasActiveSave,
    hasPersistedLocalBtPath,
    mapName,
    missionLoadError,
    missionName,
    missionStored,
    recordBtEdit,
    saveBtFile,
    saveMissionRef,
    selectedDirectory,
    selectedSpot,
    setSpots,
    spots,
  ]);

  const setDefaultXml = useCallback(async (path) => {
    if (!selectedSpot || !filePaths.includes(path)) {
      throw new Error("This XML does not belong to the selected waypoint");
    }
    if (path === defaultPath) return undefined;
    if (hasActiveSave()) throw new Error("A mission save is already in progress");
    if (operationBusyRef.current) {
      throw new Error("A Waypoint Task file operation is already in progress");
    }
    if (busy) throw new Error(`${busy} is already in progress`);
    if (missionLoadError) {
      throw new Error("Reload the mission before changing its default Waypoint Task");
    }
    if (!missionStored || !hasPersistedLocalBtPath(path)) {
      throw new Error("Save Mission before changing its default Waypoint Task");
    }

    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const targetSpotId = selectedSpot.id;
    const documentLease = captureDocumentLease();
    const busyLabel = "Set default Waypoint Task";
    const operation = beginOperation(busyLabel);
    try {
      const response = await setDefaultBtFile(
        targetMapName,
        targetSpotId,
        path,
        missionRequestName(targetMissionName),
        { expectedRevision: getPersistedRevision() },
      );
      if (!assertDocumentCurrent(documentLease, targetMapName, targetMissionName)) {
        throw new Error("Mission changed while its default Waypoint Task was being updated");
      }
      setSpots((current) => current.map((spot) => (
        spot.id === targetSpotId
          ? withLocalBtLibrary(spot, path, localBtPathsForSpot(spot))
          : spot
      )));
      setPersistedRevision(response?.revision);
      resetHistory();
      reconcileDirty();
      onMessage(`${path.split("/").pop()} set as the default Waypoint Task`);
      return response;
    } finally {
      finishOperation(operation, busyLabel);
    }
  }, [
    assertDocumentCurrent,
    beginOperation,
    busy,
    captureDocumentLease,
    defaultPath,
    filePaths,
    finishOperation,
    getPersistedRevision,
    hasActiveSave,
    hasPersistedLocalBtPath,
    mapName,
    missionLoadError,
    missionName,
    missionStored,
    onMessage,
    reconcileDirty,
    resetHistory,
    selectedSpot,
    setDefaultBtFile,
    setPersistedRevision,
    setSpots,
  ]);

  const onEditorXmlChange = useCallback((path, nextXml) => {
    if (!path) return;
    const current = getLiveBtFiles();
    // Until the server fetch resolves this path, the editor only holds the
    // fallback tree. A hydration emission must never claim that slot.
    if (current[path] === undefined || current[path] === nextXml) return;
    captureHistory();
    recordBtEdit(path, nextXml);
  }, [captureHistory, getLiveBtFiles, recordBtEdit]);

  const resetSelectedPaths = useCallback(() => {
    setSelectedPathBySpotId({});
  }, []);

  const forgetSpotSelection = useCallback((spotId) => {
    setSelectedPathBySpotId((current) => {
      if (!Object.prototype.hasOwnProperty.call(current, spotId)) return current;
      const next = { ...current };
      delete next[spotId];
      return next;
    });
  }, []);

  // Hydration is display-only, but it still owns the same operation lease as
  // explicit file actions. A full save must not snapshot fallback XML while
  // the real file is still in flight.
  useEffect(() => {
    if (!selectedSpot || !selectedPath) return undefined;
    if (missionBtFiles?.[selectedPath] !== undefined) return undefined;
    if (
      hasActiveSave()
      || operationBusyRef.current
      || hydrationExternallyBlocked
    ) return undefined;
    let cancelled = false;
    const targetMapName = String(mapName || "").trim() || DEFAULT_MAP_NAME;
    const targetMissionName = String(missionName || "").trim() || DEFAULT_MISSION_NAME;
    const documentLease = captureDocumentLease();
    const busyLabel = "Load Waypoint Task";
    const operation = beginOperation(busyLabel);
    setLoadingPath(selectedPath);
    getBtFile(
      targetMapName,
      selectedPath,
      missionRequestName(targetMissionName),
    )
      .then((response) => {
        if (
          cancelled
          || !assertDocumentCurrent(documentLease, targetMapName, targetMissionName)
        ) return;
        if (
          Number.isInteger(response?.revision)
          && response.revision !== getPersistedRevision()
        ) {
          onMessage("Mission changed in another session. Reload it before editing.");
          return;
        }
        replaceLiveBtFiles((current) => ({
          ...current,
          [selectedPath]: response?.exists && typeof response.content === "string"
            ? response.content
            : defaultLocalBtXml(selectedSpot),
        }), { bumpBtEpoch: false, reconcileDirty: false });
      })
      .catch((error) => {
        if (
          cancelled
          || !assertDocumentCurrent(documentLease, targetMapName, targetMissionName)
        ) return;
        onMessage(error instanceof Error
          ? `Failed to load ${selectedPath}: ${taskDisplayMessage(error.message)}`
          : `Failed to load ${selectedPath}`);
      })
      .finally(() => {
        if (!cancelled) setLoadingPath("");
        finishOperation(operation, busyLabel);
      });
    return () => {
      cancelled = true;
      setLoadingPath((current) => (current === selectedPath ? "" : current));
      finishOperation(operation, busyLabel);
    };
  }, [
    assertDocumentCurrent,
    beginOperation,
    captureDocumentLease,
    finishOperation,
    getBtFile,
    getPersistedRevision,
    hasActiveSave,
    hydrationExternallyBlocked,
    mapName,
    missionBtFiles,
    missionName,
    onMessage,
    replaceLiveBtFiles,
    selectedPath,
    selectedSpot,
  ]);

  return {
    selectedSpot,
    selectedDirectory,
    defaultPath,
    filePaths,
    selectedPath,
    loadingPath,
    fileActionsDisabled,
    loadXml,
    saveXml,
    selectXml,
    saveXmlAs,
    setDefaultXml,
    onEditorXmlChange,
    resetSelectedPaths,
    forgetSpotSelection,
    isBusy,
  };
}
