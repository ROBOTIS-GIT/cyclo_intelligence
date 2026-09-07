// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef, useState } from "react";
import {
  deleteNavigationMission,
  duplicateNavigationMission,
  getNavigationMissions,
  renameNavigationMission,
} from "../../../utils/navigationMissionsApi";
import { getPgmFiles } from "../../../utils/navigationApi";
import {
  DEFAULT_MISSION_NAME,
  isValidMissionName,
  mapNameFromPgmPath,
  uniqueMissionName,
} from "../lib/missionNames";

export default function useDesignMissionCatalogController({
  currentMapName,
  currentMapPath,
  currentMissionName,
  getPersistedRevision,
  setPersistedRevision,
  invalidateDocument,
  runCommand,
  onMessage,
  onPrepareOpen,
  onConfirmSelection,
  onRenamed,
  onDeleted,
  api = {},
}) {
  const listMaps = api.listMaps || getPgmFiles;
  const listMissionsApi = api.listMissions || getNavigationMissions;
  const renameMissionApi = api.renameMission || renameNavigationMission;
  const duplicateMissionApi = api.duplicateMission || duplicateNavigationMission;
  const deleteMissionApi = api.deleteMission || deleteNavigationMission;
  const [catalog, setCatalog] = useState({ mapName: "", names: [] });
  const [pickerOpen, setPickerOpen] = useState(false);
  const pickerOpenRef = useRef(false);
  const [mapFiles, setMapFiles] = useState([]);
  const [missionNames, setMissionNames] = useState([]);
  const [pendingMapPath, setPendingMapPath] = useState(currentMapPath || "");
  const [pendingMissionName, setPendingMissionName] = useState(DEFAULT_MISSION_NAME);
  const [busy, setBusy] = useState(false);
  const [catalogReady, setCatalogReady] = useState(false);
  const requestRef = useRef(0);
  const catalogGenerationRef = useRef(0);
  const pickerConfirmRef = useRef(false);
  const mutationRef = useRef(false);
  const identityRef = useRef({ mapName: currentMapName, missionName: currentMissionName });
  identityRef.current = { mapName: currentMapName, missionName: currentMissionName };
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameName, setRenameName] = useState("");
  const [duplicateOpen, setDuplicateOpen] = useState(false);
  const [duplicateName, setDuplicateName] = useState("");
  const [deleteOpen, setDeleteOpen] = useState(false);

  const listMissionNames = useCallback(async (mapName) => {
    const response = await listMissionsApi(mapName);
    return Array.isArray(response?.missions) ? response.missions : [];
  }, [listMissionsApi]);

  const refreshCatalog = useCallback(async (mapName = currentMapName) => {
    const generation = catalogGenerationRef.current + 1;
    catalogGenerationRef.current = generation;
    const names = await listMissionNames(mapName);
    if (
      catalogGenerationRef.current !== generation
      || identityRef.current.mapName !== mapName
    ) return names;
    setCatalog({ mapName, names });
    return names;
  }, [currentMapName, listMissionNames]);
  const recordSavedMission = useCallback((mapName, missionName) => {
    if (identityRef.current.mapName !== mapName) return;
    catalogGenerationRef.current += 1;
    setCatalog((current) => ({
      mapName,
      names: current.mapName === mapName
        ? [...new Set([...current.names, missionName])]
        : [missionName],
    }));
  }, []);

  const openPicker = useCallback(() => {
    const request = requestRef.current + 1;
    requestRef.current = request;
    catalogGenerationRef.current += 1;
    setCatalogReady(false);
    onPrepareOpen?.();
    pickerOpenRef.current = true;
    setPickerOpen(true);
    setPendingMapPath(currentMapPath || "");
    setBusy(true);
    onMessage("Loading saved missions");
    void listMaps()
      .then(async (response) => {
        const files = response.files || [];
        const existing = files.find((file) => file.path === currentMapPath);
        const preferred = existing
          || files.find((file) => mapNameFromPgmPath(file.path) === currentMapName)
          || files[0];
        const selectedMapName = preferred?.path ? mapNameFromPgmPath(preferred.path) : "";
        const available = selectedMapName ? await listMissionNames(selectedMapName) : [];
        if (requestRef.current !== request) return;
        setMapFiles(files);
        setPendingMapPath(preferred?.path || "");
        setMissionNames(available);
        setPendingMissionName(preferred?.path
          ? (selectedMapName === currentMapName && available.includes(currentMissionName)
            ? currentMissionName : available[0] ?? DEFAULT_MISSION_NAME)
          : "");
        if (!files.length) onMessage("No PGM files found");
        setCatalogReady(true);
      })
      .catch((error) => {
        if (requestRef.current !== request) return;
        onMessage(error instanceof Error ? error.message : "Failed to list PGM files");
      })
      .finally(() => {
        if (requestRef.current === request) setBusy(false);
      });
  }, [currentMapName, currentMapPath, currentMissionName, listMaps,
    listMissionNames, onMessage, onPrepareOpen]);

  const changePendingMap = useCallback((nextPath) => {
    const request = requestRef.current + 1;
    requestRef.current = request;
    catalogGenerationRef.current += 1;
    setCatalogReady(false);
    setPendingMapPath(nextPath);
    const selectedMapName = mapNameFromPgmPath(nextPath);
    if (!selectedMapName) {
      setMissionNames([]);
      setPendingMissionName("");
      setCatalogReady(true);
      setBusy(false);
      return;
    }
    setBusy(true);
    void listMissionNames(selectedMapName)
      .then((available) => {
        if (requestRef.current !== request) return;
        setMissionNames(available);
        setPendingMissionName(available[0] ?? DEFAULT_MISSION_NAME);
        setCatalogReady(true);
      })
      .catch((error) => {
        if (requestRef.current !== request) return;
        setMissionNames([]);
        setPendingMissionName("");
        onMessage(error instanceof Error ? error.message : "Failed to list missions");
      })
      .finally(() => {
        if (requestRef.current === request) setBusy(false);
      });
  }, [listMissionNames, onMessage]);

  const cancelPicker = useCallback(() => {
    requestRef.current += 1;
    catalogGenerationRef.current += 1;
    pickerConfirmRef.current = false;
    pickerOpenRef.current = false;
    setBusy(false);
    setPendingMapPath(currentMapPath || "");
    setPendingMissionName(currentMissionName);
    setPickerOpen(false);
  }, [currentMapPath, currentMissionName]);

  const beginDocumentRequest = useCallback(() => {
    const id = requestRef.current + 1;
    requestRef.current = id;
    catalogGenerationRef.current += 1;
    setBusy(true);
    return { id, isCurrent: () => requestRef.current === id };
  }, []);
  const finishDocumentRequest = useCallback((request) => {
    if (request?.isCurrent?.()) setBusy(false);
  }, []);
  const invalidateRequests = useCallback(() => {
    requestRef.current += 1;
    catalogGenerationRef.current += 1;
    pickerConfirmRef.current = false;
    pickerOpenRef.current = false;
    setPickerOpen(false);
    setBusy(false);
  }, []);

  const confirmSelection = useCallback(async () => {
    if (!pickerOpenRef.current || pickerConfirmRef.current || busy || !catalogReady) return;
    const mapName = mapNameFromPgmPath(pendingMapPath);
    if (!mapName) { onMessage("Map file required"); return; }
    if (!pendingMissionName) { onMessage("Mission file required"); return; }
    const documentRequest = beginDocumentRequest();
    const request = documentRequest.id;
    pickerConfirmRef.current = true;
    pickerOpenRef.current = false;
    setCatalog({ mapName, names: missionNames });
    setPickerOpen(false);
    setBusy(true);
    try {
      await onConfirmSelection({
        request,
        mapName,
        mapPath: pendingMapPath,
        missionName: pendingMissionName,
        catalogNames: missionNames,
        isCurrent: documentRequest.isCurrent,
      });
    } finally {
      if (requestRef.current === request) {
        pickerConfirmRef.current = false;
        setBusy(false);
      }
    }
  }, [beginDocumentRequest, busy, catalogReady, missionNames, onConfirmSelection, onMessage,
    pendingMapPath, pendingMissionName]);

  const openRename = useCallback(() => {
    setRenameName(currentMissionName);
    setRenameOpen(true);
  }, [currentMissionName]);
  const confirmRename = useCallback(() => {
    if (mutationRef.current) return;
    const target = renameName.trim();
    if (!isValidMissionName(target)) return;
    setRenameOpen(false);
    if (target === currentMissionName || catalog.names.includes(target)) return;
    const previousName = currentMissionName;
    const identity = { mapName: currentMapName, missionName: currentMissionName };
    mutationRef.current = true;
    void runCommand("Rename mission", async () => {
      try {
        const renamed = await renameMissionApi(currentMapName, previousName, target, {
          expectedRevision: getPersistedRevision(),
        });
        if (identityRef.current.mapName !== identity.mapName
          || identityRef.current.missionName !== identity.missionName) return undefined;
        setPersistedRevision(renamed?.revision);
        invalidateDocument({ mapName: currentMapName, missionName: target });
        const optimistic = catalog.names.map((name) => (name === previousName ? target : name));
        setCatalog({ mapName: currentMapName, names: optimistic });
        setPendingMissionName(target);
        onRenamed?.(target);
        try { await refreshCatalog(currentMapName); } catch { /* retain optimistic catalog */ }
        return `Renamed ${previousName} to ${target}`;
      } finally { mutationRef.current = false; }
    });
  }, [catalog.names, currentMapName, currentMissionName, getPersistedRevision,
    invalidateDocument, onRenamed, refreshCatalog, renameMissionApi, renameName,
    runCommand, setPersistedRevision]);

  const openDuplicate = useCallback(() => {
    setDuplicateName(uniqueMissionName(`${currentMissionName}-copy`, catalog.names));
    setDuplicateOpen(true);
  }, [catalog.names, currentMissionName]);
  const confirmDuplicate = useCallback(() => {
    if (mutationRef.current) return;
    const target = duplicateName.trim();
    if (!isValidMissionName(target) || catalog.names.includes(target)) return;
    setDuplicateOpen(false);
    const identity = { mapName: currentMapName, missionName: currentMissionName };
    mutationRef.current = true;
    void runCommand("Duplicate mission", async () => {
      try {
        await duplicateMissionApi(currentMapName, currentMissionName, target, {
          expectedRevision: getPersistedRevision(),
        });
        if (identityRef.current.mapName !== identity.mapName
          || identityRef.current.missionName !== identity.missionName) return undefined;
        setCatalog({ mapName: currentMapName, names: [...new Set([...catalog.names, target])] });
        try { await refreshCatalog(currentMapName); } catch { /* retain optimistic catalog */ }
        return `Duplicated ${currentMissionName} as ${target}`;
      } finally { mutationRef.current = false; }
    });
  }, [catalog.names, currentMapName, currentMissionName, duplicateMissionApi,
    duplicateName, getPersistedRevision, refreshCatalog, runCommand]);

  const confirmDelete = useCallback(() => {
    if (mutationRef.current) return;
    setDeleteOpen(false);
    const deletedName = currentMissionName;
    const identity = { mapName: currentMapName, missionName: currentMissionName };
    mutationRef.current = true;
    void runCommand("Delete mission", async () => {
      try {
        await deleteMissionApi(currentMapName, deletedName, {
          expectedRevision: getPersistedRevision(),
        });
        if (identityRef.current.mapName !== identity.mapName
          || identityRef.current.missionName !== identity.missionName) return undefined;
        let available = catalog.names.filter((name) => name !== deletedName);
        setCatalog({ mapName: currentMapName, names: available });
        try { available = await refreshCatalog(currentMapName); } catch { /* retain optimistic catalog */ }
        if (identityRef.current.mapName !== identity.mapName
          || identityRef.current.missionName !== identity.missionName) return undefined;
        invalidateDocument({ mapName: currentMapName, missionName: deletedName });
        await onDeleted?.({ deletedName, remainingNames: available });
        return `Deleted mission ${deletedName}`;
      } finally { mutationRef.current = false; }
    });
  }, [catalog.names, currentMapName, currentMissionName, deleteMissionApi,
    getPersistedRevision, invalidateDocument, onDeleted, refreshCatalog, runCommand]);

  return {
    catalog, refreshCatalog, recordSavedMission, listMissionNames,
    beginDocumentRequest, finishDocumentRequest, invalidateRequests,
    picker: {
      open: pickerOpen, files: mapFiles, missionNames, pendingMapPath,
      pendingMissionName, busy, catalogReady, openPicker, changePendingMap,
      setPendingMissionName, cancelPicker, confirmSelection,
    },
    rename: { open: renameOpen, name: renameName, setName: setRenameName,
      openDialog: openRename, close: () => setRenameOpen(false), confirm: confirmRename },
    duplicate: { open: duplicateOpen, name: duplicateName, setName: setDuplicateName,
      openDialog: openDuplicate, close: () => setDuplicateOpen(false), confirm: confirmDuplicate },
    deletion: { open: deleteOpen, openDialog: () => setDeleteOpen(true),
      close: () => setDeleteOpen(false), confirm: confirmDelete },
  };
}
