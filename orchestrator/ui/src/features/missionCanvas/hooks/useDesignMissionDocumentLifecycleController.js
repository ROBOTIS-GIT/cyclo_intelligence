// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef } from "react";
import { uniqueMissionName } from "../lib/missionNames";

function sameIdentity(left, right) {
  return left?.mapName === right?.mapName && left?.missionName === right?.missionName;
}

export default function useDesignMissionDocumentLifecycleController({
  getIdentity,
  setIdentity,
  loader,
  requests,
  content,
  getCatalogNames,
  setPendingMissionName,
  clearDirty,
  setLoadError,
  onPrepareChange,
  onMessage,
}) {
  const dependenciesRef = useRef({
    getIdentity,
    setIdentity,
    loader,
    requests,
    content,
    getCatalogNames,
    setPendingMissionName,
    clearDirty,
    setLoadError,
    onPrepareChange,
    onMessage,
  });
  dependenciesRef.current = {
    getIdentity,
    setIdentity,
    loader,
    requests,
    content,
    getCatalogNames,
    setPendingMissionName,
    clearDirty,
    setLoadError,
    onPrepareChange,
    onMessage,
  };

  const loadForRequest = useCallback(async ({
    identity,
    request,
    kind,
    loadLegacyDesign = true,
  }) => {
    const ports = dependenciesRef.current;
    try {
      const result = await ports.loader.load({ ...identity, loadLegacyDesign });
      if (
        !request?.isCurrent?.()
        || result?.stale
        || !ports.loader.isCurrent(result?.lease)
        || !sameIdentity(ports.getIdentity(), identity)
      ) return { ...result, stale: true };
      ports.content.commitLoadedSnapshot(result.snapshot);
      ports.setLoadError("");
      ports.onMessage(kind === "picker"
        ? result.exists
          ? `Loaded mission ${identity.missionName} for ${identity.mapName}`
          : result.loadedDesign
            ? `Loaded design for ${identity.mapName}`
            : `Started new mission ${identity.missionName} for ${identity.mapName}`
        : result.exists
          ? `Loaded mission ${identity.missionName}`
          : `Started new mission ${identity.missionName}`);
      return result;
    } catch (error) {
      if (!request?.isCurrent?.() || !sameIdentity(ports.getIdentity(), identity)) {
        return { stale: true };
      }
      const detail = error instanceof Error ? error.message : "Failed to load mission";
      ports.setLoadError(detail);
      ports.onMessage(`${detail}. Reload the mission before saving.`);
      return { error };
    }
  }, []);

  const confirmPickerSelection = useCallback(async ({
    mapName,
    mapPath,
    missionName,
    isCurrent,
  }) => {
    const ports = dependenciesRef.current;
    const identity = { mapName, mapPath, missionName };
    ports.setIdentity(identity);
    ports.setPendingMissionName(missionName);
    ports.onPrepareChange?.("picker", identity);
    ports.setLoadError("");
    ports.clearDirty();
    return loadForRequest({
      identity: { mapName, missionName },
      request: { isCurrent },
      kind: "picker",
    });
  }, [loadForRequest]);

  const switchMission = useCallback(async (missionName) => {
    const ports = dependenciesRef.current;
    const current = ports.getIdentity();
    const selectedMissionName = String(missionName || "").trim();
    if (
      !selectedMissionName
      || selectedMissionName === current?.missionName
      || !current?.mapName
    ) {
      return { skipped: true };
    }
    const identity = { ...current, missionName: selectedMissionName };
    const request = ports.requests.begin();
    ports.setIdentity(identity);
    ports.setPendingMissionName(selectedMissionName);
    ports.onPrepareChange?.("switch", identity);
    ports.clearDirty();
    ports.setLoadError("");
    try {
      return await loadForRequest({ identity, request, kind: "switch" });
    } finally {
      ports.requests.finish(request);
    }
  }, [loadForRequest]);

  const newDocument = useCallback(({ missionName, btFiles = {} } = {}) => {
    const ports = dependenciesRef.current;
    const requestedMissionName = String(missionName || "").trim();
    const nextMissionName = requestedMissionName || uniqueMissionName(
      "untitled",
      ports.getCatalogNames?.() || [],
    );
    const identity = { ...ports.getIdentity(), missionName: nextMissionName };
    ports.requests.invalidate();
    ports.loader.invalidate(identity);
    ports.setIdentity(identity);
    ports.setPendingMissionName(nextMissionName);
    ports.onPrepareChange?.("new", identity);
    ports.content.resetNewDocument({ btFiles });
    ports.onMessage("Started new mission — Save Mission to name it");
    return identity;
  }, []);

  const continueAfterDelete = useCallback(async ({ remainingNames, newDocument: nextNew }) => {
    const ports = dependenciesRef.current;
    ports.clearDirty();
    if (Array.isArray(remainingNames) && remainingNames.length > 0) {
      return switchMission(remainingNames[0]);
    }
    return newDocument(nextNew);
  }, [newDocument, switchMission]);

  return {
    confirmPickerSelection,
    switchMission,
    newDocument,
    continueAfterDelete,
  };
}
