// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef } from "react";
import persistDesignMission from "../lib/persistDesignMission";

function cloneSnapshot(value) {
  if (Array.isArray(value)) return value.map(cloneSnapshot);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, cloneSnapshot(item)]),
    );
  }
  return value;
}

function requiredSnapshot(snapshot) {
  if (!snapshot?.identity?.mapName || !snapshot?.identity?.missionName) {
    throw new Error("Design mission identity is unavailable");
  }
  return cloneSnapshot(snapshot);
}

function sameIdentity(left, right) {
  return left?.mapName === right?.mapName && left?.missionName === right?.missionName;
}

export default function useDesignMissionSaveController({
  getSnapshot,
  getIdentity,
  ledger,
  content,
  history,
  loader,
  catalog,
  setIdentity,
  runCommand,
  onMessage,
  isWaypointFileBusy,
  hasActiveSave,
  persist = persistDesignMission,
}) {
  const dependenciesRef = useRef({
    getSnapshot,
    getIdentity,
    ledger,
    content,
    history,
    loader,
    catalog,
    setIdentity,
    runCommand,
    onMessage,
    isWaypointFileBusy,
    hasActiveSave,
    persist,
  });
  dependenciesRef.current = {
    getSnapshot,
    getIdentity,
    ledger,
    content,
    history,
    loader,
    catalog,
    setIdentity,
    runCommand,
    onMessage,
    isWaypointFileBusy,
    hasActiveSave,
    persist,
  };

  const saveMission = useCallback((targetMissionName) => {
    const ports = dependenciesRef.current;
    const target = String(targetMissionName || "").trim();
    if (!target) {
      ports.onMessage("Mission name required");
      return Promise.resolve({ skipped: true, reason: "mission-name" });
    }
    if (ports.isWaypointFileBusy()) {
      ports.onMessage("A Waypoint Task file operation is already in progress");
      return Promise.resolve({ skipped: true, reason: "waypoint-file-busy" });
    }
    if (ports.hasActiveSave()) {
      ports.onMessage("A mission save is already in progress");
      return Promise.resolve({ skipped: true, reason: "save-active" });
    }

    const snapshot = requiredSnapshot(ports.getSnapshot(target));
    const sourceLease = ports.loader.captureLease();
    if (snapshot.loadError) {
      ports.onMessage(
        "Waypoint Task files did not finish loading. Reload the mission before saving.",
      );
      return Promise.resolve({ skipped: true, reason: "load-error" });
    }
    return ports.runCommand("Save mission", async () => {
      if (
        !ports.loader.isCurrent(sourceLease)
        || !sameIdentity(ports.getIdentity(), snapshot.identity)
      ) {
        return "Save canceled because the current Design document changed";
      }
      const saveOutput = await ports.persist({
        mapName: snapshot.identity.mapName,
        targetMissionName: target,
        targetKnown: snapshot.catalog?.mapName === snapshot.identity.mapName
          && (snapshot.catalog?.names || []).includes(target),
        visibleSpots: snapshot.content?.visibleSpots || [],
        behaviorNodes: snapshot.content?.behaviorNodes || [],
        missionFlowNodes: snapshot.content?.missionFlowNodes || [],
        missionFlowEdges: snapshot.content?.missionFlowEdges || [],
        historyAtStart: snapshot.historyAtStart,
        ledger: ports.ledger,
      });

      if (
        !ports.loader.isCurrent(sourceLease)
        || !sameIdentity(ports.getIdentity(), snapshot.identity)
      ) {
        return `Saved ${target}; current Design document changed`;
      }

      ports.content.applySavedCanonicalSpots(saveOutput.canonicalMissionSpots);
      if (target !== snapshot.identity.missionName) {
        ports.loader.invalidate({
          mapName: snapshot.identity.mapName,
          missionName: target,
        });
      }
      ports.setIdentity({ ...snapshot.identity, missionName: target });
      if (saveOutput.saveResult.hasNewerEdits) {
        ports.history.rebase(saveOutput.savedHistorySnapshot);
      } else {
        ports.history.reset();
      }
      ports.catalog.record(snapshot.identity.mapName, target);
      try {
        await ports.catalog.refresh(snapshot.identity.mapName);
      } catch {
        // The durable save and optimistic catalog entry remain authoritative.
      }
      return saveOutput.saveResult.hasNewerEdits
        ? `Saved ${target}; newer edits remain unsaved`
        : `Saved ${target} for ${snapshot.identity.mapName}`;
    });
  }, []);

  return { saveMission };
}
