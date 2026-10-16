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

import { useCallback, useEffect, useRef, useState } from "react";
import { DEFAULT_MAP_NAME, DEFAULT_MISSION_NAME, mapNameFromPgmPath } from "../lib/missionNames";
import { STAGE_NAVIGATE, STAGE_RUN, STAGE_OPERATE, STAGE_COLLECT } from "../lib/stages";

// Owns only the Run/Navigate/Operate asset picker and its asynchronous selection flow.
// The read-only mission document remains owned by useRunMissionSnapshot, while
// runtime/localization/session ownership remains in the workspace.
export default function useRunMapMissionSelectionController({
  getCurrentStage,
  getDefaults,
  inventory,
  snapshot,
  commits,
  onMessage,
}) {
  const [open, setOpen] = useState(false);
  const [stage, setStage] = useState(STAGE_RUN);
  const [files, setFiles] = useState([]);
  const [missionNames, setMissionNames] = useState([]);
  const [selectedPath, setSelectedPath] = useState("");
  const [selectedMission, setSelectedMission] = useState(DEFAULT_MISSION_NAME);
  const [busy, setBusy] = useState(false);
  const mountedRef = useRef(true);
  const generationRef = useRef(0);
  const snapshotLoadRequestRef = useRef(0);
  const selectedPathRef = useRef(selectedPath);
  const selectedMissionRef = useRef(selectedMission);
  selectedPathRef.current = selectedPath;
  selectedMissionRef.current = selectedMission;
  const portsRef = useRef({ getCurrentStage, getDefaults, inventory, snapshot, commits, onMessage });
  portsRef.current = { getCurrentStage, getDefaults, inventory, snapshot, commits, onMessage };

  const nextRequest = useCallback(() => {
    generationRef.current += 1;
    return generationRef.current;
  }, []);
  const isCurrent = useCallback((request) => (
    mountedRef.current && generationRef.current === request
  ), []);

  const invalidatePendingSnapshotLoad = useCallback(() => {
    if (!snapshotLoadRequestRef.current) return false;
    snapshotLoadRequestRef.current = 0;
    portsRef.current.snapshot.invalidate?.();
    return true;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      generationRef.current += 1;
      invalidatePendingSnapshotLoad();
    };
  }, [invalidatePendingSnapshotLoad]);

  const openDialog = useCallback(() => {
    invalidatePendingSnapshotLoad();
    const ports = portsRef.current;
    const currentStage = ports.getCurrentStage();
    const dialogStage = [STAGE_NAVIGATE, STAGE_OPERATE, STAGE_COLLECT].includes(currentStage) ? currentStage : STAGE_RUN;
    const mapOnly = [STAGE_NAVIGATE, STAGE_OPERATE, STAGE_COLLECT].includes(dialogStage);
    const current = ports.snapshot.get();
    const defaults = ports.getDefaults();
    const previousPath = selectedPathRef.current || defaults.selectedPath;
    const preferredMapName = current.mapName
      || mapNameFromPgmPath(previousPath)
      || defaults.mapName
      || DEFAULT_MAP_NAME;
    const preferredMission = current.mapName
      ? current.missionName
      : previousPath
        ? selectedMissionRef.current || defaults.selectedMission
        : defaults.missionName;
    const request = nextRequest();
    setStage(dialogStage);
    ports.commits.setStage(dialogStage);
    setOpen(true);
    setBusy(true);
    ports.onMessage(mapOnly ? "Loading saved maps" : "Loading saved missions");
    void ports.inventory.listMaps()
      .then(async (response) => {
        if (!isCurrent(request)) return;
        const nextFiles = Array.isArray(response?.files) ? response.files : [];
        const preferred = nextFiles.find(
          (file) => mapNameFromPgmPath(file.path) === preferredMapName,
        ) || nextFiles[0];
        setFiles(nextFiles);
        setSelectedPath(preferred?.path || "");
        if (mapOnly) {
          setMissionNames([]);
        } else if (preferred?.path) {
          const names = await ports.inventory.listMissions(mapNameFromPgmPath(preferred.path));
          if (!isCurrent(request)) return;
          setMissionNames(names);
          setSelectedMission(names.includes(preferredMission)
            ? preferredMission
            : names[0] ?? DEFAULT_MISSION_NAME);
        } else {
          setMissionNames([]);
          setSelectedMission("");
        }
        if (!nextFiles.length) ports.onMessage("No PGM files found");
      })
      .catch((error) => {
        if (!isCurrent(request)) return;
        ports.onMessage(error instanceof Error ? error.message : "Failed to list PGM files");
      })
      .finally(() => {
        if (isCurrent(request)) setBusy(false);
      });
  }, [invalidatePendingSnapshotLoad, isCurrent, nextRequest]);

  const changeMap = useCallback(async (path) => {
    invalidatePendingSnapshotLoad();
    const ports = portsRef.current;
    const request = nextRequest();
    setSelectedPath(path);
    if ([STAGE_NAVIGATE, STAGE_OPERATE, STAGE_COLLECT].includes(stage)) {
      setMissionNames([]);
      return { mapOnly: true };
    }
    const mapName = mapNameFromPgmPath(path);
    if (!mapName) {
      setMissionNames([]);
      setSelectedMission("");
      return { empty: true };
    }
    setBusy(true);
    try {
      const names = await ports.inventory.listMissions(mapName);
      if (!isCurrent(request)) return { stale: true };
      setMissionNames(names);
      setSelectedMission(names[0] ?? DEFAULT_MISSION_NAME);
      return { names };
    } catch (error) {
      if (!isCurrent(request)) return { stale: true };
      setMissionNames([]);
      setSelectedMission("");
      ports.onMessage(error instanceof Error ? error.message : "Failed to list missions");
      return { error };
    } finally {
      if (isCurrent(request)) setBusy(false);
    }
  }, [invalidatePendingSnapshotLoad, isCurrent, nextRequest, stage]);

  const cancel = useCallback(() => {
    invalidatePendingSnapshotLoad();
    nextRequest();
    setBusy(false);
    setOpen(false);
    if (stage === STAGE_RUN) {
      setSelectedMission(portsRef.current.snapshot.get().missionName);
    }
  }, [invalidatePendingSnapshotLoad, nextRequest, stage]);

  const confirm = useCallback(async () => {
    const ports = portsRef.current;
    const mapName = mapNameFromPgmPath(selectedPath);
    if (!selectedPath || !mapName) {
      ports.onMessage("Map file required");
      return { skipped: true, reason: "map-required" };
    }
    invalidatePendingSnapshotLoad();
    if ([STAGE_NAVIGATE, STAGE_OPERATE, STAGE_COLLECT].includes(stage)) {
      const current = ports.snapshot.get();
      const preserve = !current.invalid && current.catalog?.mapName === mapName;
      nextRequest();
      setOpen(false);
      ports.commits.setStage(stage);
      ports.commits.setInteractionMode("view");
      ports.commits.setMapLoaded(true);
      ports.commits.invalidateGoal();
      if (!preserve || [STAGE_OPERATE, STAGE_COLLECT].includes(stage)) {
        ports.snapshot.cancelAndClear({ mapName });
      }
      const preservePose = [STAGE_OPERATE, STAGE_COLLECT].includes(stage) ? current.mapName === mapName : preserve;
      if (!preservePose) ports.commits.resetPose();
      ports.commits.onMapLoaded?.({ mapName, stage });
      setBusy(false);
      ports.onMessage(`Loaded map ${mapName}`);
      return { mapName, preserve };
    }
    if (!selectedMission) {
      ports.onMessage("Mission file required");
      return { skipped: true, reason: "mission-required" };
    }
    const request = nextRequest();
    ports.commits.setMapLoaded(false);
    ports.snapshot.cancelAndClear();
    snapshotLoadRequestRef.current = request;
    setOpen(false);
    ports.commits.setStage(stage);
    ports.commits.setInteractionMode("view");
    setBusy(true);
    try {
      const result = await ports.snapshot.load(mapName, selectedMission, {
        catalogNames: missionNames,
      });
      if (snapshotLoadRequestRef.current === request) {
        snapshotLoadRequestRef.current = 0;
      }
      if (!isCurrent(request) || result?.stale) return { stale: true };
      ports.commits.setMapLoaded(true);
      ports.onMessage(result?.exists
        ? `Loaded mission ${selectedMission} for ${mapName}`
        : `Started new mission for ${mapName}`);
      return result;
    } catch (error) {
      if (!isCurrent(request)) return { stale: true };
      ports.onMessage(error instanceof Error ? error.message : "Failed to load mission");
      return { error };
    } finally {
      if (snapshotLoadRequestRef.current === request) {
        snapshotLoadRequestRef.current = 0;
      }
      if (isCurrent(request)) setBusy(false);
    }
  }, [invalidatePendingSnapshotLoad, isCurrent, missionNames, nextRequest,
    selectedMission, selectedPath, stage]);

  const switchMission = useCallback(async (name) => {
    const missionName = String(name || "").trim();
    const ports = portsRef.current;
    const current = ports.snapshot.get();
    if (!missionName || missionName === current.missionName) {
      return { skipped: true };
    }
    invalidatePendingSnapshotLoad();
    const request = nextRequest();
    snapshotLoadRequestRef.current = request;
    setBusy(true);
    try {
      const result = await ports.snapshot.load(current.mapName, missionName);
      if (snapshotLoadRequestRef.current === request) {
        snapshotLoadRequestRef.current = 0;
      }
      if (!isCurrent(request) || result?.stale) return { stale: true };
      ports.onMessage(result?.exists
        ? `Loaded mission ${missionName}`
        : `Started new mission ${missionName}`);
      return result;
    } catch (error) {
      if (!isCurrent(request)) return { stale: true };
      ports.onMessage(error instanceof Error ? error.message : "Failed to load mission");
      return { error };
    } finally {
      if (snapshotLoadRequestRef.current === request) {
        snapshotLoadRequestRef.current = 0;
      }
      if (isCurrent(request)) setBusy(false);
    }
  }, [invalidatePendingSnapshotLoad, isCurrent, nextRequest]);

  const invalidate = useCallback(() => {
    invalidatePendingSnapshotLoad();
    nextRequest();
    setBusy(false);
    setOpen(false);
  }, [invalidatePendingSnapshotLoad, nextRequest]);

  return {
    dialog: {
      open,
      stage,
      files,
      missionNames: [STAGE_NAVIGATE, STAGE_OPERATE, STAGE_COLLECT].includes(stage) ? null : missionNames,
      selectedPath,
      setSelectedMission,
      selectedMission,
      busy,
      openDialog,
      changeMap,
      cancel,
      confirm,
    },
    switchMission,
    invalidate,
  };
}
