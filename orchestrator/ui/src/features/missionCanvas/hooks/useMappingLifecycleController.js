// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useRef, useState } from "react";

export default function useMappingLifecycleController({
  active = true,
  inventoryRefreshToken,
  getMapName,
  runtime,
  document,
  inventory,
  runCommand,
  onMessage,
}) {
  const [savedMaps, setSavedMaps] = useState([]);
  const [inventoryLoading, setInventoryLoading] = useState(false);
  const [operation, setOperation] = useState("");
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveMapName, setSaveMapName] = useState("");
  const mountedRef = useRef(true);
  const activeRef = useRef(active);
  activeRef.current = active;
  const generationRef = useRef(0);
  const inventoryGenerationRef = useRef(0);
  const operationRef = useRef("");
  const removeRef = useRef(false);
  const dependenciesRef = useRef({
    getMapName, runtime, document, inventory, runCommand, onMessage,
  });
  dependenciesRef.current = {
    getMapName, runtime, document, inventory, runCommand, onMessage,
  };

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      generationRef.current += 1;
      inventoryGenerationRef.current += 1;
      operationRef.current = "";
      removeRef.current = false;
    };
  }, []);

  const runLocked = useCallback((label, action) => {
    if (operationRef.current) {
      dependenciesRef.current.onMessage(`${operationRef.current} is already in progress`);
      return Promise.resolve({ skipped: true, reason: "operation-active" });
    }
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    operationRef.current = label;
    setOperation(label);
    const isCurrent = () => mountedRef.current && generationRef.current === generation;
    const release = () => {
      if (!isCurrent()) return;
      operationRef.current = "";
      setOperation("");
    };
    let command;
    try {
      command = dependenciesRef.current.runCommand(label, () => action(isCurrent));
    } catch (error) {
      release();
      return Promise.reject(error);
    }
    return Promise.resolve(command).finally(release);
  }, []);

  const start = useCallback(() => {
    const ports = dependenciesRef.current;
    const mapName = String(ports.getMapName() || "").trim() || "map";
    return runLocked("Mapping", async (isCurrent) => {
      ports.runtime.prepareStart(mapName);
      const result = await ports.runtime.start(mapName);
      // Starting Nav2 is durable even when this workspace disappears before
      // the HTTP response settles. Persist that authoritative runtime fact,
      // while keeping React state commits guarded by the mounted generation.
      ports.runtime.persistStarted?.(mapName, result);
      if (!isCurrent()) return { stale: true };
      ports.runtime.commitStarted(mapName, result);
      return result;
    });
  }, [runLocked]);

  const stop = useCallback(() => runLocked("Stop", async (isCurrent) => {
    const result = await dependenciesRef.current.runtime.stop();
    if (!isCurrent()) return { stale: true };
    dependenciesRef.current.runtime.commitStopped?.(result);
    return result;
  }), [runLocked]);

  const save = useCallback((requestedName) => {
    const mapName = String(requestedName ?? saveMapName).trim();
    if (!mapName) {
      dependenciesRef.current.onMessage("Map name required");
      return Promise.resolve({ skipped: true, reason: "map-name" });
    }
    return runLocked("Save map", async (isCurrent) => {
      const result = await dependenciesRef.current.document.save(mapName);
      if (!isCurrent()) return { stale: true };
      dependenciesRef.current.document.commitSavedMap(mapName, result);
      setSaveDialogOpen(false);
      return result;
    });
  }, [runLocked, saveMapName]);

  const openSaveDialog = useCallback(() => {
    const mapName = String(dependenciesRef.current.getMapName() || "").trim() || "map";
    setSaveMapName(mapName);
    setSaveDialogOpen(true);
  }, []);

  const closeSaveDialog = useCallback(() => {
    if (operationRef.current === "Save map") return;
    setSaveDialogOpen(false);
  }, []);

  const refreshInventory = useCallback(async () => {
    if (!activeRef.current) return { skipped: true, reason: "inactive" };
    const generation = inventoryGenerationRef.current + 1;
    inventoryGenerationRef.current = generation;
    setInventoryLoading(true);
    try {
      const result = await dependenciesRef.current.inventory.list();
      if (
        !mountedRef.current
        || !activeRef.current
        || inventoryGenerationRef.current !== generation
      ) {
        return { stale: true };
      }
      const files = Array.isArray(result?.files) ? result.files : [];
      setSavedMaps(files);
      return { files };
    } catch (error) {
      if (
        !mountedRef.current
        || !activeRef.current
        || inventoryGenerationRef.current !== generation
      ) {
        return { stale: true };
      }
      setSavedMaps([]);
      return { files: [], error };
    } finally {
      if (mountedRef.current && inventoryGenerationRef.current === generation) {
        setInventoryLoading(false);
      }
    }
  }, []);

  const removeSavedMap = useCallback(async (path) => {
    const ports = dependenciesRef.current;
    if (removeRef.current) {
      return { skipped: true, reason: "remove-active" };
    }
    if (ports.inventory.isProtected?.(path)) {
      ports.onMessage("Stop navigation before deleting this map");
      return { skipped: true, reason: "protected" };
    }
    removeRef.current = true;
    try {
      await ports.inventory.remove(path);
      if (!mountedRef.current) return { stale: true };
      inventoryGenerationRef.current += 1;
      setInventoryLoading(false);
      setSavedMaps((current) => current.filter((file) => file.path !== path));
      ports.onMessage(`Deleted map ${path}`);
      return { removed: true };
    } catch (error) {
      if (mountedRef.current) {
        ports.onMessage(error instanceof Error ? error.message : "Failed to delete map");
      }
      return { removed: false, error };
    } finally {
      removeRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (!active || inventoryRefreshToken === undefined) {
      inventoryGenerationRef.current += 1;
      setInventoryLoading(false);
      return undefined;
    }
    void refreshInventory();
    return () => {
      inventoryGenerationRef.current += 1;
    };
  }, [active, inventoryRefreshToken, refreshInventory]);

  return {
    operation,
    savedMaps,
    inventoryLoading,
    start,
    stop,
    save,
    saveDialog: {
      open: saveDialogOpen,
      name: saveMapName,
      setName: setSaveMapName,
      openDialog: openSaveDialog,
      close: closeSaveDialog,
      confirm: save,
    },
    refreshInventory,
    removeSavedMap,
  };
}
