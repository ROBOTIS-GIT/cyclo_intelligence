// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useState } from "react";
import { isTextInputTarget } from "../lib/dom";
import { mapNameFromPgmPath } from "../lib/missionNames";

// Owns the stage-level UI glue around useMapEditor. The editor remains the
// source of truth for map pixels, annotations, history, files and loading.
export default function useMapEditStageCoordinator({
  active,
  currentMapName,
  currentMissionName,
  editor,
  onSelectedMapIdentity,
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pendingPath, setPendingPath] = useState("");
  const {
    files,
    selectedPath,
    setSelectedPath,
    busy,
    undo,
    redo,
  } = editor;

  const openPicker = useCallback(() => {
    setPendingPath(selectedPath || files[0]?.path || "");
    setPickerOpen(true);
  }, [files, selectedPath]);

  const cancelPicker = useCallback(() => {
    setPickerOpen(false);
  }, []);

  const confirmPicker = useCallback(() => {
    setSelectedPath(pendingPath);
    setPickerOpen(false);
  }, [pendingPath, setSelectedPath]);

  useEffect(() => {
    if (!active || !selectedPath) return;
    const selectedMapName = mapNameFromPgmPath(selectedPath);
    if (selectedMapName && selectedMapName !== currentMapName) {
      onSelectedMapIdentity({
        mapName: selectedMapName,
        missionName: currentMissionName,
      });
    }
  }, [
    active,
    currentMapName,
    currentMissionName,
    onSelectedMapIdentity,
    selectedPath,
  ]);

  useEffect(() => {
    if (!active || pickerOpen) return undefined;
    const handleKeyDown = (event) => {
      if (!(event.ctrlKey || event.metaKey) || isTextInputTarget(event.target)) return;
      if (busy) return;
      const key = event.key.toLowerCase();
      if (key === "z") {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
      } else if (key === "y" && !event.shiftKey) {
        event.preventDefault();
        redo();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [active, busy, pickerOpen, redo, undo]);

  return {
    picker: {
      open: pickerOpen,
      pendingPath,
      setPendingPath,
      openPicker,
      cancelPicker,
      confirmPicker,
    },
  };
}
