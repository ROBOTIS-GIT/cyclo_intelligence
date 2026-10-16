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

import { useCallback, useEffect, useState } from "react";
import { isTextInputTarget } from "../../mapWorkspace/lib/dom";
import { mapNameFromPgmPath } from "../../mapWorkspace/lib/missionNames";

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
