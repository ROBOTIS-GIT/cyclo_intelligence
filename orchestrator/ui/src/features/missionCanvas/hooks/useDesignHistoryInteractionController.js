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

import { useCallback, useEffect, useRef } from "react";
import { isTextInputTarget } from "../../mapWorkspace/lib/dom";

// Owns only the interaction policy around the shared Design history. Snapshot
// serialization and mutation history remain with their existing controllers.
export default function useDesignHistoryInteractionController({
  active,
  documentReady,
  busy,
  mapBusy,
  taskLayerOpen,
  canUndo,
  canRedo,
  undo,
  redo,
  onMessage,
}) {
  const locked = !active || !documentReady || Boolean(busy) || mapBusy || taskLayerOpen;
  const portsRef = useRef({ locked, canUndo, canRedo, undo, redo, onMessage });
  portsRef.current = { locked, canUndo, canRedo, undo, redo, onMessage };

  const undoAction = useCallback(() => {
    const ports = portsRef.current;
    if (ports.locked || !ports.canUndo) return false;
    ports.undo();
    ports.onMessage("Undid design change");
    return true;
  }, []);

  const redoAction = useCallback(() => {
    const ports = portsRef.current;
    if (ports.locked || !ports.canRedo) return false;
    ports.redo();
    ports.onMessage("Redid design change");
    return true;
  }, []);

  useEffect(() => {
    if (!active || taskLayerOpen) return undefined;
    const handleKeyDown = (event) => {
      if (!(event.ctrlKey || event.metaKey) || isTextInputTarget(event.target)) return;
      const key = event.key.toLowerCase();
      if (key === "z") {
        event.preventDefault();
        if (event.shiftKey) redoAction();
        else undoAction();
      } else if (key === "y" && !event.shiftKey) {
        event.preventDefault();
        redoAction();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [active, redoAction, taskLayerOpen, undoAction]);

  return { locked, undoAction, redoAction };
}
