// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useRef } from "react";
import { isTextInputTarget } from "../lib/dom";

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
