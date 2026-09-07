// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useRef, useState } from "react";

export default function useUnsavedDesignActionGuard({
  isDirty,
  clearDirty,
  save,
  documentKey = "",
}) {
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const pendingRef = useRef(null);
  const generationRef = useRef(0);
  const saveGenerationRef = useRef(null);
  const mountedRef = useRef(true);
  const dependenciesRef = useRef({ isDirty, clearDirty, save, documentKey });
  dependenciesRef.current = { isDirty, clearDirty, save, documentKey };

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      generationRef.current += 1;
      pendingRef.current = null;
    };
  }, []);

  const runGuardedAction = useCallback((action) => {
    if (typeof action !== "function") return false;
    if (!dependenciesRef.current.isDirty()) {
      action();
      return true;
    }
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    pendingRef.current = {
      action,
      generation,
      documentKey: dependenciesRef.current.documentKey,
      save: dependenciesRef.current.save,
    };
    setOpen(true);
    return false;
  }, []);

  const cancel = useCallback(() => {
    generationRef.current += 1;
    pendingRef.current = null;
    saveGenerationRef.current = null;
    setSaving(false);
    setOpen(false);
  }, []);

  useEffect(() => {
    const pending = pendingRef.current;
    if (pending && pending.documentKey !== documentKey) cancel();
  }, [cancel, documentKey]);

  const resolve = useCallback(async (mode) => {
    const pending = pendingRef.current;
    if (!pending) {
      setOpen(false);
      return false;
    }
    if (pending.documentKey !== dependenciesRef.current.documentKey) {
      cancel();
      return false;
    }

    if (mode === "discard") {
      generationRef.current += 1;
      pendingRef.current = null;
      saveGenerationRef.current = null;
      setSaving(false);
      setOpen(false);
      dependenciesRef.current.clearDirty();
      pending.action();
      return true;
    }

    if (mode !== "save" || saveGenerationRef.current !== null) return false;
    saveGenerationRef.current = pending.generation;
    setSaving(true);
    try {
      await pending.save();
    } catch {
      return false;
    } finally {
      if (mountedRef.current && saveGenerationRef.current === pending.generation) {
        saveGenerationRef.current = null;
        setSaving(false);
      }
    }

    if (
      !mountedRef.current
      || generationRef.current !== pending.generation
      || pendingRef.current !== pending
      || pending.documentKey !== dependenciesRef.current.documentKey
      || dependenciesRef.current.isDirty()
    ) return false;

    generationRef.current += 1;
    pendingRef.current = null;
    setOpen(false);
    pending.action();
    return true;
  }, [cancel]);

  return {
    open,
    saving,
    runGuardedAction,
    resolve,
    cancel,
  };
}
