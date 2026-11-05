// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef, useState } from "react";
import { isValidMissionName } from "../lib/missionNames";

export function confirmDesignMissionSave({ name, close, saveMission }) {
  const target = String(name || "").trim();
  if (!isValidMissionName(target)) return false;
  close();
  void saveMission(target);
  return true;
}

export default function useDesignMissionSaveDialog({
  missionName,
  existingNames,
  saveMission,
}) {
  const [open, setOpen] = useState(false);
  const [name, setNameState] = useState("");
  const nameRef = useRef(name);
  nameRef.current = name;
  const portsRef = useRef({ missionName, existingNames, saveMission });
  portsRef.current = { missionName, existingNames, saveMission };

  const setName = useCallback((nextName) => {
    const resolvedName = typeof nextName === "function"
      ? nextName(nameRef.current)
      : nextName;
    nameRef.current = resolvedName;
    setNameState(resolvedName);
  }, []);

  const requestSave = useCallback(() => {
    const ports = portsRef.current;
    const catalogNames = Array.isArray(ports.existingNames) ? ports.existingNames : [];
    if (catalogNames.includes(ports.missionName)) {
      void ports.saveMission(ports.missionName);
      return;
    }
    setName(ports.missionName);
    setOpen(true);
  }, [setName]);

  const cancel = useCallback(() => {
    setOpen(false);
  }, []);

  const confirm = useCallback(() => {
    confirmDesignMissionSave({
      name: nameRef.current,
      close: () => setOpen(false),
      saveMission: portsRef.current.saveMission,
    });
  }, []);

  return {
    open,
    name,
    setName,
    requestSave,
    cancel,
    confirm,
  };
}
