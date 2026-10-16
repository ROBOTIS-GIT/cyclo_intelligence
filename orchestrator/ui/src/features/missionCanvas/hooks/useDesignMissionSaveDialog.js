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

import { useCallback, useRef, useState } from "react";
import { isValidMissionName } from "../../mapWorkspace/lib/missionNames";

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
