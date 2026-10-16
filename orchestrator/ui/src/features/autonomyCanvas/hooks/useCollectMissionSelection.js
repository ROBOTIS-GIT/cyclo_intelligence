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
import { subscribeNavigationMissionChanges } from "../../../utils/navigationMissionsApi";

function missionNames(result) {
  if (Array.isArray(result)) return result;
  return Array.isArray(result?.missions) ? result.missions : [];
}

function renamedSelection(name, change) {
  return change?.mutation === "rename-mission" && change.previousMissionName === name
    ? change.missionName
    : name;
}

function validMissionResult(result) {
  return !(result?.error || result?.stale || result?.exists === false);
}

export default function useCollectMissionSelection({
  enabled,
  mapName,
  listMissions,
  loadMission,
  onMessage,
}) {
  const [state, setState] = useState({ mapName: "", names: [], name: "", loading: false });
  const generation = useRef(0);
  const pending = useRef(false);
  const stateRef = useRef(state);
  const messageRef = useRef(onMessage);
  stateRef.current = state;
  messageRef.current = onMessage;

  const refresh = useCallback(async ({ resetSelection = false, change = null } = {}) => {
    if (!enabled || !mapName) return;
    const token = ++generation.current;
    pending.current = true;
    const previous = stateRef.current.mapName === mapName && !resetSelection
      ? stateRef.current.name
      : "";
    setState((current) => ({
      mapName,
      names: current.mapName === mapName ? current.names : [],
      name: resetSelection || current.mapName !== mapName ? "" : current.name,
      loading: true,
    }));
    try {
      const names = missionNames(await listMissions(mapName));
      let name = renamedSelection(previous, change);
      if (!names.includes(name)) name = "";
      if (name) {
        try {
          const result = await loadMission(mapName, name);
          if (!validMissionResult(result)) {
            throw new Error("Unable to reload the selected mission.");
          }
        } catch (error) {
          if (token === generation.current) {
            setState({ mapName, names, name: "", loading: false });
            messageRef.current(error instanceof Error
              ? error.message
              : "Unable to reload the selected mission.");
          }
          return;
        }
      }
      if (token === generation.current) {
        setState({ mapName, names, name, loading: false });
      }
    } catch (error) {
      if (token === generation.current) {
        setState((current) => ({ ...current, loading: false }));
        messageRef.current(error instanceof Error ? error.message : "Unable to refresh missions.");
      }
    } finally {
      if (token === generation.current) pending.current = false;
    }
  }, [enabled, listMissions, loadMission, mapName]);

  useEffect(() => {
    generation.current += 1;
    pending.current = false;
    setState({ mapName, names: [], name: "", loading: Boolean(enabled && mapName) });
    if (enabled && mapName) void refresh({ resetSelection: true });
    return () => {
      generation.current += 1;
      pending.current = false;
    };
  }, [enabled, mapName, refresh]);

  useEffect(() => {
    if (!enabled || !mapName) return undefined;
    const unsubscribe = typeof subscribeNavigationMissionChanges === "function"
      ? subscribeNavigationMissionChanges((event) => {
        if (event?.detail?.mapName === mapName) void refresh({ change: event.detail });
      })
      : () => {};
    const handleFocus = () => void refresh();
    window.addEventListener("focus", handleFocus);
    return () => {
      unsubscribe();
      window.removeEventListener("focus", handleFocus);
    };
  }, [enabled, mapName, refresh]);

  const current = enabled && state.mapName === mapName;
  const select = async (name) => {
    if (!current || pending.current || state.loading || (name && !state.names.includes(name))) return;
    const token = ++generation.current;
    pending.current = true;
    setState((previous) => ({ ...previous, loading: Boolean(name) }));
    try {
      if (!name) {
        if (token === generation.current) {
          setState((previous) => ({ ...previous, name: "", loading: false }));
        }
        return;
      }
      const result = await loadMission(mapName, name);
      if (!validMissionResult(result)) throw new Error("Unable to load the selected mission.");
      if (token === generation.current) {
        setState((previous) => ({ ...previous, name, loading: false }));
      }
    } catch (error) {
      if (token === generation.current) {
        setState((previous) => ({ ...previous, loading: false }));
        messageRef.current(error instanceof Error ? error.message : "Unable to load the selected mission.");
      }
    } finally {
      if (token === generation.current) pending.current = false;
    }
  };

  return {
    names: current ? state.names : [],
    name: current ? state.name : "",
    loading: current && state.loading,
    select,
    refresh,
  };
}
