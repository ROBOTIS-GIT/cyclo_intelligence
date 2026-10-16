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

// One map-owned plan, shared by Record and Operate.
import { useCallback, useEffect, useRef, useState } from "react";
import { autonomyRequest, OBSERVATION_PLAN_CHANGED } from "./autonomyApi";
import { subscribeNavigationMissionChanges } from "../../../utils/navigationMissionsApi";

export default function useObservationPlan({ mapName, enabled = true, editable = false, locked = false }) {
  const [state, setState] = useState({ mapName: null, view: null, draft: null, dirty: false, working: false, error: null });
  const latest = useRef({}), generation = useRef(0), pending = useRef(false);
  latest.current = { mapName, enabled, editable, locked, state };
  const refresh = useCallback(async () => {
    const token = ++generation.current;
    if (!enabled || !mapName) return null;
    try {
      const data = await autonomyRequest(`/autonomy/maps/${encodeURIComponent(mapName)}`);
      if (token !== generation.current) throw new Error("The observation settings changed while refreshing. Try again.");
      setState({ mapName, view: data, draft: data.plan, dirty: false, working: false, error: null });
      return data;
    } catch (error) {
      if (token === generation.current) setState(s => ({ ...s, error: error.message }));
      throw error;
    }
  }, [enabled, mapName]);
  const invalidate = useCallback(() => { ++generation.current; }, []);
  useEffect(() => {
    setState({ mapName, view: null, draft: null, dirty: false, working: false, error: null });
    void refresh().catch(() => {});
    return invalidate;
  }, [mapName, refresh, invalidate]);
  useEffect(() => {
    if (!enabled || !mapName) return undefined;
    const sync = event => {
      if (event?.detail?.mapName && event.detail.mapName !== mapName) return;
      // Do not silently replace an editor draft after another editor saves.
      if (latest.current.state.dirty || pending.current) return;
      void refresh().catch(() => {});
    };
    window.addEventListener(OBSERVATION_PLAN_CHANGED, sync);
    window.addEventListener("focus", sync);
    const unsubscribe = subscribeNavigationMissionChanges?.(sync);
    return () => { window.removeEventListener(OBSERVATION_PLAN_CHANGED, sync); window.removeEventListener("focus", sync); unsubscribe?.(); };
  }, [enabled, mapName, refresh]);
  const edit = changes => {
    if (!editable || locked || pending.current) return;
    ++generation.current;
    setState(s => ({ ...s, draft: { ...s.draft, ...changes }, dirty: true, error: null }));
  };
  const save = async () => {
    const v = latest.current;
    if (!v.editable || v.locked || !v.state.draft || !v.state.dirty || pending.current) return false;
    pending.current = true;
    ++generation.current;
    setState(s => ({ ...s, working: true, error: null }));
    try {
      const data = await autonomyRequest(`/autonomy/maps/${encodeURIComponent(v.mapName)}`, "PUT", v.state.draft);
      if (latest.current.mapName !== v.mapName || !latest.current.enabled) return false;
      setState({ mapName: v.mapName, view: data, draft: data.plan, dirty: false, working: false, error: null });
      window.dispatchEvent(new CustomEvent(OBSERVATION_PLAN_CHANGED, { detail: { mapName: v.mapName } }));
      return true;
    } catch (error) { if (latest.current.mapName === v.mapName) setState(s => ({ ...s, error: error.message })); return false; }
    finally { pending.current = false; setState(s => ({ ...s, working: false })); }
  };
  const current = enabled && state.mapName === mapName;
  return { ...(current ? state.view : null), view: current ? state.view : null,
    draft: current ? state.draft : null, dirty: current && state.dirty, working: current && state.working,
    error: current ? state.error : null, locked: locked || state.working,
    refresh, ...(editable ? { edit, save } : {}) };
}
