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

import readApiResponse from "../../../utils/readApiResponse";
import { useCallback, useEffect, useRef, useState } from "react";

const storageKey = mapName => `autonomy.operate.model.${mapName}`;

export default function useOperateModel(mapName) {
  const [state, setState] = useState({ mapName: null, models: [], selectedId: "", loading: false, error: null });
  const requestId = useRef(0);
  const controller = useRef(null);
  const refresh = useCallback(async () => {
    controller.current?.abort();
    const id = ++requestId.current;
    const abort = new AbortController();
    controller.current = abort;
    if (!mapName) { setState({ mapName, models: [], selectedId: "", loading: false, error: null }); return; }
    setState(previous => ({ ...previous, loading: true, error: null }));
    try {
      const response = await fetch(`/api/reasoning/models?map_name=${encodeURIComponent(mapName)}`, { signal: abort.signal });
      const data = await readApiResponse(response, "Could not load reasoning models");
      if (id !== requestId.current) return;
      const models = data.models || [];
      let saved = "";
      try { saved = localStorage.getItem(storageKey(mapName)) || ""; } catch (_) { /* Storage is optional. */ }
      setState(previous => {
        const candidate = previous.mapName === mapName ? previous.selectedId : saved;
        return { mapName, models, selectedId: models.some(m => m.id === candidate) ? candidate : "", loading: false, error: null };
      });
    } catch (error) {
      if (id === requestId.current && !abort.signal.aborted) setState(previous => ({ ...previous, loading: false, error: error.message }));
    }
  }, [mapName]);
  const cancel = useCallback(() => { ++requestId.current; controller.current?.abort(); }, []);
  useEffect(() => {
    void refresh();
    return cancel;
  }, [refresh, cancel]);
  const select = useCallback(id => {
    setState(previous => {
      if (previous.mapName !== mapName || !previous.models.some(m => m.id === id)) return previous;
      return { ...previous, selectedId: id };
    });
    try { localStorage.setItem(storageKey(mapName), id); } catch (_) { /* Storage is optional. */ }
  }, [mapName]);
  const models = state.mapName === mapName ? state.models : [];
  return { models, selected: models.find(m => m.id === state.selectedId) || null,
    loading: state.loading, error: state.error, refresh, select };
}
