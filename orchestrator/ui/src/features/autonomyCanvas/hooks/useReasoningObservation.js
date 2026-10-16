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
import useQwenObservation from "./useQwenObservation";
import useOperateModel from "./useOperateModel";

// Live preview only: never write to a saved episode or dispatch mission actions.
export default function useReasoningObservation({ enabled, loadedMapName, episode, pose, localized, contextVersion, requestedModel, getExecutionContext }) {
  const mapName = loadedMapName || requestedModel?.map_name || episode?.map_name || null;
  const choice = useOperateModel(mapName);
  const { refresh: refreshModels, select: selectModel, models, loading: modelsLoading, error: modelsError } = choice;
  const [startedContext, setStartedContext] = useState(null);
  const [handoff, setHandoff] = useState(null);
  const handledRequest = useRef(null);
  const pendingRequest = Boolean(requestedModel?.id &&
    (handledRequest.current?.request !== requestedModel || handledRequest.current?.mapName !== mapName));
  const selectionError = handoff?.error ||
    (choice.selected && choice.selected.map_name !== mapName ? "The selected model belongs to a different map." : null) ||
    (startedContext && (startedContext.mapName !== mapName || startedContext.modelId !== choice.selected?.id)
      ? "Map or model changed. Stop reasoning before starting a new preview." : null);
  const available = Boolean(enabled && mapName && choice.selected && !choice.loading && !choice.error && !pendingRequest && !selectionError);
  const observation = useQwenObservation(available, {
    episode_id: null,
    context: { map_name: mapName, pose: loadedMapName ? pose : null,
      localized: loadedMapName ? localized : null, preview_active_mission: null },
  }, {
    startPayload: { model_id: choice.selected?.id, map_name: mapName },
    getContext: getExecutionContext,
    contextKey: JSON.stringify([mapName, choice.selected?.id, episode?.id, contextVersion]),
  });
  const { active, phase, start: startObservation, stop: stopObservation } = observation;
  const locked = active || observation.needsStop;

  // Dataset handoff only selects a refreshed model. Loading the model still
  // requires the operator to press Start reasoning in Live.
  useEffect(() => {
    if (!requestedModel?.id) { handledRequest.current = null; setHandoff(null); return undefined; }
    if (locked || (handledRequest.current?.request === requestedModel && handledRequest.current?.mapName === mapName)) return undefined;
    if (requestedModel.map_name !== mapName) {
      handledRequest.current = { request: requestedModel, mapName };
      setHandoff({ request: requestedModel, mapName, error: `This model belongs to ${requestedModel.map_name || "another map"}. Load that map or select a model for ${mapName}.` });
      return undefined;
    }
    let cancelled = false;
    setHandoff({ request: requestedModel, mapName, status: "loading" });
    void refreshModels().then(() => {
      if (!cancelled) setHandoff({ request: requestedModel, mapName, status: "ready" });
    });
    return () => { cancelled = true; };
  }, [requestedModel, mapName, locked, refreshModels]);

  useEffect(() => {
    if (locked || !handoff || handoff.status !== "ready" || modelsLoading ||
      handoff.request !== requestedModel || handoff.mapName !== mapName) return;
    const model = models.find(item => item.id === requestedModel.id && item.map_name === mapName);
    handledRequest.current = { request: requestedModel, mapName };
    if (model && !modelsError) {
      selectModel(model.id);
      setHandoff({ request: requestedModel, mapName, status: "selected" });
    } else {
      setHandoff({ request: requestedModel, mapName,
        error: modelsError || "This model is not available. Refresh models or choose another model." });
    }
  }, [handoff, locked, mapName, requestedModel, modelsLoading, models, modelsError, selectModel]);

  const start = useCallback(() => {
    if (!available || active || phase === "stopping") return;
    setStartedContext({ mapName, modelId: choice.selected.id });
    return startObservation();
  }, [available, mapName, choice.selected, active, phase, startObservation]);
  const stop = useCallback(async () => {
    const stopped = await stopObservation();
    if (stopped) setStartedContext(null);
    return stopped;
  }, [stopObservation]);
  const select = id => {
    if (locked || choice.loading || !choice.models.some(model => model.id === id && model.map_name === mapName)) return;
    handledRequest.current = { request: requestedModel, mapName };
    setHandoff(null);
    choice.select(id);
  };
  const refresh = () => {
    if (locked) return;
    if (handoff?.error && requestedModel?.map_name === mapName) {
      handledRequest.current = null;
      setHandoff({ request: requestedModel, mapName, status: "loading" });
      return choice.refresh().then(() => setHandoff({ request: requestedModel, mapName, status: "ready" }));
    }
    return choice.refresh();
  };
  return { ...observation, start, stop, available, mapName, selectionError,
    modelChoice: { ...choice, locked, select, refresh, pending: pendingRequest,
      error: selectionError || choice.error || (pendingRequest && locked ? "Stop reasoning to select the requested model." : null) },
  };
}
