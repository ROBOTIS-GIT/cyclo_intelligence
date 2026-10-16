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
import { collectionRequest } from "../../../utils/collectionApi";
const waitingForCaptureContext = error => error?.status === 422
  && error.message.startsWith("Capture-time execution context could not be matched.");

// Shared mutation authority: every collection action uses this lock and document generation.
export default function useCollectionSession({ enabled, mapName, context, getExecutionContext }) {
  const [episode, setEpisode] = useState(null);
  const [reasoningContextVersion, setReasoningContextVersion] = useState(0);
  const [episodes, setEpisodes] = useState([]);
  const [reasonPresets, setReasonPresets] = useState([]);
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [captureReceipt, setCaptureReceipt] = useState("");
  const sequence = useRef(null);
  const retryCaptureContext = useRef(false);
  const pending = useRef(false);
  const datasetGeneration = useRef(0);
  const latest = useRef({ episode, enabled, mapName, context, getExecutionContext });
  latest.current = { episode, enabled, mapName, context, getExecutionContext };
  const active = !!episode && !episode.ended_at;
  const presetMapName = episode?.map_name || mapName;
  const refreshList = useCallback(async () => {
    const generation = datasetGeneration.current;
    const data = await collectionRequest("episodes");
    if (generation === datasetGeneration.current) setEpisodes(data.episodes || []);
  }, []);
  const refreshReasonPresets = useCallback(async () => {
    const generation = datasetGeneration.current;
    const requestedMap = latest.current.episode?.map_name || latest.current.mapName;
    if (!requestedMap) {
      setReasonPresets([]);
      return [];
    }
    const data = await collectionRequest(
      `reason-presets?map_name=${encodeURIComponent(requestedMap)}`,
    );
    if (generation === datasetGeneration.current && (latest.current.episode?.map_name || latest.current.mapName) === requestedMap) {
      setReasonPresets(data.presets || []);
    }
    return data.presets || [];
  }, []);
  const refresh = useCallback(async () => {
    const id = latest.current.episode?.id;
    if (!id || pending.current) return;
    const generation = datasetGeneration.current;
    const data = await collectionRequest(`episodes/${id}`);
    if (generation === datasetGeneration.current && latest.current.episode?.id === id && !pending.current &&
        !(latest.current.episode.ended_at && !data.ended_at)) {
      latest.current.episode = data;
      setEpisode(data);
    }
    return data;
  }, []);
  const operation = useCallback(async (fn) => {
    if (pending.current) return false;
    pending.current = true;
    setBusy(true);
    setError("");
    try {
      await fn();
      return true;
    } catch (err) {
      retryCaptureContext.current = waitingForCaptureContext(err);
      if (!(sequence.current && retryCaptureContext.current)) setError(err.message);
      return false;
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }, []);
  useEffect(() => {
    if (enabled) Promise.all([refreshList(), refresh()]).catch((err) => setError(err.message));
  }, [enabled, refreshList, refresh]);
  useEffect(() => {
    if (enabled) refreshReasonPresets().catch((err) => setError(err.message));
    else setReasonPresets([]);
  }, [enabled, presetMapName, refreshReasonPresets]);
  useEffect(() => {
    if (!active || !enabled) return undefined;
    const timer = setInterval(() => {
      refresh().catch((err) => setError(err.message));
    }, 2000);
    return () => clearInterval(timer);
  }, [active, enabled, refresh]);
  return { episode, episodes, reasonPresets, selectedFrame, error, busy, captureReceipt,
    reasoningContextVersion, active, latest, pending, datasetGeneration, sequence,
    retryCaptureContext, setEpisode, setSelectedFrame, setCaptureReceipt, setError,
    setReasoningContextVersion, operation, refresh, refreshList, refreshReasonPresets };
}
