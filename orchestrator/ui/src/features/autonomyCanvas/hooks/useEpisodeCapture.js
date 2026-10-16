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
// Capture survives execution phase changes, but never races an episode mutation.
export default function useEpisodeCapture(session) {
  const { latest, pending, sequence, retryCaptureContext, operation, setEpisode,
    setSelectedFrame, setCaptureReceipt, setError } = session;
  const [recording, setRecording] = useState(false);
  const captureLatest = useRef(null);
  const stopRecording = useCallback(() => {
    sequence.current = null;
    setRecording(false);
  }, [sequence]);
  const saveFrame = (analysisId = null, metadata = null) =>
    operation(async () => {
      const current = latest.current;
      retryCaptureContext.current = false;
      if (!current.enabled || !current.episode || current.episode.ended_at)
        throw new Error("Start an episode before saving a frame");
      setCaptureReceipt("");
      const executionContext = current.getExecutionContext?.();
      if (!analysisId && current.episode.capture_contract === "task_aligned_v1" && !executionContext?.recording_execution) {
        throw new Error("Wait for the selected capture task to be ready before saving a frame.");
      }
      const eid = current.episode.id;
      const data = await collectionRequest(
        `episodes/${eid}/${analysisId ? "save-analysis" : "capture"}`,
        analysisId
          ? {
              analysis_id: analysisId,
            }
          : {
              metadata,
              context: {
                ...current.context,
                ...executionContext,
                preview_active_mission: null,
                map_name: current.mapName || null,
                client_sampled_at_ms: Date.now(),
              },
            },
      );
      if (latest.current.episode?.id !== eid) return;
      latest.current.episode = data.episode;
      setEpisode(data.episode);
      setSelectedFrame({
        ...data.saved_frame,
        labelRevision:
          data.episode.annotations.find((a) => a.frame_ids[0] === data.saved_frame.frame_id)
            ?.revision || 0,
      });
      setCaptureReceipt(
        `${analysisId ? "Analyzed frame and result" : "Camera frame"} saved · ${new Date(data.saved_frame.capture.received_at_ms).toLocaleTimeString()}`,
      );
    });
  captureLatest.current = (metadata) => saveFrame(null, metadata);
  useEffect(
    () => () => {
      sequence.current = null;
    },
    [sequence],
  );
  const startRecording = async (intervalMs, getMetadata, beforeFrame) => {
    if (
      sequence.current ||
      !latest.current.enabled ||
      !latest.current.episode ||
      latest.current.episode.ended_at
    )
      return;
    const token = {};
    sequence.current = token;
    setRecording(true);
    let contextRetrySince = null;
    while (sequence.current === token) {
      // A list refresh or annotation save may briefly own the storage lock.
      // Waiting for it is not a failed capture and must not consume the loop.
      if (pending.current) { await new Promise(resolve => setTimeout(resolve, 25)); continue; }
      if (beforeFrame) {
        let ready;
        try { ready = await beforeFrame(); }
        catch (err) { setError(err.message); break; }
        if (sequence.current !== token || ready === false) break;
        if (ready !== true) { await new Promise(resolve => setTimeout(resolve, 250)); continue; }
      }
      // beforeFrame can await an execution transition while another operation starts.
      if (pending.current) { await new Promise(resolve => setTimeout(resolve, 25)); continue; }
      const current = latest.current;
      if (!current.enabled || !current.episode || current.episode.ended_at) break;
      // The typed episode starts before execution, but training frames only start
      // after the actual runtime enters that task. Paused patrol/mission reports
      // remain valid; approach/reposition and terminal frames are skipped.
      if (current.episode?.capture_contract === "task_aligned_v1" && !current.getExecutionContext?.()?.recording_execution) {
        if (!current.enabled || current.episode.ended_at) break;
        contextRetrySince = null;
        await new Promise(resolve => setTimeout(resolve, 250));
        continue;
      }
      const ok = await captureLatest.current(getMetadata());
      if (!ok && retryCaptureContext.current && sequence.current === token && latest.current.enabled && !latest.current.episode?.ended_at) {
        // A cached camera frame may precede a fresh waypoint/phase report. Skip
        // only this temporal mismatch and wait for the next matching image.
        if (contextRetrySince === null) contextRetrySince = Date.now();
        if (Date.now() - contextRetrySince >= 5000) {
          setError("Camera frames could not be matched to the execution state for 5 seconds. Check camera updates and system clocks, then restart capture.");
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 250));
        continue;
      }
      if (!ok || !latest.current.enabled || latest.current.episode?.ended_at) break;
      contextRetrySince = null;
      if (sequence.current !== token) break;
      await new Promise((resolve) => setTimeout(resolve, Math.max(500, intervalMs)));
    }
    if (sequence.current === token) stopRecording();
  };
  return { recording, stopRecording, startRecording,
    isRecording: () => !!sequence.current,
    capture: metadata => saveFrame(null, metadata), saveAnalysis: id => saveFrame(id) };
}
