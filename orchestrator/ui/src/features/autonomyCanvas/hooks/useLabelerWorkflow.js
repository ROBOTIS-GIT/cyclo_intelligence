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

import { useEffect, useRef, useState } from "react";
import { collectionRequest } from "../../../utils/collectionApi";
import { jobRunning } from "../components/trainingJobStatus";

// Polling and mutation responses share a generation so old status cannot resurrect a cancelled job.
export default function useLabelerWorkflow({ family, onSampled, onJobChange, onBusyChange, onRunningChange }) {
  const [state, setState] = useState(null);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [working, setWorking] = useState(false);
  const [trainingRequest, setTrainingRequest] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const version = useRef(0);
  const pendingRequest = useRef(false);
  const pendingCancel = useRef(false);
  const pendingTraining = useRef(null);
  const jobChange = useRef(onJobChange);
  jobChange.current = onJobChange;

  useEffect(() => {
    let alive = true;
    let timer;
    async function poll() {
      const generation = version.current;
      try {
        const next = await collectionRequest(`${family}/workflow`);
        if (alive && generation === version.current) {
          setState(next);
          setConnectionError("");
          jobChange.current?.(next.batch);
        }
      } catch (err) {
        if (alive && generation === version.current) setConnectionError(err.message);
      }
      if (alive) timer = setTimeout(poll, 3000);
    }
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, [family]);

  const running = [state?.training, state?.batch, state?.reasoning_training].some(jobRunning);
  const labelingRunning = jobRunning(state?.batch);
  useEffect(() => {
    onBusyChange?.(working && !trainingRequest);
    return () => onBusyChange?.(false);
  }, [working, trainingRequest, onBusyChange]);
  useEffect(() => {
    onRunningChange?.(labelingRunning);
    return () => onRunningChange?.(false);
  }, [labelingRunning, onRunningChange]);

  async function mutate(action, training = false) {
    if (pendingRequest.current) return false;
    pendingRequest.current = true;
    version.current += 1;
    setWorking(true);
    setTrainingRequest(training);
    setError("");
    try { await action(); return true; }
    catch (err) { setError(err.message); return false; }
    finally {
      version.current += 1;
      pendingRequest.current = false;
      setWorking(false);
      setTrainingRequest(false);
    }
  }
  async function run(path, body) {
    const training = path === `${family}/auto-train`;
    return mutate(async () => {
      const result = await collectionRequest(path, body);
      version.current += 1;
      if (path.endsWith("/sample")) await onSampled?.(result);
      else {
        if (training) pendingTraining.current = result.id;
        setState(previous => ({ ...previous, [training ? "training" : "batch"]: result }));
      }
      if (family === "segmentation") setState(await collectionRequest(`${family}/workflow`));
    }, training);
  }
  async function cancelTraining() {
    const job = state?.training;
    if (!jobRunning(job) || !job?.id || pendingCancel.current) return;
    pendingCancel.current = true;
    version.current += 1;
    setCancelling(true);
    setError("");
    try {
      const path = family === "qwen" ? "qwen/jobs/training/cancel" : "segmentation/training/cancel";
      const next = await collectionRequest(path, { job_id: job.id });
      version.current += 1;
      setState(next);
    } catch (err) { setError(err.message); }
    finally { pendingCancel.current = false; setCancelling(false); }
  }
  const dismissError = kind => mutate(async () => {
    setState(await collectionRequest(`${family}/jobs/${kind}/reset`, {}));
  });
  const deleteModel = id => mutate(async () => {
    await collectionRequest(`${family}/models/${id}`, undefined, "DELETE");
    setState(await collectionRequest(`${family}/workflow`));
  });
  return { state, error, connectionError, working, running, cancelling, pendingTraining,
    run, cancelTraining, dismissError, deleteModel };
}
