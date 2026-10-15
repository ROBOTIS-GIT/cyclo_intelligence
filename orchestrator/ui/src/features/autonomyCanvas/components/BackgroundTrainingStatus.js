// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import { useEffect, useState } from "react";
import { collectionRequest } from "../../../utils/collectionApi";

const ACTIVE = new Set(["queued", "training", "running"]);
const POLL_MS = 5000;

export default function BackgroundTrainingStatus() {
  const [jobs, setJobs] = useState([]);
  useEffect(() => {
    let alive = true;
    let timer;
    async function poll() {
      const results = await Promise.allSettled([
        collectionRequest("qwen/workflow"),
        collectionRequest("segmentation/workflow"),
      ]);
      if (!alive) return;
      const qwen = results[0].status === "fulfilled" ? results[0].value : {};
      const sam = results[1].status === "fulfilled" ? results[1].value : {};
      setJobs([
        ["Qwen", qwen.reasoning_training],
        ["Qwen", qwen.training],
        ["SAM", sam.training],
      ].filter(([, job]) => ACTIVE.has(job?.status)));
      timer = setTimeout(poll, POLL_MS);
    }
    poll();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  if (!jobs.length) return null;
  return <div className="mt-5 rounded-lg border p-3 text-xs" role="status" aria-label="Background training"
    style={{ borderColor: "var(--mc-border-strong)", background: "var(--mc-surface)", color: "var(--mc-text-muted)" }}>
    <strong className="block mb-2" style={{ color: "var(--mc-text)" }}>Training in background</strong>
    {jobs.map(([kind, job]) => <div key={`${kind}-${job.id || job.model_name || "training"}`} className="mb-2 last:mb-0">
      <span className="block truncate" title={job.model_name || kind}>{kind} · {job.model_name || "Model"}</span>
      {job.steps > 0 && <progress className="w-full" aria-label={`${kind} training progress`}
        max={job.steps} value={Math.min(job.steps, job.step || 0)} />}
      <span>{job.status}{job.steps > 0 ? ` · ${job.step || 0}/${job.steps}` : ""}</span>
    </div>)}
    <span>Other pages remain available.</span>
  </div>;
}
