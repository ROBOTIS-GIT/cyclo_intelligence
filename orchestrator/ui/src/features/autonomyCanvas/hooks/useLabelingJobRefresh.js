import { useCallback, useRef, useState } from "react";

// Refresh a completed batch once, in storage-lock order, without discarding unsaved edits.
export default function useLabelingJobRefresh(draft, dirty) {
  const [version, setVersion] = useState(0);
  const lastCompleted = useRef(null);
  const latest = useRef({ draft, dirty });
  latest.current = { draft, dirty };
  const onJobChange = useCallback(job => {
    if (!job?.id || !["completed", "completed_with_errors", "failed"].includes(job.status) ||
        lastCompleted.current === job.id || latest.current.dirty) return;
    lastCompleted.current = job.id;
    (async () => {
      await latest.current.draft.refresh?.();
      await latest.current.draft.refreshEpisodes?.();
    })().then(() => setVersion(value => value + 1)).catch(() => { lastCompleted.current = null; });
  }, []);
  return { version, onJobChange };
}
