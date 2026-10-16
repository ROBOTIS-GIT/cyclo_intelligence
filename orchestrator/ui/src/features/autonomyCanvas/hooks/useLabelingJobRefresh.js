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
