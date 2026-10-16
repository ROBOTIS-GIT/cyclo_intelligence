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

// Keep reversible editor commands separate from navigation and capture state.
export default function useRecordEditHistory(scope) {
  const stacks = useRef({ past: [], future: [] });
  const pending = useRef(false), epoch = useRef(0);
  const [, render] = useState(0);
  const refresh = () => render(value => value + 1);
  const clear = useCallback(() => {
    epoch.current += 1;
    stacks.current = { past: [], future: [] };
    render(value => value + 1);
  }, []);
  useEffect(clear, [scope, clear]);
  const push = entry => {
    if (pending.current) return;
    stacks.current = { past: [...stacks.current.past, entry].slice(-50), future: [] };
    refresh();
  };
  const travel = async (direction, apply) => {
    if (pending.current) return false;
    const source = direction === "undo" ? "past" : "future";
    const target = direction === "undo" ? "future" : "past";
    const entry = stacks.current[source].at(-1);
    if (!entry) return false;
    const token = epoch.current;
    pending.current = true; refresh();
    try {
      if (await apply(entry, direction) === false || token !== epoch.current) return false;
      stacks.current[source].pop();
      stacks.current[target].push(entry);
      return true;
    } finally { pending.current = false; refresh(); }
  };
  return { push, clear, travel, busy: pending.current,
    undoEntry: stacks.current.past.at(-1), redoEntry: stacks.current.future.at(-1),
    canUndo: stacks.current.past.length > 0, canRedo: stacks.current.future.length > 0 };
}
