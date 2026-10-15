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
