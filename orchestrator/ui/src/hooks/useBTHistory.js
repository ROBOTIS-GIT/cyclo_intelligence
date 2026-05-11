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

import { useCallback, useRef, useState } from 'react';

/**
 * Snapshot-based undo/redo. The caller provides:
 *   getSnapshot   () => string | null   — serialize current state
 *   applySnapshot (xml: string) => void — restore state from a snapshot
 *
 * Usage: call `capture()` BEFORE every mutation. Each capture pushes the
 * current (pre-mutation) snapshot onto the past stack and clears the redo
 * stack.
 */
export function useBTHistory({ getSnapshot, applySnapshot, capacity = 50 }) {
  // Hold callbacks in refs so the returned actions stay stable across renders
  // even though getSnapshot/applySnapshot close over fresh state each render.
  const getSnapshotRef = useRef(getSnapshot);
  const applySnapshotRef = useRef(applySnapshot);
  getSnapshotRef.current = getSnapshot;
  applySnapshotRef.current = applySnapshot;

  const pastRef = useRef([]);
  const futureRef = useRef([]);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);

  const sync = useCallback(() => {
    setCanUndo(pastRef.current.length > 0);
    setCanRedo(futureRef.current.length > 0);
  }, []);

  const capture = useCallback(() => {
    const snap = getSnapshotRef.current();
    if (snap == null) return;
    const past = pastRef.current;
    if (past.length > 0 && past[past.length - 1] === snap) return;
    past.push(snap);
    if (past.length > capacity) past.shift();
    futureRef.current = [];
    sync();
  }, [capacity, sync]);

  const undo = useCallback(() => {
    if (pastRef.current.length === 0) return;
    const prev = pastRef.current.pop();
    const current = getSnapshotRef.current();
    if (current != null) futureRef.current.push(current);
    applySnapshotRef.current(prev);
    sync();
  }, [sync]);

  const redo = useCallback(() => {
    if (futureRef.current.length === 0) return;
    const next = futureRef.current.pop();
    const current = getSnapshotRef.current();
    if (current != null) pastRef.current.push(current);
    applySnapshotRef.current(next);
    sync();
  }, [sync]);

  const reset = useCallback(() => {
    pastRef.current = [];
    futureRef.current = [];
    sync();
  }, [sync]);

  return { capture, undo, redo, reset, canUndo, canRedo };
}
