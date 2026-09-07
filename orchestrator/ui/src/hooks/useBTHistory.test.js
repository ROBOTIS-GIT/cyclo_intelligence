// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { act, renderHook } from "@testing-library/react";
import { useBTHistory } from "./useBTHistory";

function renderHistory(initialSnapshot = "A") {
  const document = { current: initialSnapshot };
  const applySnapshot = jest.fn((snapshot) => {
    document.current = snapshot;
  });
  const view = renderHook(() => useBTHistory({
    getSnapshot: () => document.current,
    applySnapshot,
  }));
  return { applySnapshot, document, view };
}

test("capture stores pre-mutation snapshots for undo and redo", () => {
  const { applySnapshot, document, view } = renderHistory();

  act(() => {
    view.result.current.capture();
    document.current = "B";
    view.result.current.capture();
    document.current = "C";
  });

  expect(view.result.current.canUndo).toBe(true);
  expect(view.result.current.canRedo).toBe(false);

  act(() => {
    view.result.current.undo();
  });
  expect(document.current).toBe("B");
  expect(view.result.current.canUndo).toBe(true);
  expect(view.result.current.canRedo).toBe(true);

  act(() => {
    view.result.current.undo();
  });
  expect(document.current).toBe("A");
  expect(view.result.current.canUndo).toBe(false);
  expect(view.result.current.canRedo).toBe(true);

  act(() => {
    view.result.current.redo();
  });
  expect(document.current).toBe("B");

  act(() => {
    view.result.current.redo();
  });
  expect(document.current).toBe("C");
  expect(view.result.current.canUndo).toBe(true);
  expect(view.result.current.canRedo).toBe(false);
  expect(applySnapshot.mock.calls.map(([snapshot]) => snapshot)).toEqual([
    "B",
    "A",
    "B",
    "C",
  ]);
});

test("capture deduplicates an unchanged snapshot and clears redo history", () => {
  const { document, view } = renderHistory();

  act(() => {
    view.result.current.capture();
    view.result.current.capture();
    document.current = "B";
    view.result.current.undo();
  });

  expect(document.current).toBe("A");
  expect(view.result.current.canUndo).toBe(false);
  expect(view.result.current.canRedo).toBe(true);

  act(() => {
    view.result.current.capture();
    document.current = "C";
  });

  expect(view.result.current.canUndo).toBe(true);
  expect(view.result.current.canRedo).toBe(false);
  act(() => {
    view.result.current.undo();
  });
  expect(document.current).toBe("A");
});

test("rebase replaces history without applying the durable snapshot", () => {
  const { applySnapshot, document, view } = renderHistory("legacy-A");

  act(() => {
    view.result.current.capture();
    document.current = "legacy-B";
    view.result.current.capture();
    document.current = "pre-rebase-live";
    view.result.current.undo();
  });
  expect(document.current).toBe("legacy-B");
  expect(view.result.current.canUndo).toBe(true);
  expect(view.result.current.canRedo).toBe(true);

  // Model an edit that landed while an async save was completing. Rebase must
  // change only the history boundary and leave this visible document intact.
  document.current = "newer-live";
  const applyCountBeforeRebase = applySnapshot.mock.calls.length;
  act(() => {
    view.result.current.rebase("durable-saved");
  });

  expect(document.current).toBe("newer-live");
  expect(applySnapshot).toHaveBeenCalledTimes(applyCountBeforeRebase);
  expect(view.result.current.canUndo).toBe(true);
  expect(view.result.current.canRedo).toBe(false);

  act(() => {
    view.result.current.undo();
  });
  expect(document.current).toBe("durable-saved");
  expect(view.result.current.canUndo).toBe(false);
  expect(view.result.current.canRedo).toBe(true);

  act(() => {
    view.result.current.redo();
  });
  expect(document.current).toBe("newer-live");
  expect(view.result.current.canUndo).toBe(true);
  expect(view.result.current.canRedo).toBe(false);
});
