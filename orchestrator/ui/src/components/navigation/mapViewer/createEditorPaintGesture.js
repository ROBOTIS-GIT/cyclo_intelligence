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
// Author: Howon Kim, Seongwoo Kim

"use client";

export function createEditorPaintGesture({
  onEditorMapPoint,
  mapPointFromEvent,
  editorMoveRafRef,
  editorMovePendingRef
}) {
  const paintEditorPoint = (event, phase = "paint") => {
    if (typeof onEditorMapPoint !== "function") return false;
    const point = mapPointFromEvent(event);
    if (!point) return false;
    event.preventDefault();
    event.stopImmediatePropagation();
    onEditorMapPoint(point.x, point.y, phase);
    return true;
  };
  const flushEditorMove = () => {
    if (editorMoveRafRef.current != null) {
      cancelAnimationFrame(editorMoveRafRef.current);
      editorMoveRafRef.current = null;
    }
    const pending = editorMovePendingRef.current;
    editorMovePendingRef.current = null;
    if (pending) paintEditorPoint(pending, "move");
  };
  const queueEditorMove = event => {
    editorMovePendingRef.current = {
      clientX: event.clientX,
      clientY: event.clientY,
      preventDefault: () => {},
      stopImmediatePropagation: () => {}
    };
    if (editorMoveRafRef.current == null) {
      editorMoveRafRef.current = requestAnimationFrame(() => {
        editorMoveRafRef.current = null;
        const pending = editorMovePendingRef.current;
        editorMovePendingRef.current = null;
        if (pending) paintEditorPoint(pending, "move");
      });
    }
  };
  return {
    paintEditorPoint,
    flushEditorMove,
    queueEditorMove
  };
}
