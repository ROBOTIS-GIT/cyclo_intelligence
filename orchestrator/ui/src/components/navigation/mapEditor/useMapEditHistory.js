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

import { useCallback } from 'react';
export function useMapEditHistory({
  pixelsRef,
  setUndoStack,
  setRedoStack,
  setPixels,
  setDirty,
  onMessage,
  annotationsRef,
  applyAnnotationsSnapshot
}) {
  const undo = useCallback(() => {
    const currentPixels = pixelsRef.current;
    setUndoStack(stack => {
      const previous = stack[stack.length - 1];
      if (!previous) return stack;
      if (previous.type === "pixels") {
        if (!currentPixels) return stack;
        setRedoStack(redo => [...redo, {
          type: "pixels",
          pixels: currentPixels
        }]);
        pixelsRef.current = previous.pixels;
        setPixels(previous.pixels);
        setDirty(stack.slice(0, -1).some(item => item.type === "pixels"));
        onMessage("Undid last map edit");
      } else if (previous.type === "annotations") {
        const currentAnnotations = annotationsRef.current;
        setRedoStack(redo => [...redo, {
          type: "annotations",
          annotations: currentAnnotations
        }]);
        applyAnnotationsSnapshot(previous.annotations, "Undid last area edit");
      }
      return stack.slice(0, -1);
    });
  }, [annotationsRef, applyAnnotationsSnapshot, onMessage, pixelsRef, setDirty, setPixels, setRedoStack, setUndoStack]);
  const redo = useCallback(() => {
    const currentPixels = pixelsRef.current;
    setRedoStack(stack => {
      const next = stack[stack.length - 1];
      if (!next) return stack;
      if (next.type === "pixels") {
        if (!currentPixels) return stack;
        setUndoStack(undoHistory => [...undoHistory, {
          type: "pixels",
          pixels: currentPixels
        }]);
        pixelsRef.current = next.pixels;
        setPixels(next.pixels);
        setDirty(true);
        onMessage("Redid last map edit");
      } else if (next.type === "annotations") {
        const currentAnnotations = annotationsRef.current;
        setUndoStack(undoHistory => [...undoHistory, {
          type: "annotations",
          annotations: currentAnnotations
        }]);
        applyAnnotationsSnapshot(next.annotations, "Redid last area edit");
      }
      return stack.slice(0, -1);
    });
  }, [annotationsRef, applyAnnotationsSnapshot, onMessage, pixelsRef, setDirty, setPixels, setRedoStack, setUndoStack]);
  return {
    undo,
    redo
  };
}
