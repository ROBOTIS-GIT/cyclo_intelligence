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
import { savePgmImage, saveMapAnnotations } from '../../../utils/navigationApi';
import { encodePgmPixels } from './pgmPixels';
import { normalizeAnnotation } from './annotationModel';
export function useMapDocumentSave({
  image,
  pixels,
  busy,
  dirty,
  annotationsDirty,
  setBusy,
  setDirty,
  annotationsRef,
  setAnnotations,
  setAnnotationsDirty,
  setUndoStack,
  setRedoStack,
  onMessage
}) {
  const save = useCallback(() => {
    if (!image || !pixels || busy || (!dirty && !annotationsDirty)) return;
    setBusy(true);
    Promise.resolve().then(async () => {
      const savedParts = [];
      let savedPath = image.path;
      if (dirty) {
        const response = await savePgmImage(image.path, image.width, image.height, image.maxval, encodePgmPixels(pixels));
        savedPath = ((response === null) || (response === void 0) ? void 0 : response.path) || savedPath;
        setDirty(false);
        savedParts.push("map");
      }
      if (annotationsDirty) {
        const response = await saveMapAnnotations(image.path, annotationsRef.current);
        const saved = Array.isArray((response === null) || (response === void 0) ? void 0 : response.annotations) ? response.annotations.map(normalizeAnnotation) : annotationsRef.current;
        annotationsRef.current = saved;
        setAnnotations(saved);
        setAnnotationsDirty(false);
        savedParts.push("areas");
      }
      setUndoStack([]);
      setRedoStack([]);
      onMessage(savedParts.length > 1 ? `Saved map and areas ${savedPath}` : `Saved ${savedParts[0] === "areas" ? "map areas" : savedPath}`);
    }).catch(error => {
      onMessage(error instanceof Error ? error.message : "Failed to save map");
    }).finally(() => setBusy(false));
  }, [annotationsDirty, annotationsRef, busy, dirty, image, onMessage, pixels, setAnnotations, setAnnotationsDirty, setBusy, setDirty, setRedoStack, setUndoStack]);
  return {
    save
  };
}
