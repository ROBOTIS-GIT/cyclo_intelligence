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
import { coveredAnnotationCellSet, visibleAnnotationCellSnapshots, brushCellsForSegment } from './annotationCoverage';
import { cellKey, regionFromCells } from './annotationGeometry';
import { ANNOTATION_EXTEND_TOOL, ANNOTATION_ERASE_TOOL } from './constants';
import { mapPointToPgmPixel } from './pgmCoordinates';
export function useMapAreaPainting({
  pixelsRef,
  open,
  image,
  busy,
  annotationsRef,
  selectedAnnotationId,
  onMessage,
  areaPaintRef,
  brushSize,
  recordAnnotationUndo,
  applyAnnotationsSnapshot,
  tool,
  areaEraseRef
}) {
  const extendSelectedArea = useCallback((pixel, phase) => {
    const currentPixels = pixelsRef.current;
    if (!open || !image || busy || !currentPixels) return;
    const target = annotationsRef.current.find(annotation => annotation.id === selectedAnnotationId);
    if (!target) {
      onMessage("Select an area to extend");
      return;
    }
    let session = areaPaintRef.current;
    if ((phase === "start") || !session || (session.id !== target.id)) {
      // Heavy prep runs ONCE per stroke: the covered-by-others set and the
      // target's visible cells (not raw bounds, so extending a rect-created
      // area can never resurrect cells hidden under earlier areas).
      const covered = coveredAnnotationCellSet(annotationsRef.current, image, currentPixels, target.id);
      const targetSnapshot = visibleAnnotationCellSnapshots(annotationsRef.current, image, currentPixels).find(snapshot => snapshot.annotation.id === target.id);
      const cells = new Map();
      (((targetSnapshot === null) || (targetSnapshot === void 0) ? void 0 : targetSnapshot.cells) || []).forEach(cell => cells.set(cellKey(cell), cell));
      session = {
        id: target.id,
        lastPixel: pixel,
        undoPushed: false,
        covered,
        cells
      };
      areaPaintRef.current = session;
    }
    const startPixel = (phase === "move") && session.lastPixel ? session.lastPixel : pixel;
    const newCells = brushCellsForSegment(currentPixels, image, startPixel, pixel, brushSize, session.covered);
    session.lastPixel = pixel;
    if (!newCells.length) {
      onMessage("Paint over unmarked white free-space");
      return;
    }
    let added = 0;
    newCells.forEach(cell => {
      const key = cellKey(cell);
      if (session.cells.has(key)) return;
      session.cells.set(key, cell);
      added += 1;
    });
    if (!added) return;
    const geometry = regionFromCells(Array.from(session.cells.values()), image);
    if (!geometry) return;
    recordAnnotationUndo(session);
    const nextAnnotation = {
      ...target,
      pose: geometry.pose,
      region: geometry.region
    };
    applyAnnotationsSnapshot(annotationsRef.current.map(annotation => annotation.id === target.id ? nextAnnotation : annotation), `Extended area ${nextAnnotation.label}`);
  }, [annotationsRef, applyAnnotationsSnapshot, areaPaintRef, brushSize, busy, image, onMessage, open, pixelsRef, recordAnnotationUndo, selectedAnnotationId]);
  const applyAreaCellErase = useCallback((startPixel, endPixel, session) => {
    const currentPixels = pixelsRef.current;
    if (!open || !image || busy || !currentPixels) return false;
    // Snapshot every annotation's visible cells ONCE per stroke; moves then
    // only delete brush keys from these Maps instead of rebuilding the world.
    if (!session.entries) {
      session.entries = visibleAnnotationCellSnapshots(annotationsRef.current, image, currentPixels).map(({
        annotation,
        cells
      }) => ({
        annotation,
        cells: new Map(cells.map(cell => [cellKey(cell), cell])),
        changed: false
      }));
    }
    const eraseCells = brushCellsForSegment(currentPixels, image, startPixel, endPixel, brushSize);
    if (!eraseCells.length) return false;
    let changed = false;
    eraseCells.forEach(cell => {
      const key = cellKey(cell);
      session.entries.forEach(entry => {
        if (entry.cells.delete(key)) {
          entry.changed = true;
          changed = true;
        }
      });
    });
    if (!changed) return false;
    recordAnnotationUndo(session);
    session.entries = session.entries.filter(entry => entry.cells.size > 0);
    const nextAnnotations = session.entries.map(entry => {
      if (!entry.changed) return entry.annotation;
      const geometry = regionFromCells(Array.from(entry.cells.values()), image);
      if (!geometry) return entry.annotation;
      const nextAnnotation = {
        ...entry.annotation,
        pose: geometry.pose,
        region: geometry.region
      };
      entry.annotation = nextAnnotation;
      entry.changed = false;
      return nextAnnotation;
    });
    applyAnnotationsSnapshot(nextAnnotations, "Erased area pixels");
    return true;
  }, [annotationsRef, applyAnnotationsSnapshot, brushSize, busy, image, open, pixelsRef, recordAnnotationUndo]);
  const editAreaAtMapPoint = useCallback((x, y, phase = "paint") => {
    if (!open || !image || busy || (tool !== ANNOTATION_EXTEND_TOOL.id && tool !== ANNOTATION_ERASE_TOOL.id)) return;
    if (phase === "end") {
      areaPaintRef.current = null;
      areaEraseRef.current = null;
      return;
    }
    const pixel = mapPointToPgmPixel(image, x, y);
    if (!pixel) return;
    if (tool === ANNOTATION_EXTEND_TOOL.id) {
      extendSelectedArea(pixel, phase);
      return;
    }
    let session = areaEraseRef.current;
    if ((phase === "start") || !session) {
      session = {
        lastPixel: pixel,
        undoPushed: false
      };
      areaEraseRef.current = session;
      // Erase under the cursor immediately so a plain press has an effect.
      applyAreaCellErase(pixel, pixel, session);
      return;
    }
    if (applyAreaCellErase(session.lastPixel || pixel, pixel, session)) {
      session.lastPixel = pixel;
    }
  }, [applyAreaCellErase, areaEraseRef, areaPaintRef, busy, extendSelectedArea, image, open, tool]);
  return {
    editAreaAtMapPoint
  };
}
