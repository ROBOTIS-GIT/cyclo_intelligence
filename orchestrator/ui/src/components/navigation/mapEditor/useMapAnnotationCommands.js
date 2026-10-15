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
import { ANNOTATION_TOOL } from './constants';
import { mapPointToPgmPixel, pgmPixelToMapPoint } from './pgmCoordinates';
import { selectedFreePgmRegion, visibleAnnotationCellSnapshots } from './annotationCoverage';
import { nextAutoAreaLabel, makeAnnotationId, chooseAnnotationColor } from './annotationIdentity';
import { normalizeAnnotation } from './annotationModel';
import { annotationWithCells } from './annotationGeometry';
export function useMapAnnotationCommands({
  open,
  image,
  busy,
  tool,
  onMessage,
  pixelsRef,
  annotationsRef,
  annotationLabel,
  persistAnnotations,
  setSelectedAnnotationId
}) {
  const placeAnnotationAtMapArea = useCallback((startX, startY, endX = startX, endY = startY) => {
    if (!open || !image || busy || (tool !== ANNOTATION_TOOL.id)) return;
    const startPixel = mapPointToPgmPixel(image, startX, startY);
    const endPixel = mapPointToPgmPixel(image, endX, endY);
    if (!startPixel || !endPixel) {
      onMessage("Select an area inside the map");
      return;
    }
    const currentPixels = pixelsRef.current;
    if (!currentPixels) return;
    const region = selectedFreePgmRegion(currentPixels, image, startPixel, endPixel, annotationsRef.current);
    if (!region || (region.count === 0)) {
      onMessage("Drag over an unmarked white free-space area");
      return;
    }
    const label = annotationLabel.trim() || nextAutoAreaLabel(annotationsRef.current);
    const centroid = pgmPixelToMapPoint(image, region.centroidPixelX, region.centroidPixelY);
    const nextAnnotation = normalizeAnnotation({
      id: makeAnnotationId(),
      label,
      color: chooseAnnotationColor(annotationsRef.current),
      pose: {
        frame_id: "map",
        x: centroid.x,
        y: centroid.y,
        yaw: 0
      },
      region: {
        seed_cell: {
          x: region.seedCell.x,
          y: region.seedCell.y
        },
        bounds: region.bounds,
        cell_count: region.count,
        width: image.width,
        height: image.height
      }
    });
    persistAnnotations([...annotationsRef.current, nextAnnotation], `Marked area ${nextAnnotation.label}`);
    setSelectedAnnotationId(nextAnnotation.id);
  }, [annotationLabel, annotationsRef, busy, image, onMessage, open, persistAnnotations, pixelsRef, setSelectedAnnotationId, tool]);
  const deleteAnnotationById = useCallback(id => {
    if (!open || !image || busy) return;
    const currentPixels = pixelsRef.current;
    if (!currentPixels) return;
    const target = annotationsRef.current.find(annotation => annotation.id === id);
    if (!target) return;
    const nextAnnotations = visibleAnnotationCellSnapshots(annotationsRef.current, image, currentPixels).filter(snapshot => snapshot.annotation.id !== id).map(snapshot => annotationWithCells(snapshot.annotation, snapshot.cells, image)).filter(Boolean);
    persistAnnotations(nextAnnotations, `Removed area ${target.label || "Area"}`);
  }, [annotationsRef, busy, image, open, persistAnnotations, pixelsRef]);
  const renameAnnotation = useCallback((id, label) => {
    const trimmed = String(label || "").trim().slice(0, 80);
    if (!trimmed) return;
    const target = annotationsRef.current.find(annotation => annotation.id === id);
    if (!target || (target.label === trimmed)) return;
    persistAnnotations(annotationsRef.current.map(annotation => annotation.id === id ? {
      ...annotation,
      label: trimmed
    } : annotation), `Renamed area ${trimmed}`);
  }, [annotationsRef, persistAnnotations]);
  const clearAnnotations = useCallback(() => {
    if (!annotationsRef.current.length) return;
    persistAnnotations([], "Cleared map areas");
  }, [annotationsRef, persistAnnotations]);
  return {
    placeAnnotationAtMapArea,
    deleteAnnotationById,
    renameAnnotation,
    clearAnnotations
  };
}
