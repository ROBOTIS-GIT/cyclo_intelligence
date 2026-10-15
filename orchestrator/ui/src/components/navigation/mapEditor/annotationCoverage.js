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

import { annotationCells, cellKey } from './annotationGeometry';
import { FREE_THRESHOLD } from './constants';
import { pgmPixelToGridCell, gridCellToPgmIndex } from './pgmCoordinates';
export function coveredAnnotationCellSet(annotations, image, pixels, excludeId = "") {
  const covered = new Set();
  annotations.forEach(annotation => {
    if (excludeId && (annotation.id === excludeId)) return;
    annotationCells(annotation, image, pixels).forEach(cell => covered.add(cellKey(cell)));
  });
  return covered;
}
export function visibleAnnotationCellSnapshots(annotations, image, pixels) {
  const covered = new Set();
  return annotations.map(annotation => {
    const visibleCells = annotationCells(annotation, image, pixels).filter(cell => !covered.has(cellKey(cell)));
    visibleCells.forEach(cell => covered.add(cellKey(cell)));
    return {
      annotation,
      cells: visibleCells
    };
  }).filter(snapshot => snapshot.cells.length > 0);
}
export function brushCellsForSegment(pixels, image, startPixel, endPixel, brushSizeCells, coveredCells = new Set()) {
  const cells = new Map();
  const addBrush = (pixelX, pixelY) => {
    const offset = Math.floor(brushSizeCells / 2);
    for (let y = pixelY - offset; y < (pixelY - offset + brushSizeCells); y += 1) {
      if ((y < 0) || (y >= image.height)) continue;
      for (let x = pixelX - offset; x < (pixelX - offset + brushSizeCells); x += 1) {
        if ((x < 0) || (x >= image.width)) continue;
        if (pixels[x + (y * image.width)] < FREE_THRESHOLD) continue;
        const cell = pgmPixelToGridCell(image, {
          pixelX: x,
          pixelY: y
        });
        const key = cellKey(cell);
        if (coveredCells.has(key)) continue;
        cells.set(key, cell);
      }
    }
  };
  let x0 = startPixel.pixelX;
  let y0 = startPixel.pixelY;
  const x1 = endPixel.pixelX;
  const y1 = endPixel.pixelY;
  const dx = Math.abs(x1 - x0);
  const sx = x0 < x1 ? 1 : -1;
  const dy = -Math.abs(y1 - y0);
  const sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  while (true) {
    addBrush(x0, y0);
    if ((x0 === x1) && (y0 === y1)) break;
    const e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
  return Array.from(cells.values());
}
export function selectedFreePgmRegion(pixels, image, startPixel, endPixel, existingAnnotations = []) {
  if (!pixels || !image || !startPixel || !endPixel) return null;
  const xMin = Math.max(0, Math.min(startPixel.pixelX, endPixel.pixelX));
  const xMax = Math.min(image.width - 1, Math.max(startPixel.pixelX, endPixel.pixelX));
  const yMin = Math.max(0, Math.min(startPixel.pixelY, endPixel.pixelY));
  const yMax = Math.min(image.height - 1, Math.max(startPixel.pixelY, endPixel.pixelY));
  // Precompute the covered pgm-index set ONCE — checking every rect pixel
  // against every annotation's cell list is O(rect × cells) and freezes the
  // UI on large selections.
  const covered = new Set();
  (existingAnnotations || []).forEach(annotation => {
    annotationCells(annotation, image, pixels).forEach(cell => {
      const index = gridCellToPgmIndex(image, cell);
      if (index >= 0) covered.add(index);
    });
  });
  let count = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = yMin; y <= yMax; y += 1) {
    for (let x = xMin; x <= xMax; x += 1) {
      const index = x + (y * image.width);
      if (pixels[index] < FREE_THRESHOLD) continue;
      if (covered.has(index)) continue;
      count += 1;
      sumX += x;
      sumY += y;
    }
  }
  const gridYMin = image.height - 1 - yMax;
  const gridYMax = image.height - 1 - yMin;
  return {
    count,
    centroidPixelX: count ? sumX / count : (xMin + xMax) / 2,
    centroidPixelY: count ? sumY / count : (yMin + yMax) / 2,
    bounds: {
      x_min: xMin,
      y_min: gridYMin,
      x_max: xMax,
      y_max: gridYMax
    },
    seedCell: {
      x: Math.floor((xMin + xMax) / 2),
      y: Math.floor((gridYMin + gridYMax) / 2)
    }
  };
}
