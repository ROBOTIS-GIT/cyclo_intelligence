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

import { normalizeAnnotationRegion, normalizeRegionCells, normalizeAnnotation } from './annotationModel';
import { gridCellToPgmIndex, pgmPixelToMapPoint } from './pgmCoordinates';
import { FREE_THRESHOLD } from './constants';
export function annotationGridBounds(annotation, image) {
  const region = normalizeAnnotationRegion((annotation === null) || (annotation === void 0) ? void 0 : annotation.region);
  if (!region) return null;
  if ((region.width && region.width !== image.width) || (region.height && region.height !== image.height)) return null;
  const bounds = region.bounds || {
    x_min: region.seed_cell.x,
    y_min: region.seed_cell.y,
    x_max: region.seed_cell.x,
    y_max: region.seed_cell.y
  };
  return {
    x_min: Math.max(0, Math.min(bounds.x_min, bounds.x_max)),
    y_min: Math.max(0, Math.min(bounds.y_min, bounds.y_max)),
    x_max: Math.min(image.width - 1, Math.max(bounds.x_min, bounds.x_max)),
    y_max: Math.min(image.height - 1, Math.max(bounds.y_min, bounds.y_max))
  };
}
export function cellKey(cell) {
  return `${cell.x}:${cell.y}`;
}
export function annotationCells(annotation, image, pixels = null) {
  const region = normalizeAnnotationRegion((annotation === null) || (annotation === void 0) ? void 0 : annotation.region);
  if (!region) return [];
  if ((region.width && region.width !== image.width) || (region.height && region.height !== image.height)) return [];
  if (Array.isArray(region.cells) && region.cells.length) {
    return normalizeRegionCells(region.cells, image.width, image.height);
  }
  const bounds = annotationGridBounds(annotation, image);
  if (!bounds) return [];
  const cells = [];
  for (let y = bounds.y_min; y <= bounds.y_max; y += 1) {
    for (let x = bounds.x_min; x <= bounds.x_max; x += 1) {
      const cell = {
        x,
        y
      };
      const index = pixels ? gridCellToPgmIndex(image, cell) : -1;
      if (pixels && (index < 0 || pixels[index] < FREE_THRESHOLD)) continue;
      cells.push(cell);
    }
  }
  return cells;
}
export function cellsToRegion(cells, image) {
  const normalizedCells = normalizeRegionCells(cells, image.width, image.height);
  if (!normalizedCells.length) return null;
  let xMin = normalizedCells[0].x;
  let xMax = normalizedCells[0].x;
  let yMin = normalizedCells[0].y;
  let yMax = normalizedCells[0].y;
  let sumX = 0;
  let sumY = 0;
  normalizedCells.forEach(cell => {
    xMin = Math.min(xMin, cell.x);
    xMax = Math.max(xMax, cell.x);
    yMin = Math.min(yMin, cell.y);
    yMax = Math.max(yMax, cell.y);
    sumX += cell.x;
    sumY += cell.y;
  });
  const seedCell = {
    x: Math.floor(sumX / normalizedCells.length),
    y: Math.floor(sumY / normalizedCells.length)
  };
  const centroidPgmY = image.height - 1 - (sumY / normalizedCells.length);
  const centroid = pgmPixelToMapPoint(image, sumX / normalizedCells.length, centroidPgmY);
  return {
    pose: {
      frame_id: "map",
      x: centroid.x,
      y: centroid.y,
      yaw: 0
    },
    region: {
      seed_cell: seedCell,
      bounds: {
        x_min: xMin,
        y_min: yMin,
        x_max: xMax,
        y_max: yMax
      },
      cells: normalizedCells,
      cell_count: normalizedCells.length,
      width: image.width,
      height: image.height
    }
  };
}
export function regionFromCells(cells, image) {
  if (!cells.length) return null;
  let xMin = cells[0].x;
  let xMax = cells[0].x;
  let yMin = cells[0].y;
  let yMax = cells[0].y;
  let sumX = 0;
  let sumY = 0;
  cells.forEach(cell => {
    if (cell.x < xMin) xMin = cell.x;
    if (cell.x > xMax) xMax = cell.x;
    if (cell.y < yMin) yMin = cell.y;
    if (cell.y > yMax) yMax = cell.y;
    sumX += cell.x;
    sumY += cell.y;
  });
  const centroidX = sumX / cells.length;
  const centroidY = sumY / cells.length;
  const centroidPgmY = image.height - 1 - centroidY;
  const centroid = pgmPixelToMapPoint(image, centroidX, centroidPgmY);
  return {
    pose: {
      frame_id: "map",
      x: centroid.x,
      y: centroid.y,
      yaw: 0
    },
    region: {
      seed_cell: {
        x: Math.floor(centroidX),
        y: Math.floor(centroidY)
      },
      bounds: {
        x_min: xMin,
        y_min: yMin,
        x_max: xMax,
        y_max: yMax
      },
      cells,
      cell_count: cells.length,
      width: image.width,
      height: image.height
    }
  };
}
export function annotationWithCells(annotation, cells, image) {
  const geometry = cellsToRegion(cells, image);
  if (!geometry) return null;
  return normalizeAnnotation({
    ...annotation,
    pose: geometry.pose,
    region: geometry.region
  });
}
