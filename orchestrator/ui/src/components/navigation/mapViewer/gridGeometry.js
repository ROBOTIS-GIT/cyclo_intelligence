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

import { yawFromPose } from '../../../utils/navigationTf';
import { mapAreaSelectionBounds } from '../../../utils/mapAreaGeometry';
export function gridMeta(grid) {
  var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k;
  const info = (grid === null) || (grid === void 0) ? void 0 : grid.info;
  const width = Number(((_a = info === null || info === void 0 ? void 0 : info.width) !== null) && (_a !== void 0) ? _a : 0);
  const height = Number(((_b = info === null || info === void 0 ? void 0 : info.height) !== null) && (_b !== void 0) ? _b : 0);
  const resolution = Number(((_c = info === null || info === void 0 ? void 0 : info.resolution) !== null) && (_c !== void 0) ? _c : 0);
  const originX = Number(((_f = (_e = (_d = info === null || info === void 0 ? void 0 : info.origin) === null || _d === void 0 ? void 0 : _d.position) === null || _e === void 0 ? void 0 : _e.x) !== null) && (_f !== void 0) ? _f : 0);
  const originY = Number(((_j = (_h = (_g = info === null || info === void 0 ? void 0 : info.origin) === null || _g === void 0 ? void 0 : _g.position) === null || _h === void 0 ? void 0 : _h.y) !== null) && (_j !== void 0) ? _j : 0);
  const originYaw = yawFromPose(((_k = info === null || info === void 0 ? void 0 : info.origin) !== null) && (_k !== void 0) ? _k : null);
  if (!width || !height || !resolution) return null;
  return {
    width,
    height,
    resolution,
    originX,
    originY,
    originYaw
  };
}
export function mapPointToGridCell(meta, x, y) {
  if (!meta || !Number.isFinite(x) || !Number.isFinite(y)) return null;
  const dx = x - meta.originX;
  const dy = y - meta.originY;
  const localX = (Math.cos(meta.originYaw) * dx) + (Math.sin(meta.originYaw) * dy);
  const localY = (-Math.sin(meta.originYaw) * dx) + (Math.cos(meta.originYaw) * dy);
  const cellX = Math.floor(localX / meta.resolution);
  const cellY = Math.floor(localY / meta.resolution);
  if ((cellX < 0) || (cellX >= meta.width) || (cellY < 0) || (cellY >= meta.height)) return null;
  return {
    x: cellX,
    y: cellY
  };
}
export function gridCellToMapPoint(meta, cellX, cellY) {
  const localX = (cellX + 0.5) * meta.resolution;
  const localY = (cellY + 0.5) * meta.resolution;
  return {
    x: (meta.originX + Math.cos(meta.originYaw) * localX) - (Math.sin(meta.originYaw) * localY),
    y: meta.originY + (Math.sin(meta.originYaw) * localX) + (Math.cos(meta.originYaw) * localY)
  };
}
export function boundedFreeRegion(grid, regionSpec, excludedCells = null) {
  const meta = gridMeta(grid);
  if (!meta || !grid.data || !regionSpec) return null;
  if (Array.isArray(regionSpec.cells) && regionSpec.cells.length) {
    const cells = [];
    const seen = new Set();
    let sumX = 0;
    let sumY = 0;
    regionSpec.cells.forEach(cell => {
      const x = Math.floor(Number((cell === null) || (cell === void 0) ? void 0 : cell.x));
      const y = Math.floor(Number((cell === null) || (cell === void 0) ? void 0 : cell.y));
      if (!Number.isFinite(x) || !Number.isFinite(y) || (x < 0) || (y < 0) || (x >= meta.width) || (y >= meta.height)) return;
      const index = x + (y * meta.width);
      if (seen.has(index) || (grid.data[index] !== 0)) return;
      if ((excludedCells === null) || (excludedCells === void 0) ? void 0 : excludedCells.has(index)) return;
      seen.add(index);
      cells.push(index);
      sumX += x;
      sumY += y;
    });
    if (!cells.length) return null;
    return {
      cells,
      centroid: {
        x: sumX / cells.length,
        y: sumY / cells.length
      }
    };
  }
  const seed = regionSpec.seed_cell;
  const bounds = regionSpec.bounds || (seed ? {
    x_min: seed.x,
    y_min: seed.y,
    x_max: seed.x,
    y_max: seed.y
  } : null);
  if (!bounds) return null;
  const rawXMin = Math.floor(Number(bounds.x_min));
  const rawXMax = Math.floor(Number(bounds.x_max));
  const rawYMin = Math.floor(Number(bounds.y_min));
  const rawYMax = Math.floor(Number(bounds.y_max));
  if (![rawXMin, rawXMax, rawYMin, rawYMax].every(Number.isFinite)) return null;
  const xMin = Math.max(0, Math.min(rawXMin, rawXMax));
  const xMax = Math.min(meta.width - 1, Math.max(rawXMin, rawXMax));
  const yMin = Math.max(0, Math.min(rawYMin, rawYMax));
  const yMax = Math.min(meta.height - 1, Math.max(rawYMin, rawYMax));
  const cells = [];
  let sumX = 0;
  let sumY = 0;
  for (let y = yMin; y <= yMax; y += 1) {
    for (let x = xMin; x <= xMax; x += 1) {
      const index = x + (y * meta.width);
      if (grid.data[index] !== 0) continue;
      if ((excludedCells === null) || (excludedCells === void 0) ? void 0 : excludedCells.has(index)) continue;
      cells.push(index);
      sumX += x;
      sumY += y;
    }
  }
  if (!cells.length) return null;
  return {
    cells,
    centroid: {
      x: sumX / cells.length,
      y: sumY / cells.length
    }
  };
}
export function mapAreaPreviewCellIndices(grid, selection, excludedCells = null) {
  const bounds = mapAreaSelectionBounds(grid, selection);
  if (!bounds) return [];
  return boundedFreeRegion(grid, {
    bounds
  }, excludedCells)?.cells ?? [];
}
