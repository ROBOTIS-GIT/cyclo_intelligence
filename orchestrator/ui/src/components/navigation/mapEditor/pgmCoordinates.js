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

import { mapPointToAreaGridCell } from '../../../utils/mapAreaGeometry';
import { yawFromPose } from '../../../utils/navigationTf';
import { DEFAULT_BRUSH_SIZE_CELLS, MAX_BRUSH_SIZE_CELLS } from './constants';
export function pgmPixelToGridCell(image, pixel) {
  return {
    x: pixel.pixelX,
    y: image.height - 1 - pixel.pixelY
  };
}
export function gridCellToPgmIndex(image, cell) {
  const pgmY = image.height - 1 - cell.y;
  if ((cell.x < 0) || (cell.x >= image.width) || (pgmY < 0) || (pgmY >= image.height)) return -1;
  return cell.x + (pgmY * image.width);
}
export function mapPointToPgmPixel(image, x, y) {
  const cell = mapPointToAreaGridCell(image, x, y);
  if (!cell) return null;
  return {
    pixelX: cell.x,
    pixelY: image.height - 1 - cell.y
  };
}
export function pgmPixelToMapPoint(image, pixelX, pixelY) {
  const resolution = Number(image.resolution ?? 1) || 1;
  const origin = image.origin ?? {
    position: {
      x: 0,
      y: 0,
      z: 0
    },
    orientation: {
      x: 0,
      y: 0,
      z: 0,
      w: 1
    }
  };
  const originX = Number(origin.position?.x ?? 0);
  const originY = Number(origin.position?.y ?? 0);
  const originYaw = yawFromPose(origin);
  const localX = (pixelX + 0.5) * resolution;
  const localY = (image.height - 1 - pixelY + 0.5) * resolution;
  return {
    x: (originX + Math.cos(originYaw) * localX) - (Math.sin(originYaw) * localY),
    y: originY + (Math.sin(originYaw) * localX) + (Math.cos(originYaw) * localY)
  };
}
export function normalizeBrushSize(value) {
  if (!Number.isFinite(value)) return DEFAULT_BRUSH_SIZE_CELLS;
  return Math.min(Math.max(Math.floor(value), 1), MAX_BRUSH_SIZE_CELLS);
}
export function mapNameFromPgmPath(path) {
  return path.split("/").pop().replace(/\.pgm$/i, "");
}
export function preferredPgmFile(files, mapName) {
  const exact = files.find(file => mapNameFromPgmPath(file.path) === mapName);
  return exact || files.find(file => file.path.includes(mapName));
}
