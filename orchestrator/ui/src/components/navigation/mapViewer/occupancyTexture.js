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

import { gridMeta } from './gridGeometry';
import { OCC_COLORS, OCC_ROUNDED, occupancyCellType, OCC_CELL_WALL, OCC_CELL_FREE, OCC_CELL_UNKNOWN, occRgba, drawRoundedOccupancyLayer } from './occupancyDrawing';
import { OCC_REFINE, occRefineComponents, occThickenWalls, occSilhouetteShadow, ROOM_TINTS, ROOM_TINT_ALPHA } from './occupancyRefinement';
import { segmentFreeRooms, roomSegmentationParams } from '../../../utils/roomSegmentation';
import { rgbaArrayFromHex } from './labelSprites';
import * as THREE from 'three';
import { GLOBAL_COSTMAP_FULL_UPLOAD_RATIO } from './constants';
export function makeOccupancyTexture(grid, alpha, mode, highlightedCells = null, refined = false) {
  var _a;
  const meta = gridMeta(grid);
  if (!meta || !grid.data || (grid.data.length < meta.width * meta.height)) return null;
  const palette = OCC_COLORS;
  const w = meta.width;
  const h = meta.height;

  // ---- Rounded exact-cell rendering for the base map ----
  if (mode === "map") {
    const scale = Math.max(1, Math.min(OCC_ROUNDED.scale, Math.floor(OCC_ROUNDED.maxDim / Math.max(w, h)) || 1));
    const cells = new Uint8Array(w * h);
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const srcIndex = (w - 1 - x) + ((h - 1 - y) * w); // same flip as before
        const value = ((_a = grid.data[srcIndex]) !== null) && (_a !== void 0) ? _a : -1;
        cells[x + (y * w)] = occupancyCellType(value);
      }
    }
    const refine = refined && (w * h >= OCC_REFINE.minGridCells);
    if (refine) {
      // Despeckle: wall specks inside the floor become floor, stray free
      // islands vanish, and small enclosed unknown holes are filled in.
      occRefineComponents(cells, w, h, {
        type: OCC_CELL_WALL,
        minSize: OCC_REFINE.minWallCells,
        replacement: OCC_CELL_FREE
      });
      occRefineComponents(cells, w, h, {
        type: OCC_CELL_FREE,
        minSize: OCC_REFINE.minFreeCells,
        replacement: OCC_CELL_UNKNOWN
      });
      occRefineComponents(cells, w, h, {
        type: OCC_CELL_UNKNOWN,
        minSize: OCC_REFINE.holeFillCells,
        replacement: OCC_CELL_FREE,
        interiorOnly: true
      });
      occThickenWalls(cells, w, h, OCC_REFINE.wallThickenPasses);
    }
    const out = document.createElement("canvas");
    out.width = w * scale;
    out.height = h * scale;
    const ctx = out.getContext("2d");
    if (!ctx) return null;
    let roomLabels = null;
    if (refine) {
      // Room segmentation for the pastel tint (2+ rooms only).
      const freeMask = new Uint8Array(w * h);
      for (let i = 0; i < (w * h); i += 1) freeMask[i] = cells[i] === OCC_CELL_FREE ? 1 : 0;
      const segmented = segmentFreeRooms(freeMask, w, h, roomSegmentationParams(meta.resolution));
      if (segmented.roomCount >= 2) roomLabels = segmented.labels;
    }
    if (refine) {
      // Floating silhouette: unknown stays transparent, the known floor
      // casts a soft shadow onto the warm scene background.
      const shadow = occSilhouetteShadow(cells, w, h, scale);
      if (shadow) ctx.drawImage(shadow, scale * 1.2, scale * 1.6);
    } else {
      ctx.fillStyle = occRgba(palette.unknown);
      ctx.fillRect(0, 0, out.width, out.height);
    }
    const freeLayer = drawRoundedOccupancyLayer(ctx, cells, w, h, scale, OCC_CELL_FREE, palette.free);
    if (roomLabels && freeLayer) {
      const tintSmall = document.createElement("canvas");
      tintSmall.width = w;
      tintSmall.height = h;
      const tctx = tintSmall.getContext("2d");
      if (tctx) {
        const tintPalette = ROOM_TINTS.map(hex => rgbaArrayFromHex(hex, 255));
        const tintImage = tctx.createImageData(w, h);
        for (let i = 0; i < (w * h); i += 1) {
          const label = roomLabels[i];
          if (label <= 0) continue;
          const [r, g, b] = tintPalette[(label - 1) % tintPalette.length];
          tintImage.data[i * 4] = r;
          tintImage.data[(i * 4) + 1] = g;
          tintImage.data[(i * 4) + 2] = b;
          tintImage.data[(i * 4) + 3] = 255;
        }
        tctx.putImageData(tintImage, 0, 0);
        const tintBig = document.createElement("canvas");
        tintBig.width = w * scale;
        tintBig.height = h * scale;
        const btx = tintBig.getContext("2d");
        if (btx) {
          btx.imageSmoothingEnabled = true;
          btx.drawImage(tintSmall, 0, 0, tintBig.width, tintBig.height);
          // Clip the tint to the rounded free-space silhouette.
          btx.globalCompositeOperation = "destination-in";
          btx.drawImage(freeLayer, 0, 0);
          ctx.globalAlpha = ROOM_TINT_ALPHA;
          ctx.drawImage(tintBig, 0, 0);
          ctx.globalAlpha = 1;
        }
      }
    }
    drawRoundedOccupancyLayer(ctx, cells, w, h, scale, OCC_CELL_WALL, palette.wall);
    const texture = new THREE.CanvasTexture(out);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;
    texture.flipY = false;
    texture.needsUpdate = true;
    return texture;
  }

  // ---- costmaps: crisp per-cell (safety-accurate), warm colors ----
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const srcIndex = (w - 1 - x) + ((h - 1 - y) * w);
      const dstIndex = (x + y * w) * 4;
      const value = ((_a = grid.data[srcIndex]) !== null) && (_a !== void 0) ? _a : -1;
      let r = 118,
        g = 118,
        b = 118,
        a = alpha;
      if (mode === "globalCostmap") {
        if (value <= 0) {
          r = 0;
          g = 0;
          b = 0;
          a = 0;
        } else {
          // Neutral gray at ~1/3 the old opacity — a hint under the
          // mission markers rather than a poster over the map.
          const normalized = Math.min(Math.max(value, 0), 100) / 100;
          const gray = Math.round(200 - (normalized * 130));
          r = gray;
          g = gray;
          b = gray;
          a = Math.round(40 + (normalized * 70));
        }
      } else if (mode === "localCostmap") {
        if ((highlightedCells === null) || (highlightedCells === void 0) ? void 0 : highlightedCells.has(srcIndex)) {
          [r, g, b, a] = palette.lethal;
        } else if (value <= 20) {
          a = 0;
        } else if (value < 70) {
          [r, g, b, a] = palette.lethal;
        } else {
          [r, g, b, a] = palette.inflate;
        }
      }
      data[dstIndex] = r;
      data[dstIndex + 1] = g;
      data[dstIndex + 2] = b;
      data[dstIndex + 3] = a;
    }
  }
  const texture = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  if (mode === "globalCostmap") {
    texture.userData.globalCostmapFullUploadPending = false;
    texture.onUpdate = () => {
      texture.userData.globalCostmapFullUploadPending = false;
    };
  }
  texture.needsUpdate = true;
  return texture;
}
export function writeGlobalCostmapPixel(data, index, value) {
  if (value <= 0) {
    data[index] = 0;
    data[index + 1] = 0;
    data[index + 2] = 0;
    data[index + 3] = 0;
    return;
  }
  const normalized = Math.min(Math.max(value, 0), 100) / 100;
  const gray = Math.round(200 - (normalized * 130));
  data[index] = gray;
  data[index + 1] = gray;
  data[index + 2] = gray;
  data[index + 3] = Math.round(40 + (normalized * 70));
}
export function updateGlobalCostmapTexture(texture, grid, updateRegion) {
  const meta = gridMeta(grid);
  const image = texture?.image;
  const textureData = image?.data;
  if (!meta || !textureData || !updateRegion || !texture?.addUpdateRange) return false;
  const x = Number(updateRegion.x);
  const y = Number(updateRegion.y);
  const width = Number(updateRegion.width);
  const height = Number(updateRegion.height);
  if (!Number.isInteger(x) || !Number.isInteger(y) || !Number.isInteger(width) || !Number.isInteger(height) || (x < 0) || (y < 0) || (width <= 0) || (height <= 0) || (x + width > meta.width) || (y + height > meta.height) || (image.width !== meta.width) || (image.height !== meta.height) || (textureData.length < meta.width * meta.height * 4)) {
    return false;
  }
  const useFullUpload = (width * height) > (meta.width * meta.height * GLOBAL_COSTMAP_FULL_UPLOAD_RATIO);
  const fullUploadPending = Boolean(texture.userData?.globalCostmapFullUploadPending);
  if (useFullUpload) {
    texture.clearUpdateRanges?.();
    texture.userData.globalCostmapFullUploadPending = true;
  }
  for (let gridY = y; gridY < (y + height); gridY += 1) {
    const textureY = meta.height - 1 - gridY;
    const textureX = meta.width - x - width;
    for (let gridX = x; gridX < (x + width); gridX += 1) {
      const sourceIndex = (gridY * meta.width) + gridX;
      const targetX = meta.width - 1 - gridX;
      const targetIndex = (textureY * meta.width + targetX) * 4;
      writeGlobalCostmapPixel(textureData, targetIndex, grid.data[sourceIndex] ?? -1);
    }
    if (!useFullUpload && !fullUploadPending) {
      texture.addUpdateRange((textureY * meta.width + textureX) * 4, width * 4);
    }
  }
  texture.needsUpdate = true;
  return true;
}
