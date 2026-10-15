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

export const OCC_COLORS = {
  unknown: [227, 221, 207, 220],
  // #E3DDCF, blends into paper scene bg
  free: [250, 248, 243, 255],
  // #FAF8F3
  wall: [42, 38, 32, 255],
  // #2A2620 warm ink
  // Local costmap in muted sand-amber (the UI's warning family) at low
  // opacity: semantically "caution zone", warm enough to sit apart from
  // the neutral-gray global costmap, quiet enough to stay under markers.
  lethal: [172, 116, 58, 100],
  // desaturated --mc-warning
  inflate: [188, 148, 96, 70] // lighter sand
};
export const OCC_ROUNDED = {
  scale: 4,
  maxDim: 2048,
  radiusCells: 0.38
};
export function occRgba([r, g, b, a]) {
  return `rgba(${r},${g},${b},${a / 255})`;
}
export const OCC_CELL_UNKNOWN = 0;
export const OCC_CELL_FREE = 1;
export const OCC_CELL_WALL = 2;
export function occupancyCellType(value) {
  if (value === 0) return OCC_CELL_FREE;
  if (value > 0) return OCC_CELL_WALL;
  return OCC_CELL_UNKNOWN;
}
export function sameOccupancyCell(cells, w, h, type, x, y) {
  return (x >= 0) && (x < w) && (y >= 0) && (y < h) && (cells[x + y * w] === type);
}
export function addCornerCut(ctx, x0, y0, x1, y1, r, corner) {
  if (corner === "tl") {
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + r, y0);
    ctx.quadraticCurveTo(x0, y0, x0, y0 + r);
  } else if (corner === "tr") {
    ctx.moveTo(x1, y0);
    ctx.lineTo(x1, y0 + r);
    ctx.quadraticCurveTo(x1, y0, x1 - r, y0);
  } else if (corner === "br") {
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - r, y1);
    ctx.quadraticCurveTo(x1, y1, x1, y1 - r);
  } else {
    ctx.moveTo(x0, y1);
    ctx.lineTo(x0, y1 - r);
    ctx.quadraticCurveTo(x0, y1, x0 + r, y1);
  }
  ctx.closePath();
}
export function drawRoundedOccupancyLayer(targetCtx, cells, w, h, scale, type, color) {
  const layer = document.createElement("canvas");
  layer.width = w * scale;
  layer.height = h * scale;
  const ctx = layer.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = occRgba(color);
  for (let y = 0; y < h; y += 1) {
    let runStart = -1;
    for (let x = 0; x <= w; x += 1) {
      const matches = (x < w) && (cells[x + y * w] === type);
      if (matches && (runStart < 0)) {
        runStart = x;
      } else if (!matches && (runStart >= 0)) {
        ctx.fillRect(runStart * scale, y * scale, (x - runStart) * scale, scale);
        runStart = -1;
      }
    }
  }
  const r = Math.min(scale * OCC_ROUNDED.radiusCells, scale / 2);
  if (r > 0.1) {
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "#000";
    ctx.beginPath();
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        if (cells[x + (y * w)] !== type) continue;
        const left = sameOccupancyCell(cells, w, h, type, x - 1, y);
        const right = sameOccupancyCell(cells, w, h, type, x + 1, y);
        const top = sameOccupancyCell(cells, w, h, type, x, y - 1);
        const bottom = sameOccupancyCell(cells, w, h, type, x, y + 1);
        const x0 = x * scale;
        const y0 = y * scale;
        const x1 = x0 + scale;
        const y1 = y0 + scale;
        if (!left && !top) addCornerCut(ctx, x0, y0, x1, y1, r, "tl");
        if (!right && !top) addCornerCut(ctx, x0, y0, x1, y1, r, "tr");
        if (!right && !bottom) addCornerCut(ctx, x0, y0, x1, y1, r, "br");
        if (!left && !bottom) addCornerCut(ctx, x0, y0, x1, y1, r, "bl");
      }
    }
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  }
  targetCtx.drawImage(layer, 0, 0);
  return layer;
}
