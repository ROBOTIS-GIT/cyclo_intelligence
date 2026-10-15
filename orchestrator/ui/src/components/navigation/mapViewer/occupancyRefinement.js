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

import { OCC_CELL_WALL, OCC_CELL_UNKNOWN } from './occupancyDrawing';
export const OCC_REFINE = {
  minWallCells: 6,
  // wall specks smaller than this become floor
  minFreeCells: 24,
  // stray free islands smaller than this become unknown
  holeFillCells: 24,
  // enclosed unknown holes smaller than this become floor
  minGridCells: 400,
  // skip refinement for tiny grids
  wallThickenPasses: 1
};
export const ROOM_TINTS = ["#D7E3F0", "#F0DCE3", "#E4DEF0", "#D9EAE8", "#DFE6EF", "#EFDCE9"];
export const ROOM_TINT_ALPHA = 0.5;
export function occRefineComponents(cells, w, h, {
  type,
  minSize,
  replacement,
  interiorOnly = false
}) {
  const labels = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  const component = [];
  for (let start = 0; start < (w * h); start += 1) {
    if ((cells[start] !== type) || labels[start]) continue;
    let head = 0;
    let tail = 0;
    queue[tail] = start;
    tail += 1;
    labels[start] = 1;
    component.length = 0;
    let touchesBorder = false;
    while (head < tail) {
      const index = queue[head];
      head += 1;
      component.push(index);
      const x = index % w;
      const y = (index / w) | 0;
      if ((x === 0) || (y === 0) || (x === w - 1) || (y === h - 1)) touchesBorder = true;
      if ((x > 0) && (cells[index - 1] === type) && !labels[index - 1]) {
        labels[index - 1] = 1;
        queue[tail] = index - 1;
        tail += 1;
      }
      if ((x < w - 1) && (cells[index + 1] === type) && !labels[index + 1]) {
        labels[index + 1] = 1;
        queue[tail] = index + 1;
        tail += 1;
      }
      if ((y > 0) && (cells[index - w] === type) && !labels[index - w]) {
        labels[index - w] = 1;
        queue[tail] = index - w;
        tail += 1;
      }
      if ((y < h - 1) && (cells[index + w] === type) && !labels[index + w]) {
        labels[index + w] = 1;
        queue[tail] = index + w;
        tail += 1;
      }
    }
    if ((component.length < minSize) && (!interiorOnly || !touchesBorder)) {
      for (let i = 0; i < component.length; i += 1) cells[component[i]] = replacement;
    }
  }
}
export function occThickenWalls(cells, w, h, passes) {
  // Thicken OUTWARD only (into unknown): room interiors keep their exact
  // edges, so walls read as a clean stroke without visually shrinking rooms.
  for (let pass = 0; pass < passes; pass += 1) {
    const source = new Uint8Array(cells);
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        if (source[x + (y * w)] !== OCC_CELL_WALL) continue;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            const nx = x + dx;
            const ny = y + dy;
            if ((nx < 0) || (nx >= w) || (ny < 0) || (ny >= h)) continue;
            const index = nx + (ny * w);
            if (cells[index] === OCC_CELL_UNKNOWN) cells[index] = OCC_CELL_WALL;
          }
        }
      }
    }
  }
}
export function occSilhouetteShadow(cells, w, h, scale) {
  const silhouette = document.createElement("canvas");
  silhouette.width = w;
  silhouette.height = h;
  const sctx = silhouette.getContext("2d");
  if (!sctx) return null;
  const image = sctx.createImageData(w, h);
  for (let i = 0; i < (w * h); i += 1) {
    if (cells[i] !== OCC_CELL_UNKNOWN) image.data[(i * 4) + 3] = 255;
  }
  sctx.putImageData(image, 0, 0);
  const big = document.createElement("canvas");
  big.width = w * scale;
  big.height = h * scale;
  const bctx = big.getContext("2d");
  if (!bctx) return null;
  bctx.imageSmoothingEnabled = true;
  bctx.filter = `blur(${Math.max(2, scale * 1.6)}px)`;
  bctx.drawImage(silhouette, 0, 0, big.width, big.height);
  bctx.filter = "none";
  bctx.globalCompositeOperation = "source-in";
  bctx.fillStyle = "rgba(28,26,23,0.26)";
  bctx.fillRect(0, 0, big.width, big.height);
  return big;
}
