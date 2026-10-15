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

import { FREE_VALUE, UNKNOWN_VALUE, OCCUPIED_VALUE, OCCUPIED_THRESHOLD, FREE_THRESHOLD } from './constants';
export function decodePgmPixels(image) {
  const binary = window.atob(image.pixels_base64);
  const pixels = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    pixels[index] = binary.charCodeAt(index);
  }
  return pixels;
}
export function encodePgmPixels(pixels) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < pixels.length; index += chunkSize) {
    const chunk = pixels.subarray(index, index + chunkSize);
    binary += String.fromCharCode.apply(null, Array.from(chunk));
  }
  return window.btoa(binary);
}
export function applyPgmBrush(pixels, width, height, pixelX, pixelY, operation, brushSizeCells) {
  const value = operation === "erase_black" ? FREE_VALUE : operation === "draw_unknown" ? UNKNOWN_VALUE : OCCUPIED_VALUE;
  const offset = Math.floor(brushSizeCells / 2);
  const startX = pixelX - offset;
  const startY = pixelY - offset;
  for (let y = startY; y < (startY + brushSizeCells); y += 1) {
    if ((y < 0) || (y >= height)) continue;
    for (let x = startX; x < (startX + brushSizeCells); x += 1) {
      if ((x < 0) || (x >= width)) continue;
      pixels[x + (y * width)] = value;
    }
  }
}
export function paintPgmPixels(pixels, width, height, pixelX, pixelY, operation, brushSizeCells) {
  const next = new Uint8Array(pixels);
  applyPgmBrush(next, width, height, pixelX, pixelY, operation, brushSizeCells);
  return next;
}
export function paintPgmPixelSegment(pixels, width, height, startPixel, endPixel, operation, brushSizeCells) {
  const next = new Uint8Array(pixels);
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
    applyPgmBrush(next, width, height, x0, y0, operation, brushSizeCells);
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
  return next;
}
export function pgmPixelsToGrid(image, pixels) {
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
  const data = new Array(image.width * image.height);
  for (let pgmY = 0; pgmY < image.height; pgmY += 1) {
    for (let x = 0; x < image.width; x += 1) {
      const value = pixels[x + (pgmY * image.width)];
      const occupancy = value <= OCCUPIED_THRESHOLD ? 100 : value >= FREE_THRESHOLD ? 0 : -1;
      const gridY = image.height - 1 - pgmY;
      data[x + (gridY * image.width)] = occupancy;
    }
  }
  return {
    header: {
      frame_id: "map"
    },
    info: {
      resolution,
      width: image.width,
      height: image.height,
      origin
    },
    data
  };
}
