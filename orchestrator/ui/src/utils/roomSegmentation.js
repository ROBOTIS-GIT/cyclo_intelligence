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
// Author: Seongwoo Kim

// Morphological room segmentation of an occupancy-grid free mask (the classic
// robot-vacuum approach): erode free space until doorways pinch shut, take the
// surviving blobs as room cores, then grow every core back over the original
// free mask with a multi-source BFS so each free cell joins its nearest room.
//
// Inputs are display/derived data only — segmentation never alters the map.

export function segmentFreeRooms(freeMask, width, height, { erosionCells = 6, minCoreCells = 24 } = {}) {
  const size = width * height;
  let eroded = Uint8Array.from(freeMask);
  let scratch = new Uint8Array(size);
  for (let pass = 0; pass < erosionCells; pass += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = x + y * width;
        scratch[index] = eroded[index]
          && x > 0 && eroded[index - 1]
          && x < width - 1 && eroded[index + 1]
          && y > 0 && eroded[index - width]
          && y < height - 1 && eroded[index + width]
          ? 1 : 0;
      }
    }
    const swap = eroded;
    eroded = scratch;
    scratch = swap;
  }

  const labels = new Int32Array(size);
  const queue = new Int32Array(size);
  const component = [];
  let roomCount = 0;
  for (let start = 0; start < size; start += 1) {
    if (!eroded[start] || labels[start] !== 0) continue;
    let head = 0;
    let tail = 0;
    queue[tail] = start;
    tail += 1;
    labels[start] = -1;
    component.length = 0;
    while (head < tail) {
      const index = queue[head];
      head += 1;
      component.push(index);
      const x = index % width;
      const y = (index / width) | 0;
      if (x > 0 && eroded[index - 1] && labels[index - 1] === 0) { labels[index - 1] = -1; queue[tail] = index - 1; tail += 1; }
      if (x < width - 1 && eroded[index + 1] && labels[index + 1] === 0) { labels[index + 1] = -1; queue[tail] = index + 1; tail += 1; }
      if (y > 0 && eroded[index - width] && labels[index - width] === 0) { labels[index - width] = -1; queue[tail] = index - width; tail += 1; }
      if (y < height - 1 && eroded[index + width] && labels[index + width] === 0) { labels[index + width] = -1; queue[tail] = index + width; tail += 1; }
    }
    if (component.length >= minCoreCells) {
      roomCount += 1;
      for (let i = 0; i < component.length; i += 1) labels[component[i]] = roomCount;
    }
  }
  for (let i = 0; i < size; i += 1) {
    if (labels[i] === -1) labels[i] = 0;
  }
  if (roomCount === 0) {
    return { labels, roomCount: 0 };
  }

  // Grow the cores back over the full free mask (nearest-core wins).
  let head = 0;
  let tail = 0;
  for (let i = 0; i < size; i += 1) {
    if (labels[i] > 0) {
      queue[tail] = i;
      tail += 1;
    }
  }
  while (head < tail) {
    const index = queue[head];
    head += 1;
    const label = labels[index];
    const x = index % width;
    const y = (index / width) | 0;
    if (x > 0 && freeMask[index - 1] && labels[index - 1] === 0) { labels[index - 1] = label; queue[tail] = index - 1; tail += 1; }
    if (x < width - 1 && freeMask[index + 1] && labels[index + 1] === 0) { labels[index + 1] = label; queue[tail] = index + 1; tail += 1; }
    if (y > 0 && freeMask[index - width] && labels[index - width] === 0) { labels[index - width] = label; queue[tail] = index - width; tail += 1; }
    if (y < height - 1 && freeMask[index + width] && labels[index + width] === 0) { labels[index + width] = label; queue[tail] = index + width; tail += 1; }
  }
  return { labels, roomCount };
}

// Shared parameter derivation so the viewer tint and the editor's room
// suggestions segment identically for a given map resolution.
export function roomSegmentationParams(resolution) {
  const res = Number(resolution) || 1;
  return {
    // Half a doorway (~0.9 m opening) so doors pinch shut during erosion.
    erosionCells: Math.min(14, Math.max(3, Math.round(0.45 / res))),
    // A real room core is at least ~1.2 m² after erosion.
    minCoreCells: Math.max(9, Math.round(1.2 / (res * res))),
  };
}
