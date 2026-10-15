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

import { SCAN_VISUALIZATION_STRIDE } from './constants';
import { gridMeta } from './gridGeometry';
import { yawFromPose } from '../../../utils/navigationTf';
export function forEachVisualizedScanRange(ranges, callback) {
  let lastVisited = -1;
  for (let index = 0; index < ranges.length; index += SCAN_VISUALIZATION_STRIDE) {
    callback(ranges[index], index);
    lastVisited = index;
  }
  // Preserve the far edge of the scanner's field of view when stride does
  // not naturally land on the final ray.
  const lastIndex = ranges.length - 1;
  if ((lastIndex >= 0) && (lastVisited !== lastIndex)) {
    callback(ranges[lastIndex], lastIndex);
  }
}
export function scanCellsForGrid(grid, scan, scanPose, framePose) {
  var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o;
  const meta = gridMeta(grid);
  if (!meta || !(((_a = scan === null || scan === void 0 ? void 0 : scan.ranges) === null) || (_a === void 0) ? void 0 : _a.length) || !scanPose) return null;
  const frameYaw = framePose ? yawFromPose(framePose) : 0;
  const frameCos = Math.cos(frameYaw);
  const frameSin = Math.sin(frameYaw);
  const frameX = Number(((_c = (_b = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _b === void 0 ? void 0 : _b.x) !== null) && (_c !== void 0) ? _c : 0);
  const frameY = Number(((_e = (_d = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _d === void 0 ? void 0 : _d.y) !== null) && (_e !== void 0) ? _e : 0);
  const originYaw = meta.originYaw;
  const originCos = Math.cos(originYaw);
  const originSin = Math.sin(originYaw);
  const scanX = Number(((_g = (_f = scanPose.position) === null || _f === void 0 ? void 0 : _f.x) !== null) && (_g !== void 0) ? _g : 0);
  const scanY = Number(((_j = (_h = scanPose.position) === null || _h === void 0 ? void 0 : _h.y) !== null) && (_j !== void 0) ? _j : 0);
  const scanYaw = yawFromPose(scanPose);
  const min = Number(((_k = scan.range_min) !== null) && (_k !== void 0) ? _k : 0.02);
  const max = Number(((_l = scan.range_max) !== null) && (_l !== void 0) ? _l : 20);
  const angleMin = Number(((_m = scan.angle_min) !== null) && (_m !== void 0) ? _m : 0);
  const inc = Number(((_o = scan.angle_increment) !== null) && (_o !== void 0) ? _o : 0);
  const cells = new Set();
  forEachVisualizedScanRange(scan.ranges, (range, index) => {
    const r = Number(range);
    if (!Number.isFinite(r) || (r < min) || (r > max)) return;
    const angle = scanYaw + angleMin + (inc * index);
    const mapX = scanX + (Math.cos(angle) * r);
    const mapY = scanY + (Math.sin(angle) * r);
    const frameDx = mapX - frameX;
    const frameDy = mapY - frameY;
    const gridFrameX = (frameCos * frameDx) + (frameSin * frameDy);
    const gridFrameY = (-frameSin * frameDx) + (frameCos * frameDy);
    const originDx = gridFrameX - meta.originX;
    const originDy = gridFrameY - meta.originY;
    const localX = (originCos * originDx) + (originSin * originDy);
    const localY = (-originSin * originDx) + (originCos * originDy);
    const cellX = Math.floor(localX / meta.resolution);
    const cellY = Math.floor(localY / meta.resolution);
    if ((cellX < 0) || (cellX >= meta.width) || (cellY < 0) || (cellY >= meta.height)) return;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        const x = cellX + dx;
        const y = cellY + dy;
        if ((x < 0) || (x >= meta.width) || (y < 0) || (y >= meta.height)) continue;
        cells.add(x + (y * meta.width));
      }
    }
  });
  return cells;
}
